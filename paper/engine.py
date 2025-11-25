from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, Tuple
import os
import logging

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from paper.models import PaperTradingSession, PaperTrade
from trading.strategies import StrategyInput, run_quant_pipeline, QuantStrategyError
from trading.market_data import fetch_latest_quote, fetch_recent_window
from trading.observability import record_metric
from django.conf import settings

try:  # Optional dependency used only for cache extraction helpers
    import pandas as pd
except Exception:  # pragma: no cover - fallback when pandas unavailable
    pd = None

LOGGER = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = int(os.environ.get("PAPER_TRADING_INTERVAL_SECONDS", 300))
DEFAULT_INITIAL_CASH = Decimal(os.environ.get("PAPER_TRADING_INITIAL_CASH", "100000"))
DEFAULT_SLIPPAGE_BPS = float(os.environ.get("PAPER_TRADING_SLIPPAGE_BPS", "5.0"))
DEFAULT_COMMISSION_BPS = float(os.environ.get("PAPER_TRADING_COMMISSION_BPS", "8.0"))
FALLBACK_SIGNAL_TTL_SECONDS = int(os.environ.get("PAPER_TRADING_SIGNAL_TTL", "3600") or 0)
FALLBACK_SIGNAL_MAX_AGE_SECONDS = int(os.environ.get("PAPER_TRADING_SIGNAL_MAX_AGE", str(86400 * 3)) or 0)
ALLOW_LIGHT_HEARTBEAT = os.environ.get("PAPER_TRADING_ALLOW_LIGHT", "1") in {"1", "true", "True"}


class PaperTradingError(RuntimeError):
    pass


def _parse_date_safe(value: Any, fallback: date | None = None) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass
    return fallback or date.today()


def _build_strategy_from_config(config: Dict[str, Any], user_id: str) -> StrategyInput:
    payload = {k: v for k, v in dict(config).items() if not str(k).startswith("__")}
    payload["user_id"] = user_id
    payload["request_id"] = payload.get("request_id") or f"paper-{user_id}"
    end = _parse_date_safe(payload.get("end_date"), date.today())
    start = _parse_date_safe(payload.get("start_date"), end - timedelta(days=365))
    if start >= end:
        start = end - timedelta(days=365)
    payload["start_date"] = start
    payload["end_date"] = end
    params = StrategyInput(**payload)
    # 防止历史样本不足，确保最少 lookback
    min_days = max(
        params.train_window + params.test_window + 60,
        params.long_window * 5,
        540,  # 至少 ~18 个月
    )
    span = (params.end_date - params.start_date).days
    if span < min_days:
        params = params.__class__(
            **{
                **asdict(params),
                "start_date": params.end_date - timedelta(days=min_days),
            }
        )
    return params


def _ensure_curve(curve: list[dict[str, Any]], entry: dict[str, Any], max_points: int = 500) -> list[dict[str, Any]]:
    curve = (curve or []) + [entry]
    if len(curve) > max_points:
        curve = curve[-max_points:]
    return curve


def _extract_last_signal(config: dict[str, Any] | None) -> tuple[str | None, str | None]:
    last = (config or {}).get("__last_signal") or {}
    return last.get("source"), last.get("at")


def _record_skip(session: PaperTradingSession, reason: str, payload: dict[str, Any] | None = None) -> None:
    """Persist the latest跳过原因，便于前端展示。"""
    cfg = dict(session.config or {})
    cfg["__last_skip"] = {
        "reason": reason,
        "at": timezone.now().isoformat(),
        **(payload or {}),
    }
    PaperTradingSession.objects.filter(pk=session.pk).update(config=cfg, updated_at=timezone.now())


def _apply_risk_guard(weight: float, stats: dict[str, Any] | None, confidence: float | None) -> tuple[float, dict[str, Any] | None]:
    """Downscale target weight based on volatility/VaR/confidence."""
    if stats is None:
        stats = {}
    factor = 1.0
    reasons: list[str] = []
    vol = float(stats.get("volatility") or stats.get("annual_volatility") or 0.0)
    var95 = float(stats.get("var_95") or 0.0)
    cvar95 = float(stats.get("cvar_95") or 0.0)
    if vol > 0.35:
        factor *= 0.6
        reasons.append(f"volatility {vol:.2f}")
    elif vol > 0.25:
        factor *= 0.8
        reasons.append(f"volatility {vol:.2f}")
    if var95 > 0.08 or cvar95 > 0.1:
        factor *= 0.8
        reasons.append(f"VaR/CVaR {max(var95,cvar95):.2f}")
    if confidence is not None:
        if confidence < 0.4:
            factor *= 0.6
            reasons.append(f"confidence {confidence:.2f}")
        elif confidence < 0.6:
            factor *= 0.85
            reasons.append(f"confidence {confidence:.2f}")
    if factor >= 0.999:
        return weight, None
    return weight * factor, {"factor": round(factor, 3), "reasons": reasons}


def _to_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return default


def _serialize_strategy_config(params: StrategyInput) -> dict[str, Any]:
    payload = asdict(params)
    for key in ("start_date", "end_date"):
        val = payload.get(key)
        if isinstance(val, (date, datetime)):
            payload[key] = val.isoformat()
    return payload


def _compute_account_value(positions: Dict[str, float], price_lookup: Dict[str, float], cash: Decimal | float) -> Decimal:
    total = _to_decimal(cash)
    for symbol, qty in positions.items():
        price = price_lookup.get(symbol)
        if price is None:
            continue
        total += _to_decimal(qty) * _to_decimal(price)
    return total


def _session_interval(session: PaperTradingSession) -> str:
    cfg = session.config or {}
    interval = cfg.get("interval")
    if interval:
        return str(interval)
    # Map interval_seconds (if present) to yfinance-friendly interval strings
    try:
        sec = int(cfg.get("interval_seconds") or session.interval_seconds or 86400)
    except Exception:
        sec = 86400
    if sec <= 60:
        return "1m"
    if sec <= 300:
        return "5m"
    if sec <= 900:
        return "15m"
    if sec <= 1800:
        return "30m"
    if sec <= 3600:
        return "1h"
    return "1d"


def _extract_quote_from_cache(ticker: str, interval: str, cache: dict[str, object] | None) -> dict[str, object]:
    if not cache:
        return {}
    data = cache.get((ticker, interval)) or cache.get(ticker)
    if data is None:
        return {}
    # Allow both {'price': ..} dicts and DataFrame snapshots
    if isinstance(data, dict) and "price" in data:
        return data
    if pd is not None and isinstance(data, pd.DataFrame) and not data.empty:
        try:
            col = "Adj Close" if "Adj Close" in data.columns else "Close"
            series = data[col] if col in data.columns else data.iloc[:, 0]
            series = series.dropna()
            if series.empty:
                return {}
            price = float(series.iloc[-1])
            ts = series.index[-1]
            try:
                ts = ts.to_pydatetime()
            except Exception:
                pass
            quote: dict[str, object] = {"price": price, "as_of": ts}
            if "Volume" in data.columns:
                vol = data["Volume"].fillna(0)
                adv = (vol * series).rolling(20, min_periods=5).mean()
                if not adv.empty:
                    quote["adv"] = float(adv.iloc[-1])
                    quote["adv_median"] = float(adv.median())
                    quote["volume"] = float(vol.iloc[-1])
            return quote
        except Exception:
            return {}
    return {}


def create_session(user, params: StrategyInput, *, name: str | None = None, initial_cash: Decimal | float | None = None) -> PaperTradingSession:
    cash = Decimal(str(initial_cash if initial_cash is not None else DEFAULT_INITIAL_CASH))
    session = PaperTradingSession.objects.create(
        user=user,
        name=name or f"{params.ticker.upper()} 模拟盘",
        ticker=params.ticker.upper(),
        benchmark=params.benchmark_ticker or "",
        status="running",
        config=_serialize_strategy_config(params),
        current_cash=cash,
        initial_cash=cash,
        last_equity=cash,
        interval_seconds=max(30, int(getattr(settings, "PAPER_TRADING_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS))),
        next_run_at=None,
    )
    return session


def _rebalance_position(
    session: PaperTradingSession,
    *,
    price: float,
    target_weight: float,
    as_of: datetime,
    slippage_bps: float,
    commission_bps: float,
) -> Tuple[Dict[str, float], Decimal, list[PaperTrade]]:
    positions = dict(session.current_positions or {})
    cash = _to_decimal(session.current_cash)
    price_dec = _to_decimal(price)
    slip_rate = Decimal(str(slippage_bps / 10000.0))
    comm_rate = Decimal(str(commission_bps / 10000.0))

    # Use Decimal for all monetary math to avoid rounding drift
    price_lookup = {session.ticker: float(price_dec)}
    equity = _to_decimal(_compute_account_value(positions, price_lookup, float(cash)))
    target_notional = equity * Decimal(str(target_weight))
    current_qty = _to_decimal(positions.get(session.ticker, 0.0))
    desired_qty = Decimal("0") if price_dec == 0 else target_notional / price_dec
    trade_qty = desired_qty - current_qty
    trades: list[PaperTrade] = []

    if abs(trade_qty) > Decimal("1e-9"):
        is_buy = trade_qty > 0
        side = "buy" if is_buy else "sell"
        exec_price = price_dec * (Decimal("1") + slip_rate if is_buy else Decimal("1") - slip_rate)
        notional = trade_qty * exec_price
        commission = abs(notional) * comm_rate
        cash -= notional
        cash -= commission
        new_qty = current_qty + trade_qty
        if abs(new_qty) < Decimal("1e-9"):
            positions.pop(session.ticker, None)
            new_qty = Decimal("0")
        else:
            positions[session.ticker] = float(new_qty.quantize(Decimal("0.00000001")))
        trade = PaperTrade(
            session=session,
            symbol=session.ticker,
            side=side,
            quantity=trade_qty.copy_abs(),
            price=exec_price,
            notional=abs(notional),
            metadata={
                "target_weight": target_weight,
                "equity_before": float(equity),
                "slippage_bps": slippage_bps,
                "commission_bps": commission_bps,
                "commission": float(commission),
            },
            executed_at=as_of,
        )
        trades.append(trade)
        equity = _to_decimal(_compute_account_value(positions, price_lookup, float(cash)))

    return positions, cash, trades


def process_session(
    session: PaperTradingSession,
    *,
    now: datetime | None = None,
    price_cache: dict[str, object] | None = None,
) -> dict[str, Any]:
    if session.status != "running":
        record_metric("paper_session_skipped", session_id=str(session.session_id), reason="not_running")
        return {"skipped": True, "reason": "not_running"}

    now = now or timezone.now()
    if timezone.is_naive(now):  # normalize for comparisons
        now = timezone.make_aware(now, timezone.get_current_timezone())
    if session.next_run_at and session.next_run_at > now:
        record_metric("paper_session_skipped", session_id=str(session.session_id), reason="not_due")
        return {"skipped": True, "reason": "not_due"}

    params = _build_strategy_from_config(session.config or {}, str(session.user_id))
    # 每次以最新日期作为回测终点，以便拿到最新信号
    params = params.__class__(**{**asdict(params), "end_date": date.today()})

    target_weight = 0.0
    signal_at = now
    signal_source = "fresh"
    signal_confidence: float | None = None

    stats: dict[str, Any] | None = None
    try:
        # Optional light heartbeat: reuse cached signal if still fresh
        if ALLOW_LIGHT_HEARTBEAT and session.config.get("__last_signal"):
            last_signal = session.config["__last_signal"]
            try:
                ts = datetime.fromisoformat(last_signal.get("at"))
                if timezone.is_naive(ts):
                    ts = timezone.make_aware(ts, timezone.get_current_timezone())
            except Exception:
                ts = None
            ttl = max(60, FALLBACK_SIGNAL_TTL_SECONDS) if FALLBACK_SIGNAL_TTL_SECONDS > 0 else 0
            if ts and (ttl == 0 or (now - ts).total_seconds() <= ttl):
                target_weight = float(last_signal.get("weight", 0.0))
                signal_at = ts
                signal_source = "light_cached"
                signal_confidence = float(last_signal.get("confidence", 0.0)) if "confidence" in last_signal else None
                record_metric("paper_signal_source", session_id=str(session.session_id), source="light_cached")
            else:
                raise QuantStrategyError("cached_signal_expired")
        else:
            raise QuantStrategyError("no_light_mode")
    except QuantStrategyError:
        try:
            result = run_quant_pipeline(params)
            stats = result.get("stats") or {}
            recent_rows = result.get("recent_rows") or []
            if not recent_rows:
                raise PaperTradingError("策略未返回有效的 recent_rows，无法生成交易信号。")
            last_bar = recent_rows[-1]
            target_weight = float(last_bar.get("position", 0.0))
            signal_at = now
            signal_source = "fresh"
            signal_confidence = float(last_bar.get("confidence")) if last_bar.get("confidence") is not None else None
            record_metric("paper_signal_source", session_id=str(session.session_id), source="fresh")
        except Exception as exc:
            last_signal = (session.config or {}).get("__last_signal") or {}
            if last_signal:
                try:
                    ts = datetime.fromisoformat(last_signal.get("at"))
                    if timezone.is_naive(ts):
                        ts = timezone.make_aware(ts, timezone.get_current_timezone())
                except Exception:
                    ts = None
                age = (now - ts).total_seconds() if ts else None
                max_age = FALLBACK_SIGNAL_MAX_AGE_SECONDS or FALLBACK_SIGNAL_TTL_SECONDS
                if ts and (max_age == 0 or (age is not None and age <= max_age)):
                    target_weight = float(last_signal.get("weight", 0.0))
                    signal_at = ts
                    signal_source = "fallback_cached"
                    signal_confidence = float(last_signal.get("confidence", 0.0)) if "confidence" in last_signal else None
                    record_metric("paper_signal_source", session_id=str(session.session_id), source="fallback_cached")
                else:
                    record_metric("paper_signal_source", session_id=str(session.session_id), source="failure", error=str(exc))
                    raise
            else:
                record_metric("paper_signal_source", session_id=str(session.session_id), source="failure", error=str(exc))
                raise

    data_interval = _session_interval(session)
    adjusted_weight = target_weight
    risk_guard = None
    if signal_source == "fresh":
        adjusted_weight, risk_guard = _apply_risk_guard(target_weight, stats, signal_confidence)

    last_signal_payload = {"weight": float(adjusted_weight), "at": signal_at.isoformat(), "source": signal_source}
    if signal_confidence is not None:
        last_signal_payload["confidence"] = float(signal_confidence)
    if risk_guard:
        last_signal_payload["risk_guard"] = risk_guard

    quote = _extract_quote_from_cache(params.ticker, data_interval, price_cache)
    if not quote:
        quote = fetch_latest_quote(params.ticker, interval=data_interval)
    price = quote.get("price")
    if price is None:
        record_metric("paper_session_skipped", session_id=str(session.session_id), reason="quote_unavailable")
        _record_skip(session, "quote_unavailable")
        return {"skipped": True, "reason": "quote_unavailable"}
    as_of = quote.get("as_of") or now
    if isinstance(as_of, str):
        try:
            as_of = datetime.fromisoformat(as_of)
        except ValueError:
            as_of = now
    if timezone.is_naive(as_of):
        as_of = timezone.make_aware(as_of, timezone.get_current_timezone())

    # Liquidity guard: skip rebalance when latest ADV is extremely low to avoid unrealistic fills
    try:
        adv = float(quote.get("adv") or 0.0)
        adv_median = float(quote.get("adv_median") or 0.0)
        volume = float(quote.get("volume") or 0.0)
    except Exception:
        adv = adv_median = volume = 0.0
    if adv_median > 0 and (adv <= adv_median * 0.1 or volume <= 0):
        record_metric(
            "paper_session_skipped",
            session_id=str(session.session_id),
            reason="illiquid",
            adv=adv,
            adv_median=adv_median,
            volume=volume,
        )
        _record_skip(session, "illiquid", {"adv": adv, "adv_median": adv_median, "volume": volume})
        return {"skipped": True, "reason": "illiquid", "adv": adv, "adv_median": adv_median, "volume": volume}

    with transaction.atomic():
        locked = PaperTradingSession.objects.select_for_update().get(pk=session.pk)
        positions, cash, trades = _rebalance_position(
            locked,
            price=float(price),
            target_weight=adjusted_weight,
            as_of=now,
            slippage_bps=getattr(params, "slippage_bps", DEFAULT_SLIPPAGE_BPS),
            commission_bps=getattr(params, "transaction_cost_bps", DEFAULT_COMMISSION_BPS),
        )
        price_lookup = {locked.ticker: float(price)}
        equity = _compute_account_value(positions, price_lookup, cash)
        curve_entry = {"ts": now.isoformat(), "equity": float(equity.quantize(Decimal("0.01")))}
        cfg = dict(locked.config or {})
        cfg["__last_signal"] = last_signal_payload

        locked.current_positions = positions
        locked.current_cash = Decimal(str(round(cash, 2)))
        locked.last_equity = Decimal(str(round(equity, 2)))
        locked.equity_curve = _ensure_curve(list(locked.equity_curve or []), curve_entry)
        locked.last_run_at = now
        locked.next_run_at = now + timedelta(seconds=max(30, locked.interval_seconds or DEFAULT_INTERVAL_SECONDS))
        locked.config = cfg
        locked.save(
            update_fields=[
                "current_positions",
                "current_cash",
                "last_equity",
                "equity_curve",
                "last_run_at",
                "next_run_at",
                "config",
                "updated_at",
            ]
        )
        if trades:
            PaperTrade.objects.bulk_create(trades)

    payload = {
        "session_id": str(session.session_id),
        "ticker": session.ticker,
        "target_weight": adjusted_weight,
        "price": float(price),
        "equity": float(equity),
        "trades": [{"side": t.side, "quantity": float(t.quantity), "price": float(t.price)} for t in trades],
        "signal_source": signal_source,
        "last_signal_at": signal_at.isoformat(),
        "signal_confidence": signal_confidence,
        "risk_guard": risk_guard,
    }
    record_metric(
        "paper_session_processed",
        session_id=str(session.session_id),
        ticker=session.ticker,
        status=session.status,
        trades=len(trades),
        interval_seconds=locked.interval_seconds if "locked" in locals() else session.interval_seconds,
    )
    return payload


def _infer_window_limit(session: PaperTradingSession) -> int:
    cfg = session.config or {}
    def _as_int(val: Any, default: int) -> int:
        try:
            return int(val)
        except Exception:
            return default
    long_win = _as_int(cfg.get("long_window"), 20)
    rsi = _as_int(cfg.get("rsi_period"), 14)
    train = _as_int(cfg.get("train_window"), 252)
    test = _as_int(cfg.get("test_window"), 21)
    # cover最大指标周期，保底 120，封顶 1000
    return max(120, min(1000, long_win * 5, rsi * 8, train + test + 60))


def _build_price_cache(sessions: list[PaperTradingSession]) -> dict[str, object]:
    """Batch-fetch a recent window for all tickers/intervals to reduce duplicate requests."""
    # group by interval to avoid mixing resolutions
    grouped: dict[str, list[str]] = {}
    for s in sessions:
        if not s.ticker:
            continue
        interval = _session_interval(s)
        grouped.setdefault(interval, []).append(s.ticker)
    if not grouped:
        return {}
    cache: dict[str, object] = {}
    for interval, tickers in grouped.items():
        unique = list({t for t in tickers if t})
        if not unique:
            continue
        limit = max(_infer_window_limit(s) for s in sessions if _session_interval(s) == interval)
        try:
            frames = fetch_recent_window(unique, interval=interval, limit=limit)
        except Exception:
            frames = {}
        if isinstance(frames, dict):
            for ticker, frame in frames.items():
                if frame is None:
                    continue
                cache[(ticker, interval)] = frame
    LOGGER.info("Paper trading price cache built: %s groups, %s symbols", len(grouped), len(cache))
    record_metric(
        "paper_price_cache_built",
        groups=len(grouped),
        symbols=len(cache),
        intervals=",".join(grouped.keys()) if grouped else "",
    )
    return cache


def run_pending_sessions(limit: int = 20, price_cache: dict[str, object] | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    now = timezone.now()
    qs = (
        PaperTradingSession.objects.filter(status="running")
        .order_by("next_run_at")
        .select_related("user")[:limit]
    )
    if not qs:
        LOGGER.info("Paper trading: no running sessions to process.")
        return [{"skipped": True, "reason": "no_running_sessions"}]

    price_cache = price_cache or _build_price_cache(list(qs))

    for session in qs:
        try:
            result = process_session(session, now=now, price_cache=price_cache)
            results.append(result)
        except PaperTradingError as exc:
            LOGGER.warning("Paper trading failed for session %s: %s", session.pk, exc)
            PaperTradingSession.objects.filter(pk=session.pk).update(status="error", ended_at=timezone.now())
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Paper trading unexpected error for session %s", session.pk)
            PaperTradingSession.objects.filter(pk=session.pk).update(status="error", ended_at=timezone.now())
    return results


def serialize_session(session: PaperTradingSession, *, include_details: bool = False, trades_limit: int = 20) -> dict[str, Any]:
    sig_source, sig_at = _extract_last_signal(session.config or {})
    last_signal = (session.config or {}).get("__last_signal") or {}
    last_skip = (session.config or {}).get("__last_skip") or {}
    payload = {
        "session_id": str(session.session_id),
        "name": session.name,
        "ticker": session.ticker,
        "benchmark": session.benchmark,
        "status": session.status,
        "current_cash": float(session.current_cash),
        "last_equity": float(session.last_equity),
        "initial_cash": float(session.initial_cash),
        "positions": session.current_positions or {},
        "interval_seconds": session.interval_seconds,
        "last_run_at": session.last_run_at.isoformat() if session.last_run_at else None,
        "next_run_at": session.next_run_at.isoformat() if session.next_run_at else None,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "config": session.config or {},
        "pnl_pct": float(session.last_equity) / float(session.initial_cash or 1) - 1 if session.initial_cash else None,
        "signal_source": sig_source,
        "last_signal_at": sig_at,
        "signal_confidence": last_signal.get("confidence"),
        "risk_guard": last_signal.get("risk_guard"),
        "last_skip": last_skip or None,
    }
    if include_details:
        curve = session.equity_curve or []
        payload["equity_curve"] = curve[-200:] if len(curve) > 200 else curve
        trades = session.trades.all().order_by("-executed_at")[:trades_limit]
        last_trade_at = trades[0].executed_at.isoformat() if trades else None
        payload["recent_trades"] = [
            {
                "side": trade.side,
                "symbol": trade.symbol,
                "quantity": float(trade.quantity),
                "price": float(trade.price),
                "notional": float(trade.notional),
                "executed_at": trade.executed_at.isoformat(),
                "metadata": trade.metadata or {},
            }
            for trade in trades
        ]
        payload["last_trade_at"] = last_trade_at
    return payload
