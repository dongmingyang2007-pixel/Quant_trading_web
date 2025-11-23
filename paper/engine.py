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
from trading.strategies import StrategyInput, run_quant_pipeline
from trading.market_data import fetch_latest_quote

LOGGER = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = int(os.environ.get("PAPER_TRADING_INTERVAL_SECONDS", 300))
DEFAULT_INITIAL_CASH = Decimal(os.environ.get("PAPER_TRADING_INITIAL_CASH", "100000"))


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
    payload = dict(config)
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


def _serialize_strategy_config(params: StrategyInput) -> dict[str, Any]:
    payload = asdict(params)
    for key in ("start_date", "end_date"):
        val = payload.get(key)
        if isinstance(val, (date, datetime)):
            payload[key] = val.isoformat()
    return payload


def _compute_account_value(positions: Dict[str, float], price_lookup: Dict[str, float], cash: float) -> float:
    total = cash
    for symbol, qty in positions.items():
        price = price_lookup.get(symbol)
        if price is None:
            continue
        total += float(qty) * float(price)
    return float(total)


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


def _rebalance_position(session: PaperTradingSession, *, price: float, target_weight: float, as_of: datetime) -> Tuple[Dict[str, float], float, list[PaperTrade]]:
    positions = dict(session.current_positions or {})
    cash = float(session.current_cash)
    price_lookup = {session.ticker: price}
    equity = _compute_account_value(positions, price_lookup, cash)
    target_notional = equity * target_weight
    current_qty = float(positions.get(session.ticker, 0.0))
    desired_qty = target_notional / price if price else 0.0
    trade_qty = desired_qty - current_qty
    trades: list[PaperTrade] = []

    if abs(trade_qty) > 1e-6:
        side = "buy" if trade_qty > 0 else "sell"
        notional = trade_qty * price
        cash -= notional
        positions[session.ticker] = desired_qty
        trade = PaperTrade(
            session=session,
            symbol=session.ticker,
            side=side,
            quantity=Decimal(str(trade_qty)),
            price=Decimal(str(price)),
            notional=Decimal(str(abs(notional))),
            metadata={"target_weight": target_weight, "equity_before": equity},
            executed_at=as_of,
        )
        trades.append(trade)
        equity = _compute_account_value(positions, price_lookup, cash)

    return positions, cash, trades


def process_session(session: PaperTradingSession, *, now: datetime | None = None) -> dict[str, Any]:
    if session.status != "running":
        return {"skipped": True, "reason": "not_running"}

    now = now or timezone.now()
    if timezone.is_naive(now):  # normalize for comparisons
        now = timezone.make_aware(now, timezone.get_current_timezone())
    if session.next_run_at and session.next_run_at > now:
        return {"skipped": True, "reason": "not_due"}

    params = _build_strategy_from_config(session.config or {}, str(session.user_id))
    # 每次以最新日期作为回测终点，以便拿到最新信号
    params = params.__class__(**{**asdict(params), "end_date": date.today()})
    result = run_quant_pipeline(params)
    recent_rows = result.get("recent_rows") or []
    if not recent_rows:
        raise PaperTradingError("策略未返回有效的 recent_rows，无法生成交易信号。")
    last_bar = recent_rows[-1]
    target_weight = float(last_bar.get("position", 0.0))

    quote = fetch_latest_quote(params.ticker)
    price = quote.get("price")
    if price is None:
        raise PaperTradingError(f"未获取到 {params.ticker} 最新价格。")
    as_of = quote.get("as_of") or now
    if isinstance(as_of, str):
        try:
            as_of = datetime.fromisoformat(as_of)
        except ValueError:
            as_of = now
    if timezone.is_naive(as_of):
        as_of = timezone.make_aware(as_of, timezone.get_current_timezone())

    with transaction.atomic():
        locked = PaperTradingSession.objects.select_for_update().get(pk=session.pk)
        positions, cash, trades = _rebalance_position(locked, price=float(price), target_weight=target_weight, as_of=now)
        price_lookup = {locked.ticker: float(price)}
        equity = _compute_account_value(positions, price_lookup, cash)
        curve_entry = {"ts": now.isoformat(), "equity": round(equity, 2)}

        locked.current_positions = positions
        locked.current_cash = Decimal(str(round(cash, 2)))
        locked.last_equity = Decimal(str(round(equity, 2)))
        locked.equity_curve = _ensure_curve(list(locked.equity_curve or []), curve_entry)
        locked.last_run_at = now
        locked.next_run_at = now + timedelta(seconds=max(30, locked.interval_seconds or DEFAULT_INTERVAL_SECONDS))
        locked.save(
            update_fields=[
                "current_positions",
                "current_cash",
                "last_equity",
                "equity_curve",
                "last_run_at",
                "next_run_at",
                "updated_at",
            ]
        )
        if trades:
            PaperTrade.objects.bulk_create(trades)

    return {
        "session_id": str(session.session_id),
        "ticker": session.ticker,
        "target_weight": target_weight,
        "price": float(price),
        "equity": float(equity),
        "trades": [{"side": t.side, "quantity": float(t.quantity), "price": float(t.price)} for t in trades],
    }


def run_pending_sessions(limit: int = 20) -> list[dict[str, Any]]:
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

    for session in qs:
        try:
            result = process_session(session, now=now)
            results.append(result)
        except PaperTradingError as exc:
            LOGGER.warning("Paper trading failed for session %s: %s", session.pk, exc)
            PaperTradingSession.objects.filter(pk=session.pk).update(status="error", ended_at=timezone.now())
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Paper trading unexpected error for session %s", session.pk)
            PaperTradingSession.objects.filter(pk=session.pk).update(status="error", ended_at=timezone.now())
    return results


def serialize_session(session: PaperTradingSession, *, include_details: bool = False, trades_limit: int = 20) -> dict[str, Any]:
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
