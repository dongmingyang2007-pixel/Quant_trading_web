from __future__ import annotations

from collections import deque
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Iterable

import pandas as pd
from django.conf import settings

from .cache_utils import build_cache_key, cache_memoize
from .network import get_requests_session, resolve_retry_config, retry_call_result
from trading.observability import record_metric
from .alpaca_data import fetch_stock_bars_frame, fetch_stock_snapshots, resolve_alpaca_credentials

try:  # optional dependency
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore

RATE_LIMIT_PER_WINDOW = int(os.environ.get("MARKET_FETCH_RATE_LIMIT", "10000") or 0)
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("MARKET_FETCH_RATE_WINDOW", "60") or 0)

_RATE_LOCK = threading.Lock()
_RATE_BUCKET: deque[float] = deque()
LOGGER = logging.getLogger(__name__)
_PARQUET_AVAILABLE: bool | None = None
_PARQUET_WARNED = False

DATA_CACHE_DIR: Path = getattr(settings, "DATA_CACHE_DIR", Path(settings.DATA_ROOT) / "data_cache")
DISK_CACHE_DIR = DATA_CACHE_DIR / "market_snapshots"
DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DISK_CACHE_TTL_SECONDS = int(getattr(settings, "MARKET_DISK_CACHE_TTL", 6 * 3600))

_ALPACA_INTERVAL_MAP = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "60m": "1Hour",
    "1h": "1Hour",
    "1d": "1Day",
}


def _has_parquet_engine() -> bool:
    global _PARQUET_AVAILABLE
    if _PARQUET_AVAILABLE is not None:
        return _PARQUET_AVAILABLE
    for module in ("pyarrow", "fastparquet"):
        try:
            __import__(module)
            _PARQUET_AVAILABLE = True
            return True
        except Exception:
            continue
    _PARQUET_AVAILABLE = False
    return False


def _warn_parquet_missing(context: str) -> None:
    global _PARQUET_WARNED
    if _PARQUET_WARNED:
        return
    LOGGER.warning(
        "Parquet engine missing; %s skipped. Install pyarrow>=14 or fastparquet.",
        context,
    )
    _PARQUET_WARNED = True


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _market_retry_config(timeout: int | None = None):
    return resolve_retry_config(
        timeout=timeout,
        retries=os.environ.get("MARKET_FETCH_MAX_RETRIES"),
        backoff=os.environ.get("MARKET_FETCH_RETRY_BACKOFF"),
        default_timeout=getattr(settings, "MARKET_DATA_TIMEOUT_SECONDS", None),
    )


def _empty_frame(value: object) -> bool:
    return not isinstance(value, pd.DataFrame) or value.empty


def _rate_limited() -> bool:
    """Simple in-memory rate limiter: returns True if allowed, False if over budget."""
    if RATE_LIMIT_PER_WINDOW <= 0 or RATE_LIMIT_WINDOW_SECONDS <= 0:
        return True
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    with _RATE_LOCK:
        # drop old
        while _RATE_BUCKET and _RATE_BUCKET[0] < window_start:
            _RATE_BUCKET.popleft()
        if len(_RATE_BUCKET) >= RATE_LIMIT_PER_WINDOW:
            LOGGER.warning("market_data rate limit hit: %s in %ss", len(_RATE_BUCKET), RATE_LIMIT_WINDOW_SECONDS)
            record_metric("market_rate_limited", count=len(_RATE_BUCKET), window=RATE_LIMIT_WINDOW_SECONDS)
            return False
        _RATE_BUCKET.append(now)
        return True


def _resolve_alpaca_timeframe(interval: str | None) -> str:
    if not interval:
        return "1Day"
    return _ALPACA_INTERVAL_MAP.get(interval.lower().strip(), "1Day")


def _period_to_timedelta(period: str | None) -> timedelta | None:
    if not period:
        return None
    text = str(period).strip().lower()
    if not text:
        return None
    if text.endswith("mo"):
        try:
            value = int(text[:-2])
        except ValueError:
            return None
        return timedelta(days=max(1, value) * 30)
    if text.endswith("y"):
        try:
            value = int(text[:-1])
        except ValueError:
            return None
        return timedelta(days=max(1, value) * 365)
    if text.endswith("d"):
        try:
            value = int(text[:-1])
        except ValueError:
            return None
        return timedelta(days=max(1, value))
    return None


def _resolve_alpaca_range(
    *,
    period: str | None,
    start: date | datetime | None,
    end: date | datetime | None,
    timeframe: str,
) -> tuple[date | datetime | None, date | datetime | None]:
    start_val = start
    end_val = end
    if not start_val and period:
        delta = _period_to_timedelta(period)
        if delta is not None:
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - delta
            start_val = start_dt
            end_val = end_dt
    if isinstance(end_val, date) and not isinstance(end_val, datetime):
        end_val = end_val + timedelta(days=1)
    return start_val, end_val


def fetch(
    symbols: Iterable[str],
    *,
    period: str | None = None,
    interval: str | None = None,
    start: date | None = None,
    end: date | None = None,
    fields: str = "Adj Close",
    cache: bool = True,
    ttl: int | None = None,
    timeout: int | None = None,
    cache_alias: str | None = None,
    user_id: str | None = None,
) -> pd.DataFrame:
    """Fetch market data with simple TTL caching and graceful fallback."""

    unique = [sym for sym in dict.fromkeys(symbols) if sym]
    cache_key = build_cache_key("market-data", sorted(unique), period or start, end, interval, fields)
    retry_config = _market_retry_config(timeout)
    session = get_requests_session(retry_config.timeout)

    def _cache_paths(key: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        base = digest[:24]
        return DISK_CACHE_DIR / f"{base}.parquet", DISK_CACHE_DIR / f"{base}.json"

    def _candidate_cache_paths(key: str) -> list[tuple[Path, Path]]:
        hashed = _cache_paths(key)
        legacy = (DISK_CACHE_DIR / f"{key}.parquet", DISK_CACHE_DIR / f"{key}.json")
        if hashed == legacy:
            return [hashed]
        return [hashed, legacy]

    def _load_disk_cache() -> pd.DataFrame:
        for path, meta_path in _candidate_cache_paths(cache_key):
            if not path.exists() or not meta_path.exists():
                continue
            try:
                if not _has_parquet_engine():
                    _warn_parquet_missing("market cache read")
                    return pd.DataFrame()
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                ts = float(meta.get("timestamp", 0))
                if time.time() - ts > DISK_CACHE_TTL_SECONDS:
                    try:
                        path.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    continue
                df = pd.read_parquet(path)
                if isinstance(df, pd.DataFrame):
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    def _persist_disk_cache(df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        try:
            if not _has_parquet_engine():
                _warn_parquet_missing("market cache write")
                return
            path, meta_path = _cache_paths(cache_key)
            meta = {"timestamp": time.time(), "symbols": unique, "fields": fields, "cache_key": cache_key}
            _write_parquet_atomic(df, path)
            _write_json_atomic(meta_path, meta)
        except Exception:
            pass

    if not unique:
        return _load_disk_cache()
    def _builder() -> pd.DataFrame:
        timeframe = _resolve_alpaca_timeframe(interval)
        alpaca_timeout = _market_retry_config(timeout).timeout
        alpaca_start, alpaca_end = _resolve_alpaca_range(
            period=period,
            start=start,
            end=end,
            timeframe=timeframe,
        )
        key_id, secret = resolve_alpaca_credentials(user_id=user_id)
        if key_id and secret:
            try:
                frame = fetch_stock_bars_frame(
                    unique,
                    start=alpaca_start,
                    end=alpaca_end,
                    timeframe=timeframe,
                    feed="sip",
                    adjustment="split",
                    user_id=user_id,
                    timeout=alpaca_timeout,
                )
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    return frame
            except Exception:
                pass
        if yf is None or not _rate_limited():
            return pd.DataFrame()

        def _download() -> pd.DataFrame:
            return yf.download(
                tickers=" ".join(unique),
                period=period,
                interval=interval,
                start=start,
                end=end,
                auto_adjust=False,
                group_by="ticker",
                threads=False,
                progress=False,
                timeout=retry_config.timeout,
                session=session,
            )

        try:
            data = retry_call_result(_download, config=retry_config, should_retry=_empty_frame)
        except Exception:
            return pd.DataFrame()
        if not isinstance(data, pd.DataFrame):
            return pd.DataFrame()
        return data

    if not cache:
        frame = _builder()
        if frame.empty:
            fallback = _load_disk_cache()
            return fallback
        _persist_disk_cache(frame)
        return frame

    cache_ttl = ttl if ttl is not None else getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300)
    result: pd.DataFrame | None = None
    try:
        result = cache_memoize(cache_key, _builder, cache_ttl, cache_alias=cache_alias)  # type: ignore[arg-type]
    except Exception:
        result = _builder()

    if isinstance(result, pd.DataFrame) and not result.empty:
        _persist_disk_cache(result)
        return result
    fallback = _load_disk_cache()
    return fallback if isinstance(fallback, pd.DataFrame) else pd.DataFrame()


def _extract_price_from_frame(frame: pd.DataFrame) -> dict[str, object]:
    """Internal helper: pick latest price/timestamp from a price frame."""
    try:
        col = "Adj Close" if "Adj Close" in frame.columns else "Close"
        if isinstance(frame.columns, pd.MultiIndex):
            level0 = frame.columns.get_level_values(0)
            col = "Adj Close" if "Adj Close" in level0 else level0[0]
            series = frame.xs(col, level=0, axis=1)
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
        else:
            series = frame[col] if col in frame.columns else frame.iloc[:, 0]
        series = series.dropna()
        if series.empty:
            return {}
        price = float(series.iloc[-1])
        ts = pd.to_datetime(series.index[-1]).to_pydatetime()
        return {"price": price, "as_of": ts}
    except Exception:
        return {}


def fetch_recent_window(
    symbols: list[str] | tuple[str, ...] | str,
    *,
    interval: str = "1d",
    limit: int = 120,
    user_id: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Batch-fetch a recent window of price data for one or more symbols.

    Returns a dict of symbol -> DataFrame limited to `limit` rows (most recent first).
    This is used by paper trading heartbeat to avoid N duplicate requests. Falls back to disk cache when rate limited.
    """
    if not symbols:
        return {}
    if isinstance(symbols, str):
        symbols = [symbols]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_symbols: list[str] = []
    for sym in symbols:
        if sym and sym not in seen:
            seen.add(sym)
            unique_symbols.append(sym)
    key_id, secret = resolve_alpaca_credentials(user_id=user_id)
    if key_id and secret:
        timeframe = _resolve_alpaca_timeframe(interval)
        try:
            data = fetch_stock_bars_frame(
                unique_symbols,
                timeframe=timeframe,
                limit=limit,
                feed="sip",
                adjustment="split",
                user_id=user_id,
            )
        except Exception:
            data = pd.DataFrame()
        if isinstance(data, pd.DataFrame) and not data.empty:
            result: dict[str, pd.DataFrame] = {}
            if isinstance(data.columns, pd.MultiIndex):
                for sym in unique_symbols:
                    try:
                        sub = data.xs(sym, level=1, axis=1).dropna().tail(limit)
                        if not sub.empty:
                            result[sym] = sub
                    except Exception:
                        continue
            else:
                try:
                    frame = data.dropna().tail(limit)
                    if not frame.empty:
                        result[unique_symbols[0]] = frame
                except Exception:
                    pass
            if result:
                return result
    if not unique_symbols or yf is None:  # pragma: no cover - network dependency
        return {}
    retry_config = _market_retry_config(None)
    session = get_requests_session(retry_config.timeout)
    def _resolve_period(interval: str, limit: int) -> str:
        """Choose a compact period for yf.download to avoid over-fetching."""
        interval = (interval or "1d").lower()
        if interval.endswith("m") or interval.endswith("h"):
            return "5d"  # intraday endpoints accept up to ~60d; 5d is enough for rolling signals here
        days = max(30, limit * 2)
        if days >= 365 * 10:
            return "10y"
        return f"{days}d"

    def _download_recent() -> pd.DataFrame:
        return yf.download(
            tickers=" ".join(unique_symbols),
            period=_resolve_period(interval, limit),
            interval=interval,
            progress=False,
            threads=False,
            timeout=retry_config.timeout,
            session=session,
        )

    try:
        data = retry_call_result(_download_recent, config=retry_config, should_retry=_empty_frame)
    except Exception:
        data = pd.DataFrame()
    result: dict[str, pd.DataFrame] = {}
    if not isinstance(data, pd.DataFrame) or data.empty:
        # degrade to daily interval as a fallback if intraday fails
        if interval.endswith("m") or interval.endswith("h"):
            try:
                degraded = fetch_recent_window(unique_symbols, interval="1d", limit=limit)
                if degraded:
                    LOGGER.warning("market_data degraded interval %s -> 1d for symbols=%s", interval, unique_symbols)
                    record_metric("market_fetch_degraded", interval_from=interval, interval_to="1d", symbols=len(unique_symbols))
                    return degraded
            except Exception:
                return {}
        return result
    if isinstance(data.columns, pd.MultiIndex):
        # data has MultiIndex (field, symbol)
        for sym in unique_symbols:
            try:
                sub = data.xs(sym, level=1, axis=1).dropna().tail(limit)
                if not sub.empty:
                    result[sym] = sub
            except Exception:
                continue
    else:
        # Single symbol only
        try:
            frame = data.dropna().tail(limit)
            if not frame.empty:
                result[unique_symbols[0]] = frame
        except Exception:
            pass
    return result


def fetch_latest_quote(
    symbol: str,
    *,
    interval: str = "1m",
    user_id: str | None = None,
) -> dict[str, object]:
    """Fetch the latest quote for a symbol. Returns {'price': float, 'as_of': datetime}."""
    if not symbol:
        return {}
    key_id, secret = resolve_alpaca_credentials(user_id=user_id)
    if key_id and secret:
        snapshots = fetch_stock_snapshots([symbol], user_id=user_id)
        snap = snapshots.get(symbol.upper()) if isinstance(snapshots, dict) else None
        if isinstance(snap, dict):
            latest_trade = snap.get("latestTrade") or snap.get("latest_trade") or {}
            latest_quote = snap.get("latestQuote") or snap.get("latest_quote") or {}
            daily_bar = snap.get("dailyBar") or snap.get("daily_bar") or {}
            prev_bar = snap.get("prevDailyBar") or snap.get("prev_daily_bar") or {}
            price = (
                latest_trade.get("p")
                or daily_bar.get("c")
                or prev_bar.get("c")
                or latest_quote.get("ap")
                or latest_quote.get("bp")
            )
            ts = latest_trade.get("t") or latest_quote.get("t") or daily_bar.get("t")
            if price is not None:
                as_of = pd.to_datetime(ts, utc=True, errors="coerce") if ts else None
                as_of_val = as_of.to_pydatetime() if as_of is not None and not pd.isna(as_of) else None
                return {"price": float(price), "as_of": as_of_val, "source": "alpaca"}
    if yf is None or not _rate_limited():  # pragma: no cover - network dependency
        return {}
    retry_config = _market_retry_config(None)
    session = get_requests_session(retry_config.timeout)
    source = "yf"
    def _download_latest() -> pd.DataFrame:
        return yf.download(
            tickers=symbol,
            period="5d",
            interval=interval,
            progress=False,
            threads=False,
            timeout=retry_config.timeout,
            session=session,
        )

    try:
        data = retry_call_result(_download_latest, config=retry_config, should_retry=_empty_frame)
        if isinstance(data, pd.DataFrame) and not data.empty:
            extracted = _extract_price_from_frame(data)
            if extracted:
                extracted["source"] = source
                return extracted
    except Exception:
        pass
    # Fallback to end-of-day fetch
    try:
        frame = fetch([symbol], period="5d", interval="1d", cache=False)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            extracted = _extract_price_from_frame(frame)
            if extracted:
                extracted["source"] = "fallback"
                return extracted
    except Exception:
        pass
    return {}
