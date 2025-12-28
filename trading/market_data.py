from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable
from pathlib import Path
import json
import time
import hashlib
import os
import threading
import logging
from collections import deque

MAX_RETRIES = int(os.environ.get("MARKET_FETCH_MAX_RETRIES", "2") or 0)
RETRY_BACKOFF_SECONDS = float(os.environ.get("MARKET_FETCH_RETRY_BACKOFF", "1.0") or 0)
RATE_LIMIT_PER_WINDOW = int(os.environ.get("MARKET_FETCH_RATE_LIMIT", "120") or 0)
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("MARKET_FETCH_RATE_WINDOW", "60") or 0)

_RATE_LOCK = threading.Lock()
_RATE_BUCKET: deque[float] = deque()
LOGGER = logging.getLogger(__name__)

import pandas as pd

from django.conf import settings

from .cache_utils import build_cache_key, cache_memoize
from trading.observability import record_metric

try:  # optional dependency
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore

DATA_CACHE_DIR: Path = getattr(settings, "DATA_CACHE_DIR", Path(settings.DATA_ROOT) / "data_cache")
DISK_CACHE_DIR = DATA_CACHE_DIR / "market_snapshots"
DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DISK_CACHE_TTL_SECONDS = int(getattr(settings, "MARKET_DISK_CACHE_TTL", 6 * 3600))

def _with_retries(func, *args, **kwargs):
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - network/IO
            last_exc = exc
            if attempt >= MAX_RETRIES:
                break
            try:
                time.sleep(RETRY_BACKOFF_SECONDS)
            except Exception:
                pass
    if last_exc:
        raise last_exc
    return None


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
) -> pd.DataFrame:
    """Fetch market data with simple TTL caching and graceful fallback."""

    unique = [sym for sym in dict.fromkeys(symbols) if sym]
    cache_key = build_cache_key("market-data", sorted(unique), period or start, end, interval, fields)

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
            path, meta_path = _cache_paths(cache_key)
            df.to_parquet(path)
            meta = {"timestamp": time.time(), "symbols": unique, "fields": fields, "cache_key": cache_key}
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except Exception:
            pass

    if not unique:
        return _load_disk_cache()
    if yf is None or not _rate_limited():
        fallback = _load_disk_cache()
        return fallback

    def _builder() -> pd.DataFrame:
        try:
            data = _with_retries(
                yf.download,
                tickers=" ".join(unique),
                period=period,
                interval=interval,
                start=start,
                end=end,
                auto_adjust=False,
                group_by="ticker",
                threads=False,
                progress=False,
                timeout=timeout or getattr(settings, "MARKET_DATA_TIMEOUT_SECONDS", None),
            )
            if not isinstance(data, pd.DataFrame):
                return pd.DataFrame()
            return data
        except Exception:
            return pd.DataFrame()

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


def fetch_recent_window(symbols: list[str] | tuple[str, ...] | str, *, interval: str = "1d", limit: int = 120) -> dict[str, pd.DataFrame]:
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
    if not unique_symbols or yf is None:  # pragma: no cover - network dependency
        return {}
    def _resolve_period(interval: str, limit: int) -> str:
        """Choose a compact period for yf.download to avoid over-fetching."""
        interval = (interval or "1d").lower()
        if interval.endswith("m") or interval.endswith("h"):
            return "5d"  # intraday endpoints accept up to ~60d; 5d is enough for rolling signals here
        days = max(30, limit * 2)
        if days >= 365 * 10:
            return "10y"
        return f"{days}d"

    try:
        data = _with_retries(
            yf.download,
            tickers=" ".join(unique_symbols),
            period=_resolve_period(interval, limit),
            interval=interval,
            progress=False,
            threads=False,
        )
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


def fetch_latest_quote(symbol: str, *, interval: str = "1m") -> dict[str, object]:
    """Fetch the latest quote for a symbol. Returns {'price': float, 'as_of': datetime}."""
    if not symbol:
        return {}
    if yf is None or not _rate_limited():  # pragma: no cover - network dependency
        return {}
    source = "yf"
    try:
        data = _with_retries(
            yf.download,
            tickers=symbol,
            period="5d",
            interval=interval,
            progress=False,
            threads=False,
        )
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
