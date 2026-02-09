from __future__ import annotations

from collections import deque
from threading import Lock
import time

from django.conf import settings

MAX_TRADES_PER_SYMBOL = int(getattr(settings, "MARKET_CHART_STORE_MAX_TRADES", 60000))
MAX_TRADE_AGE_SECONDS = int(getattr(settings, "MARKET_CHART_STORE_MAX_AGE_SECONDS", 30 * 60))

_LOCK = Lock()
_TRADES: dict[str, deque[dict[str, float]]] = {}
_LATEST: dict[str, dict[str, float]] = {}


def _normalize_symbol(symbol: str | None) -> str | None:
    if not symbol:
        return None
    normalized = str(symbol).strip().upper()
    return normalized or None


def add_trade(symbol: str | None, price: float, size: float | None, ts: float | None) -> None:
    sym = _normalize_symbol(symbol)
    if not sym:
        return
    if price is None:
        return
    try:
        price_val = float(price)
    except (TypeError, ValueError):
        return
    size_val = 0.0
    if size is not None:
        try:
            size_val = float(size)
        except (TypeError, ValueError):
            size_val = 0.0
    timestamp = float(ts) if ts is not None else time.time()
    entry = {"ts": timestamp, "price": price_val, "size": size_val}
    with _LOCK:
        bucket = _TRADES.get(sym)
        if bucket is None:
            bucket = deque(maxlen=MAX_TRADES_PER_SYMBOL)
            _TRADES[sym] = bucket
        bucket.append(entry)
        _LATEST[sym] = entry
        _prune_bucket(bucket, timestamp)


def get_trades(
    symbol: str | None,
    *,
    start: float | None = None,
    end: float | None = None,
    limit: int | None = None,
) -> list[dict[str, float]]:
    sym = _normalize_symbol(symbol)
    if not sym:
        return []
    now_ts = time.time()
    with _LOCK:
        bucket = _TRADES.get(sym)
        if not bucket:
            return []
        _prune_bucket(bucket, now_ts)
        if not bucket:
            return []
        use_limit = limit is not None and limit > 0
        filtered = deque(maxlen=limit if use_limit else None)
        for item in bucket:
            ts = item.get("ts")
            if ts is None:
                continue
            if start is not None and ts < start:
                continue
            if end is not None and ts > end:
                continue
            filtered.append(item)
    return list(filtered)


def get_latest_trade(symbol: str | None) -> dict[str, float] | None:
    sym = _normalize_symbol(symbol)
    if not sym:
        return None
    with _LOCK:
        latest = _LATEST.get(sym)
    return dict(latest) if latest else None


def _prune_bucket(bucket: deque[dict[str, float]], now_ts: float) -> None:
    if not bucket:
        return
    cutoff = now_ts - MAX_TRADE_AGE_SECONDS
    while bucket:
        ts = bucket[0].get("ts")
        if ts is None or ts >= cutoff:
            break
        bucket.popleft()
