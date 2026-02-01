from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable
import math

from .realtime.bars import parse_timestamp


@dataclass(slots=True)
class TradePoint:
    ts: float
    price: float
    size: float


def _normalize_trade(trade: Any) -> TradePoint | None:
    if trade is None:
        return None
    if isinstance(trade, (list, tuple)) and len(trade) >= 2:
        ts_val = parse_timestamp(trade[0])
        price_val = trade[1]
        size_val = trade[2] if len(trade) > 2 else 0
    elif isinstance(trade, dict):
        ts_val = parse_timestamp(trade.get("t") or trade.get("timestamp") or trade.get("ts"))
        price_val = trade.get("p") or trade.get("price")
        size_val = trade.get("s") or trade.get("size") or trade.get("v") or 0
    else:
        return None
    if ts_val is None or price_val is None:
        return None
    try:
        price = float(price_val)
    except (TypeError, ValueError):
        return None
    try:
        size = float(size_val) if size_val is not None else 0.0
    except (TypeError, ValueError):
        size = 0.0
    return TradePoint(ts=float(ts_val), price=price, size=size)


def _ensure_monotonic_time(bars: list[dict[str, Any]], *, epsilon: float = 1e-6) -> list[dict[str, Any]]:
    if not bars:
        return bars
    last_time: float | None = None
    for bar in bars:
        time_val = bar.get("time")
        if time_val is None:
            continue
        try:
            time_num = float(time_val)
        except (TypeError, ValueError):
            continue
        if last_time is not None and time_num <= last_time:
            time_num = last_time + epsilon
            bar["time"] = time_num
        last_time = time_num
    return bars


def aggregate_trades_to_tick_bars(
    trades: Iterable[Any],
    *,
    ticks_per_bar: int,
    max_bars: int = 1000,
) -> list[dict[str, Any]]:
    if ticks_per_bar <= 0:
        return []
    normalized: list[TradePoint] = []
    for trade in trades:
        point = _normalize_trade(trade)
        if point is None:
            continue
        normalized.append(point)
    if not normalized:
        return []
    normalized.sort(key=lambda item: item.ts)

    bars: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    count = 0
    for point in normalized:
        if current is None or count >= ticks_per_bar:
            if current is not None:
                bars.append(current)
            current = {
                "time": point.ts,
                "open": point.price,
                "high": point.price,
                "low": point.price,
                "close": point.price,
                "volume": point.size,
                "trade_count": 1,
            }
            count = 1
            continue
        current["high"] = max(float(current.get("high") or point.price), point.price)
        current["low"] = min(float(current.get("low") or point.price), point.price)
        current["close"] = point.price
        current["time"] = point.ts
        current["volume"] = float(current.get("volume") or 0.0) + point.size
        current["trade_count"] = int(current.get("trade_count") or 0) + 1
        count += 1
    if current is not None:
        bars.append(current)
    if max_bars > 0 and len(bars) > max_bars:
        bars = bars[-max_bars:]
    return _ensure_monotonic_time(bars)


def aggregate_trades_to_time_bars(
    trades: Iterable[Any],
    *,
    interval_seconds: int,
    max_bars: int = 1000,
) -> list[dict[str, Any]]:
    interval = max(1, int(interval_seconds))
    normalized: list[TradePoint] = []
    for trade in trades:
        point = _normalize_trade(trade)
        if point is None:
            continue
        normalized.append(point)
    if not normalized:
        return []
    normalized.sort(key=lambda item: item.ts)

    bars: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_bucket: int | None = None

    for point in normalized:
        bucket = int(math.floor(point.ts / interval))
        if current is None or current_bucket != bucket:
            if current is not None:
                bars.append(current)
            current_bucket = bucket
            current = {
                "time": float(current_bucket * interval),
                "open": point.price,
                "high": point.price,
                "low": point.price,
                "close": point.price,
                "volume": point.size,
                "trade_count": 1,
            }
            continue
        current["high"] = max(float(current.get("high") or point.price), point.price)
        current["low"] = min(float(current.get("low") or point.price), point.price)
        current["close"] = point.price
        current["time"] = float(current_bucket * interval)
        current["volume"] = float(current.get("volume") or 0.0) + point.size
        current["trade_count"] = int(current.get("trade_count") or 0) + 1

    if current is not None:
        bars.append(current)
    if max_bars > 0 and len(bars) > max_bars:
        bars = bars[-max_bars:]
    return _ensure_monotonic_time(bars)


def format_trade_timestamp(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
