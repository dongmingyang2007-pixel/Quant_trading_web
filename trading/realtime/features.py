from __future__ import annotations

from typing import Any


def compute_features(bar: dict[str, Any], *, prev_close: float | None) -> dict[str, float | None]:
    close = bar.get("close")
    high = bar.get("high")
    low = bar.get("low")
    volume = bar.get("volume")
    bid = bar.get("bid")
    ask = bar.get("ask")
    bid_size = bar.get("bid_size")
    ask_size = bar.get("ask_size")
    mid = bar.get("mid")

    ret = None
    if prev_close and close:
        try:
            ret = float(close) / float(prev_close) - 1.0
        except (TypeError, ValueError, ZeroDivisionError):
            ret = None

    range_pct = None
    if close and high is not None and low is not None:
        try:
            range_pct = (float(high) - float(low)) / float(close)
        except (TypeError, ValueError, ZeroDivisionError):
            range_pct = None

    spread_bps = None
    if bid is not None and ask is not None and mid:
        try:
            spread_bps = ((float(ask) - float(bid)) / float(mid)) * 10000.0
        except (TypeError, ValueError, ZeroDivisionError):
            spread_bps = None

    imbalance = None
    if bid_size is not None and ask_size is not None:
        try:
            bid_val = float(bid_size)
            ask_val = float(ask_size)
            denom = bid_val + ask_val
            if denom > 0:
                imbalance = (bid_val - ask_val) / denom
        except (TypeError, ValueError):
            imbalance = None

    activity = None
    if bar.get("tick_count") is not None:
        try:
            activity = float(bar.get("tick_count"))
        except (TypeError, ValueError):
            activity = None

    return {
        "ret_5s": ret,
        "range_pct": range_pct,
        "spread_bps": spread_bps,
        "imbalance": imbalance,
        "volume": float(volume) if volume is not None else None,
        "activity": activity,
    }
