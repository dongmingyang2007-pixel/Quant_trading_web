from __future__ import annotations

import threading
import time
from typing import Any

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from .config import load_realtime_config
from .universe import build_universe
from .alpaca.stream import AlpacaStreamClient
from ..alpaca_data import fetch_stock_snapshots
from ..observability import record_metric

GROUP_NAME = "market-data"
MAX_STREAM_SYMBOLS = 50
REFRESH_SECONDS = 300

_LOCK = threading.Lock()
_CLIENT: AlpacaStreamClient | None = None
_REF_COUNT = 0
_LAST_REFRESH_TS = 0.0
_LAST_SEND_TS: dict[str, float] = {}
_PREV_CLOSE: dict[str, float] = {}
_USER_ID: str | None = None


def _resolve_top_symbols(user_id: str | None) -> tuple[list[str], str]:
    config = load_realtime_config()
    entries = build_universe(config.universe, user_id=user_id, feed=config.engine.feed)
    symbols = [entry.symbol for entry in entries][:MAX_STREAM_SYMBOLS]
    return symbols, config.engine.feed


def _seed_prev_close(symbols: list[str], *, user_id: str | None, feed: str) -> None:
    if not symbols:
        return
    snapshots = fetch_stock_snapshots(symbols, feed=feed, user_id=user_id)
    for symbol, snapshot in (snapshots or {}).items():
        if not isinstance(snapshot, dict):
            continue
        daily = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
        prev = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar") or {}
        prev_close = prev.get("c") or prev.get("close") or 0.0
        if not prev_close:
            prev_close = daily.get("o") or daily.get("open") or daily.get("c") or daily.get("close") or 0.0
        try:
            prev_close_val = float(prev_close)
        except (TypeError, ValueError):
            continue
        if prev_close_val > 0:
            _PREV_CLOSE[str(symbol).upper()] = prev_close_val


def _ensure_stream(user_id: str | None) -> None:
    global _CLIENT, _LAST_REFRESH_TS, _USER_ID
    now = time.time()
    if _CLIENT and (now - _LAST_REFRESH_TS) < REFRESH_SECONDS:
        return
    symbols, feed = _resolve_top_symbols(user_id)
    _seed_prev_close(symbols, user_id=user_id, feed=feed)
    if _CLIENT is None:
        _USER_ID = user_id
        _CLIENT = AlpacaStreamClient(
            user_id=user_id,
            feed=feed or "sip",
            stream_url=None,
            stream_trades=True,
            stream_quotes=False,
            reconnect_seconds=5,
            on_trade=_handle_trade,
            on_quote=None,
        )
        if not _CLIENT.start():
            record_metric("market.ws.stream_failed")
            _CLIENT = None
            return
    if symbols:
        _CLIENT.set_symbols(symbols)
    _LAST_REFRESH_TS = now


def subscribe(user_id: str | None) -> None:
    global _REF_COUNT
    with _LOCK:
        _REF_COUNT += 1
        _ensure_stream(user_id)


def unsubscribe() -> None:
    global _REF_COUNT, _CLIENT
    with _LOCK:
        _REF_COUNT = max(0, _REF_COUNT - 1)
        if _REF_COUNT == 0 and _CLIENT:
            _CLIENT.stop()
            _CLIENT = None
            _LAST_SEND_TS.clear()
            _PREV_CLOSE.clear()


def _handle_trade(symbol: str, price: float | None, _size: float | None, _ts: float | None) -> None:
    if not symbol or price is None:
        return
    symbol = str(symbol).upper()
    prev_close = _PREV_CLOSE.get(symbol)
    if not prev_close:
        return
    try:
        price_val = float(price)
    except (TypeError, ValueError):
        return
    if price_val <= 0:
        return
    now = time.time()
    last_ts = _LAST_SEND_TS.get(symbol, 0.0)
    if now - last_ts < 0.5:
        return
    _LAST_SEND_TS[symbol] = now
    change_pct = ((price_val / prev_close) - 1.0) * 100.0
    channel_layer = get_channel_layer()
    if not channel_layer:
        return
    payload = {
        "symbol": symbol,
        "price": price_val,
        "change_pct": change_pct,
    }
    async_to_sync(channel_layer.group_send)(
        GROUP_NAME,
        {"type": "market.update", "payload": payload},
    )
