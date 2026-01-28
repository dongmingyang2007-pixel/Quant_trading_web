from __future__ import annotations

import threading
import time
from typing import Iterable

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from .alpaca.stream import AlpacaStreamClient
from .config import load_realtime_config
from .stream_registry import acquire_stream, release_stream
from ..observability import record_metric

GROUP_NAME = "market-chart"
MAX_STREAM_SYMBOLS = 20

_LOCK = threading.Lock()
_CLIENT: AlpacaStreamClient | None = None
_REF_COUNT = 0
_USER_ID: str | None = None
_SYMBOL_REFS: dict[str, int] = {}
_SYMBOL_ORDER: list[str] = []
_CURRENT_SYMBOLS: list[str] = []


def _normalize_symbol(value: str | None) -> str | None:
    if not value:
        return None
    symbol = str(value).strip().upper()
    return symbol or None


def _resolve_active_symbols() -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for sym in _SYMBOL_ORDER:
        if sym in seen:
            continue
        if _SYMBOL_REFS.get(sym, 0) <= 0:
            continue
        seen.add(sym)
        symbols.append(sym)
        if len(symbols) >= MAX_STREAM_SYMBOLS:
            break
    return symbols


def _apply_symbols(symbols: list[str]) -> None:
    global _CURRENT_SYMBOLS
    if _CLIENT is None:
        return
    if symbols == _CURRENT_SYMBOLS:
        return
    _CLIENT.set_symbols(symbols)
    _CURRENT_SYMBOLS = list(symbols)


def _ensure_stream(user_id: str | None) -> None:
    global _CLIENT, _USER_ID
    symbols = _resolve_active_symbols()
    if _CLIENT is None:
        if not acquire_stream("market_chart_stream"):
            record_metric("market.chart.stream_locked")
            return
        config = load_realtime_config()
        _USER_ID = user_id
        _CLIENT = AlpacaStreamClient(
            user_id=user_id,
            feed=config.engine.feed or "sip",
            stream_url=None,
            stream_trades=True,
            stream_quotes=False,
            reconnect_seconds=5,
            on_trade=_handle_trade,
            on_quote=None,
        )
        if not _CLIENT.start():
            record_metric("market.chart.stream_failed")
            release_stream("market_chart_stream")
            _CLIENT = None
            return
    _apply_symbols(symbols)


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
            release_stream("market_chart_stream")
            _SYMBOL_REFS.clear()
            _SYMBOL_ORDER.clear()
            _CURRENT_SYMBOLS.clear()


def add_symbol(symbol: str | None, *, user_id: str | None = None) -> None:
    sym = _normalize_symbol(symbol)
    if not sym:
        return
    with _LOCK:
        if sym not in _SYMBOL_REFS:
            _SYMBOL_REFS[sym] = 1
            _SYMBOL_ORDER.append(sym)
        else:
            _SYMBOL_REFS[sym] += 1
        _ensure_stream(user_id or _USER_ID)


def remove_symbol(symbol: str | None) -> None:
    sym = _normalize_symbol(symbol)
    if not sym:
        return
    with _LOCK:
        if sym not in _SYMBOL_REFS:
            return
        _SYMBOL_REFS[sym] = max(0, _SYMBOL_REFS[sym] - 1)
        if _SYMBOL_REFS[sym] <= 0:
            _SYMBOL_REFS.pop(sym, None)
        _apply_symbols(_resolve_active_symbols())


def _handle_trade(symbol: str, price: float | None, size: float | None, ts: float | None) -> None:
    if not symbol or price is None:
        return
    sym = str(symbol).upper()
    if sym not in _CURRENT_SYMBOLS:
        return
    try:
        price_val = float(price)
    except (TypeError, ValueError):
        return
    size_val: float | None = None
    if size is not None:
        try:
            size_val = float(size)
        except (TypeError, ValueError):
            size_val = None
    timestamp = float(ts) if ts is not None else time.time()
    channel_layer = get_channel_layer()
    if not channel_layer:
        return
    payload = {
        "symbol": sym,
        "price": price_val,
        "size": size_val,
        "ts": timestamp,
        "source": "trade",
    }
    async_to_sync(channel_layer.group_send)(GROUP_NAME, {"type": "chart_trade", **payload})
