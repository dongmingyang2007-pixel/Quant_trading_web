from __future__ import annotations

import threading
import time

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from .config import load_realtime_config
from .stream_registry import acquire_stream, release_stream
from .subscriptions import update_subscription_state
from .universe import build_universe
from .alpaca.stream import AlpacaStreamClient
from ..alpaca_data import fetch_stock_snapshots
from ..observability import record_metric

GROUP_NAME = "market-data"
MAX_STREAM_SYMBOLS = 50
EXTRA_SYMBOLS_MAX = 50
REFRESH_SECONDS = 300

_LOCK = threading.Lock()
_CLIENT: AlpacaStreamClient | None = None
_REF_COUNT = 0
_LAST_REFRESH_TS = 0.0
_LAST_SEND_TS: dict[str, float] = {}
_PREV_CLOSE: dict[str, float] = {}
_USER_ID: str | None = None
_EXTRA_SYMBOLS: list[str] = []
_CURRENT_SYMBOLS: list[str] = []


def _resolve_top_symbols(user_id: str | None) -> tuple[list[str], str]:
    config = load_realtime_config()
    entries = build_universe(config.universe, user_id=user_id, feed=config.engine.feed)
    symbols = [entry.symbol for entry in entries][:MAX_STREAM_SYMBOLS]
    return symbols, config.engine.feed


def _resolve_stream_symbols(user_id: str | None) -> tuple[list[str], str]:
    symbols, feed = _resolve_top_symbols(user_id)
    if not _EXTRA_SYMBOLS:
        return symbols, feed
    combined: list[str] = []
    seen: set[str] = set()
    for raw in _EXTRA_SYMBOLS + symbols:
        sym = str(raw or "").upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        combined.append(sym)
        if len(combined) >= MAX_STREAM_SYMBOLS:
            break
    return combined, feed


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


def _seed_missing_prev_close(symbols: list[str], *, user_id: str | None, feed: str) -> None:
    missing = [symbol for symbol in symbols if symbol not in _PREV_CLOSE]
    if missing:
        _seed_prev_close(missing, user_id=user_id, feed=feed)


def _apply_symbols(symbols: list[str]) -> None:
    global _CURRENT_SYMBOLS
    if _CLIENT is None:
        return
    if symbols == _CURRENT_SYMBOLS:
        return
    _CLIENT.set_symbols(symbols)
    _CURRENT_SYMBOLS = list(symbols)


def _ensure_stream(user_id: str | None) -> None:
    global _CLIENT, _LAST_REFRESH_TS, _USER_ID
    now = time.time()
    symbols, feed = _resolve_stream_symbols(user_id)
    if _CLIENT is None:
        if not acquire_stream("market_stream"):
            record_metric("market.ws.stream_locked")
            return
        _USER_ID = user_id
        _CLIENT = AlpacaStreamClient(
            user_id=user_id,
            feed=feed or "sip",
            stream_url=None,
            stream_trades=True,
            stream_quotes=True,
            reconnect_seconds=5,
            on_trade=_handle_trade,
            on_quote=_handle_quote,
        )
        if not _CLIENT.start():
            record_metric("market.ws.stream_failed")
            release_stream("market_stream")
            _CLIENT = None
            return
    if symbols:
        _apply_symbols(symbols)
    _seed_missing_prev_close(symbols, user_id=user_id, feed=feed or "sip")
    _LAST_REFRESH_TS = now


def subscribe(user_id: str | None) -> None:
    global _REF_COUNT
    with _LOCK:
        _REF_COUNT += 1
        _ensure_stream(user_id)


def request_symbol(symbol: str | None, *, user_id: str | None = None) -> None:
    if not symbol:
        return
    sym = str(symbol).upper()
    if not sym:
        return
    updated = update_subscription_state([sym], replace=False)
    with _LOCK:
        _EXTRA_SYMBOLS.clear()
        _EXTRA_SYMBOLS.extend(updated)
        _ensure_stream(user_id or _USER_ID)


def request_symbols(symbols: list[str] | None, *, user_id: str | None = None) -> None:
    if not symbols:
        return
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        sym = str(raw or "").upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        cleaned.append(sym)
        if len(cleaned) >= EXTRA_SYMBOLS_MAX:
            break
    if not cleaned:
        return
    updated = update_subscription_state(cleaned, replace=True)
    with _LOCK:
        _EXTRA_SYMBOLS.clear()
        _EXTRA_SYMBOLS.extend(updated)
        _ensure_stream(user_id or _USER_ID)


def unsubscribe() -> None:
    global _REF_COUNT, _CLIENT
    with _LOCK:
        _REF_COUNT = max(0, _REF_COUNT - 1)
        if _REF_COUNT == 0 and _CLIENT:
            _CLIENT.stop()
            _CLIENT = None
            release_stream("market_stream")
            _LAST_SEND_TS.clear()
            _PREV_CLOSE.clear()
            _EXTRA_SYMBOLS.clear()
            _CURRENT_SYMBOLS.clear()


def _emit_update(symbol: str, price_val: float, *, source: str) -> None:
    if not symbol or price_val <= 0:
        return
    prev_close = _PREV_CLOSE.get(symbol)
    now = time.time()
    last_ts = _LAST_SEND_TS.get(symbol, 0.0)
    if now - last_ts < 0.1:
        return
    _LAST_SEND_TS[symbol] = now
    change_pct = None
    if prev_close:
        change_pct = ((price_val / prev_close) - 1.0) * 100.0
    channel_layer = get_channel_layer()
    if not channel_layer:
        return
    payload = {
        "symbol": symbol,
        "price": price_val,
        "change_pct": change_pct,
        "server_ts": now,
        "source": source,
    }
    async_to_sync(channel_layer.group_send)(GROUP_NAME, {"type": "market_update", **payload})


def _handle_trade(symbol: str, price: float | None, _size: float | None, _ts: float | None) -> None:
    if not symbol or price is None:
        return
    symbol = str(symbol).upper()
    try:
        price_val = float(price)
    except (TypeError, ValueError):
        return
    _emit_update(symbol, price_val, source="trade")


def _handle_quote(
    symbol: str,
    bid: float | None,
    ask: float | None,
    _bid_size: float | None,
    _ask_size: float | None,
    _ts: float | None,
) -> None:
    if not symbol:
        return
    symbol = str(symbol).upper()
    price_val = None
    if bid is not None and ask is not None:
        price_val = (bid + ask) / 2.0
    elif ask is not None:
        price_val = ask
    elif bid is not None:
        price_val = bid
    if price_val is None:
        return
    _emit_update(symbol, float(price_val), source="quote")
