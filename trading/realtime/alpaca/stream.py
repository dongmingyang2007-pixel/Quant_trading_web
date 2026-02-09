from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import random
import threading
from typing import Callable, Iterable

import websockets

from ...alpaca_data import resolve_alpaca_data_credentials
from ...observability import record_metric
from ..bars import parse_timestamp

LOGGER = logging.getLogger(__name__)


def _default_stream_url(feed: str) -> str:
    base = os.environ.get("ALPACA_DATA_WS_URL")
    if base:
        return base
    suffix = feed.lower().strip() if feed else "sip"
    return f"wss://stream.data.alpaca.markets/v2/{suffix}"


class AlpacaStreamClient:
    def __init__(
        self,
        *,
        user_id: str | None,
        feed: str,
        stream_url: str | None,
        stream_trades: bool,
        stream_quotes: bool,
        reconnect_seconds: int,
        on_trade: Callable[[str, float | None, float | None, float | None], None] | None,
        on_quote: Callable[[str, float | None, float | None, float | None, float | None, float | None], None] | None,
        on_status: Callable[[str, str], None] | None = None,
    ) -> None:
        self.user_id = user_id
        self.feed = feed or "sip"
        self.stream_url = stream_url or _default_stream_url(self.feed)
        self.stream_trades = stream_trades
        self.stream_quotes = stream_quotes
        self.reconnect_seconds = max(1, int(reconnect_seconds))
        self.on_trade = on_trade
        self.on_quote = on_quote
        self.on_status = on_status
        self._key_id, self._secret = resolve_alpaca_data_credentials(user_id=user_id)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._command_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._symbols: set[str] = set()
        self._connected = False
        self._fallback_feed_attempted = False

    def start(self) -> bool:
        if self._thread:
            return True
        if not self._key_id or not self._secret:
            record_metric("realtime.stream.credentials_missing")
            return False
        self._thread = threading.Thread(target=self._run, name="alpaca-stream", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        self._command_queue.put(("stop", None))

    def set_symbols(self, symbols: Iterable[str]) -> None:
        normalized = {str(sym).strip().upper() for sym in symbols if str(sym).strip()}
        self._symbols = normalized
        self._command_queue.put(("symbols", normalized))

    def is_connected(self) -> bool:
        return self._connected

    def _run(self) -> None:
        asyncio.run(self._run_loop())

    async def _run_loop(self) -> None:
        delay = self.reconnect_seconds
        max_delay = max(self.reconnect_seconds * 6, 30)
        while not self._stop_event.is_set():
            try:
                await self._connect_once()
                delay = self.reconnect_seconds
            except Exception as exc:
                record_metric("realtime.stream.error", error=str(exc))
                delay = min(max_delay, max(self.reconnect_seconds, delay * 2))
            if not self._stop_event.is_set():
                jitter = random.random() * 0.3 * delay
                await asyncio.sleep(delay + jitter)

    async def _connect_once(self) -> None:
        async with websockets.connect(
            self.stream_url,
            ping_interval=20,
            ping_timeout=20,
            max_queue=512,
        ) as ws:
            self._connected = True
            record_metric("realtime.stream.connected", url=self.stream_url)
            if self.on_status:
                self.on_status("connected", self.stream_url)
            await self._send_auth(ws)
            await self._send_subscribe(ws, self._symbols, set())
            recv_task = asyncio.create_task(self._recv_loop(ws))
            cmd_task = asyncio.create_task(self._command_loop(ws))
            done, pending = await asyncio.wait(
                [recv_task, cmd_task],
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in pending:
                task.cancel()
            for task in done:
                if task.exception():
                    raise task.exception()
        self._connected = False
        record_metric("realtime.stream.disconnected", url=self.stream_url)
        if self.on_status:
            self.on_status("disconnected", self.stream_url)

    async def _send_auth(self, ws) -> None:
        payload = {"action": "auth", "key": self._key_id, "secret": self._secret}
        await ws.send(json.dumps(payload))

    async def _send_subscribe(self, ws, add: set[str], remove: set[str]) -> None:
        if add and (self.stream_trades or self.stream_quotes):
            payload: dict[str, object] = {"action": "subscribe"}
            if self.stream_trades:
                payload["trades"] = sorted(add)
            if self.stream_quotes:
                payload["quotes"] = sorted(add)
            await ws.send(json.dumps(payload))
        if remove and (self.stream_trades or self.stream_quotes):
            payload = {"action": "unsubscribe"}
            if self.stream_trades:
                payload["trades"] = sorted(remove)
            if self.stream_quotes:
                payload["quotes"] = sorted(remove)
            await ws.send(json.dumps(payload))

    async def _command_loop(self, ws) -> None:
        current = set(self._symbols)
        while not self._stop_event.is_set():
            cmd, payload = await asyncio.to_thread(self._command_queue.get)
            if cmd == "stop":
                break
            if cmd == "symbols":
                new_symbols = set(payload) if isinstance(payload, set) else set()
                add = new_symbols - current
                remove = current - new_symbols
                current = new_symbols
                await self._send_subscribe(ws, add, remove)

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            events = payload if isinstance(payload, list) else [payload]
            for event in events:
                if not isinstance(event, dict):
                    continue
                event_type = str(event.get("T") or event.get("t") or event.get("type") or "").lower()
                if event_type in {"success", "error", "subscription"}:
                    message = str(event.get("msg") or "")
                    record_metric("realtime.stream.status", status=event_type, message=message)
                    if self.on_status:
                        self.on_status(event_type, message)
                    if event_type in {"error", "subscription"} and self._maybe_request_feed_fallback(message):
                        raise RuntimeError("alpaca_feed_fallback")
                    continue
                if event_type in {"t", "trade"}:
                    self._handle_trade(event)
                elif event_type in {"q", "quote"}:
                    self._handle_quote(event)

    def _handle_trade(self, event: dict[str, object]) -> None:
        if not self.on_trade:
            return
        symbol = str(event.get("S") or event.get("symbol") or "").upper()
        price = event.get("p") or event.get("price")
        size = event.get("s") or event.get("size") or event.get("v")
        ts = parse_timestamp(event.get("t") or event.get("timestamp"))
        if not symbol:
            return
        try:
            price_val = float(price) if price is not None else None
        except (TypeError, ValueError):
            price_val = None
        try:
            size_val = float(size) if size is not None else None
        except (TypeError, ValueError):
            size_val = None
        self.on_trade(symbol, price_val, size_val, ts)

    def _handle_quote(self, event: dict[str, object]) -> None:
        if not self.on_quote:
            return
        symbol = str(event.get("S") or event.get("symbol") or "").upper()
        bid = event.get("bp") or event.get("bid_price") or event.get("bid")
        ask = event.get("ap") or event.get("ask_price") or event.get("ask")
        bid_size = event.get("bs") or event.get("bid_size")
        ask_size = event.get("as") or event.get("ask_size")
        ts = parse_timestamp(event.get("t") or event.get("timestamp"))
        if not symbol:
            return
        try:
            bid_val = float(bid) if bid is not None else None
        except (TypeError, ValueError):
            bid_val = None
        try:
            ask_val = float(ask) if ask is not None else None
        except (TypeError, ValueError):
            ask_val = None
        try:
            bid_size_val = float(bid_size) if bid_size is not None else None
        except (TypeError, ValueError):
            bid_size_val = None
        try:
            ask_size_val = float(ask_size) if ask_size is not None else None
        except (TypeError, ValueError):
            ask_size_val = None
        self.on_quote(symbol, bid_val, ask_val, bid_size_val, ask_size_val, ts)

    def _maybe_request_feed_fallback(self, message: str) -> bool:
        if self._fallback_feed_attempted:
            return False
        if str(self.feed).lower() != "sip":
            return False
        lowered = str(message or "").lower()
        if not lowered:
            return False
        triggers = ("not authorized", "not permitted", "subscription", "forbidden", "unauthorized", "insufficient")
        if not any(trigger in lowered for trigger in triggers):
            return False
        self._fallback_feed_attempted = True
        self.feed = "iex"
        self.stream_url = _default_stream_url("iex")
        record_metric("realtime.stream.fallback", from_feed="sip", to_feed="iex")
        if self.on_status:
            self.on_status("fallback", "SIP feed denied, switched to IEX.")
        return True
