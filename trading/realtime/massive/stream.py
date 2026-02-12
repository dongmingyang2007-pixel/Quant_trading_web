from __future__ import annotations

import asyncio
import json
import queue
import random
import threading
from typing import Callable, Iterable

import websockets

from ...massive_data import resolve_massive_credentials, resolve_massive_ws_url
from ...observability import record_metric
from ..bars import parse_timestamp


class MassiveStreamClient:
    def __init__(
        self,
        *,
        user_id: str | None,
        stream_url: str | None,
        stream_trades: bool,
        stream_quotes: bool,
        reconnect_seconds: int,
        on_trade: Callable[[str, float | None, float | None, float | None], None] | None,
        on_quote: Callable[[str, float | None, float | None, float | None, float | None, float | None], None] | None,
        on_status: Callable[[str, str], None] | None = None,
    ) -> None:
        self.user_id = user_id
        self.stream_url = stream_url or resolve_massive_ws_url(user_id=user_id)
        self.stream_trades = stream_trades
        self.stream_quotes = stream_quotes
        self.reconnect_seconds = max(1, int(reconnect_seconds))
        self.on_trade = on_trade
        self.on_quote = on_quote
        self.on_status = on_status
        self._api_key, _ = resolve_massive_credentials(user_id=user_id)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._command_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._symbols: set[str] = set()
        self._connected = False

    def start(self) -> bool:
        if self._thread:
            return True
        if not self._api_key:
            record_metric("realtime.massive.stream.credentials_missing")
            return False
        self._thread = threading.Thread(target=self._run, name="massive-stream", daemon=True)
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
                record_metric("realtime.massive.stream.error", error=str(exc))
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
            record_metric("realtime.massive.stream.connected", url=self.stream_url)
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
        record_metric("realtime.massive.stream.disconnected", url=self.stream_url)
        if self.on_status:
            self.on_status("disconnected", self.stream_url)

    async def _send_auth(self, ws) -> None:
        payload = {"action": "auth", "params": self._api_key}
        await ws.send(json.dumps(payload))

    def _encode_subscription(self, symbols: set[str]) -> str:
        channels: list[str] = []
        for symbol in sorted(symbols):
            if self.stream_trades:
                channels.append(f"T.{symbol}")
            if self.stream_quotes:
                channels.append(f"Q.{symbol}")
        return ",".join(channels)

    async def _send_subscribe(self, ws, add: set[str], remove: set[str]) -> None:
        if add and (self.stream_trades or self.stream_quotes):
            params = self._encode_subscription(add)
            if params:
                await ws.send(json.dumps({"action": "subscribe", "params": params}))
        if remove and (self.stream_trades or self.stream_quotes):
            params = self._encode_subscription(remove)
            if params:
                await ws.send(json.dumps({"action": "unsubscribe", "params": params}))

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
                event_type = str(event.get("ev") or event.get("type") or "").lower()
                if event_type == "status":
                    status = str(event.get("status") or "status").lower()
                    message = str(event.get("message") or event.get("msg") or "")
                    record_metric("realtime.massive.stream.status", status=status, message=message)
                    if self.on_status:
                        self.on_status(status, message)
                    continue
                if event_type in {"t", "trade"}:
                    self._handle_trade(event)
                elif event_type in {"q", "quote"}:
                    self._handle_quote(event)

    def _handle_trade(self, event: dict[str, object]) -> None:
        if not self.on_trade:
            return
        symbol = str(event.get("sym") or event.get("symbol") or event.get("S") or "").upper()
        price = event.get("p") or event.get("price")
        size = event.get("s") or event.get("size")
        ts = parse_timestamp(event.get("t") or event.get("timestamp") or event.get("sip_timestamp"))
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
        symbol = str(event.get("sym") or event.get("symbol") or event.get("S") or "").upper()
        bid = event.get("bp") or event.get("bid_price") or event.get("bid")
        ask = event.get("ap") or event.get("ask_price") or event.get("ask")
        bid_size = event.get("bs") or event.get("bid_size")
        ask_size = event.get("as") or event.get("ask_size")
        ts = parse_timestamp(event.get("t") or event.get("timestamp") or event.get("sip_timestamp"))
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
