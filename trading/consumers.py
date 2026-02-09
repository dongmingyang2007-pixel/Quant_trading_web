from __future__ import annotations

import json
import time

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from .realtime.market_stream import GROUP_NAME as MARKET_STREAM_GROUP, subscribe, unsubscribe
from .realtime.chart_stream import (
    GROUP_PREFIX as CHART_STREAM_GROUP_PREFIX,
    MAX_STREAM_SYMBOLS as CHART_MAX_SYMBOLS,
    add_symbol as chart_add_symbol,
    remove_symbol as chart_remove_symbol,
    subscribe as chart_subscribe,
    unsubscribe as chart_unsubscribe,
)

MARKET_GLOBAL_GROUP = "market_global"


class MarketDataConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        await self.channel_layer.group_add(MARKET_STREAM_GROUP, self.channel_name)
        await self.channel_layer.group_add(MARKET_GLOBAL_GROUP, self.channel_name)
        await self.accept()
        user = self.scope.get("user")
        user_id = str(user.id) if user and getattr(user, "is_authenticated", False) else None
        await sync_to_async(subscribe)(user_id)

    async def disconnect(self, close_code: int) -> None:
        await self.channel_layer.group_discard(MARKET_STREAM_GROUP, self.channel_name)
        await self.channel_layer.group_discard(MARKET_GLOBAL_GROUP, self.channel_name)
        await sync_to_async(unsubscribe)()

    async def market_update(self, event: dict) -> None:
        payload = event.get("payload") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            payload = dict(event or {})
        payload.pop("type", None)
        if "server_ts" not in payload:
            payload["server_ts"] = time.time()
        await self.send(text_data=json.dumps(payload))


class MarketChartConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._symbols: set[str] = set()

    async def connect(self) -> None:
        await self.accept()
        user = self.scope.get("user")
        user_id = str(user.id) if user and getattr(user, "is_authenticated", False) else None
        await sync_to_async(chart_subscribe)(user_id)

    async def disconnect(self, close_code: int) -> None:
        if self._symbols:
            for symbol in list(self._symbols):
                await self.channel_layer.group_discard(
                    f"{CHART_STREAM_GROUP_PREFIX}-{symbol}", self.channel_name
                )
                await sync_to_async(chart_remove_symbol)(symbol)
            self._symbols.clear()
        await sync_to_async(chart_unsubscribe)()

    async def receive(self, text_data: str | None = None, bytes_data: bytes | None = None) -> None:
        if not text_data:
            return
        try:
            payload = json.loads(text_data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        action = str(payload.get("action") or "").lower()
        symbol = str(payload.get("symbol") or payload.get("ticker") or "").upper()
        if action == "ping":
            await self.send(text_data=json.dumps({"type": "pong", "server_ts": time.time()}))
            return
        if action in {"subscribe", "set_symbol"} and symbol:
            if symbol in self._symbols:
                return
            if len(self._symbols) >= CHART_MAX_SYMBOLS:
                await self.send(
                    text_data=json.dumps(
                        {
                            "type": "notice",
                            "source": "system",
                            "code": "subscription_limit",
                            "limit": CHART_MAX_SYMBOLS,
                            "message": "Too many subscriptions. Reduce symbols or switch to polling.",
                            "message_zh": "订阅数量过多，请减少订阅或改为轮询。",
                        }
                    )
                )
                return
            self._symbols.add(symbol)
            await self.channel_layer.group_add(f"{CHART_STREAM_GROUP_PREFIX}-{symbol}", self.channel_name)
            user = self.scope.get("user")
            user_id = str(user.id) if user and getattr(user, "is_authenticated", False) else None
            await sync_to_async(chart_add_symbol)(symbol, user_id=user_id)
        elif action in {"unsubscribe", "clear"}:
            if symbol:
                if symbol in self._symbols:
                    await self.channel_layer.group_discard(
                        f"{CHART_STREAM_GROUP_PREFIX}-{symbol}", self.channel_name
                    )
                    await sync_to_async(chart_remove_symbol)(symbol)
                    self._symbols.discard(symbol)
            else:
                for sym in list(self._symbols):
                    await self.channel_layer.group_discard(
                        f"{CHART_STREAM_GROUP_PREFIX}-{sym}", self.channel_name
                    )
                    await sync_to_async(chart_remove_symbol)(sym)
                self._symbols.clear()

    async def chart_trade(self, event: dict) -> None:
        payload = event.get("payload") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            payload = dict(event or {})
        payload.pop("type", None)
        symbol = str(payload.get("symbol") or "").upper()
        if symbol and self._symbols and symbol not in self._symbols:
            return
        trades = payload.get("trades")
        if not isinstance(trades, list):
            trade = {
                "price": payload.get("price"),
                "size": payload.get("size"),
                "ts": payload.get("ts"),
            }
            trades = [trade]
        normalized = {
            "source": "trade_batch",
            "symbol": symbol,
            "trades": trades,
            "server_ts": payload.get("server_ts") or time.time(),
        }
        await self.send(text_data=json.dumps(normalized))
