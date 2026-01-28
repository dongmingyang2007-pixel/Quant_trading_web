from __future__ import annotations

import json
import time

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from .realtime.market_stream import GROUP_NAME as MARKET_STREAM_GROUP, subscribe, unsubscribe
from .realtime.chart_stream import (
    GROUP_NAME as CHART_STREAM_GROUP,
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
        self._symbol: str | None = None

    async def connect(self) -> None:
        await self.channel_layer.group_add(CHART_STREAM_GROUP, self.channel_name)
        await self.accept()
        user = self.scope.get("user")
        user_id = str(user.id) if user and getattr(user, "is_authenticated", False) else None
        await sync_to_async(chart_subscribe)(user_id)

    async def disconnect(self, close_code: int) -> None:
        await self.channel_layer.group_discard(CHART_STREAM_GROUP, self.channel_name)
        if self._symbol:
            await sync_to_async(chart_remove_symbol)(self._symbol)
            self._symbol = None
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
        if action in {"subscribe", "set_symbol"} and symbol:
            if self._symbol and self._symbol != symbol:
                await sync_to_async(chart_remove_symbol)(self._symbol)
            self._symbol = symbol
            user = self.scope.get("user")
            user_id = str(user.id) if user and getattr(user, "is_authenticated", False) else None
            await sync_to_async(chart_add_symbol)(symbol, user_id=user_id)
        elif action in {"unsubscribe", "clear"}:
            if self._symbol:
                await sync_to_async(chart_remove_symbol)(self._symbol)
                self._symbol = None

    async def chart_trade(self, event: dict) -> None:
        payload = event.get("payload") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            payload = dict(event or {})
        payload.pop("type", None)
        symbol = str(payload.get("symbol") or "").upper()
        if self._symbol and symbol and symbol != self._symbol:
            return
        await self.send(text_data=json.dumps(payload))
