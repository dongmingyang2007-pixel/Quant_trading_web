from __future__ import annotations

import json

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from .realtime.market_stream import GROUP_NAME, subscribe, unsubscribe


class MarketDataConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        await self.channel_layer.group_add(GROUP_NAME, self.channel_name)
        await self.accept()
        user = self.scope.get("user")
        user_id = str(user.id) if user and getattr(user, "is_authenticated", False) else None
        await sync_to_async(subscribe)(user_id)

    async def disconnect(self, close_code: int) -> None:
        await self.channel_layer.group_discard(GROUP_NAME, self.channel_name)
        await sync_to_async(unsubscribe)()

    async def market_update(self, event: dict) -> None:
        payload = event.get("payload") or {}
        await self.send(text_data=json.dumps(payload))
