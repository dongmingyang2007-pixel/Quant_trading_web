from __future__ import annotations

import json

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from .realtime.market_stream import GROUP_NAME as MARKET_STREAM_GROUP, subscribe, unsubscribe

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
        await self.send(text_data=json.dumps(event))
