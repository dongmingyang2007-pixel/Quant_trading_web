from __future__ import annotations

import asyncio
import json
import os
import threading
import time

from django.test import SimpleTestCase

from trading.realtime.alpaca.stream import AlpacaStreamClient
from trading.realtime.massive.stream import MassiveStreamClient


try:  # pragma: no cover - optional dependency
    import websockets

    WEBSOCKETS_AVAILABLE = True
except Exception:  # pragma: no cover
    WEBSOCKETS_AVAILABLE = False


class _TestStreamServer:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread: threading.Thread | None = None
        self.server = None
        self.port = None

    async def _handler(self, websocket):
        async for message in websocket:
            payload = json.loads(message)
            if payload.get("action") == "auth":
                await websocket.send(json.dumps([{"T": "success", "msg": "authenticated"}]))
            if payload.get("action") == "subscribe":
                await websocket.send(
                    json.dumps(
                        [
                            {
                                "T": "t",
                                "S": "AAPL",
                                "p": 123.4,
                                "s": 1,
                                "t": "2024-01-01T00:00:00Z",
                            }
                        ]
                    )
                )

    def start(self):
        def _run():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._start_server())
            self.ready.set()
            self.loop.run_forever()

        self.thread = threading.Thread(target=_run, name="ws-test-server", daemon=True)
        self.thread.start()
        self.ready.wait(timeout=2)

    async def _start_server(self):
        self.server = await websockets.serve(self._handler, "127.0.0.1", 0)
        socket = self.server.sockets[0]
        self.port = socket.getsockname()[1]

    def stop(self):
        if not self.server or not self.loop.is_running():
            return

        async def _shutdown():
            self.server.close()
            await self.server.wait_closed()
            self.loop.stop()

        asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)
        if self.thread:
            self.thread.join(timeout=2)


class _TestMassiveStreamServer:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread: threading.Thread | None = None
        self.server = None
        self.port = None

    async def _handler(self, websocket):
        async for message in websocket:
            payload = json.loads(message)
            action = payload.get("action")
            if action == "auth":
                await websocket.send(
                    json.dumps([{"ev": "status", "status": "auth_success", "message": "authenticated"}])
                )
            if action == "subscribe":
                await websocket.send(
                    json.dumps(
                        [
                            {
                                "ev": "T",
                                "sym": "AAPL",
                                "p": 234.5,
                                "s": 2,
                                "t": "2024-01-01T00:00:00Z",
                            }
                        ]
                    )
                )

    def start(self):
        def _run():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._start_server())
            self.ready.set()
            self.loop.run_forever()

        self.thread = threading.Thread(target=_run, name="massive-ws-test-server", daemon=True)
        self.thread.start()
        self.ready.wait(timeout=2)

    async def _start_server(self):
        self.server = await websockets.serve(self._handler, "127.0.0.1", 0)
        socket = self.server.sockets[0]
        self.port = socket.getsockname()[1]

    def stop(self):
        if not self.server or not self.loop.is_running():
            return

        async def _shutdown():
            self.server.close()
            await self.server.wait_closed()
            self.loop.stop()

        asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)
        if self.thread:
            self.thread.join(timeout=2)


class RealtimeStreamTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        self._old_key = os.environ.get("ALPACA_API_KEY_ID")
        self._old_secret = os.environ.get("ALPACA_API_SECRET_KEY")
        self._old_massive_key = os.environ.get("MASSIVE_API_KEY")
        os.environ["ALPACA_API_KEY_ID"] = "test_key"
        os.environ["ALPACA_API_SECRET_KEY"] = "test_secret"
        os.environ["MASSIVE_API_KEY"] = "massive_test_key"

    def tearDown(self):
        if self._old_key is None:
            os.environ.pop("ALPACA_API_KEY_ID", None)
        else:
            os.environ["ALPACA_API_KEY_ID"] = self._old_key
        if self._old_secret is None:
            os.environ.pop("ALPACA_API_SECRET_KEY", None)
        else:
            os.environ["ALPACA_API_SECRET_KEY"] = self._old_secret
        if self._old_massive_key is None:
            os.environ.pop("MASSIVE_API_KEY", None)
        else:
            os.environ["MASSIVE_API_KEY"] = self._old_massive_key
        super().tearDown()

    def test_stream_receives_trade(self):
        if not WEBSOCKETS_AVAILABLE:
            self.skipTest("websockets not installed")
        server = _TestStreamServer()
        server.start()
        self.addCleanup(server.stop)
        self.assertIsNotNone(server.port)

        received = threading.Event()
        payload = {}

        def _on_trade(symbol, price, size, ts):
            payload["symbol"] = symbol
            payload["price"] = price
            payload["size"] = size
            payload["ts"] = ts
            received.set()

        client = AlpacaStreamClient(
            user_id=None,
            feed="iex",
            stream_url=f"ws://127.0.0.1:{server.port}",
            stream_trades=True,
            stream_quotes=False,
            reconnect_seconds=1,
            on_trade=_on_trade,
            on_quote=None,
        )
        self.assertTrue(client.start())
        client.set_symbols(["AAPL"])
        self.assertTrue(received.wait(timeout=3))
        client.stop()
        if client._thread:
            client._thread.join(timeout=2)

        self.assertEqual(payload.get("symbol"), "AAPL")
        self.assertEqual(payload.get("price"), 123.4)

    def test_stream_connects_with_fakepaca(self):
        fake_url = os.environ.get("ALPACA_TEST_WS_URL")
        if not fake_url:
            self.skipTest("ALPACA_TEST_WS_URL not set")
        if not WEBSOCKETS_AVAILABLE:
            self.skipTest("websockets not installed")

        client = AlpacaStreamClient(
            user_id=None,
            feed="iex",
            stream_url=fake_url,
            stream_trades=True,
            stream_quotes=False,
            reconnect_seconds=1,
            on_trade=None,
            on_quote=None,
        )
        self.assertTrue(client.start())
        client.set_symbols(["AAPL"])
        connected = False
        for _ in range(30):
            if client.is_connected():
                connected = True
                break
            time.sleep(0.1)
        client.stop()
        if client._thread:
            client._thread.join(timeout=2)
        self.assertTrue(connected)

    def test_massive_stream_receives_trade(self):
        if not WEBSOCKETS_AVAILABLE:
            self.skipTest("websockets not installed")
        server = _TestMassiveStreamServer()
        server.start()
        self.addCleanup(server.stop)
        self.assertIsNotNone(server.port)

        received = threading.Event()
        payload = {}

        def _on_trade(symbol, price, size, ts):
            payload["symbol"] = symbol
            payload["price"] = price
            payload["size"] = size
            payload["ts"] = ts
            received.set()

        client = MassiveStreamClient(
            user_id=None,
            stream_url=f"ws://127.0.0.1:{server.port}",
            stream_trades=True,
            stream_quotes=False,
            reconnect_seconds=1,
            on_trade=_on_trade,
            on_quote=None,
        )
        self.assertTrue(client.start())
        client.set_symbols(["AAPL"])
        self.assertTrue(received.wait(timeout=3))
        client.stop()
        if client._thread:
            client._thread.join(timeout=2)

        self.assertEqual(payload.get("symbol"), "AAPL")
        self.assertEqual(payload.get("price"), 234.5)
