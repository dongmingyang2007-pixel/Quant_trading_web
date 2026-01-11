from __future__ import annotations

import os
import tempfile
from unittest import mock

from django.core.management import call_command
from django.test import SimpleTestCase

from trading.realtime.storage import read_state


class RealtimeRefreshAssetsTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self._old_state = os.environ.get("REALTIME_STATE_DIR")
        os.environ["REALTIME_STATE_DIR"] = os.path.join(self._temp.name, "state")

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("REALTIME_STATE_DIR", None)
        else:
            os.environ["REALTIME_STATE_DIR"] = self._old_state
        super().tearDown()

    def test_refresh_assets_writes_state(self):
        assets = [
            {"symbol": "AAPL", "status": "active", "tradable": True},
            {"symbol": "MSFT", "status": "active", "tradable": True},
        ]
        with mock.patch("trading.realtime.alpaca.fetch_assets", return_value=assets):
            call_command("realtime_refresh_assets", user_id="1")

        payload = read_state("assets_master.json", default={})
        self.assertEqual(payload.get("count"), 2)
        self.assertEqual(len(payload.get("assets", [])), 2)
