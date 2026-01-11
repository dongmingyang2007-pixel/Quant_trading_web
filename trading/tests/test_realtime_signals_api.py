from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from trading.realtime.storage import append_ndjson, write_state


class RealtimeSignalsApiTests(TestCase):
    def setUp(self):
        super().setUp()
        User = get_user_model()
        self.user = User.objects.create_user(username="signals_user", password="secret123")
        self.client.force_login(self.user)
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self._old_state = os.environ.get("REALTIME_STATE_DIR")
        self._old_data = os.environ.get("REALTIME_DATA_DIR")
        os.environ["REALTIME_STATE_DIR"] = os.path.join(self._temp.name, "state")
        os.environ["REALTIME_DATA_DIR"] = os.path.join(self._temp.name, "data")

    def tearDown(self):
        if self._old_state is None:
            os.environ.pop("REALTIME_STATE_DIR", None)
        else:
            os.environ["REALTIME_STATE_DIR"] = self._old_state
        if self._old_data is None:
            os.environ.pop("REALTIME_DATA_DIR", None)
        else:
            os.environ["REALTIME_DATA_DIR"] = self._old_data
        super().tearDown()

    def test_signals_api_reads_latest(self):
        write_state(
            "signals_latest.json",
            {"signals": [{"symbol": "AAPL", "signal": "long", "timestamp": "2024-01-01T00:00:00Z"}]},
        )
        response = self.client.get(reverse("trading:api_v1_realtime_signals"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["signals"]), 1)

    def test_signals_api_reads_ndjson(self):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        append_ndjson(
            f"signals_{stamp}.ndjson",
            [{"symbol": "MSFT", "signal": "short", "timestamp": "2024-01-01T00:00:05Z"}],
        )
        response = self.client.get(
            reverse("trading:api_v1_realtime_signals"),
            data={"source": "ndjson", "date": stamp},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["signals"]), 1)
