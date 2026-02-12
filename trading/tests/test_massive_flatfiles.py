from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd
from django.test import TestCase, override_settings

from trading import massive_flatfiles


@override_settings(SECRET_KEY="test-secret")
class MassiveFlatfilesTests(TestCase):
    def setUp(self):
        super().setUp()
        self._tmp = TemporaryDirectory()
        self._override = override_settings(DATA_CACHE_DIR=Path(self._tmp.name))
        self._override.enable()

    def tearDown(self):
        self._override.disable()
        self._tmp.cleanup()
        super().tearDown()

    def test_intraday_only_window_returns_empty_history(self):
        with mock.patch(
            "trading.massive_flatfiles._today_start_utc",
            return_value=datetime(2026, 2, 10, tzinfo=timezone.utc),
        ):
            frame, meta = massive_flatfiles.fetch_historical_bars(
                symbol="AAPL",
                start=datetime(2026, 2, 10, 12, 0, tzinfo=timezone.utc),
                end=datetime(2026, 2, 10, 13, 0, tzinfo=timezone.utc),
                interval="1m",
                user_id="1",
            )
        self.assertTrue(frame.empty)
        self.assertEqual(meta.get("historical_source"), "none")
        self.assertTrue((meta.get("history_coverage") or {}).get("complete"))

    def test_cached_history_still_works_when_s3_credentials_missing(self):
        day = datetime(2026, 2, 6, tzinfo=timezone.utc).date()
        cache_path, _empty_path = massive_flatfiles._symbol_day_cache_paths("AAPL", day)
        frame_day = pd.DataFrame(
            [
                {"Open": 100.0, "High": 101.0, "Low": 99.5, "Close": 100.6, "Volume": 1200.0},
                {"Open": 100.6, "High": 101.4, "Low": 100.1, "Close": 101.1, "Volume": 950.0},
            ],
            index=pd.to_datetime(
                [
                    "2026-02-06T14:30:00Z",
                    "2026-02-06T14:31:00Z",
                ],
                utc=True,
            ),
        )
        massive_flatfiles._write_cached_day(cache_path, frame_day)

        with mock.patch(
            "trading.massive_flatfiles._today_start_utc",
            return_value=datetime(2026, 2, 10, tzinfo=timezone.utc),
        ), mock.patch(
            "trading.massive_flatfiles._build_s3_client",
            return_value=(None, "credentials_missing"),
        ):
            frame, meta = massive_flatfiles.fetch_historical_bars(
                symbol="AAPL",
                start=datetime(2026, 2, 6, 0, 0, tzinfo=timezone.utc),
                end=datetime(2026, 2, 7, 0, 0, tzinfo=timezone.utc),
                interval="1m",
                user_id="1",
            )

        self.assertFalse(frame.empty)
        self.assertEqual(meta.get("historical_source"), "massive_flatfiles")
        self.assertEqual(meta.get("cache_miss"), 0)
        self.assertGreaterEqual(meta.get("cache_hit", 0), 1)
        self.assertTrue((meta.get("history_coverage") or {}).get("complete"))
        self.assertIsNone(meta.get("error"))
