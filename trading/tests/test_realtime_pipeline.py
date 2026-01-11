from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from unittest import mock

from django.test import SimpleTestCase

from trading.realtime.bars import BarsProcessor
from trading.realtime.config import FocusConfig, SignalConfig, UniverseConfig
from trading.realtime.focus import update_focus
from trading.realtime.signals import SignalEngine
from trading.realtime.storage import read_ndjson_tail, read_state, write_state
from trading.realtime.universe import build_universe


class RealtimePipelineTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
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

    def test_bars_processor_aggregates(self):
        collected = []
        processor = BarsProcessor(
            bar_interval_seconds=1,
            bar_aggregate_seconds=5,
            stale_seconds=999,
            on_bar_5s=collected.append,
        )
        base_ts = 1000.0
        for idx in range(7):
            processor.on_trade("AAPL", 100 + idx, 10, base_ts + idx)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        rows_1s = read_ndjson_tail(f"bars_1s_{stamp}.ndjson", limit=5)
        rows_5s = read_ndjson_tail(f"bars_5s_{stamp}.ndjson", limit=5)
        latest = read_state("bars_latest.json", default={})

        self.assertTrue(rows_1s)
        self.assertTrue(rows_5s)
        self.assertTrue(collected)
        self.assertIn("bars", latest)

    def test_signal_engine_generates_signal(self):
        config = SignalConfig(lookback_bars=2, entry_threshold=0.0005, min_volume=0.0, max_spread_bps=999.0)
        engine = SignalEngine(config)
        bars = [
            {"timestamp": "2024-01-01T00:00:00Z", "symbol": "AAPL", "close": 100, "volume": 50000},
            {"timestamp": "2024-01-01T00:00:05Z", "symbol": "AAPL", "close": 101, "volume": 50000},
            {"timestamp": "2024-01-01T00:00:10Z", "symbol": "AAPL", "close": 102, "volume": 50000},
        ]
        for bar in bars:
            engine.on_bar(bar)

        latest = read_state("signals_latest.json", default={})
        signals = latest.get("signals")
        self.assertTrue(signals)
        self.assertEqual(signals[-1]["signal"], "long")

    def test_focus_hysteresis(self):
        now = datetime.now(timezone.utc).timestamp()
        write_state(
            "focus_state.json",
            {
                "updated_at": now,
                "symbols": [
                    {"symbol": "AAA", "since_ts": now},
                    {"symbol": "BBB", "since_ts": now - 1000},
                ],
            },
        )
        config = FocusConfig(size=2, max_churn_per_refresh=1, min_residence_seconds=300)
        updated = update_focus(["CCC", "DDD", "EEE"], config)
        symbols = [entry.symbol for entry in updated]
        self.assertIn("AAA", symbols)
        self.assertEqual(len(symbols), 2)

    def test_universe_filters_and_scores(self):
        config = UniverseConfig(min_price=1.0, min_volume=10, min_dollar_volume=100, top_n=2, max_symbols=2)
        snapshots = {
            "AAA": {"dailyBar": {"c": 10.0, "v": 20}, "prevDailyBar": {"c": 9.0}},
            "BBB": {"dailyBar": {"c": 3.0, "v": 5}, "prevDailyBar": {"c": 3.0}},
        }
        with mock.patch("trading.realtime.universe.fetch_stock_snapshots", return_value=snapshots):
            entries = build_universe(config, user_id=None, feed="iex")
        symbols = [entry.symbol for entry in entries]
        self.assertEqual(symbols, ["AAA"])
