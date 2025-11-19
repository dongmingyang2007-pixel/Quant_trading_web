from __future__ import annotations

from datetime import date

from django.test import SimpleTestCase

from trading.validation import collect_repro_metadata
from trading.strategies import StrategyInput


class MetadataLoggingTests(SimpleTestCase):
    def test_collect_repro_metadata_includes_versions_and_seeds(self):
        params = StrategyInput(
            ticker="TEST",
            benchmark_ticker=None,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 1),
            short_window=5,
            long_window=10,
            rsi_period=5,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
        )
        meta = collect_repro_metadata(params)
        self.assertIn("versions", meta)
        self.assertIn("seeds", meta)
        self.assertIn("numpy", meta["versions"])
        self.assertIn("python", meta["seeds"])
        # Optional deps may be missing, but yfinance key should exist even if None
        self.assertIn("yfinance", meta["versions"])
        self.assertEqual(meta.get("return_path"), "close_to_close")
