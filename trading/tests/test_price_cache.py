from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd
from django.test import SimpleTestCase

from trading.strategies import core as core_module


class PriceCacheTests(SimpleTestCase):
    def test_fetch_price_data_writes_csv_with_date(self):
        tmp_dir = Path(tempfile.mkdtemp(prefix="price-cache-"))
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Adj Close": [101.0, 102.0],
                "Volume": [1_000_000.0, 1_100_000.0],
            },
            index=pd.date_range("2024-01-02", periods=2, freq="B"),
        )

        with mock.patch.object(core_module, "DATA_CACHE_DIR", tmp_dir), mock.patch(
            "trading.strategies.core.resolve_market_provider",
            return_value="alpaca",
        ), mock.patch(
            "trading.strategies.core.fetch_stock_bars_frame",
            return_value=df,
        ):
            data, _ = core_module.fetch_price_data("AAPL", date(2024, 1, 1), date(2024, 1, 10))

        cache_file = tmp_dir / "AAPL.csv"
        self.assertTrue(cache_file.exists())
        loaded = pd.read_csv(cache_file, parse_dates=["date"])
        self.assertTrue(
            {"date", "open", "high", "low", "close", "adj close", "volume"}.issubset(set(loaded.columns))
        )
        self.assertEqual(data.attrs.get("cache_path"), str(cache_file))
