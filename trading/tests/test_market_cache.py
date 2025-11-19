from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from django.test import SimpleTestCase

from trading import market_data
from trading.cache_utils import build_cache_key


class MarketCacheTests(SimpleTestCase):
    def test_disk_cache_fallback_used_when_no_api(self):
        tmp_dir = Path(self._get_tempdir())
        market_data.DISK_CACHE_DIR = tmp_dir
        market_data.DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        symbols = ["TST"]
        cache_key = build_cache_key("market-data", symbols, None, None, None, "Adj Close")
        df = pd.DataFrame({"Adj Close": [1.0, 1.1, 1.2]}, index=pd.date_range("2024-01-01", periods=3, freq="B"))
        df.to_parquet(tmp_dir / f"{cache_key}.parquet")
        meta = {"timestamp": time.time(), "symbols": symbols, "fields": "Adj Close"}
        (tmp_dir / f"{cache_key}.json").write_text(json.dumps(meta), encoding="utf-8")

        # Simulate missing yfinance by forcing yf to None
        yf_backup = market_data.yf
        market_data.yf = None
        try:
            result = market_data.fetch(symbols, cache=True)
            self.assertFalse(result.empty)
            self.assertIn("Adj Close", result.columns)
        finally:
            market_data.yf = yf_backup

    def _get_tempdir(self) -> str:
        import tempfile

        return tempfile.mkdtemp(prefix="market-cache-test-")
