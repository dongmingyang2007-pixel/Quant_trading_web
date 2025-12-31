from __future__ import annotations

import json
import threading
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

    def test_atomic_disk_cache_write_remains_readable(self):
        tmp_dir = Path(self._get_tempdir())
        market_data.DISK_CACHE_DIR = tmp_dir
        market_data.DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        cache_key = "atomic-cache-test"
        path = tmp_dir / f"{cache_key}.parquet"
        meta_path = tmp_dir / f"{cache_key}.json"

        df_small = pd.DataFrame({"Adj Close": [1.0, 1.1, 1.2]}, index=pd.date_range("2024-01-01", periods=3, freq="B"))
        df_large = pd.DataFrame({"Adj Close": [2.0, 2.2, 2.4, 2.6]}, index=pd.date_range("2024-02-01", periods=4, freq="B"))

        def _writer(frame: pd.DataFrame, delay: float) -> None:
            if delay:
                time.sleep(delay)
            market_data._write_parquet_atomic(frame, path)
            meta = {"timestamp": time.time(), "symbols": ["TST"], "fields": "Adj Close", "cache_key": cache_key}
            market_data._write_json_atomic(meta_path, meta)

        threads = [
            threading.Thread(target=_writer, args=(df_small, 0.0)),
            threading.Thread(target=_writer, args=(df_large, 0.01)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        loaded = pd.read_parquet(path)
        self.assertFalse(loaded.empty)
