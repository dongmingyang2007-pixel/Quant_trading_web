from __future__ import annotations

from datetime import date
import json

import pandas as pd
from django.test import SimpleTestCase
from django.conf import settings

from trading.history import BacktestRecord, append_history


class HistoryLoggingTests(SimpleTestCase):
    def test_append_history_fallback_logs_locally(self):
        payload = {
            "ticker": "TEST",
            "benchmark_ticker": "SPY",
            "engine": "sma",
            "start_date": "2024-01-01",
            "end_date": "2024-05-01",
            "metrics": [],
            "stats": {
                "sharpe": 1.2,
                "total_return": 0.15,
                "max_drawdown": -0.05,
                "execution_cost_total": 0.01,
            },
            "params": {"request_id": "test-req"},
            "warnings": ["unit-test"],
            "snapshot": {"demo": True},
        }
        record = BacktestRecord.from_payload(payload, user_id=None)
        append_history(record)

        log_path = settings.DATA_CACHE_DIR / "reports" / f"{payload['ticker']}_benchmarks.json"
        self.assertTrue(log_path.exists())
        history = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(history), 1)
        entry = history[0]
        self.assertEqual(entry.get("record_id"), record.record_id)
        self.assertEqual(entry.get("ticker"), payload["ticker"])
        self.assertAlmostEqual(entry.get("sharpe"), 1.2)
