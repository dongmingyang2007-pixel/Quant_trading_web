from __future__ import annotations

from unittest import mock

from django.test import SimpleTestCase

from trading.history import compact_history_snapshot
from trading.tasks import _persist_history


class HistoryCompactTests(SimpleTestCase):
    def test_compact_snapshot_includes_metadata(self):
        payload = {
            "ticker": "AAPL",
            "metadata": {"data_signature": {"source": "yfinance", "rows": 10}},
            "repro": {"git_commit": "abc123"},
            "market_context": {"headline": "demo"},
        }
        compacted = compact_history_snapshot(payload)
        self.assertIn("metadata", compacted)
        self.assertIn("repro", compacted)
        self.assertIn("market_context", compacted)
        self.assertEqual(compacted["metadata"]["data_signature"]["source"], "yfinance")

    @mock.patch("trading.tasks.append_fallback_history")
    @mock.patch("trading.tasks.append_history")
    def test_persist_history_reuses_record_id_after_compact(self, mock_append_history, mock_append_fallback):
        record_ids: list[str] = []

        def _side_effect(record):
            record_ids.append(record.record_id)
            return len(record_ids) > 1

        mock_append_history.side_effect = _side_effect
        mock_append_fallback.return_value = True

        result = {
            "ticker": "AAPL",
            "benchmark_ticker": "SPY",
            "engine": "sma",
            "start_date": "2024-01-01",
            "end_date": "2024-05-01",
            "metrics": [],
            "stats": {"sharpe": 1.1},
            "params": {"request_id": "req-1"},
            "warnings": [],
            "metadata": {"data_signature": {"source": "yfinance", "rows": 10}},
        }
        history_id = _persist_history(result, user_id="1")
        self.assertEqual(history_id, record_ids[0])
        self.assertEqual(record_ids[0], record_ids[1])
        mock_append_fallback.assert_not_called()

        second_record = mock_append_history.call_args_list[1][0][0]
        self.assertIn("metadata", second_record.snapshot)
