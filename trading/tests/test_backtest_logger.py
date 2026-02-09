from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase

from trading import backtest_logger


class BacktestLoggerTests(SimpleTestCase):
    def test_top_runs_reads_and_sorts_entries(self):
        with tempfile.TemporaryDirectory(prefix="backtest-log-") as tmp:
            log_dir = Path(tmp)
            payload = [
                {"timestamp": "2026-01-01T00:00:00Z", "sharpe": 0.8, "total_return": 0.12},
                {"timestamp": "2026-01-02T00:00:00Z", "sharpe": 1.2, "total_return": 0.08},
                {"timestamp": "2026-01-03T00:00:00Z", "sharpe": 1.2, "total_return": 0.2},
            ]
            with mock.patch.object(backtest_logger, "LOG_DIR", log_dir):
                path = backtest_logger._log_file_for("AAPL")
                path.write_text(json.dumps(payload), encoding="utf-8")
                runs = backtest_logger.top_runs("AAPL", limit=2)

            self.assertEqual(len(runs), 2)
            self.assertEqual(runs[0]["sharpe"], 1.2)
            self.assertEqual(runs[0]["total_return"], 0.2)
            self.assertEqual(runs[1]["sharpe"], 1.2)
            self.assertEqual(runs[1]["total_return"], 0.08)
