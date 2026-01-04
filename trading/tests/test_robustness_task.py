from __future__ import annotations

from unittest import mock

from django.test import SimpleTestCase

from trading.tasks import execute_robustness_job


class RobustnessTaskTests(SimpleTestCase):
    @mock.patch("trading.tasks.run_quant_pipeline")
    def test_execute_robustness_job_returns_grid(self, mock_run):
        mock_run.return_value = {
            "stats": {
                "sharpe": 1.2,
                "max_drawdown": -0.2,
                "execution_stats": {"avg_coverage": 0.85},
                "total_return": 0.4,
                "cost_ratio": 0.05,
            }
        }
        payload = {
            "ticker": "AAPL",
            "benchmark_ticker": "SPY",
            "start_date": "2024-01-01",
            "end_date": "2024-06-01",
            "short_window": 20,
            "long_window": 50,
            "rsi_period": 14,
            "include_plots": False,
            "show_ai_thoughts": False,
            "risk_profile": "balanced",
            "capital": 250000,
            "user_id": "1",
            "robustness": {
                "max_runs": 6,
                "cost_rates": [0.0006],
                "adv_participation": [0.1],
                "thresholds": [0.55, 0.58],
            },
        }
        result = execute_robustness_job(payload)
        grid = result.get("grid")
        self.assertIsInstance(grid, dict)
        self.assertIn("cells", grid)
        self.assertLessEqual(len(grid["cells"]), 6)
        self.assertIn("best", grid)
