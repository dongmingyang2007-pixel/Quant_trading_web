from __future__ import annotations

import pandas as pd
from django.test import SimpleTestCase

from trading.strategies.ml_engine import _apply_execution_model
from trading.strategies.config import StrategyInput


class ExecutionModelTests(SimpleTestCase):
    def _params(self) -> StrategyInput:
        return StrategyInput(
            ticker="AAPL",
            benchmark_ticker="SPY",
            start_date=pd.Timestamp("2023-01-01").date(),
            end_date=pd.Timestamp("2023-06-01").date(),
            short_window=5,
            long_window=20,
            rsi_period=14,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
            slippage_bps=5.0,
            transaction_cost_bps=8.0,
        )

    def test_execution_model_limits_fills_on_high_impact(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        exposure = pd.Series([0, 1, 2, 3, 4], index=idx, dtype=float)
        prices = pd.Series([100, 101, 102, 103, 104], index=idx, dtype=float)
        # Very low ADV to force high impact
        adv = pd.Series([0.01] * 5, index=idx, dtype=float)
        params = self._params()
        adjusted, txn, exec_cost, coverage, stats = _apply_execution_model(exposure, prices, adv, params)

        self.assertTrue((adjusted.abs() <= exposure.abs() + 1e-9).all())
        self.assertGreater(exec_cost.sum(), 0.0)
        self.assertLess(stats["avg_coverage"], 0.6)
        self.assertGreater(stats["unfilled_ratio"], 0.2)
        self.assertEqual(len(adjusted), len(exposure))
        self.assertEqual(len(txn), len(exposure))
        self.assertEqual(len(coverage), len(exposure))

    def test_execution_model_near_full_fill_on_low_impact(self):
        idx = pd.date_range("2024-02-01", periods=5, freq="D")
        exposure = pd.Series([0, 0.2, 0.4, 0.6, 0.8], index=idx, dtype=float)
        prices = pd.Series([50, 50.5, 51, 51.5, 52], index=idx, dtype=float)
        # High ADV to allow fills
        adv = pd.Series([1000] * 5, index=idx, dtype=float)
        params = self._params()
        adjusted, txn, exec_cost, coverage, stats = _apply_execution_model(exposure, prices, adv, params)

        self.assertAlmostEqual(stats["avg_coverage"], 1.0, places=2)
        self.assertLess(stats["unfilled_ratio"], 0.01)
        self.assertTrue((adjusted.abs() <= exposure.abs() + 1e-9).all())
        self.assertGreaterEqual(exec_cost.sum(), 0.0)
        self.assertEqual(len(txn), len(exposure))
