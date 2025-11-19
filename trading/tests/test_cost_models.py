from __future__ import annotations

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from datetime import date

from trading.optimization import _simulate_returns
from trading.optimization import _compute_slippage_cost  # type: ignore
from trading.strategies import summarize_backtest, StrategyInput


class CostModelTests(SimpleTestCase):
    def test_linear_slippage_reduces_pnl(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        probs = np.array([0.2, 0.8, 0.8, 0.2, 0.8])
        future = pd.Series([0.01] * 5, index=idx)
        pnl, detail = _simulate_returns(
            probs,
            idx,
            future,
            entry=0.6,
            exit=0.4,
            cost_rate=0.0,
            slippage={"type": "linear", "bps": 100},
            prices=pd.Series(100.0, index=idx),
            volume=pd.Series(1_000_000.0, index=idx),
            borrow_cost_bps=0.0,
        )
        self.assertLess(pnl.sum(), future.sum())  # slippage should reduce profit
        self.assertGreater(detail["slippage_cost"], 0.0)

    def test_borrow_cost_applied(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        probs = np.array([0.9, 0.9, 0.9])  # long exposure
        future = pd.Series([0.0, 0.0, 0.0], index=idx)
        pnl, detail = _simulate_returns(
            probs,
            idx,
            future,
            entry=0.6,
            exit=0.4,
            cost_rate=0.0,
            slippage={"type": "linear", "bps": 0},
            prices=pd.Series(100.0, index=idx),
            volume=pd.Series(1_000_000.0, index=idx),
            borrow_cost_bps=50.0,
        )
        self.assertLessEqual(pnl.sum(), 0.0)
        self.assertGreater(detail["borrow_cost"], 0.0)

    def test_long_short_borrow_diff(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        probs = np.array([0.9, 0.2, 0.2])  # long then flip short
        future = pd.Series([0.0, 0.0, 0.0], index=idx)
        pnl, detail = _simulate_returns(
            probs,
            idx,
            future,
            entry=0.6,
            exit=0.4,
            cost_rate=0.0,
            slippage={"type": "linear", "bps": 0},
            prices=pd.Series(100.0, index=idx),
            volume=pd.Series(1_000_000.0, index=idx),
            long_borrow_cost_bps=10.0,
            short_borrow_cost_bps=80.0,
        )
        self.assertGreater(detail["borrow_cost"], 0.0)

    def test_future_return_is_not_delayed(self):
        idx = pd.date_range("2024-03-01", periods=3, freq="B")
        probs = np.array([0.9, 0.1, 0.1])  # first day triggers long
        future = pd.Series([0.02, -0.01, 0.0], index=idx)  # gain shows up immediately if no lag
        pnl, _ = _simulate_returns(
            probs,
            idx,
            future,
            entry=0.6,
            exit=0.4,
            cost_rate=0.0,
            slippage={"type": "linear", "bps": 0},
            prices=pd.Series(100.0, index=idx),
            volume=pd.Series(1_000_000.0, index=idx),
        )
        # Without an artificial one-day lag, first day's forward return should be captured.
        self.assertGreater(pnl.sum(), 0.0)

    def test_compute_slippage_cost_sqrt_model(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        changes = pd.Series([0.0, 0.5, -0.5, 0.2], index=idx)
        prices = pd.Series(50.0, index=idx)
        vol = pd.Series(1_000_000.0, index=idx)
        cost = _compute_slippage_cost(changes, prices, vol, {"type": "sqrt", "eta": 1.5})
        self.assertTrue((cost >= 0).all())
        self.assertGreater(cost.sum(), 0.0)

    def test_execution_and_borrow_costs_counted_in_stats(self):
        idx = pd.date_range("2024-04-01", periods=3, freq="B")
        backtest = pd.DataFrame(
            {
                "strategy_return": pd.Series([0.01, -0.005, 0.0], index=idx),
                "asset_return": pd.Series([0.0, 0.0, 0.0], index=idx),
                "position": pd.Series([0.0, 1.0, 0.0], index=idx),
                "exposure": pd.Series([0.0, 1.0, 0.0], index=idx),
                "transaction_cost": pd.Series([0.0, 0.001, 0.0], index=idx),
                "execution_cost": pd.Series([0.002, 0.0, 0.0], index=idx),
                "borrow_cost": pd.Series([0.0, 0.0005, 0.0005], index=idx),
            }
        )
        params = StrategyInput(
            ticker="TEST",
            benchmark_ticker=None,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 4, 1),
            short_window=5,
            long_window=20,
            rsi_period=14,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
        )
        _, stats = summarize_backtest(backtest, params)
        expected_total_cost = 0.001 + 0.002 + 0.001
        self.assertAlmostEqual(stats["transaction_cost_total"], 0.001)
        self.assertAlmostEqual(stats["execution_cost_total"], 0.002)
        self.assertAlmostEqual(stats["borrow_cost_total"], 0.001)
        self.assertAlmostEqual(stats["total_cost"], expected_total_cost)
        self.assertGreater(stats["cost_ratio"], 0.0)

    def test_cost_assumptions_exposed(self):
        idx = pd.date_range("2024-05-01", periods=3, freq="B")
        backtest = pd.DataFrame(
            {
                "strategy_return": pd.Series([0.0, 0.0, 0.0], index=idx),
                "asset_return": pd.Series([0.0, 0.0, 0.0], index=idx),
            }
        )
        params = StrategyInput(
            ticker="TEST",
            benchmark_ticker=None,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 5, 1),
            short_window=5,
            long_window=20,
            rsi_period=14,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
            slippage_model={"type": "linear", "bps": 50},
            execution_mode="limit",
            max_adv_participation=0.15,
        )
        _, stats = summarize_backtest(backtest, params)
        assumptions = stats.get("cost_assumptions") or {}
        self.assertEqual(assumptions.get("slippage_model", {}).get("type"), "linear")
        self.assertEqual(assumptions.get("execution_mode"), "limit")
