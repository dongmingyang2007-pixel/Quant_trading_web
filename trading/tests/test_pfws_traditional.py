from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from django.test import SimpleTestCase

from trading.strategies import StrategyInput, compute_indicators, backtest_sma_strategy


def _make_prices(rows: int = 180) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    base = 100 + np.cumsum(np.random.default_rng(42).normal(0.05, 0.8, rows))
    frame = pd.DataFrame(
        {
            "adj close": base,
            "close": base * 0.999,
            "open": base * 1.001,
            "high": base * 1.01,
            "low": base * 0.99,
            "volume": np.linspace(1_000_000, 2_000_000, rows),
        },
        index=idx,
    )
    return frame


class PFWSTraditionalTests(SimpleTestCase):
    def _params(self) -> StrategyInput:
        today = date.today()
        return StrategyInput(
            ticker="TEST",
            benchmark_ticker=None,
            start_date=today - timedelta(days=365),
            end_date=today,
            short_window=5,
            long_window=20,
            rsi_period=14,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
            enforce_pfws_only=True,
            train_window=60,
            test_window=20,
            embargo_days=5,
        )

    def test_sma_backtest_generates_oos_report_when_pfws_enforced(self):
        prices = compute_indicators(_make_prices(), 5, 20, 14)
        backtest, _, stats = backtest_sma_strategy(prices, self._params())
        self.assertIn("validation_penalized_sharpe", stats)
        self.assertIn("validation_oos_folds", stats)
        self.assertEqual(stats.get("validation_train_window"), 60)
        self.assertEqual(stats.get("validation_test_window"), 20)
        self.assertEqual(stats.get("validation_embargo"), 5)

    def test_sma_backtest_outputs_oos_even_when_not_enforced(self):
        prices = compute_indicators(_make_prices(), 5, 20, 14)
        params = self._params()
        params.enforce_pfws_only = False
        _, _, stats = backtest_sma_strategy(prices, params)
        self.assertIn("validation_penalized_sharpe", stats)
        self.assertIn("validation_oos_folds", stats)
