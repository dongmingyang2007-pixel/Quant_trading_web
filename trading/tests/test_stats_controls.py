from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date

from django.test import SimpleTestCase

from trading.strategies import StrategyInput, summarize_backtest


def _build_dummy_backtest(length: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=length, freq="B")
    rng = np.random.default_rng(123)
    strategy_ret = pd.Series(rng.normal(0.0005, 0.01, length), index=idx)
    asset_ret = pd.Series(rng.normal(0.0004, 0.009, length), index=idx)
    return pd.DataFrame(
        {
            "strategy_return": strategy_ret,
            "asset_return": asset_ret,
            "position": pd.Series(1.0, index=idx),
            "exposure": pd.Series(1.0, index=idx),
        }
    )


class StatsControlTests(SimpleTestCase):
    def _params(self, **kwargs) -> StrategyInput:
        base = dict(
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
        base.update(kwargs)
        return StrategyInput(**base)

    def test_disable_bootstrap_stats(self):
        backtest = _build_dummy_backtest()
        params = self._params(stats_enable_bootstrap=False)
        _, stats = summarize_backtest(backtest, params)
        self.assertNotIn("sharpe_pvalue_bootstrap", stats)
        self.assertNotIn("sharpe_pvalue_spa", stats)
        self.assertGreater(stats["sharpe"], -10.0)  # baseline stats still computed

    def test_custom_bootstrap_samples_and_block(self):
        backtest = _build_dummy_backtest()
        params = self._params(stats_enable_bootstrap=True, stats_bootstrap_samples=50, stats_bootstrap_block=5)
        _, stats = summarize_backtest(backtest, params)
        self.assertEqual(stats.get("sharpe_bootstrap_block"), 5)
        self.assertEqual(stats.get("sharpe_spa_block"), 5)
        # ensure bootstrap actually executed
        self.assertIn("sharpe_pvalue_bootstrap", stats)
        self.assertIn("sharpe_pvalue_spa", stats)
