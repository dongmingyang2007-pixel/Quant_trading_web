from __future__ import annotations

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from trading.strategies import StrategyInput, compute_indicators, backtest_sma_strategy


class GoldenBacktestTests(SimpleTestCase):
    def test_constant_price_series_produces_zero_returns(self):
        idx = pd.date_range("2024-01-01", periods=180, freq="B")
        price = np.full(len(idx), 100.0)
        prices = pd.DataFrame(
            {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "adj close": price,
                "volume": np.full(len(idx), 1_000_000.0),
            },
            index=idx,
        )
        prices = compute_indicators(prices, short_window=5, long_window=20, rsi_period=14)
        params = StrategyInput(
            ticker="TEST",
            benchmark_ticker=None,
            start_date=idx[0].date(),
            end_date=idx[-1].date(),
            short_window=5,
            long_window=20,
            rsi_period=14,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
            execution_penalty_bps=0.0,
            execution_liquidity_buffer=0.5,
            max_adv_participation=0.5,
            borrow_cost_bps=0.0,
            long_borrow_cost_bps=0.0,
            short_borrow_cost_bps=0.0,
            max_drawdown_stop=1.0,
            daily_exposure_limit=10.0,
        )
        _backtest, _metrics, stats = backtest_sma_strategy(prices, params)
        self.assertAlmostEqual(stats.get("total_return", 0.0), 0.0, places=6)
        self.assertAlmostEqual(stats.get("max_drawdown", 0.0), 0.0, places=6)
        self.assertAlmostEqual(stats.get("sharpe", 0.0), 0.0, places=6)
