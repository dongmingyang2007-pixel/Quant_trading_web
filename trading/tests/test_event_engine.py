from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from django.test import SimpleTestCase

from trading.strategies import StrategyInput
from trading.strategies.event_engine import compute_realized_returns, run_event_backtest
from trading.strategies.ma_cross import backtest_sma_strategy


class EventEngineTests(SimpleTestCase):
    def _params(self, **overrides) -> StrategyInput:
        today = date.today()
        base = StrategyInput(
            ticker="TEST",
            benchmark_ticker=None,
            start_date=today - timedelta(days=30),
            end_date=today,
            short_window=3,
            long_window=8,
            rsi_period=5,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=1000.0,
            transaction_cost_bps=0.0,
            slippage_bps=0.0,
            execution_penalty_bps=0.0,
            execution_liquidity_buffer=0.5,
            max_adv_participation=0.5,
            execution_delay_days=1,
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return base

    def test_event_engine_respects_execution_delay(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        price = [100.0, 110.0, 121.0, 133.1]
        frame = pd.DataFrame(
            {
                "adj close": price,
                "close": price,
                "open": price,
                "volume": np.full(len(idx), 1_000_000.0),
                "adv": np.full(len(idx), 1_000_000_000.0),
            },
            index=idx,
        )
        target = pd.Series(1.0, index=idx)
        params = self._params(return_path="close_to_close", execution_delay_days=1)
        backtest, _, _ = run_event_backtest(frame, target, params, leverage=pd.Series(1.0, index=idx))

        self.assertEqual(float(backtest["exposure"].iloc[0]), 0.0)
        self.assertGreater(float(backtest["exposure"].iloc[1]), 0.9)

    def test_compute_realized_returns_close_to_open(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        frame = pd.DataFrame(
            {
                "adj close": [100.0, 110.0, 121.0],
                "close": [100.0, 110.0, 121.0],
                "open": [101.0, 111.0, 122.0],
            },
            index=idx,
        )
        params = self._params(return_path="close_to_open", execution_delay_days=0)
        realized = compute_realized_returns(frame, params)
        self.assertEqual(float(realized.iloc[0]), 0.0)
        self.assertAlmostEqual(float(realized.iloc[1]), 0.11, places=6)

    def test_backtest_sma_sets_execution_stats(self):
        idx = pd.date_range("2024-01-01", periods=12, freq="B")
        base = np.linspace(100.0, 112.0, len(idx))
        frame = pd.DataFrame(
            {
                "adj close": base,
                "close": base,
                "open": base,
                "volume": np.full(len(idx), 1_000_000.0),
                "adv": np.full(len(idx), 1_000_000_000.0),
                "sma_short": base,
                "sma_long": base - 1.0,
                "rsi": np.full(len(idx), 55.0),
            },
            index=idx,
        )
        params = self._params(return_path="close_to_close", execution_delay_days=1)

        backtest, _, stats = backtest_sma_strategy(
            frame,
            params,
            summarize_backtest_fn=lambda *_args, **_kwargs: ([], {}),
            compute_oos_report=lambda *_args, **_kwargs: None,
        )

        self.assertFalse(backtest.empty)
        self.assertIn("execution_stats", stats)
        self.assertIn("avg_coverage", stats["execution_stats"])
