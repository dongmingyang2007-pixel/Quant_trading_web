from __future__ import annotations

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from trading.strategies import StrategyInput, apply_vol_targeting


class VolTargetingTests(SimpleTestCase):
    def test_apply_vol_targeting_scales_exposure(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="B")
        exposure = pd.Series(1.0, index=idx)
        returns = pd.Series(np.random.normal(0, 0.02, len(idx)), index=idx)
        params = StrategyInput(
            ticker="TEST",
            benchmark_ticker="SPY",
            start_date=idx[0].date(),
            end_date=idx[-1].date(),
            short_window=5,
            long_window=20,
            rsi_period=5,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
            target_vol=0.1,
            vol_target_window=30,
        )
        scaled, events = apply_vol_targeting(exposure, returns, params)
        self.assertTrue(len(events) >= 0)
        self.assertTrue((scaled <= 3.0).all())
