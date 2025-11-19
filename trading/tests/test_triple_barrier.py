from __future__ import annotations

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from trading.strategies import (
    StrategyInput,
    build_feature_matrix,
    compute_indicators,
    compute_triple_barrier_labels,
)


class TripleBarrierTests(SimpleTestCase):
    def test_compute_triple_barrier_labels_multiclass(self):
        prices = pd.Series([100, 105, 110, 90, 95, 108], index=pd.date_range("2024-01-01", periods=6))
        binary, multiclass = compute_triple_barrier_labels(prices, up=0.04, down=0.04, max_holding=3)
        self.assertEqual(binary.iloc[0], 1)
        self.assertIn(multiclass.iloc[0], {1})
        self.assertIn(multiclass.iloc[2], {-1, 0})

    def test_triple_barrier_dynamic_thresholds_recorded(self):
        idx = pd.date_range("2023-01-01", periods=180, freq="B")
        base = pd.DataFrame(
            {
                "open": np.linspace(100, 130, len(idx)),
                "high": np.linspace(101, 131, len(idx)),
                "low": np.linspace(99, 129, len(idx)),
                "close": np.linspace(100, 130, len(idx)),
                "adj close": np.linspace(100, 130, len(idx)),
                "volume": np.linspace(1e6, 2e6, len(idx)),
            },
            index=idx,
        )
        prices = compute_indicators(base, short_window=10, long_window=30, rsi_period=14)
        params = StrategyInput(
            ticker="TEST",
            benchmark_ticker=None,
            start_date=idx[0].date(),
            end_date=idx[-1].date(),
            short_window=10,
            long_window=30,
            rsi_period=14,
            include_plots=False,
            show_ai_thoughts=False,
            risk_profile="balanced",
            capital=100000.0,
            label_style="triple_barrier",
            tb_dynamic=True,
            tb_vol_multiplier=2.0,
            tb_up=0.02,
            tb_down=0.02,
            tb_max_holding=5,
        )
        dataset, features = build_feature_matrix(prices, params)
        self.assertIn("tb_up_active", dataset.columns)
        self.assertIn("tb_down_active", dataset.columns)
        self.assertTrue(dataset["target"].notna().any())
