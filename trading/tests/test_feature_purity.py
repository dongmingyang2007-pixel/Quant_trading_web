from __future__ import annotations

import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from trading.strategies import build_feature_frame, compute_indicators


class FeaturePurityTests(SimpleTestCase):
    def _make_prices(self, rows: int = 90) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=rows, freq="B")
        base = 100 + np.cumsum(np.random.default_rng(1).normal(0.02, 0.3, rows))
        return pd.DataFrame(
            {
                "open": base * 0.999,
                "high": base * 1.01,
                "low": base * 0.99,
                "close": base,
                "adj close": base,
                "volume": np.linspace(1e6, 1.2e6, rows),
            },
            index=idx,
        )

    def test_build_feature_frame_does_not_mutate_input(self):
        base = self._make_prices()
        base_copy = base.copy(deep=True)
        features = build_feature_frame(base, short_window=5, long_window=20, rsi_period=14)
        pd.testing.assert_frame_equal(base, base_copy)
        self.assertNotIn("forward_return", features.columns)
        self.assertNotIn("label", features.columns)

    def test_compute_indicators_can_skip_labels(self):
        base = self._make_prices()
        frame = compute_indicators(base, short_window=5, long_window=20, rsi_period=14, include_labels=False)
        self.assertNotIn("forward_return", frame.columns)
        self.assertNotIn("label", frame.columns)
