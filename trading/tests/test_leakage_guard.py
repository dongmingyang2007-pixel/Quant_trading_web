from __future__ import annotations

import pandas as pd
from django.test import SimpleTestCase

from trading.validation import assert_no_feature_leakage


class LeakageGuardTests(SimpleTestCase):
    def test_rejects_future_columns_in_features(self):
        df = pd.DataFrame(
            {
                "return_1d": [0.01, -0.02],
                "forward_return": [0.03, -0.01],
                "target": [1, 0],
            }
        )
        with self.assertRaises(ValueError):
            assert_no_feature_leakage(df, ["return_1d", "forward_return"])

    def test_accepts_regular_feature_columns(self):
        df = pd.DataFrame(
            {
                "return_1d": [0.01, -0.02],
                "rsi": [55.0, 48.0],
                "target": [1, 0],
            }
        )
        assert_no_feature_leakage(df, ["return_1d", "rsi"])

    def test_missing_feature_columns_raise(self):
        df = pd.DataFrame({"return_1d": [0.01, -0.02]})
        with self.assertRaises(ValueError):
            assert_no_feature_leakage(df, ["return_1d", "missing_col"])
