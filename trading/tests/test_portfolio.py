from __future__ import annotations

import pandas as pd
from django.test import SimpleTestCase

from trading.portfolio import combine


class PortfolioTests(SimpleTestCase):
    def test_combine_equal_weights(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        a = pd.Series([0.01] * 5, index=idx)
        b = pd.Series([-0.005] * 5, index=idx)
        combined, weights = combine({"A": a, "B": b}, scheme="equal")
        self.assertAlmostEqual(weights["A"], 0.5)
        self.assertEqual(len(combined), 5)

    def test_combine_erc_fallback(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        a = pd.Series([0.0] * 5, index=idx)
        combined, weights = combine({"A": a}, scheme="erc")
        self.assertIn("A", weights)
        self.assertEqual(len(combined), 5)
