from __future__ import annotations

import pandas as pd
from django.test import SimpleTestCase

from trading.portfolio import build_target_weights, build_trade_list, cap_turnover, combine


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

    def test_build_target_weights_long_only(self):
        weights = build_target_weights({"A": 1.0, "B": -0.5}, allow_short=False)
        self.assertGreaterEqual(weights.get("A", 0.0), 0.0)
        self.assertNotIn("B", [k for k, v in weights.items() if v < 0])

    def test_cap_turnover_limits_changes(self):
        prev = {"A": 0.1}
        target = {"A": 0.9}
        capped = cap_turnover(prev, target, turnover_cap=0.2)
        self.assertLess(abs(capped["A"] - prev["A"]), abs(target["A"] - prev["A"]))

    def test_build_trade_list_respects_lot_size(self):
        trades = build_trade_list(
            {"A": 0.5},
            {"A": 0.0},
            {"A": 100.0},
            capital=10000.0,
            lot_size=10,
        )
        self.assertTrue(trades)
        self.assertEqual(trades[0]["quantity"] % 10, 0)
