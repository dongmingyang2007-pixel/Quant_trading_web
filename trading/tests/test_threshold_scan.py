from __future__ import annotations

import numpy as np
import pandas as pd

from django.test import SimpleTestCase

from trading.strategies import _scan_threshold_stability


class ThresholdScanTests(SimpleTestCase):
    def test_scan_threshold_stability_returns_summary(self):
        idx = pd.date_range("2024-01-01", periods=160, freq="B")
        proba = pd.Series(np.linspace(0.3, 0.8, len(idx)), index=idx)
        rng = np.random.default_rng(42)
        future = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
        summary = _scan_threshold_stability(proba, future, cost_rate=0.0005, base_entry=0.6, base_exit=0.4)
        self.assertIn("best", summary)
        self.assertIn("grid", summary)
        self.assertGreater(summary["count"], 0)

    def test_scan_threshold_stability_vectorized_counts(self):
        idx = pd.date_range("2024-02-01", periods=200, freq="B")
        proba = pd.Series(np.linspace(0.2, 0.9, len(idx)), index=idx)
        future = pd.Series(0.001, index=idx)
        summary = _scan_threshold_stability(proba, future, cost_rate=0.0002, base_entry=0.65, base_exit=0.35)
        entry_grid = np.linspace(max(0.5, 0.65 - 0.05), min(0.95, 0.65 + 0.05), 5)
        exit_grid = np.linspace(max(0.05, 0.35 - 0.05), min(0.45, 0.35 + 0.05), 5)
        expected_grid = sum(1 for e in entry_grid for x in exit_grid if x < e - 0.02)
        self.assertEqual(summary["grid"]["entry"][0], round(entry_grid[0], 4))
        self.assertEqual(summary["count"], expected_grid)
