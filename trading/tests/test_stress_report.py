from __future__ import annotations

import pandas as pd
from django.test import SimpleTestCase

from trading.validation import build_stress_report


class StressReportTests(SimpleTestCase):
    def test_stress_report_generates_worst_stats(self):
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        returns = pd.Series([0.002] * 20, index=idx)
        report = build_stress_report(returns)
        self.assertIn("scenarios", report)
        self.assertLess(report["worst_mdd"], 0)  # gap 应产生回撤
        self.assertLess(report["worst_sharpe"], 1.0)
        self.assertGreater(len(report["scenarios"]), 0)

    def test_psi_detects_drift(self):
        base = pd.Series([0.01] * 50)
        recent = pd.Series([-0.02] * 50)
        from trading.validation import compute_psi
        psi = compute_psi(base, recent)
        self.assertGreater(psi, 0.2)
