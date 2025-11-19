from __future__ import annotations

import pandas as pd

from django.test import SimpleTestCase

from trading.risk_stats import compute_robust_sharpe


class RobustSharpeTests(SimpleTestCase):
    def test_compute_robust_sharpe_returns_ci_and_dsr(self):
        series = pd.Series([0.001] * 500)
        stats = compute_robust_sharpe(series, annual_factor=252, trials=5)
        self.assertIn("std_error", stats)
        self.assertIn("ci", stats)
        self.assertLessEqual(stats["ci"][0], stats["ci"][1])
        self.assertIn("deflated_sharpe", stats)
