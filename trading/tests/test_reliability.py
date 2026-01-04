from __future__ import annotations

from django.test import SimpleTestCase

from trading.strategies.pipeline import _build_reliability


class ReliabilityTests(SimpleTestCase):
    def test_build_reliability_scores_and_actions(self):
        stats = {
            "data_quality": {"missing_ratio": 0.12, "zero_volume_days": 3, "stale_price_days": 10},
            "execution_stats": {"avg_coverage": 0.62, "unfilled_ratio": 0.4, "adv_hard_cap_hits": 4, "halt_days": 2, "limit_days": 1},
            "cost_ratio": 0.2,
            "validation_penalized_sharpe": -0.1,
            "validation_summary_compact": {"sharpe": {"mean": 0.2, "std": 0.6}},
            "auc": 0.52,
            "calibration": {"brier": 0.3},
            "cpcv": {"p10_sharpe": -0.1, "worst_sharpe": -0.5},
            "drift": {"psi_returns": 0.3, "psi_probabilities": 0.1},
            "threshold_stability": {"worst": {"sharpe": -0.2}},
        }
        metadata = {
            "data_signature": {"source": "csv_cache"},
            "requested_start": "2020-01-01",
            "requested_end": "2020-12-31",
            "effective_start": "2020-02-01",
            "effective_end": "2020-11-30",
        }
        reliability = _build_reliability(stats, metadata)
        self.assertIn(reliability["label"], {"Excellent", "Good", "Caution", "High risk"})
        self.assertGreaterEqual(reliability["score"], 0)
        self.assertLessEqual(reliability["score"], 100)
        self.assertTrue(reliability["reasons"])
        self.assertTrue(reliability["actions"])
