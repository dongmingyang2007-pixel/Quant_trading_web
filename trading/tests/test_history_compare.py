from __future__ import annotations

from datetime import date

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from trading.models import BacktestRecord


class HistoryCompareTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="arena_user", password="secret")
        self.client.force_login(self.user)

    def _make_record(self, record_id: str, ticker: str, returns: list[float]) -> BacktestRecord:
        recent_rows = []
        cum_value = 1.0
        for idx, value in enumerate(returns):
            cum_value *= 1 + value
            recent_rows.append(
                {
                    "date": date(2024, 1, idx + 1).isoformat(),
                    "daily_return": value,
                    "cum_strategy": round(cum_value, 4),
                }
            )
        return BacktestRecord.objects.create(
            record_id=record_id,
            user=self.user,
            timestamp=timezone.now(),
            ticker=ticker,
            benchmark="SPY",
            engine="sma",
            start_date="2024-01-01",
            end_date="2024-01-05",
            metrics=[],
            stats={"total_return": 0.05, "max_drawdown": -0.02, "volatility": 0.2, "sharpe": 1.2},
            params={},
            warnings=[],
            snapshot={"recent_rows": recent_rows},
        )

    @override_settings(STATICFILES_STORAGE="django.contrib.staticfiles.storage.StaticFilesStorage")
    def test_compare_includes_correlation_and_risk(self):
        self._make_record("rec-1", "AAPL", [0.01, -0.005, 0.004, 0.0])
        self._make_record("rec-2", "MSFT", [0.008, -0.004, 0.006, 0.001])
        url = reverse("trading:history_compare") + "?records=rec-1,rec-2"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        context = response.context
        self.assertIn("correlation_matrix", context)
        self.assertIn("risk_rows", context)
        self.assertEqual(len(context["correlation_matrix"]["labels"]), 2)
        corr_value = context["correlation_matrix"]["rows"][0]["cells"][1]["value"]
        self.assertIsNotNone(corr_value)
        self.assertEqual(len(context["risk_rows"]), 2)
        self.assertIn("avg_return_pct", context["risk_rows"][0]["metrics"])
