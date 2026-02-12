from __future__ import annotations

from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


class StrategyWorkbenchApiTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="strategy-workbench", password="secret123")
        self.client.force_login(self.user)

    def test_workbench_default_workspace_is_trade(self):
        response = self.client.get(reverse("trading:api_v1_strategy_workbench"))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("workspace"), "trade")
        self.assertIn(payload.get("market_data_source"), {"alpaca", "massive"})
        self.assertEqual(payload.get("source"), payload.get("market_data_source"))
        self.assertEqual(payload.get("execution_source"), "alpaca")
        self.assertIn("trade", payload)
        self.assertIn("backtest", payload)
        self.assertIn("review", payload)

    def test_workbench_rejects_invalid_workspace(self):
        response = self.client.get(reverse("trading:api_v1_strategy_workbench"), {"workspace": "abc"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error_code"), "invalid_workspace")

    @mock.patch("trading.api.views_v1.trade_source_unavailable", return_value=True)
    @mock.patch("trading.api.views_v1.build_strategy_workbench_payload")
    def test_trade_workspace_returns_market_data_unavailable(self, mock_build, _mock_unavailable):
        mock_build.return_value = {
            "request_id": "req-1",
            "workspace": "trade",
            "source": "alpaca",
            "market_data_source": "alpaca",
            "execution_source": "alpaca",
            "deprecated": False,
            "trade": {"engine": {"online": False}, "summary": {"focus_count": 0, "signals_count": 0}},
            "backtest": {"latest_run": {}, "tasks": [], "history_briefs": []},
            "review": {"execution_diagnostics": [], "risk_diagnostics": [], "ai_briefs": []},
        }
        response = self.client.get(reverse("trading:api_v1_strategy_workbench"), {"workspace": "trade"})
        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload.get("error_code"), "market_data_unavailable")
        self.assertEqual(payload.get("legacy_error_code"), "alpaca_unavailable")
