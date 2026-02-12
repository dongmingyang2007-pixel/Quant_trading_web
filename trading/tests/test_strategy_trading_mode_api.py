from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


class StrategyTradingModeApiTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="strategy-mode-user", password="secret123")
        self.client.force_login(self.user)

    def test_live_mode_requires_double_confirmation(self):
        response = self.client.post(
            reverse("trading:api_v1_strategy_trading_mode"),
            data=json.dumps({"mode": "live"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json().get("error_code"), "live_confirmation_required")

    def test_get_returns_mode_contract(self):
        response = self.client.get(reverse("trading:api_v1_strategy_trading_mode"))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload.get("market_data_source"), {"alpaca", "massive"})
        self.assertEqual(payload.get("source"), payload.get("market_data_source"))
        self.assertEqual(payload.get("execution_source"), "alpaca")
        self.assertIn("allowed_modes", payload)
        self.assertIn("templates", payload)
        self.assertIn("confirmation_rules", payload)

    def test_live_mode_switches_after_confirmation(self):
        response = self.client.post(
            reverse("trading:api_v1_strategy_trading_mode"),
            data=json.dumps({"mode": "live", "confirm_live": True, "confirm_phrase": "LIVE"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("mode"), "live")

    def test_template_apply_returns_success(self):
        response = self.client.post(
            reverse("trading:api_v1_strategy_trading_mode"),
            data=json.dumps({"template_key": "retail_second", "mode": "paper"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("applied_template"), "retail_second")
        self.assertEqual(payload.get("mode"), "paper")
