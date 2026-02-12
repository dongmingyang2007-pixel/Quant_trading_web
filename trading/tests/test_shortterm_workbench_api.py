from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from trading.models import RealtimeProfile


class ShortTermWorkbenchApiTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="shortterm-user", password="secret123")
        self.client.force_login(self.user)

    def test_realtime_pages_redirect_to_shortterm_panel(self):
        expected = f"{reverse('trading:backtest')}?workspace=trade"
        settings_resp = self.client.get(reverse("trading:realtime_settings"))
        monitor_resp = self.client.get(reverse("trading:realtime_monitor"))

        self.assertEqual(settings_resp.status_code, 302)
        self.assertEqual(settings_resp.url, expected)
        self.assertEqual(monitor_resp.status_code, 302)
        self.assertEqual(monitor_resp.url, expected)

    def test_shortterm_workbench_returns_alpaca_source(self):
        response = self.client.get(reverse("trading:api_v1_shortterm_workbench"))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload.get("market_data_source"), {"alpaca", "massive"})
        self.assertEqual(payload.get("source"), payload.get("market_data_source"))
        self.assertEqual(payload.get("execution_source"), "alpaca")
        self.assertTrue(payload.get("deprecated"))
        self.assertIn("trade", payload)
        self.assertIn("engine", payload["trade"])
        self.assertIn("trading", payload["trade"])
        self.assertIn("summary", payload["trade"])

    def test_live_mode_requires_double_confirmation(self):
        response = self.client.post(
            reverse("trading:api_v1_shortterm_trading_mode"),
            data=json.dumps({"mode": "live"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload.get("error_code"), "live_confirmation_required")

    def test_shortterm_trading_mode_get_returns_deprecated_contract(self):
        response = self.client.get(reverse("trading:api_v1_shortterm_trading_mode"))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload.get("deprecated"))
        self.assertEqual(payload.get("migration"), "/api/v1/strategy/workbench/trading-mode/")
        self.assertIn("allowed_modes", payload)

    def test_live_mode_updates_after_confirmation(self):
        response = self.client.post(
            reverse("trading:api_v1_shortterm_trading_mode"),
            data=json.dumps({"mode": "live", "confirm_live": True, "confirm_phrase": "LIVE"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("mode"), "live")

        profile = RealtimeProfile.objects.filter(user=self.user, is_active=True).first()
        self.assertIsNotNone(profile)
        self.assertEqual((profile.payload.get("trading") or {}).get("mode"), "live")

    def test_template_switch_is_supported(self):
        response = self.client.post(
            reverse("trading:api_v1_shortterm_trading_mode"),
            data=json.dumps({"template_key": "retail_second", "mode": "paper"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("applied_template"), "retail_second")
        self.assertEqual(payload.get("mode"), "paper")
        self.assertTrue(payload.get("deprecated"))
