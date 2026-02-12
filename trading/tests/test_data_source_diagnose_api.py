from __future__ import annotations

import json
from unittest import mock

import pandas as pd
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


class DataSourceDiagnoseApiTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="diagnose-user", password="secret123")
        self.client.force_login(self.user)

    def test_rejects_unsupported_provider(self):
        response = self.client.post(
            reverse("trading:api_v1_data_source_diagnose"),
            data=json.dumps({"provider": "foo"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload.get("error_code"), "provider_not_supported")

    @mock.patch("trading.api.views_v1.has_market_data_credentials", return_value=False)
    def test_massive_missing_credentials_returns_specific_error(self, _mock_has_creds):
        response = self.client.post(
            reverse("trading:api_v1_data_source_diagnose"),
            data=json.dumps({"provider": "massive", "check_news": False, "check_ws": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload.get("error_code"), "massive_credentials_missing")
        self.assertIn("Massive credentials are missing", payload.get("message", ""))
        failed_checks = payload.get("failed_checks") or []
        self.assertTrue(any(item.get("name") == "credentials" for item in failed_checks))

    @mock.patch("trading.api.views_v1.provider_fetch_stock_trades", return_value=([], None, None, None))
    @mock.patch("trading.api.views_v1.provider_fetch_stock_snapshots", return_value={"SPY": {"dailyBar": {"c": 500}}})
    @mock.patch("trading.api.views_v1.provider_fetch_stock_bars_frame")
    @mock.patch("trading.api.views_v1.has_market_data_credentials", return_value=True)
    def test_success_payload_contains_checks(self, _mock_has_creds, mock_bars_frame, _mock_snapshots, _mock_trades):
        mock_bars_frame.return_value = pd.DataFrame(
            [{"Open": 1.0, "High": 1.2, "Low": 0.9, "Close": 1.1, "Volume": 1000}],
            index=[pd.Timestamp("2026-01-01", tz="UTC")],
        )

        response = self.client.post(
            reverse("trading:api_v1_data_source_diagnose"),
            data=json.dumps({"provider": "alpaca", "check_news": False, "check_ws": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload.get("ok"))
        checks = payload.get("checks") or {}
        self.assertTrue(checks.get("credentials", {}).get("ok"))
        self.assertTrue(checks.get("bars", {}).get("ok"))
        self.assertTrue(checks.get("snapshot", {}).get("ok"))
        self.assertTrue(checks.get("trades", {}).get("ok"))

    @mock.patch("trading.api.views_v1.provider_fetch_stock_trades", return_value=([], None, None, None))
    @mock.patch("trading.api.views_v1.provider_fetch_stock_snapshots", return_value={"SPY": {"dailyBar": {"c": 500}}})
    @mock.patch("trading.api.views_v1.provider_fetch_stock_bars_frame")
    @mock.patch("trading.api.views_v1.has_market_data_credentials", return_value=True)
    def test_diagnose_bars_call_does_not_pass_period_kwarg(
        self,
        _mock_has_creds,
        mock_bars_frame,
        _mock_snapshots,
        _mock_trades,
    ):
        mock_bars_frame.return_value = pd.DataFrame(
            [{"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100}],
            index=[pd.Timestamp("2026-01-01", tz="UTC")],
        )

        response = self.client.post(
            reverse("trading:api_v1_data_source_diagnose"),
            data=json.dumps({"provider": "massive", "check_news": False, "check_ws": False}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(mock_bars_frame.called)
        _args, kwargs = mock_bars_frame.call_args
        self.assertNotIn("period", kwargs)
        self.assertIn("start", kwargs)
        self.assertIn("end", kwargs)
