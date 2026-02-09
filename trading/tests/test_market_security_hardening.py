from __future__ import annotations

import json
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from trading.models import RealtimeProfile
from trading.realtime.schema import RealtimePayloadError, validate_realtime_payload
from trading.views import market as market_views


class MarketInsightsDebugVisibilityTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="market_member", password="secret123")
        self.staff = user_model.objects.create_user(username="market_staff", password="secret123", is_staff=True)
        self.url = reverse("trading:market_insights_data")

    def _detail_params(self) -> dict[str, str]:
        return {
            "detail": "1",
            "symbol": "AAPL",
            "include_bars": "0",
            "include_ai": "0",
            "debug": "1",
        }

    @override_settings(DEBUG=True)
    def test_non_staff_cannot_read_market_debug_payload(self):
        self.client.force_login(self.user)
        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market._fetch_company_profile",
            return_value={},
        ), mock.patch(
            "trading.views.market._fetch_yfinance_fundamentals_debug",
            side_effect=AssertionError("non-staff should not hit fundamentals debug path"),
        ) as mock_debug, mock.patch(
            "trading.views.market._fetch_yfinance_fundamentals",
            return_value={},
        ), mock.patch(
            "trading.views.market._fetch_symbol_news_page",
            return_value=([], {"offset": 0, "limit": 10, "count": 0, "has_more": False, "next_offset": None}),
        ), mock.patch(
            "trading.views.market._fetch_52w_stats",
            return_value={"high_52w": None, "low_52w": None, "as_of": None},
        ):
            response = self.client.get(self.url, data=self._detail_params())

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotIn("debug", payload)
        mock_debug.assert_not_called()

    @override_settings(DEBUG=False)
    def test_staff_cannot_read_market_debug_payload_when_debug_off(self):
        self.client.force_login(self.staff)
        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market._fetch_company_profile",
            return_value={},
        ), mock.patch(
            "trading.views.market._fetch_yfinance_fundamentals_debug",
            side_effect=AssertionError("debug=false should not hit fundamentals debug path"),
        ) as mock_debug, mock.patch(
            "trading.views.market._fetch_yfinance_fundamentals",
            return_value={},
        ), mock.patch(
            "trading.views.market._fetch_symbol_news_page",
            return_value=([], {"offset": 0, "limit": 10, "count": 0, "has_more": False, "next_offset": None}),
        ), mock.patch(
            "trading.views.market._fetch_52w_stats",
            return_value={"high_52w": None, "low_52w": None, "as_of": None},
        ):
            response = self.client.get(self.url, data=self._detail_params())

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertNotIn("debug", payload)
        mock_debug.assert_not_called()

    @override_settings(DEBUG=True)
    def test_staff_can_read_market_debug_payload_in_debug(self):
        self.client.force_login(self.staff)
        with mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market._fetch_company_profile",
            return_value={},
        ), mock.patch(
            "trading.views.market._fetch_yfinance_fundamentals_debug",
            return_value=({}, {"symbol": "AAPL", "cached": False, "keys": [], "has_values": False}),
        ) as mock_debug, mock.patch(
            "trading.views.market._fetch_symbol_news_page",
            return_value=([], {"offset": 0, "limit": 10, "count": 0, "has_more": False, "next_offset": None}),
        ), mock.patch(
            "trading.views.market._fetch_52w_stats",
            return_value={"high_52w": None, "low_52w": None, "as_of": None},
        ):
            response = self.client.get(self.url, data=self._detail_params())

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("debug", payload)
        self.assertIsInstance(payload["debug"], dict)
        mock_debug.assert_called_once()


class MarketTradingModeErrorContractTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="realtime_member", password="secret123")
        self.client.force_login(self.user)
        self.url = reverse("trading:market_trading_mode")
        RealtimeProfile.objects.create(
            user=self.user,
            name="Realtime Profile",
            description="",
            payload=validate_realtime_payload({}),
            is_active=True,
        )

    def test_market_trading_mode_does_not_expose_internal_validation_error(self):
        with mock.patch(
            "trading.views.market.validate_realtime_payload",
            side_effect=RealtimePayloadError("internal-secret-config"),
        ):
            response = self.client.post(
                self.url,
                data=json.dumps({"mode": "live"}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload.get("error_code"), "invalid_realtime_config")
        self.assertNotIn("internal-secret-config", json.dumps(payload, ensure_ascii=False))


class SnapshotRefreshSanitizationTests(TestCase):
    def test_refresh_snapshot_rankings_sanitizes_unexpected_errors(self):
        class _DummyLock:
            def __init__(self):
                self.released = False

            def acquire(self):
                return True

            def release(self):
                self.released = True

        lock = _DummyLock()
        with mock.patch("trading.views.market.InstanceLock", return_value=lock), mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market._load_assets_master",
            side_effect=RuntimeError("upstream-secret-token"),
        ), mock.patch("trading.views.market.write_state") as mock_write:
            payload = market_views.refresh_snapshot_rankings(user_id="42")

        self.assertEqual(payload.get("status"), "error")
        self.assertEqual(payload.get("error"), "snapshot_refresh_failed")
        self.assertNotIn("upstream-secret-token", json.dumps(payload, ensure_ascii=False))
        self.assertTrue(lock.released)
        self.assertTrue(
            any(
                call.args
                and call.args[0] == market_views.SNAPSHOT_RANKINGS_PROGRESS_STATE
                and isinstance(call.args[1], dict)
                and call.args[1].get("error") == "snapshot_refresh_failed"
                for call in mock_write.call_args_list
            )
        )
