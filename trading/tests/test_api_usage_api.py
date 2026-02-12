from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from trading.api_usage import record_provider_api_call, reset_provider_api_usage_for_tests


class ApiUsageViewTests(TestCase):
    def setUp(self) -> None:
        reset_provider_api_usage_for_tests()
        self.user = get_user_model().objects.create_user(username="usage-user", password="secret123")
        self.client.force_login(self.user)

    def tearDown(self) -> None:
        reset_provider_api_usage_for_tests()

    def test_returns_sliding_minute_usage_for_current_user(self) -> None:
        record_provider_api_call("massive", user_id=str(self.user.id))
        record_provider_api_call("massive", user_id=str(self.user.id))
        record_provider_api_call("alpaca", user_id=str(self.user.id))
        record_provider_api_call("alpaca", user_id="someone-else")

        response = self.client.get(reverse("trading:api_v1_api_usage"))
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertEqual(payload.get("window_seconds"), 60)
        self.assertEqual(payload.get("user_total"), 3)
        by_provider = payload.get("user_by_provider") or {}
        self.assertEqual(by_provider.get("massive"), 2)
        self.assertEqual(by_provider.get("alpaca"), 1)

