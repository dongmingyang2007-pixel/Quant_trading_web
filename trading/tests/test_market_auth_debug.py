from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse


@override_settings(DEBUG=True)
class MarketAuthDebugTests(TestCase):
    def test_market_auth_debug_returns_404_for_anonymous(self):
        response = self.client.get(reverse("trading:market_auth_debug"))
        self.assertEqual(response.status_code, 404)

    def test_market_auth_debug_returns_404_for_non_staff(self):
        user = get_user_model().objects.create_user(username="member", password="secret123")
        self.client.force_login(user)
        response = self.client.get(reverse("trading:market_auth_debug"))
        self.assertEqual(response.status_code, 404)

    def test_market_auth_debug_allows_staff_in_debug(self):
        staff = get_user_model().objects.create_user(
            username="staff_user",
            password="secret123",
            is_staff=True,
        )
        self.client.force_login(staff)
        response = self.client.get(reverse("trading:market_auth_debug"))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["is_authenticated"])
        self.assertEqual(payload["user_id"], staff.id)
