from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from trading.realtime.schema import PYDANTIC_AVAILABLE


def _profile_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "universe": {"max_symbols": 800, "top_n": 200, "min_price": 1.0},
        "focus": {"size": 120, "max_churn_per_refresh": 15},
        "engine": {"stream_enabled": False, "feed": "iex"},
        "signals": {"lookback_bars": 3, "entry_threshold": 0.0025},
    }
    payload.update(overrides)
    return payload


class RealtimeProfileApiTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="rt_user", password="secret123")
        self.client.force_login(self.user)

    def test_profile_crud_and_active_switch(self):
        response = self.client.post(
            reverse("trading:api_v1_realtime_profiles"),
            data=json.dumps(
                {
                    "name": "Primary",
                    "description": "Focus 200 default",
                    "payload": _profile_payload(),
                    "is_active": True,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        first_id = data["profile_id"]
        self.assertTrue(data["is_active"])

        response = self.client.post(
            reverse("trading:api_v1_realtime_profiles"),
            data=json.dumps(
                {
                    "name": "Secondary",
                    "description": "",
                    "payload": _profile_payload(engine={"stream_enabled": True}),
                    "is_active": True,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        second_id = response.json()["profile_id"]

        response = self.client.get(reverse("trading:api_v1_realtime_profiles"))
        payload = response.json()
        profiles = payload["profiles"]
        active = [item for item in profiles if item["is_active"]]
        self.assertEqual(len(active), 1)
        self.assertEqual(str(active[0]["profile_id"]), str(second_id))

        response = self.client.patch(
            reverse("trading:api_v1_realtime_profile_detail", kwargs={"profile_id": first_id}),
            data=json.dumps({"is_active": True}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(reverse("trading:api_v1_realtime_profiles"))
        payload = response.json()
        active = [item for item in payload["profiles"] if item["is_active"]]
        self.assertEqual(len(active), 1)
        self.assertEqual(str(active[0]["profile_id"]), str(first_id))

    def test_profile_validation_error(self):
        if not PYDANTIC_AVAILABLE:
            self.skipTest("pydantic not installed")
        invalid_payload = _profile_payload(universe={"max_symbols": 100, "top_n": 500})
        response = self.client.post(
            reverse("trading:api_v1_realtime_profiles"),
            data=json.dumps(
                {
                    "name": "Invalid",
                    "description": "",
                    "payload": invalid_payload,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
