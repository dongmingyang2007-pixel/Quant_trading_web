from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


def _preset_payload(ticker: str = "AAPL", *, include_ticker_dates: bool = False) -> dict[str, object]:
    payload = {
        "ticker": ticker,
        "benchmark_ticker": "SPY",
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "capital": "250000",
        "ml_mode": "light",
    }
    if include_ticker_dates:
        payload["include_ticker_dates"] = True
    return payload


class StrategyPresetApiTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="preset_user", password="secret123")
        self.client.force_login(self.user)

    def test_preset_crud_and_default(self):
        create_payload = {
            "name": "Core Trend",
            "description": "Primary setup",
            "payload": _preset_payload(include_ticker_dates=True),
            "is_default": True,
        }
        response = self.client.post(
            reverse("trading:api_v1_presets"),
            data=json.dumps(create_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        preset_id = data["preset_id"]
        self.assertTrue(data["is_default"])
        self.assertEqual(data["payload"]["ticker"], "AAPL")

        response = self.client.get(reverse("trading:api_v1_presets"))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["presets"]), 1)

        response = self.client.post(
            reverse("trading:api_v1_presets"),
            data=json.dumps(
                {
                    "name": "Swing Setup",
                    "description": "",
                    "payload": _preset_payload("MSFT", include_ticker_dates=True),
                    "is_default": True,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        second_id = response.json()["preset_id"]

        response = self.client.get(reverse("trading:api_v1_presets"))
        presets = response.json()["presets"]
        defaults = [preset for preset in presets if preset["is_default"]]
        self.assertEqual(len(defaults), 1)
        self.assertEqual(str(defaults[0]["preset_id"]), str(second_id))

        response = self.client.patch(
            reverse("trading:api_v1_preset_detail", kwargs={"preset_id": preset_id}),
            data=json.dumps({"name": "Core Trend v2", "is_default": True}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        updated = response.json()
        self.assertEqual(updated["name"], "Core Trend v2")
        self.assertTrue(updated["is_default"])

        response = self.client.delete(reverse("trading:api_v1_preset_detail", kwargs={"preset_id": preset_id}))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["deleted"])

    def test_preset_strips_ticker_dates_by_default(self):
        create_payload = {
            "name": "No Dates",
            "description": "",
            "payload": _preset_payload(),
            "is_default": False,
        }
        response = self.client.post(
            reverse("trading:api_v1_presets"),
            data=json.dumps(create_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()
        payload = data["payload"]
        self.assertNotIn("ticker", payload)
        self.assertNotIn("start_date", payload)
        self.assertNotIn("end_date", payload)
