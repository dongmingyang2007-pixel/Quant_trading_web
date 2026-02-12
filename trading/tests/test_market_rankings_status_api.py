from __future__ import annotations

from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


class MarketRankingsStatusApiTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="rankings_status", password="secret123")
        self.url = reverse("trading:market_rankings_status")

    def test_requires_authentication(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)

    def test_returns_status_items_for_each_supported_combination(self):
        self.client.force_login(self.user)

        support = {
            "gainers": ["1d", "5d"],
            "losers": ["1d"],
            "most_active": ["1d"],
            "top_turnover": ["1d"],
        }

        def _snapshot_state(*, timeframe_key, provider, used_active_snapshot, list_type=None):
            self.assertEqual(provider, "massive")
            self.assertTrue(used_active_snapshot)
            building = timeframe_key == "5d"
            return {
                "served_from": "building_fallback" if building else "active",
                "active_generated_at": 1_707_000_000,
                "building_progress": 42 if building else 100,
                "building": building,
                "build_state": "running" if building else "idle",
                "stale_seconds": 25 if building else 2,
                "provider": provider,
            }

        with mock.patch(
            "trading.views.market.resolve_market_provider",
            return_value="massive",
        ), mock.patch(
            "trading.views.market._resolve_list_timeframe_support",
            return_value=support,
        ), mock.patch(
            "trading.views.market._snapshot_state_payload",
            side_effect=_snapshot_state,
        ), mock.patch(
            "trading.views.market._resolve_building_snapshot_payload",
            side_effect=lambda timeframe_key, provider=None: (
                {
                    "status": "running",
                    "source": "massive",
                    "chunks_completed": 42,
                    "total_chunks": 100,
                    "progress": {"status": "running", "chunks_completed": 42, "total_chunks": 100},
                }
                if timeframe_key == "5d"
                else {"status": "idle", "source": "massive"}
            ),
        ):
            response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("provider"), "massive")
        self.assertIn("request_id", payload)
        items = payload.get("items", [])
        self.assertEqual(len(items), 5)
        by_pair = {(item.get("list_type"), item.get("timeframe", {}).get("key")): item for item in items}
        self.assertIn(("gainers", "5d"), by_pair)
        gainers_5d = by_pair[("gainers", "5d")]
        self.assertTrue(gainers_5d.get("building"))
        self.assertEqual(gainers_5d.get("progress"), 42)
        self.assertEqual(gainers_5d.get("build_progress"), 42)
        self.assertEqual(gainers_5d.get("build_state"), "running")
        self.assertIsInstance(gainers_5d.get("cycle_progress"), int)
        self.assertEqual(gainers_5d.get("provider"), "massive")
        self.assertEqual(gainers_5d.get("stale_seconds"), 25)

        groups = payload.get("groups", [])
        self.assertEqual(len(groups), 4)
        by_timeframe = {group.get("timeframe", {}).get("key"): group for group in groups}
        self.assertIn("5d", by_timeframe)
        group_5d_items = by_timeframe["5d"].get("items", [])
        self.assertEqual(len(group_5d_items), 4)
        group_5d_by_list = {item.get("list_type"): item for item in group_5d_items}
        self.assertTrue(group_5d_by_list["gainers"].get("supported"))
        self.assertTrue(group_5d_by_list["gainers"].get("building"))
        self.assertEqual(group_5d_by_list["gainers"].get("build_state"), "running")
        self.assertIsInstance(group_5d_by_list["gainers"].get("build_progress"), int)
        self.assertIsInstance(group_5d_by_list["gainers"].get("cycle_progress"), int)
        self.assertFalse(group_5d_by_list["losers"].get("supported"))

    def test_stalled_build_reports_stalled_state_with_zero_build_progress(self):
        self.client.force_login(self.user)

        support = {
            "gainers": ["5d"],
            "losers": [],
            "most_active": [],
            "top_turnover": [],
        }

        with mock.patch(
            "trading.views.market.resolve_market_provider",
            return_value="massive",
        ), mock.patch(
            "trading.views.market._resolve_list_timeframe_support",
            return_value=support,
        ), mock.patch(
            "trading.views.market._snapshot_state_payload",
            return_value={
                "served_from": "active",
                "active_generated_at": 1_707_000_100,
                "building_progress": 100,
                "building": False,
                "build_state": "stalled",
                "stale_seconds": 180,
                "stale_threshold_seconds": 120,
                "provider": "massive",
                "active_schema_valid": True,
            },
        ), mock.patch(
            "trading.views.market._resolve_building_snapshot_payload",
            return_value={
                "status": "stalled",
                "started_at": 1_707_000_000,
                "updated_at": 1_707_000_060,
                "error": "stalled_timeout",
            },
        ):
            response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        items = payload.get("items", [])
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.get("build_state"), "stalled")
        self.assertEqual(item.get("build_progress"), 0)
        self.assertEqual(item.get("progress"), 0)
        self.assertEqual(item.get("build_error_code"), "stalled_timeout")
        self.assertGreaterEqual(item.get("cycle_progress") or 0, 100)
