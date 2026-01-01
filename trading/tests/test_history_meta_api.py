from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from trading.models import BacktestRecord as BacktestRecordModel


class HistoryMetaApiTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="history_user", password="secret123")
        self.client.force_login(self.user)
        self.record = BacktestRecordModel.objects.create(
            record_id="hist-101",
            user=self.user,
            timestamp=timezone.now(),
            ticker="AAPL",
            benchmark="SPY",
            engine="sma",
            start_date="2023-01-01",
            end_date="2023-06-30",
            metrics=[],
            stats={},
            params={},
            warnings=[],
            snapshot={"ticker": "AAPL"},
        )

    def test_patch_history_meta(self):
        payload = {"title": "Momentum Core", "tags": ["swing", "ml"], "notes": "Focus on 1M", "starred": True}
        response = self.client.patch(
            reverse("trading:api_v1_history_meta", kwargs={"record_id": self.record.record_id}),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["record_id"], self.record.record_id)
        self.assertEqual(data["title"], "Momentum Core")
        self.assertEqual(data["tags"], ["swing", "ml"])
        self.assertEqual(data["notes"], "Focus on 1M")
        self.assertTrue(data["starred"])
