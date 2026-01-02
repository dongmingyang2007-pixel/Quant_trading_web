from __future__ import annotations

import json
from unittest import mock

import pandas as pd

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


class PreflightViewTests(TestCase):
    def setUp(self):
        user = get_user_model().objects.create_user(username="preflight", password="secret123")
        self.client.force_login(user)

    def _payload(self) -> dict[str, str]:
        return {
            "ticker": "AAPL",
            "benchmark_ticker": "SPY",
            "start_date": "2023-01-01",
            "end_date": "2023-06-30",
            "capital": "250000",
            "ml_mode": "light",
        }

    @mock.patch("trading.api.views_v1.fetch_price_data")
    def test_preflight_returns_rows_and_quality(self, mock_fetch):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        frame = pd.DataFrame(
            {
                "open": [100.0 + i for i in range(len(idx))],
                "high": [101.0 + i for i in range(len(idx))],
                "low": [99.0 + i for i in range(len(idx))],
                "close": [100.5 + i for i in range(len(idx))],
                "adj close": [100.5 + i for i in range(len(idx))],
                "volume": [1_000_000.0 for _ in range(len(idx))],
            },
            index=idx,
        )
        frame.attrs["data_source"] = "yfinance"
        frame.attrs["cache_path"] = "/tmp/AAPL.csv"
        mock_fetch.return_value = (frame, ["cache hit"])

        response = self.client.post(
            reverse("trading:api_v1_backtest_preflight"),
            data=json.dumps(self._payload()),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["rows"], 10)
        self.assertIn("data_quality", payload)
        self.assertEqual(payload["source"], "yfinance")
        self.assertEqual(payload["cache_path"], "/tmp/AAPL.csv")
