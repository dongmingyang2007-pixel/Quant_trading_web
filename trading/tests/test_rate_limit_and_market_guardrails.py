from __future__ import annotations

from unittest import mock

from django.contrib.auth import get_user_model
from django.core.cache import caches
from django.test import TestCase
from django.urls import reverse

from trading.rate_limit import check_rate_limit


class RateLimitHelperTests(TestCase):
    def setUp(self):
        caches["default"].clear()

    def test_check_rate_limit_blocks_after_max(self):
        state1 = check_rate_limit(cache_alias="default", key="rl:test", window=30, max_calls=1)
        self.assertFalse(state1.limited)
        state2 = check_rate_limit(cache_alias="default", key="rl:test", window=30, max_calls=1)
        self.assertTrue(state2.limited)
        self.assertGreaterEqual(state2.retry_after, 1)


class MarketInsightsGuardsTests(TestCase):
    def setUp(self):
        caches["default"].clear()
        User = get_user_model()
        self.user = User.objects.create_user(username="tester", email="t@example.com", password="pwd")
        self.client.force_login(self.user)
        self.url = reverse("trading:market_insights_data")

    def _dummy_future(self, value):
        class _Future:
            def __init__(self, val):
                self.val = val
                self.cancelled = False

            def result(self, timeout=None):
                return self.val

            def cancel(self):
                self.cancelled = True

        return _Future(value)

    def test_market_params_clamped_and_sanitized(self):
        with mock.patch("trading.views.market._rank_symbols") as mock_rank, mock.patch(
            "trading.views.market._MARKET_EXECUTOR"
        ) as executor, mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ):
            mock_rank.return_value = [
                {
                    "symbol": "ABC",
                    "price": 10.0,
                    "change_pct_period": 1.0,
                    "change_pct_day": 0.5,
                    "period_label": "x",
                    "period_label_en": "x",
                    "series": [],
                    "timestamps": [],
                }
            ]
            executor.submit.return_value = self._dummy_future(({"ABC": object()}, "alpaca"))
            resp = self.client.get(self.url, {"timeframe": "bad", "limit": "9999", "query": "abc$%^"}, follow=True)

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["timeframe"]["clamped"])
            self.assertTrue(data["limit_clamped"])
            self.assertEqual(data["query"], "ABC")

    def test_market_rate_limit_blocks_follow_up_calls(self):
        caches["default"].clear()
        with mock.patch("trading.views.market._rank_symbols") as mock_rank, mock.patch(
            "trading.views.market._MARKET_EXECUTOR"
        ) as executor, mock.patch(
            "trading.views.market.resolve_alpaca_data_credentials",
            return_value=("key", "secret"),
        ), mock.patch(
            "trading.views.market.MARKET_RATE_WINDOW", 60
        ), mock.patch(
            "trading.views.market.MARKET_RATE_MAX_CALLS", 1
        ):
            mock_rank.return_value = [
                {
                    "symbol": "XYZ",
                    "price": 5.0,
                    "change_pct_period": 0.1,
                    "change_pct_day": 0.2,
                    "period_label": "x",
                    "period_label_en": "x",
                    "series": [],
                    "timestamps": [],
                }
            ]
            executor.submit.side_effect = [
                self._dummy_future(({"XYZ": object()}, "alpaca")),
                self._dummy_future(({"XYZ": object()}, "alpaca")),
            ]
            first = self.client.get(self.url, {"query": "xyz"})
            second = self.client.get(self.url, {"query": "xyz"})

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 429)
