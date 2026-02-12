from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase

from trading.market_provider import resolve_market_context, resolve_market_provider, resolve_news_provider
from trading.profile import save_api_credentials


class MarketProviderResolutionTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="provider-user", password="secret123")

    def test_market_provider_reads_user_setting(self):
        save_api_credentials(str(self.user.id), {"market_data_provider": "massive"})
        provider = resolve_market_provider(user=self.user)
        self.assertEqual(provider, "massive")

    def test_news_provider_follow_data_tracks_market_provider(self):
        save_api_credentials(
            str(self.user.id),
            {
                "market_data_provider": "massive",
                "market_news_provider": "follow_data",
            },
        )
        news_provider = resolve_news_provider(user=self.user)
        self.assertEqual(news_provider, "massive")

    def test_market_context_contains_execution_source(self):
        save_api_credentials(str(self.user.id), {"market_data_provider": "massive"})
        context = resolve_market_context(user=self.user)
        self.assertEqual(context.get("market_data_source"), "massive")
        self.assertEqual(context.get("execution_source"), "alpaca")
