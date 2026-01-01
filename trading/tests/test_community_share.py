from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from trading.community import list_posts
from trading.models import BacktestRecord, CommunityPost, CommunityTopic


class CommunityShareTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="share_user", password="secret")
        self.topic = CommunityTopic.objects.create(topic_id="topic-share", name="Share")

    def _create_record(self) -> BacktestRecord:
        return BacktestRecord.objects.create(
            record_id="hist-001",
            user=self.user,
            timestamp=timezone.now(),
            ticker="AAPL",
            benchmark="SPY",
            engine="sma",
            start_date="2024-01-01",
            end_date="2024-06-01",
            stats={"total_return": 0.12, "sharpe": 1.4, "max_drawdown": -0.08, "volatility": 0.2},
            params={},
            metrics=[],
            warnings=[],
            snapshot={},
        )

    def test_list_posts_includes_backtest_summary(self):
        record = self._create_record()
        post = CommunityPost.objects.create(
            post_id="post-share",
            topic=self.topic,
            author=self.user,
            author_display_name="ShareUser",
            content="Sharing",
            backtest_record_id=record.record_id,
        )
        results = list_posts(limit=10, topic_id=self.topic.topic_id)
        payload = next(item for item in results if item["post_id"] == post.post_id)
        summary = payload["backtest_summary"]
        self.assertIsNotNone(summary)
        self.assertEqual(summary["record_id"], record.record_id)
        self.assertEqual(summary["ticker"], record.ticker)
        self.assertIn("%", summary["total_return"])

    @override_settings(STATICFILES_STORAGE="django.contrib.staticfiles.storage.StaticFilesStorage")
    def test_share_history_prefills_form(self):
        record = self._create_record()
        self.client.force_login(self.user)
        url = reverse("trading:community") + f"?share_history_id={record.record_id}"
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["share_record"]["record_id"], record.record_id)
        content = response.context["post_form"].initial.get("content", "")
        self.assertIn("Backtest Summary", content)
        self.assertIn(record.ticker, content)
