from __future__ import annotations

import json
import tempfile
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase

from trading.models import BacktestRecord, CommunityPost, CommunityTopic, UserProfile


class ImportLegacyCacheCommandTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="legacy", password="pass123")

    def _write_json(self, base_dir: Path, name: str, payload):
        (base_dir / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_imports_profiles_community_and_backtests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            self._write_json(
                base,
                "user_profiles.json",
                {str(self.user.id): {"display_name": "Legacy User", "bio": "old data"}},
            )
            self._write_json(
                base,
                "community_topics.json",
                [{"topic_id": "topic-1", "name": "Legacy Topic", "creator_id": self.user.id}],
            )
            self._write_json(
                base,
                "community_posts.json",
                [
                    {
                        "post_id": "post-1",
                        "topic_id": "topic-1",
                        "user_id": self.user.id,
                        "content": "Hello world",
                        "liked_by": [self.user.id],
                        "comments": [
                            {"comment_id": "c-1", "user_id": self.user.id, "content": "Nice"}
                        ],
                    }
                ],
            )
            self._write_json(
                base,
                "backtest_history.json",
                [
                    {
                        "record_id": "hist-1",
                        "user_id": self.user.id,
                        "ticker": "AAPL",
                        "start_date": "2020-01-01",
                        "end_date": "2020-06-30",
                    }
                ],
            )
            call_command("import_legacy_cache", "--base-dir", tmpdir)

        profile = UserProfile.objects.get(user=self.user)
        self.assertEqual(profile.display_name, "Legacy User")
        self.assertTrue(CommunityTopic.objects.filter(topic_id="topic-1").exists())
        self.assertTrue(CommunityPost.objects.filter(post_id="post-1").exists())
        self.assertTrue(BacktestRecord.objects.filter(record_id="hist-1").exists())

    def test_handles_missing_files_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            call_command("import_legacy_cache", "--base-dir", tmpdir, "--profiles")
        self.assertEqual(UserProfile.objects.count(), 0)
