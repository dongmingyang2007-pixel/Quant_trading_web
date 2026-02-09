from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from trading.models import CommunityPost, CommunityPostLike, CommunityTopic, Notification
from trading.notifications_cache import get_unread_notifications_count


@override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
)
class NotificationCacheTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.author = user_model.objects.create_user(username="author", password="secret123")
        self.actor = user_model.objects.create_user(username="actor", password="secret123")
        self.topic = CommunityTopic.objects.create(
            topic_id="topic-cache",
            name="Cache Topic",
            description="",
            creator=self.author,
            creator_name=self.author.username,
        )
        self.post = CommunityPost.objects.create(
            topic=self.topic,
            author=self.author,
            author_display_name=self.author.username,
            content="hello",
        )

    def test_like_signal_invalidates_unread_cache(self):
        self.assertEqual(get_unread_notifications_count(self.author), 0)
        self.assertEqual(get_unread_notifications_count(self.author), 0)  # prime cache
        CommunityPostLike.objects.create(post=self.post, user=self.actor)
        self.assertEqual(get_unread_notifications_count(self.author), 1)

    def test_notifications_page_marks_read_and_invalidates_cache(self):
        Notification.objects.create(
            recipient=self.author,
            actor=self.actor,
            verb=Notification.VERB_LIKED,
            target_post=self.post,
        )
        self.assertEqual(get_unread_notifications_count(self.author), 1)
        self.client.force_login(self.author)
        response = self.client.get(reverse("trading:community_notifications"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(get_unread_notifications_count(self.author), 0)
