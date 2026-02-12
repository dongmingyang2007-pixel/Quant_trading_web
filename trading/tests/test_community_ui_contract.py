from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from trading.models import CommunityPost, CommunityTopic, UserProfile


@override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
)
class CommunityUiContractTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls.user = User.objects.create_user(username="community_ui", password="secret")
        UserProfile.objects.create(user=cls.user, display_name="Community UI")
        cls.topic = CommunityTopic.objects.create(topic_id="topic-ui", name="UI")
        for idx in range(25):
            CommunityPost.objects.create(
                post_id=f"ui-post-{idx}",
                topic=cls.topic,
                author=cls.user,
                author_display_name="Community UI",
                content=f"Post {idx}",
            )

    def setUp(self):
        self.client.force_login(self.user)

    def test_community_hub_contains_v2_nodes_and_load_more_button(self):
        response = self.client.get(reverse("trading:community") + "?topic=topic-ui")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "community-page-v2")
        self.assertContains(response, 'data-role="topic-list"')
        self.assertContains(response, 'id="community-post-list"')
        self.assertContains(response, 'data-role="load-more-button"')
        self.assertNotContains(response, 'hx-trigger="revealed"')

    def test_htmx_feed_chunk_includes_oob_pagination(self):
        response = self.client.get(reverse("trading:community") + "?page=2", HTTP_HX_REQUEST="true")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'hx-swap-oob="true"')
        self.assertContains(response, 'id="community-feed-pagination"')

    def test_community_write_contract_nodes_exist(self):
        response = self.client.get(reverse("trading:community_write"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-role="write-form"')
        self.assertContains(response, 'data-role="editor"')
        self.assertContains(response, 'data-role="drafts-list"')
        self.assertContains(response, 'id="mathModal"')
        self.assertContains(response, 'id="backtestSelectModal"')
        self.assertContains(response, 'data-role="cover-input"')
