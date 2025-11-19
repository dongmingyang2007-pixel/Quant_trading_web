from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase

from trading.community import list_posts
from trading.models import (
    CommunityPost,
    CommunityPostComment,
    CommunityTopic,
    UserProfile,
)


class CommunityQueryTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls.user = User.objects.create_user(username="alice", password="pw")
        cls.other = User.objects.create_user(username="bob", password="pw")
        UserProfile.objects.create(user=cls.user, display_name="Alice", avatar_path="avatars/alice.jpg")
        UserProfile.objects.create(user=cls.other, display_name="Bob", avatar_path="avatars/bob.jpg")

        cls.topic_a = CommunityTopic.objects.create(topic_id="topic-a", name="Topic A")
        cls.topic_b = CommunityTopic.objects.create(topic_id="topic-b", name="Topic B")

        cls.post_a = CommunityPost.objects.create(
            post_id="post-a",
            topic=cls.topic_a,
            author=cls.user,
            author_display_name="Alice",
            content="Post in topic A",
        )
        cls.post_b = CommunityPost.objects.create(
            post_id="post-b",
            topic=cls.topic_b,
            author=cls.user,
            author_display_name="Alice",
            content="Post in topic B",
        )
        cls.post_c = CommunityPost.objects.create(
            post_id="post-c",
            topic=cls.topic_a,
            author=cls.other,
            author_display_name="Bob",
            content="Second post in topic A",
        )
        cls.comment = CommunityPostComment.objects.create(
            comment_id="comment-a",
            post=cls.post_a,
            author=cls.other,
            author_display_name="Bob",
            content="Nice idea",
        )

    def test_list_posts_filters_by_topic_and_user(self):
        topic_results = list_posts(limit=10, topic_id=self.topic_a.topic_id)
        self.assertEqual(len(topic_results), 2)
        self.assertTrue(all(post["topic_id"] == self.topic_a.topic_id for post in topic_results))

        user_results = list_posts(limit=10, user_id=str(self.user.id))
        self.assertEqual(len(user_results), 2)
        self.assertTrue(all(post["user_id"] == str(self.user.id) for post in user_results))

    def test_list_posts_includes_avatar_data(self):
        results = list_posts(limit=5, topic_id=self.topic_a.topic_id)
        sample = next(post for post in results if post["post_id"] == "post-a")
        self.assertEqual(sample["avatar_path"], "avatars/alice.jpg")
        self.assertEqual(sample["author"], "Alice")
        comment = next(item for item in sample["comments"] if item["comment_id"] == "comment-a")
        self.assertEqual(comment["avatar_path"], "avatars/bob.jpg")
