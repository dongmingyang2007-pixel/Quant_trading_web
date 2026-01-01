from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models


class BacktestRecord(models.Model):
    record_id = models.CharField(max_length=64, unique=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="backtests")
    timestamp = models.DateTimeField()
    ticker = models.CharField(max_length=32)
    benchmark = models.CharField(max_length=32, blank=True, default="")
    engine = models.CharField(max_length=64, blank=True, default="")
    start_date = models.CharField(max_length=16, blank=True, default="")
    end_date = models.CharField(max_length=16, blank=True, default="")
    metrics = models.JSONField(default=list)
    stats = models.JSONField(default=dict)
    params = models.JSONField(default=dict)
    warnings = models.JSONField(default=list)
    snapshot = models.JSONField(default=dict)
    title = models.CharField(max_length=120, blank=True, default="")
    tags = models.JSONField(default=list, blank=True)
    notes = models.TextField(blank=True, default="")
    starred = models.BooleanField(default=False)

    class Meta:
        indexes = [
            models.Index(fields=["user", "timestamp"]),
        ]
        ordering = ["-timestamp"]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.ticker} {self.start_date}->{self.end_date} ({self.user_id})"


class StrategyPreset(models.Model):
    preset_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="strategy_presets")
    name = models.CharField(max_length=80)
    description = models.CharField(max_length=200, blank=True, default="")
    payload = models.JSONField(default=dict)
    is_default = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["user", "-updated_at"]),
        ]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.name} ({self.user_id})"


def generate_post_id() -> str:
    return f"post-{uuid.uuid4().hex[:10]}"


def generate_comment_id() -> str:
    return f"comment-{uuid.uuid4().hex[:10]}"


class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    slug = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    display_name = models.CharField(max_length=64, blank=True, default="")
    cover_color = models.CharField(max_length=16, default="#116e5f", blank=True)
    bio = models.TextField(blank=True, default="")
    avatar_path = models.CharField(max_length=255, blank=True, default="")
    feature_image_path = models.CharField(max_length=255, blank=True, default="")
    gallery_paths = models.JSONField(default=list, blank=True)
    market_watchlist = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.user.username} profile"


class CommunityTopic(models.Model):
    topic_id = models.SlugField(max_length=64, unique=True)
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True, default="")
    creator = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="created_topics"
    )
    creator_name = models.CharField(max_length=120, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:  # pragma: no cover
        return self.name


class CommunityPost(models.Model):
    post_id = models.CharField(max_length=40, unique=True, default=generate_post_id)
    topic = models.ForeignKey(CommunityTopic, on_delete=models.CASCADE, related_name="posts")
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="community_posts")
    author_display_name = models.CharField(max_length=120)
    content = models.TextField()
    image_path = models.CharField(max_length=255, blank=True, default="")
    backtest_record_id = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    liked_by = models.ManyToManyField(
        settings.AUTH_USER_MODEL, through="CommunityPostLike", related_name="liked_community_posts", blank=True
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["topic", "-created_at"]),
        ]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.author_display_name}: {self.content[:20]}"


class CommunityPostLike(models.Model):
    post = models.ForeignKey(CommunityPost, on_delete=models.CASCADE, related_name="like_entries")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("post", "user")


class CommunityPostComment(models.Model):
    comment_id = models.CharField(max_length=48, unique=True, default=generate_comment_id)
    post = models.ForeignKey(CommunityPost, on_delete=models.CASCADE, related_name="comments")
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    author_display_name = models.CharField(max_length=120)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
