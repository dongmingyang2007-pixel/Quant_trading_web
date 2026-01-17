from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Iterable, Optional
import uuid

from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.db.models import QuerySet
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from .models import (
    CommunityPost as CommunityPostModel,
    CommunityPostComment,
    CommunityPostLike,
    CommunityTopic as CommunityTopicModel,
    BacktestRecord as BacktestRecordModel,
)
from .storage_utils import delete_media_file
from .observability import record_metric

DEFAULT_TOPIC_ID = "topic-general"
DEFAULT_TOPIC_NAME = _("闲聊大厅")


@dataclass(slots=True)
class CommunityPost:
    topic_id: str
    topic_name: str
    post_id: str
    user_id: str
    author: str
    content: str
    image_path: str | None
    created_at: str
    backtest_record_id: str | None = None
    liked_by: list[str] = field(default_factory=list)
    comments: list[dict[str, Any]] = field(default_factory=list)
    like_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CommunityTopic:
    topic_id: str
    name: str
    description: str
    creator_id: str
    creator_name: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CommunityComment:
    comment_id: str
    user_id: str
    author: str
    content: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_pct(value: Any, *, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "--"


def _format_float(value: Any, *, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _record_field(record: Any, key: str, default: Any = "") -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def build_backtest_summary(record: Any) -> dict[str, Any]:
    stats = _record_field(record, "stats") or {}
    return {
        "record_id": _record_field(record, "record_id", ""),
        "ticker": _record_field(record, "ticker", ""),
        "benchmark": _record_field(record, "benchmark", ""),
        "engine": _record_field(record, "engine", ""),
        "start_date": _record_field(record, "start_date", ""),
        "end_date": _record_field(record, "end_date", ""),
        "total_return": _format_pct(stats.get("total_return")),
        "sharpe": _format_float(stats.get("sharpe")),
        "max_drawdown": _format_pct(stats.get("max_drawdown")),
        "volatility": _format_pct(stats.get("volatility")),
        "cagr": _format_pct(stats.get("cagr")),
    }


def _ensure_default_topic() -> CommunityTopicModel:
    obj, created = CommunityTopicModel.objects.get_or_create(
        topic_id=DEFAULT_TOPIC_ID,
        defaults={
            "name": DEFAULT_TOPIC_NAME,
            "description": _("自由讨论市场洞察、策略灵感与生活日常。"),
            "creator_name": _("系统"),
        },
    )
    return obj


def _serialize_topic(topic: CommunityTopicModel) -> dict[str, Any]:
    return {
        "topic_id": topic.topic_id,
        "name": topic.name,
        "description": topic.description,
        "creator_id": str(topic.creator_id) if topic.creator_id else "",
        "creator_name": topic.creator_name,
        "created_at": topic.created_at.strftime("%Y-%m-%d %H:%M UTC"),
    }


def list_topics(limit: Optional[int] = None) -> list[dict[str, Any]]:
    _ensure_default_topic()
    qs = CommunityTopicModel.objects.order_by("-created_at")
    if limit:
        qs = qs[:limit]
    return [_serialize_topic(topic) for topic in qs]


def create_topic(name: str, description: str, *, creator_id: str, creator_name: str) -> dict[str, Any]:
    _ensure_default_topic()
    UserModel = get_user_model()
    creator = None
    if creator_id:
        try:
            creator = UserModel.objects.get(pk=creator_id)
        except UserModel.DoesNotExist:
            creator = None
    topic = CommunityTopicModel.objects.create(
        topic_id=f"topic-{uuid.uuid4().hex[:10]}",
        name=name.strip(),
        description=description.strip(),
        creator=creator,
        creator_name=creator_name,
    )
    return _serialize_topic(topic)


def get_topic(topic_id: str | None) -> dict[str, Any]:
    _ensure_default_topic()
    if topic_id:
        topic = CommunityTopicModel.objects.filter(topic_id=topic_id).first()
        if topic:
            return _serialize_topic(topic)
    topic = CommunityTopicModel.objects.filter(topic_id=DEFAULT_TOPIC_ID).first()
    return _serialize_topic(topic)


def _serialize_comment(comment: CommunityPostComment) -> dict[str, Any]:
    profile = getattr(comment.author, "profile", None)
    user_slug = str(profile.slug) if profile else ""
    avatar_path = ""
    if profile and profile.avatar_path:
        avatar_path = profile.avatar_path
    return {
        "comment_id": comment.comment_id,
        "user_id": str(comment.author_id),
        "user_slug": user_slug,
        "author": comment.author_display_name,
        "content": comment.content,
        "created_at": comment.created_at.strftime("%Y-%m-%d %H:%M UTC"),
        "avatar_path": avatar_path,
    }


def _serialize_post(post: CommunityPostModel, *, backtest_summaries: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    profile = getattr(post.author, "profile", None)
    user_slug = str(profile.slug) if profile else ""
    avatar_path = ""
    if profile and profile.avatar_path:
        avatar_path = profile.avatar_path
    comments_qs = post.comments.all()
    comment_entries = [_serialize_comment(comment) for comment in comments_qs]
    liked_ids = [str(pk) for pk in post.liked_by.values_list("id", flat=True)]
    summary = None
    if backtest_summaries and post.backtest_record_id:
        summary = backtest_summaries.get(post.backtest_record_id)
    return {
        "topic_id": post.topic.topic_id,
        "topic_name": post.topic.name,
        "post_id": post.post_id,
        "user_id": str(post.author_id),
        "user_slug": user_slug,
        "avatar_path": avatar_path,
        "author": post.author_display_name,
        "content": post.content,
        "image_path": post.image_path or "",
        "backtest_record_id": post.backtest_record_id or "",
        "backtest_summary": summary,
        "created_at": post.created_at.strftime("%Y-%m-%d %H:%M UTC"),
        "liked_by": liked_ids,
        "like_count": len(liked_ids),
        "like_events": [
            {"user_id": str(entry.user_id), "created_at": entry.created_at.strftime("%Y-%m-%d %H:%M UTC")}
            for entry in post.like_entries.order_by("-created_at")[:20]
        ],
        "comments": comment_entries,
        "comment_count": len(comment_entries),
    }


def _build_posts_queryset(*, topic_id: str | None = None, user_id: str | None = None):
    qs = (
        CommunityPostModel.objects.select_related("topic", "author", "author__profile")
        .prefetch_related(
            "liked_by",
            "like_entries",
            "comments",
            "comments__author",
            "comments__author__profile",
        )
        .order_by("-created_at")
    )
    if topic_id and topic_id != "all":
        qs = qs.filter(topic__topic_id=topic_id)
    if user_id:
        qs = qs.filter(author_id=user_id)
    return qs


def serialize_posts(posts: Iterable[CommunityPostModel]) -> list[dict[str, Any]]:
    posts_list = list(posts)
    backtest_summaries: dict[str, dict[str, Any]] = {}
    record_ids = {post.backtest_record_id for post in posts_list if post.backtest_record_id}
    if record_ids:
        for record in BacktestRecordModel.objects.filter(record_id__in=record_ids):
            backtest_summaries[record.record_id] = build_backtest_summary(record)
    return [_serialize_post(post, backtest_summaries=backtest_summaries) for post in posts_list]


def list_posts(
    limit: int | None = 50,
    *,
    topic_id: str | None = None,
    user_id: str | None = None,
    offset: int | None = None,
    return_queryset: bool = False,
) -> list[dict[str, Any]] | QuerySet[CommunityPostModel]:
    qs = _build_posts_queryset(topic_id=topic_id, user_id=user_id)
    if offset is not None:
        if limit is not None:
            qs = qs[offset : offset + limit]
        else:
            qs = qs[offset:]
    elif limit is not None:
        qs = qs[:limit]
    if return_queryset:
        return qs
    return serialize_posts(qs)


def append_post(post: CommunityPost) -> None:
    topic = CommunityTopicModel.objects.filter(topic_id=post.topic_id).first() or _ensure_default_topic()
    UserModel = get_user_model()
    author = UserModel.objects.get(pk=post.user_id)
    CommunityPostModel.objects.create(
        post_id=post.post_id,
        topic=topic,
        author=author,
        author_display_name=post.author,
        content=post.content,
        image_path=post.image_path or "",
        backtest_record_id=post.backtest_record_id or "",
    )
    record_metric(
        "community.post.create",
        topic_id=topic.topic_id,
        user_id=post.user_id,
        has_image=bool(post.image_path),
    )


def remove_post(post_id: str, user_id: str) -> tuple[bool, dict[str, str]]:
    if not post_id or not user_id:
        return False, {"error": "invalid"}
    post = (
        CommunityPostModel.objects.select_related("author", "topic")
        .filter(post_id=post_id)
        .first()
    )
    if not post:
        return False, {"error": "not_found"}
    if str(post.author_id) != str(user_id):
        return False, {"error": "forbidden"}
    topic_id = post.topic.topic_id
    image_path = post.image_path
    post.delete()
    if image_path and not image_path.startswith(("http://", "https://")):
        delete_media_file(image_path)
    record_metric(
        "community.post.delete",
        topic_id=topic_id,
        user_id=user_id,
    )
    return True, {"topic_id": topic_id}


def build_post(
    *,
    topic_id: str,
    topic_name: str,
    user_id: str,
    author: str,
    content: str,
    image_path: str | None,
    backtest_record_id: str | None = None,
) -> CommunityPost:
    return CommunityPost(
        topic_id=topic_id or DEFAULT_TOPIC_ID,
        topic_name=topic_name or DEFAULT_TOPIC_NAME,
        user_id=user_id,
        post_id=f"post-{uuid.uuid4().hex[:10]}",
        author=author,
        content=content,
        image_path=image_path,
        backtest_record_id=backtest_record_id or "",
        created_at=timezone.now().strftime("%Y-%m-%d %H:%M UTC"),
    )


def toggle_like(post_id: str, user_id: str) -> dict[str, Any] | None:
    try:
        post = CommunityPostModel.objects.get(post_id=post_id)
    except CommunityPostModel.DoesNotExist:
        return None
    UserModel = get_user_model()
    try:
        user = UserModel.objects.get(pk=user_id)
    except UserModel.DoesNotExist:
        return None
    try:
        like, created = CommunityPostLike.objects.get_or_create(post=post, user=user)
        if not created:
            like.delete()
            liked = False
        else:
            liked = True
    except IntegrityError:
        return None
    like_count = post.liked_by.count()
    record_metric(
        "community.post.toggle_like",
        post_id=post_id,
        user_id=user_id,
        liked=liked,
        like_count=like_count,
    )
    return {"liked": liked, "like_count": like_count}


def add_comment(post_id: str, comment: CommunityComment) -> dict[str, Any] | None:
    try:
        post = CommunityPostModel.objects.get(post_id=post_id)
    except CommunityPostModel.DoesNotExist:
        return None
    UserModel = get_user_model()
    try:
        author = UserModel.objects.get(pk=comment.user_id)
    except UserModel.DoesNotExist:
        return None
    obj = CommunityPostComment.objects.create(
        comment_id=comment.comment_id,
        post=post,
        author=author,
        author_display_name=comment.author,
        content=comment.content,
    )
    payload = _serialize_comment(obj)
    record_metric(
        "community.post.comment",
        post_id=post_id,
        user_id=comment.user_id,
    )
    return payload
