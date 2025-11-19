from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import transaction
from django.utils import timezone

from .models import (
    BacktestRecord as BacktestRecordModel,
    CommunityPost,
    CommunityPostComment,
    CommunityPostLike,
    CommunityTopic,
    UserProfile,
)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def import_user_profiles(base_dir: Path) -> dict[str, int]:
    payload = _load_json(base_dir / "user_profiles.json")
    if not isinstance(payload, dict):
        return {"profiles_created": 0}
    created = 0
    User = get_user_model()
    for user_id, data in payload.items():
        try:
            user = User.objects.get(pk=int(user_id))
        except (User.DoesNotExist, ValueError, TypeError):
            continue
        profile, is_created = UserProfile.objects.get_or_create(user=user)
        profile.display_name = data.get("display_name") or profile.display_name
        profile.cover_color = data.get("cover_color") or profile.cover_color
        profile.bio = data.get("bio") or profile.bio
        profile.avatar_path = data.get("avatar_path") or profile.avatar_path
        profile.feature_image_path = data.get("feature_image_path") or profile.feature_image_path
        gallery = data.get("gallery_paths")
        if isinstance(gallery, str):
            gallery = [gallery] if gallery else []
        elif not isinstance(gallery, list):
            gallery = []
        profile.gallery_paths = gallery
        profile.save()
        if is_created:
            created += 1
    return {"profiles_created": created}


def _parse_ts(value: Any) -> datetime:
    if not value:
        return timezone.now()
    if isinstance(value, datetime):
        return value
    text = str(value).strip().rstrip("Z")
    for fmt in ("%Y-%m-%d %H:%M UTC", "%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    except ValueError:
        return timezone.now()


@transaction.atomic
def import_community(base_dir: Path) -> dict[str, int]:
    stats = {
        "topics_created": 0,
        "posts_created": 0,
        "comments_created": 0,
        "likes_created": 0,
    }
    User = get_user_model()
    topics_payload = _load_json(base_dir / "community_topics.json") or []
    for entry in topics_payload:
        if not isinstance(entry, dict):
            continue
        topic_id = entry.get("topic_id") or f"topic-{entry.get('name', 'legacy')}"
        creator = None
        creator_id = entry.get("creator_id")
        if creator_id:
            try:
                creator = User.objects.filter(pk=int(creator_id)).first()
            except (ValueError, TypeError):
                creator = User.objects.filter(username=str(creator_id)).first()
        obj, created = CommunityTopic.objects.get_or_create(
            topic_id=topic_id,
            defaults={
                "name": entry.get("name") or topic_id,
                "description": entry.get("description", ""),
                "creator": creator,
                "creator_name": entry.get("creator_name")
                or (creator.get_full_name() if creator else ""),
            },
        )
        if created:
            stats["topics_created"] += 1
    posts_payload = _load_json(base_dir / "community_posts.json") or []
    for post in posts_payload:
        if not isinstance(post, dict):
            continue
        author = None
        user_id = post.get("user_id")
        if user_id:
            try:
                author = User.objects.filter(pk=int(user_id)).first()
            except (ValueError, TypeError):
                author = User.objects.filter(username=str(user_id)).first()
        if not author:
            continue
        topic = CommunityTopic.objects.filter(topic_id=post.get("topic_id")).first()
        if topic is None:
            topic = CommunityTopic.objects.create(
                topic_id=post.get("topic_id") or f"topic-{author.id}",
                name=post.get("topic_name") or "Legacy Topic",
            )
        obj, created = CommunityPost.objects.get_or_create(
            post_id=post.get("post_id") or f"post-{author.id}",
            defaults={
                "topic": topic,
                "author": author,
                "author_display_name": post.get("author") or author.get_full_name() or author.username,
                "content": post.get("content") or "",
                "image_path": post.get("image_path") or "",
                "created_at": _parse_ts(post.get("created_at")),
            },
        )
        if created:
            stats["posts_created"] += 1
        for liker in post.get("liked_by") or []:
            liker_obj = User.objects.filter(pk=liker).first()
            if not liker_obj:
                continue
            _, like_created = CommunityPostLike.objects.get_or_create(post=obj, user=liker_obj)
            if like_created:
                stats["likes_created"] += 1
        for comment in post.get("comments") or []:
            commenter = User.objects.filter(pk=comment.get("user_id")).first()
            if not commenter:
                continue
            comment_obj, comment_created = CommunityPostComment.objects.get_or_create(
                comment_id=comment.get("comment_id") or f"comment-{obj.id}",
                defaults={
                    "post": obj,
                    "author": commenter,
                    "author_display_name": comment.get("author") or commenter.username,
                    "content": comment.get("content") or "",
                    "created_at": _parse_ts(comment.get("created_at")),
                },
            )
            if comment_created:
                stats["comments_created"] += 1
    return stats


def import_backtests(base_dir: Path) -> dict[str, int]:
    payload = _load_json(base_dir / "backtest_history.json") or []
    if not isinstance(payload, list):
        return {"backtests_imported": 0}
    User = get_user_model()
    imported = 0
    for entry in payload:
        user_id = entry.get("user_id")
        if not user_id:
            continue
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            continue
        timestamp = entry.get("timestamp")
        try:
            parsed_ts = datetime.fromisoformat(str(timestamp).rstrip("Z"))
        except Exception:
            parsed_ts = timezone.now()
        BacktestRecordModel.objects.update_or_create(
            record_id=entry.get("record_id") or f"legacy-{user_id}-{imported}",
            defaults={
                "user": user,
                "timestamp": parsed_ts,
                "ticker": entry.get("ticker", "UNKNOWN"),
                "benchmark": entry.get("benchmark", ""),
                "engine": entry.get("engine", ""),
                "start_date": entry.get("start_date", ""),
                "end_date": entry.get("end_date", ""),
                "metrics": entry.get("metrics", []),
                "stats": entry.get("stats", {}),
                "params": entry.get("params", {}),
                "warnings": entry.get("warnings", []),
                "snapshot": entry.get("snapshot", {}),
            },
        )
        imported += 1
    return {"backtests_imported": imported}


def import_all(base_dir: Path | None = None, *, include_profiles=True, include_community=True, include_backtests=True) -> dict[str, int]:
    base_dir = Path(base_dir or settings.DATA_CACHE_DIR)
    stats: dict[str, int] = {}
    if include_profiles:
        stats.update(import_user_profiles(base_dir))
    if include_community:
        stats.update(import_community(base_dir))
    if include_backtests:
        stats.update(import_backtests(base_dir))
    return stats
