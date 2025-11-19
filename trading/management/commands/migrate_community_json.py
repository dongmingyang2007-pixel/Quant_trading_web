from __future__ import annotations

import json
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from trading.models import (
    CommunityPost as CommunityPostModel,
    CommunityPostComment,
    CommunityPostLike,
    CommunityTopic as CommunityTopicModel,
)


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M UTC", "%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


class Command(BaseCommand):
    help = "Import legacy community JSON data (topics/posts) into the database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--topics",
            action="store_true",
            help="Only import topics.",
        )
        parser.add_argument(
            "--posts",
            action="store_true",
            help="Only import posts.",
        )

    def handle(self, *args, **options):
        import_topics = options["topics"] or not options["posts"]
        import_posts = options["posts"] or not options["topics"]

        data_dir = Path(settings.DATA_CACHE_DIR)
        topic_file = data_dir / "community_topics.json"
        post_file = data_dir / "community_posts.json"

        if import_topics and not topic_file.exists():
            raise CommandError(f"Topic file not found: {topic_file}")
        if import_posts and not post_file.exists():
            raise CommandError(f"Post file not found: {post_file}")

        stats: dict[str, int] = {
            "topics_created": 0,
            "topics_updated": 0,
            "posts_created": 0,
            "posts_updated": 0,
            "likes_created": 0,
            "comments_created": 0,
            "posts_skipped": 0,
            "comments_skipped": 0,
        }

        UserModel = get_user_model()

        @transaction.atomic
        def _import_topics():
            payload = json.loads(topic_file.read_text(encoding="utf-8"))
            for entry in payload:
                topic_id = entry.get("topic_id")
                if not topic_id:
                    continue
                creator = None
                creator_id = entry.get("creator_id")
                if creator_id:
                    try:
                        creator = UserModel.objects.filter(pk=int(creator_id)).first()
                    except (ValueError, TypeError):
                        creator = UserModel.objects.filter(username=str(creator_id)).first()
                defaults = {
                    "name": entry.get("name") or topic_id,
                    "description": entry.get("description", ""),
                    "creator": creator,
                    "creator_name": entry.get("creator_name", "") or (creator.get_full_name() if creator else ""),
                }
                created = _parse_dt(entry.get("created_at"))
                obj, was_created = CommunityTopicModel.objects.update_or_create(
                    topic_id=topic_id,
                    defaults=defaults,
                )
                if created:
                    obj.created_at = created
                    obj.save(update_fields=["created_at"])
                if was_created:
                    stats["topics_created"] += 1
                else:
                    stats["topics_updated"] += 1

        @transaction.atomic
        def _import_posts():
            topics = {topic.topic_id: topic for topic in CommunityTopicModel.objects.all()}
            payload = json.loads(post_file.read_text(encoding="utf-8"))
            for entry in payload:
                post_id = entry.get("post_id")
                user_id = entry.get("user_id")
                if not post_id or not user_id:
                    stats["posts_skipped"] += 1
                    continue
                try:
                    author = UserModel.objects.filter(pk=int(user_id)).first()
                except (ValueError, TypeError):
                    author = UserModel.objects.filter(username=str(user_id)).first()
                if not author:
                    stats["posts_skipped"] += 1
                    continue
                topic = topics.get(entry.get("topic_id"))
                if not topic:
                    topic = CommunityTopicModel.objects.create(
                        topic_id=entry.get("topic_id") or f"topic-{uuid.uuid4().hex[:10]}",
                        name=entry.get("topic_name") or entry.get("topic_id") or "Legacy Topic",
                        creator=author,
                        creator_name=entry.get("author") or author.get_full_name() or author.username,
                    )
                    topics[topic.topic_id] = topic
                created_at = _parse_dt(entry.get("created_at")) or datetime.utcnow()
                post_obj, was_created = CommunityPostModel.objects.update_or_create(
                    post_id=post_id,
                    defaults={
                        "topic": topic,
                        "author": author,
                        "author_display_name": entry.get("author") or author.get_full_name() or author.username,
                        "content": entry.get("content", ""),
                        "image_path": entry.get("image_path") or "",
                        "created_at": created_at,
                    },
                )
                if was_created:
                    stats["posts_created"] += 1
                else:
                    stats["posts_updated"] += 1
                    post_obj.created_at = created_at
                    post_obj.save(update_fields=["created_at"])

                # Likes
                liked_by = entry.get("liked_by") or []
                existing_likes = set(
                    CommunityPostLike.objects.filter(post=post_obj).values_list("user_id", flat=True)
                )
                for liker_id in liked_by:
                    liker = UserModel.objects.filter(pk=liker_id).first()
                    if not liker or liker.id in existing_likes:
                        continue
                    CommunityPostLike.objects.create(post=post_obj, user=liker)
                    stats["likes_created"] += 1

                # Comments
                for comment_entry in entry.get("comments") or []:
                    comment_id = comment_entry.get("comment_id")
                    comment_user_id = comment_entry.get("user_id")
                    if not comment_id or not comment_user_id:
                        stats["comments_skipped"] += 1
                        continue
                    commenter = UserModel.objects.filter(pk=comment_user_id).first()
                    if not commenter:
                        stats["comments_skipped"] += 1
                        continue
                    comment_obj, created_comment = CommunityPostComment.objects.get_or_create(
                        comment_id=comment_id,
                        defaults={
                            "post": post_obj,
                            "author": commenter,
                            "author_display_name": comment_entry.get("author") or commenter.username,
                            "content": comment_entry.get("content", ""),
                        },
                    )
                    ts = _parse_dt(comment_entry.get("created_at"))
                    if ts:
                        comment_obj.created_at = ts
                        comment_obj.save(update_fields=["created_at"])
                    if created_comment:
                        stats["comments_created"] += 1

        if import_topics:
            self.stdout.write("Importing topics...")
            _import_topics()
        if import_posts:
            self.stdout.write("Importing posts...")
            _import_posts()

        self.stdout.write(self.style.SUCCESS(json.dumps(stats, ensure_ascii=False, indent=2)))
