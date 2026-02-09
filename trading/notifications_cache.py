from __future__ import annotations

import os

from django.core.cache import caches

from .models import Notification

UNREAD_NOTIFICATIONS_CACHE_ALIAS = os.environ.get("UNREAD_NOTIFICATIONS_CACHE_ALIAS", "default")
UNREAD_NOTIFICATIONS_CACHE_TTL = max(5, int(os.environ.get("UNREAD_NOTIFICATIONS_CACHE_TTL", "20") or 20))


def _cache_key(user_id: str | int) -> str:
    return f"notifications:unread:{user_id}"


def get_unread_notifications_count(user) -> int:
    if not user or not getattr(user, "is_authenticated", False):
        return 0
    key = _cache_key(user.id)
    cache = caches[UNREAD_NOTIFICATIONS_CACHE_ALIAS]
    cached = cache.get(key)
    if isinstance(cached, int):
        return cached
    count = Notification.objects.filter(recipient=user, is_read=False).count()
    cache.set(key, count, timeout=UNREAD_NOTIFICATIONS_CACHE_TTL)
    return count


def invalidate_unread_notifications_cache(user_id: str | int | None) -> None:
    if user_id is None:
        return
    key = _cache_key(user_id)
    cache = caches[UNREAD_NOTIFICATIONS_CACHE_ALIAS]
    cache.delete(key)
