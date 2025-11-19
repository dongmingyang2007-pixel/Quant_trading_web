from __future__ import annotations

import time
from dataclasses import dataclass

from django.core.cache import caches, InvalidCacheBackendError
from django.http import HttpRequest


def _get_cache(alias: str):
    try:
        return caches[alias]
    except InvalidCacheBackendError:  # pragma: no cover - fallback to default cache
        return caches["default"]


def rate_limit_key(request: HttpRequest) -> str:
    """
    Build a stable cache key for the current caller (优先用户，其次 IP)。
    """
    if getattr(request, "user", None) and request.user.is_authenticated:
        return f"user:{request.user.id}"
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded:
        return f"anon:{forwarded.split(',')[0].strip()}"
    return f"anon:{request.META.get('REMOTE_ADDR', 'unknown')}"


@dataclass(frozen=True)
class RateLimitState:
    limited: bool
    retry_after: int
    window: int
    count: int


def check_rate_limit(
    *,
    cache_alias: str,
    key: str,
    window: int,
    max_calls: int,
) -> RateLimitState:
    """
    简单时间窗限流：在 window 秒内最多允许 max_calls 次调用。
    返回 RateLimitState，调用方可据此返回 429。
    """
    cache = _get_cache(cache_alias)
    now = time.time()
    record = cache.get(key)
    if not isinstance(record, dict) or now >= record.get("reset", 0):
        record = {"reset": now + window, "count": 0}
    if record["count"] >= max_calls:
        retry = max(1, int(record["reset"] - now))
        cache.set(key, record, retry)
        return RateLimitState(True, retry, window, record["count"])

    record["count"] += 1
    ttl = max(1, int(record["reset"] - now))
    cache.set(key, record, ttl)
    return RateLimitState(False, ttl, window, record["count"])
