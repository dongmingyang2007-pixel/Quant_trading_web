from __future__ import annotations

import os
import pickle
import threading
import time
from typing import Any, Callable

from django.conf import settings

try:  # optional dependency
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


class BaseCache:
    def get(self, key: str) -> bytes | None:
        raise NotImplementedError

    def set(self, key: str, value: bytes, ttl: int) -> None:
        raise NotImplementedError


class InMemoryCache(BaseCache):
    def __init__(self):
        self._store: dict[str, tuple[float, bytes]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        now = time.time()
        with self._lock:
            record = self._store.get(key)
            if not record:
                return None
            expires, payload = record
            if expires < now:
                self._store.pop(key, None)
                return None
            return payload

    def set(self, key: str, value: bytes, ttl: int) -> None:
        with self._lock:
            self._store[key] = (time.time() + max(ttl, 1), value)


class RedisCache(BaseCache):
    def __init__(self, url: str):
        if redis is None:  # pragma: no cover - handled by caller
            raise RuntimeError("redis package is not installed.")
        self._client = redis.Redis.from_url(url)

    def get(self, key: str) -> bytes | None:
        try:
            data = self._client.get(key)
            return data if isinstance(data, (bytes, bytearray)) else None
        except Exception:
            return None

    def set(self, key: str, value: bytes, ttl: int) -> None:
        try:
            self._client.setex(key, ttl, value)
        except Exception:
            pass


_CACHE: BaseCache | None = None


def _init_cache() -> BaseCache:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    url = getattr(settings, "REDIS_URL", None)
    if url and redis is not None:
        try:
            client = RedisCache(url)
            client.set("__cache_test__", b"1", 1)
            _CACHE = client
            return _CACHE
        except Exception:
            pass
    _CACHE = InMemoryCache()
    return _CACHE


def get_cache() -> BaseCache:
    return _init_cache()


def build_cache_key(*parts: Any) -> str:
    normalized = []
    for item in parts:
        if item is None:
            continue
        if isinstance(item, (list, tuple, set)):
            normalized.append("-".join(sorted(str(v) for v in item if v is not None)))
        else:
            normalized.append(str(item))
    return ":".join(normalized)


def cache_get_object(key: str) -> Any | None:
    payload = get_cache().get(key)
    if payload is None:
        return None
    try:
        return pickle.loads(payload)
    except Exception:
        return None


def cache_set_object(key: str, value: Any, ttl: int) -> None:
    try:
        payload = pickle.dumps(value)
    except Exception:
        return
    get_cache().set(key, payload, ttl)


def cache_memoize(key: str, builder: Callable[[], Any], ttl: int) -> Any:
    cached = cache_get_object(key)
    if cached is not None:
        return cached
    value = builder()
    if value is not None:
        cache_set_object(key, value, ttl)
    return value
