from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable
from io import BytesIO

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional
    pd = None  # type: ignore

from django.conf import settings

try:  # optional dependency
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore

LOGGER = logging.getLogger(__name__)
_PARQUET_AVAILABLE: bool | None = None
_PARQUET_WARNED = False


def _has_parquet_engine() -> bool:
    global _PARQUET_AVAILABLE
    if _PARQUET_AVAILABLE is not None:
        return _PARQUET_AVAILABLE
    for module in ("pyarrow", "fastparquet"):
        try:
            __import__(module)
            _PARQUET_AVAILABLE = True
            return True
        except Exception:
            continue
    _PARQUET_AVAILABLE = False
    return False


def _warn_parquet_missing(context: str) -> None:
    global _PARQUET_WARNED
    if _PARQUET_WARNED:
        return
    LOGGER.warning(
        "Parquet engine missing; %s skipped. Install pyarrow>=14 or fastparquet.",
        context,
    )
    _PARQUET_WARNED = True


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


def _scoped_cache_key(key: str, cache_alias: str | None) -> str:
    if not cache_alias or cache_alias == "default":
        return key
    return f"{cache_alias}:{key}"


def _is_json_safe(value: Any) -> bool:
    """Allow only primitive/structured JSON types; reject everything else."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_safe(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(item) for item in value)
    return False


def _to_json_bytes(value: Any) -> bytes | None:
    """Serialize JSON-safe objects to bytes; skip complex/unsafe types."""
    if not _is_json_safe(value):
        return None
    try:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return payload.encode("utf-8")


def _encode_value(value: Any) -> bytes | None:
    """Encode cache value to bytes with a simple format marker."""
    if pd is not None:
        try:
            if isinstance(value, pd.Series):
                if not _has_parquet_engine():
                    _warn_parquet_missing("cache parquet serialization (Series)")
                    return None
                buffer = BytesIO()
                value.to_frame("value").to_parquet(buffer, index=True)
                header = json.dumps({"type": "series"}, separators=(",", ":")).encode("utf-8")
                return b"PQT1" + header + b"\n" + buffer.getvalue()
            if isinstance(value, pd.DataFrame):
                if not _has_parquet_engine():
                    _warn_parquet_missing("cache parquet serialization (DataFrame)")
                    return None
                buffer = BytesIO()
                value.to_parquet(buffer, index=True)
                header = json.dumps({"type": "dataframe"}, separators=(",", ":")).encode("utf-8")
                return b"PQT1" + header + b"\n" + buffer.getvalue()
        except Exception:
            return None

    json_bytes = _to_json_bytes(value)
    if json_bytes is not None:
        return b"JSON" + json_bytes
    return None


def _decode_value(payload: bytes | None) -> Any | None:
    if payload is None:
        return None
    if isinstance(payload, str):  # compatibility with potential string cache backends
        payload = payload.encode("utf-8")
    if payload.startswith(b"PQT1") and pd is not None:
        try:
            if not _has_parquet_engine():
                _warn_parquet_missing("cache parquet deserialization")
                return None
            header_and_body = payload[4:]
            split = header_and_body.find(b"\n")
            if split == -1:
                return None
            header = json.loads(header_and_body[:split].decode("utf-8"))
            body = header_and_body[split + 1 :]
            df = pd.read_parquet(BytesIO(body))
            if header.get("type") == "series" and isinstance(df, pd.DataFrame) and df.shape[1] == 1:
                return df.iloc[:, 0]
            return df
        except Exception:
            return None
    if payload.startswith(b"JSON"):
        try:
            return json.loads(payload[4:].decode("utf-8"))
        except Exception:
            return None
    # Legacy payloads without marker
    try:
        text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)
        return json.loads(text)
    except Exception:
        return None


def cache_get_object(key: str, *, cache_alias: str | None = None) -> Any | None:
    scoped_key = _scoped_cache_key(key, cache_alias)
    payload = get_cache().get(scoped_key)
    return _decode_value(payload)


def cache_set_object(key: str, value: Any, ttl: int, *, cache_alias: str | None = None) -> None:
    payload = _encode_value(value)
    if payload is None:
        return
    scoped_key = _scoped_cache_key(key, cache_alias)
    get_cache().set(scoped_key, payload, ttl)


def cache_memoize(key: str, builder: Callable[[], Any], ttl: int, *, cache_alias: str | None = None) -> Any:
    cached = cache_get_object(key, cache_alias=cache_alias)
    if cached is not None:
        return cached
    value = builder()
    if value is not None:
        cache_set_object(key, value, ttl, cache_alias=cache_alias)
    return value
