from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Any

WINDOW_SECONDS = 60.0
_MAX_EVENTS = 20_000

_LOCK = Lock()
_GLOBAL_EVENTS: deque[tuple[float, str]] = deque()
_USER_EVENTS: dict[str, deque[tuple[float, str]]] = {}


def _normalize_provider(provider: str | None) -> str:
    text = str(provider or "").strip().lower()
    if text in {"alpaca", "massive"}:
        return text
    return "unknown"


def _normalize_user_id(user_id: str | int | None) -> str:
    text = str(user_id or "").strip()
    return text or "anonymous"


def _prune(queue: deque[tuple[float, str]], *, now: float) -> None:
    cutoff = now - WINDOW_SECONDS
    while queue and queue[0][0] < cutoff:
        queue.popleft()


def _count_by_provider(queue: deque[tuple[float, str]]) -> dict[str, int]:
    counts: dict[str, int] = {"alpaca": 0, "massive": 0, "unknown": 0}
    for _ts, provider in queue:
        key = _normalize_provider(provider)
        counts[key] = counts.get(key, 0) + 1
    return counts


def record_provider_api_call(provider: str | None, *, user_id: str | int | None = None) -> None:
    now = time.time()
    normalized_provider = _normalize_provider(provider)
    normalized_user_id = _normalize_user_id(user_id)
    with _LOCK:
        _prune(_GLOBAL_EVENTS, now=now)
        _GLOBAL_EVENTS.append((now, normalized_provider))
        while len(_GLOBAL_EVENTS) > _MAX_EVENTS:
            _GLOBAL_EVENTS.popleft()

        user_queue = _USER_EVENTS.get(normalized_user_id)
        if user_queue is None:
            user_queue = deque()
            _USER_EVENTS[normalized_user_id] = user_queue
        _prune(user_queue, now=now)
        user_queue.append((now, normalized_provider))
        while len(user_queue) > _MAX_EVENTS:
            user_queue.popleft()

        stale_users = [uid for uid, queue in _USER_EVENTS.items() if not queue]
        for stale_uid in stale_users:
            _USER_EVENTS.pop(stale_uid, None)


def get_provider_api_usage(*, user_id: str | int | None = None) -> dict[str, Any]:
    now = time.time()
    normalized_user_id = _normalize_user_id(user_id)
    with _LOCK:
        _prune(_GLOBAL_EVENTS, now=now)
        global_counts = _count_by_provider(_GLOBAL_EVENTS)

        user_queue = _USER_EVENTS.get(normalized_user_id)
        if user_queue is None:
            user_queue = deque()
        else:
            _prune(user_queue, now=now)
        user_counts = _count_by_provider(user_queue)

    return {
        "window_seconds": int(WINDOW_SECONDS),
        "global_total": sum(global_counts.values()),
        "global_by_provider": global_counts,
        "user_total": sum(user_counts.values()),
        "user_by_provider": user_counts,
    }


def reset_provider_api_usage_for_tests() -> None:
    with _LOCK:
        _GLOBAL_EVENTS.clear()
        _USER_EVENTS.clear()
