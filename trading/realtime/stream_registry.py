from __future__ import annotations

import threading

from .lock import InstanceLock
from .storage import resolve_state_dir

_STREAM_OWNER: str | None = None
_STREAM_LOCK = threading.Lock()
_STREAM_INSTANCE_LOCK = InstanceLock(resolve_state_dir() / "alpaca_stream.pid")


def acquire_stream(owner: str) -> bool:
    if not owner:
        return False
    with _STREAM_LOCK:
        global _STREAM_OWNER
        if _STREAM_OWNER:
            return _STREAM_OWNER == owner
        if not _STREAM_INSTANCE_LOCK.acquire():
            return False
        _STREAM_OWNER = owner
        return True


def release_stream(owner: str) -> None:
    with _STREAM_LOCK:
        global _STREAM_OWNER
        if _STREAM_OWNER != owner:
            return
        _STREAM_OWNER = None
        _STREAM_INSTANCE_LOCK.release()


def current_owner() -> str | None:
    with _STREAM_LOCK:
        global _STREAM_OWNER
        return _STREAM_OWNER
