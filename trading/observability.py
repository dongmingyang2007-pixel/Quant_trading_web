from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator
import json
import threading
import time
import uuid
import logging
from typing import IO
import os

try:  # pragma: no cover - platform specific
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore
try:  # pragma: no cover - windows fallback
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover
    msvcrt = None  # type: ignore

from django.conf import settings

DATA_CACHE_DIR = settings.DATA_CACHE_DIR
METRICS_PATH = DATA_CACHE_DIR / "telemetry.ndjson"
METRICS_MAX_BYTES = int(os.environ.get("METRICS_MAX_BYTES", str(5 * 1024 * 1024)) or 0)
_METRIC_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)


def _lock_file_handle(handle: IO[str]) -> None:
    """Best-effort cross-platform advisory lock."""
    fd = handle.fileno()
    if fcntl:  # Unix-like
        fcntl.flock(fd, fcntl.LOCK_EX)
    elif msvcrt:  # Windows
        # Lock a large range from current position
        handle.seek(0, 2)
        size = handle.tell() or 1
        handle.seek(0, 2)
        msvcrt.locking(fd, msvcrt.LK_LOCK, size)


def _unlock_file_handle(handle: IO[str]) -> None:
    fd = handle.fileno()
    if fcntl:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt:
        handle.seek(0, 2)
        size = max(handle.tell(), 1)
        handle.seek(0, 2)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, size)


def ensure_request_id(request: Any | None = None) -> str:
    """Read or create a request id (compatible with Amazon style headers)."""
    header_keys = ("HTTP_X_REQUEST_ID", "HTTP_X_AMZN_TRACE_ID", "HTTP_CF_RAY")
    if request is not None:
        for key in header_keys:
            value = request.META.get(key) if hasattr(request, "META") else None
            if value:
                return value.strip()
        attr = getattr(request, "_generated_request_id", None)
        if attr:
            return attr
        new_id = uuid.uuid4().hex
        setattr(request, "_generated_request_id", new_id)
        return new_id
    return uuid.uuid4().hex


def record_metric(event: str, **fields: Any) -> None:
    ts = datetime.now(timezone.utc)
    entry = {
        "event": event,
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }
    entry.update({k: v for k, v in fields.items() if v is not None})
    try:
        payload = json.dumps(entry, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - serialization guard
        LOGGER.warning("Failed to serialize metric %s: %s", event, exc)
        return
    try:
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _METRIC_LOCK:
            _rotate_metrics_file()
            with METRICS_PATH.open("a", encoding="utf-8") as fh:
                try:
                    _lock_file_handle(fh)
                    fh.write(payload + "\n")
                    fh.flush()
                finally:
                    try:
                        _unlock_file_handle(fh)
                    except Exception:
                        pass
    except Exception as exc:  # pragma: no cover - IO errors
        LOGGER.warning("Failed to write metric %s: %s", event, exc)


def _rotate_metrics_file() -> None:
    """Simple size-based rotation to avoid unbounded telemetry growth."""
    if METRICS_MAX_BYTES <= 0:
        return
    try:
        if not METRICS_PATH.exists():
            return
        if METRICS_PATH.stat().st_size <= METRICS_MAX_BYTES:
            return
        backup = METRICS_PATH.with_name(f"{METRICS_PATH.name}.1")
        try:
            backup.unlink(missing_ok=True)
        except Exception:
            pass
        METRICS_PATH.rename(backup)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to rotate metrics file: %s", exc)


@contextmanager
def track_latency(event: str, **fields: Any) -> Iterator[None]:
    """Context manager that records duration & success flag for an event."""
    start = time.perf_counter()
    success = True
    try:
        yield
    except Exception as exc:
        success = False
        fields.setdefault("error", str(exc))
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        fields["duration_ms"] = round(duration_ms, 2)
        fields["success"] = success
        record_metric(event, **fields)
