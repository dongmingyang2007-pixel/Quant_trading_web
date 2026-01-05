from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, IO, Iterator

try:  # pragma: no cover - platform specific
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore
try:  # pragma: no cover - windows fallback
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover
    msvcrt = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def _lock_file_handle(handle: IO[str]) -> None:
    fd = handle.fileno()
    if fcntl:
        fcntl.flock(fd, fcntl.LOCK_EX)
    elif msvcrt:
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


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as fh:
        _lock_file_handle(fh)
        try:
            yield
        finally:
            try:
                _unlock_file_handle(fh)
            except Exception:
                pass


def read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to read JSON from %s: %s", path, exc)
        return default


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, payload: Any, *, indent: int | None = 2) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    atomic_write_text(path, text)


def update_json_file(
    path: Path,
    *,
    default: Any,
    update_fn: Callable[[Any], Any],
    indent: int | None = 2,
) -> Any:
    with file_lock(path):
        current = read_json(path, default=default)
        updated = update_fn(current)
        if updated is None:
            updated = default
        atomic_write_json(path, updated, indent=indent)
        return updated

