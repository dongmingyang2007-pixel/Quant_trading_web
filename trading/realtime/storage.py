from __future__ import annotations

import json
import os
from pathlib import Path
from collections import deque
from typing import Any, Iterable

from django.conf import settings

from ..file_utils import atomic_write_json, file_lock, read_json


def _resolve_root(default_suffix: str, env_key: str) -> Path:
    override = os.environ.get(env_key)
    if override:
        return Path(override).expanduser().resolve()
    base = Path(getattr(settings, "DATA_CACHE_DIR", Path(settings.DATA_ROOT) / "data_cache"))
    return base / "realtime" / default_suffix


def resolve_state_dir() -> Path:
    root = _resolve_root("state", "REALTIME_STATE_DIR")
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_data_dir() -> Path:
    root = _resolve_root("data", "REALTIME_DATA_DIR")
    root.mkdir(parents=True, exist_ok=True)
    return root


def state_path(filename: str) -> Path:
    return resolve_state_dir() / filename


def data_path(filename: str) -> Path:
    return resolve_data_dir() / filename


def read_state(filename: str, *, default: Any) -> Any:
    return read_json(state_path(filename), default=default)


def write_state(filename: str, payload: Any) -> None:
    path = state_path(filename)
    with file_lock(path):
        atomic_write_json(path, payload, indent=2)


NDJSON_MAX_BYTES = int(os.environ.get("REALTIME_NDJSON_MAX_BYTES", str(10 * 1024 * 1024)) or 0)


def _rotate_ndjson(path: Path) -> None:
    if NDJSON_MAX_BYTES <= 0:
        return
    try:
        if not path.exists():
            return
        if path.stat().st_size <= NDJSON_MAX_BYTES:
            return
        backup = path.with_name(f"{path.name}.1")
        try:
            backup.unlink(missing_ok=True)
        except Exception:
            pass
        path.rename(backup)
    except Exception:
        return


def append_ndjson(filename: str, rows: Iterable[dict[str, Any]]) -> None:
    path = data_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        _rotate_ndjson(path)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_ndjson_tail(filename: str, *, limit: int = 100) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    path = data_path(filename)
    if not path.exists():
        return []
    buffer: deque[dict[str, Any]] = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    buffer.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return list(buffer)
