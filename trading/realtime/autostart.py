from __future__ import annotations

import os
import sys
import threading

from django.db import close_old_connections

from . import RealtimeEngine, load_realtime_config, load_realtime_config_from_payload
from .lock import InstanceLock
from .storage import resolve_state_dir
from ..models import RealtimeProfile

_AUTO_START_LOCK = threading.Lock()
_AUTO_STARTED = False

_SKIP_COMMANDS = {
    "migrate",
    "makemigrations",
    "collectstatic",
    "shell",
    "dbshell",
    "createsuperuser",
    "realtime_run",
    "realtime_refresh_assets",
    "test",
    "pytest",
}


def _should_autostart() -> bool:
    if os.environ.get("REALTIME_AUTO_START", "1") in {"0", "false", "False"}:
        return False
    if len(sys.argv) > 1 and sys.argv[1] in _SKIP_COMMANDS:
        return False
    if "runserver" in sys.argv and os.environ.get("RUN_MAIN") not in {"true", "1"}:
        return False
    return True


def _run_engine() -> None:
    close_old_connections()
    config = None
    user_id = None
    try:
        profile = (
            RealtimeProfile.objects.filter(is_active=True)
            .order_by("-updated_at")
            .first()
        )
        if profile:
            config = load_realtime_config_from_payload(profile.payload)
            user_id = str(profile.user_id)
    except Exception:
        config = None
        user_id = None
    if config is None:
        config = load_realtime_config()
    engine = RealtimeEngine(config, user_id=user_id)
    lock = InstanceLock(resolve_state_dir() / "realtime.pid")
    if not lock.acquire():
        return
    try:
        engine.run()
    finally:
        lock.release()


def start_realtime_engine() -> None:
    global _AUTO_STARTED
    if not _should_autostart():
        return
    with _AUTO_START_LOCK:
        if _AUTO_STARTED:
            return
        _AUTO_STARTED = True
    thread = threading.Thread(target=_run_engine, name="realtime-engine", daemon=True)
    thread.start()
