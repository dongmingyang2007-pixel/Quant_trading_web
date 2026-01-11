from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from ...models import RealtimeProfile
from ...realtime import RealtimeEngine, load_realtime_config, load_realtime_config_from_payload
from ...realtime.lock import InstanceLock
from ...realtime.storage import resolve_state_dir


class Command(BaseCommand):
    help = "Run the realtime market engine loop."

    def add_arguments(self, parser):
        parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to realtime config JSON (default: realtime_profile.json in state dir).",
        )
        parser.add_argument(
            "--once",
            action="store_true",
            help="Run a single refresh cycle then exit.",
        )
        parser.add_argument(
            "--user-id",
            type=str,
            default=None,
            help="User id used to resolve Alpaca credentials.",
        )
        parser.add_argument(
            "--no-lock",
            action="store_true",
            help="Skip single-instance lock (not recommended).",
        )

    def handle(self, *args, **options):
        config_path = options.get("config")
        once = bool(options.get("once"))
        user_id = options.get("user_id")
        no_lock = bool(options.get("no_lock"))

        config = None
        if config_path:
            path = Path(config_path).expanduser().resolve()
            config = load_realtime_config(path)
        elif user_id:
            profile = (
                RealtimeProfile.objects.filter(user_id=user_id, is_active=True)
                .order_by("-updated_at")
                .first()
            )
            if profile:
                config = load_realtime_config_from_payload(profile.payload)
        if config is None:
            config = load_realtime_config()
        engine = RealtimeEngine(config, user_id=user_id)

        lock_path = resolve_state_dir() / "realtime.pid"
        if no_lock:
            if once:
                engine.run_once()
            else:
                engine.run()
            return

        lock = InstanceLock(lock_path)
        if not lock.acquire():
            self.stdout.write(self.style.ERROR(f"Realtime engine already running (lock: {lock_path})."))
            return
        try:
            if once:
                engine.run_once()
            else:
                engine.run()
        finally:
            lock.release()
