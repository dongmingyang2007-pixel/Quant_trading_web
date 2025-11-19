from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from trading.models import BacktestRecord as BacktestRecordModel


def _parse_iso(timestamp: str | None) -> datetime:
    if not timestamp:
        return datetime.utcnow()
    text = timestamp.strip()
    if text.endswith("Z"):
        text = text[:-1]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.utcnow()


class Command(BaseCommand):
    help = "Import legacy backtest_history.json entries into the BacktestRecord table."

    def add_arguments(self, parser):
        parser.add_argument(
            "--user",
            required=True,
            help="User ID or username that owns the imported records.",
        )
        parser.add_argument(
            "--path",
            default=str(Path(settings.DATA_CACHE_DIR) / "backtest_history.json"),
            help="Path to legacy JSON file (defaults to storage_bundle/data_cache/backtest_history.json).",
        )

    def handle(self, *args, **options):
        user_ref = options["user"]
        data_path = Path(options["path"]).expanduser()
        if not data_path.exists():
            raise CommandError(f"Legacy file not found: {data_path}")

        User = get_user_model()
        user = User.objects.filter(pk=user_ref).first() or User.objects.filter(username=user_ref).first()
        if not user:
            raise CommandError(f"User '{user_ref}' not found (searched by pk and username).")

        records = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise CommandError("Legacy file format is invalid (expected list).")

        stats = {"created": 0, "updated": 0, "skipped": 0}

        @transaction.atomic
        def _import():
            for entry in records:
                if not isinstance(entry, dict):
                    stats["skipped"] += 1
                    continue
                record_id = entry.get("record_id") or uuid.uuid4().hex
                timestamp = _parse_iso(entry.get("timestamp"))
                defaults = {
                    "user": user,
                    "timestamp": timestamp,
                    "ticker": entry.get("ticker", ""),
                    "benchmark": entry.get("benchmark", ""),
                    "engine": entry.get("engine", ""),
                    "start_date": entry.get("start_date", ""),
                    "end_date": entry.get("end_date", ""),
                    "metrics": entry.get("metrics") or [],
                    "stats": entry.get("stats") or {},
                    "params": entry.get("params") or {},
                    "warnings": entry.get("warnings") or [],
                    "snapshot": entry.get("snapshot") or {},
                }
                obj, created = BacktestRecordModel.objects.update_or_create(
                    record_id=record_id,
                    defaults=defaults,
                )
                if created:
                    stats["created"] += 1
                else:
                    stats["updated"] += 1

        _import()
        self.stdout.write(self.style.SUCCESS(json.dumps(stats, ensure_ascii=False, indent=2)))
