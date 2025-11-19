from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Any

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.shortcuts import render
from django.utils import timezone


METRICS_PATH = Path(settings.DATA_CACHE_DIR) / "telemetry.ndjson"
LIMIT_CHOICES = [50, 100, 200, 400, 800, 1200, 2000]
DEFAULT_LIMIT = 400
MIN_LIMIT = LIMIT_CHOICES[0]
MAX_LIMIT = LIMIT_CHOICES[-1]
BASE_DETAIL_KEYS = {"event", "ts", "timestamp", "duration_ms", "success", "error", "request_id", "user_id"}


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidate: datetime | None = None
    try:
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        candidate = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                candidate = datetime.strptime(text.rstrip("Z"), fmt)
                break
            except ValueError:
                continue
    if candidate is None:
        return None
    if timezone.is_naive(candidate):
        candidate = candidate.replace(tzinfo=dt_timezone.utc)
    return timezone.localtime(candidate)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@login_required
def observability_dashboard(request):
    if not request.user.is_superuser:
        raise PermissionDenied
    try:
        limit = int(request.GET.get("limit", DEFAULT_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT
    limit = max(MIN_LIMIT, min(MAX_LIMIT, limit))
    active_event = (request.GET.get("event") or "").strip()

    entries: list[dict[str, Any]] = []
    available_events: set[str] = set()
    event_stats: dict[str, dict[str, Any]] = {}
    total_lines = 0
    buffer_len = 0

    if METRICS_PATH.exists():
        with METRICS_PATH.open("r", encoding="utf-8") as fh:
            buffer = deque(maxlen=limit)
            for line in fh:
                total_lines += 1
                buffer.append(line)
        buffer_len = len(buffer)

        for raw in reversed(buffer):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event = str(record.get("event") or "unknown").strip() or "unknown"
            available_events.add(event)
            if active_event and event != active_event:
                continue

            ts_obj = _parse_ts(record.get("ts") or record.get("timestamp"))
            duration = _as_float(record.get("duration_ms"))
            success_value = record.get("success")
            success = None
            if isinstance(success_value, bool):
                success = success_value
            elif isinstance(success_value, (int, float)):
                success = bool(success_value)
            error_text = str(record.get("error") or "").strip()

            details: list[tuple[str, Any]] = []
            for key, value in record.items():
                if key in BASE_DETAIL_KEYS:
                    continue
                details.append((key, value))
                if len(details) >= 6:
                    break

            entries.append(
                {
                    "event": event,
                    "ts": record.get("ts") or record.get("timestamp"),
                    "ts_obj": ts_obj,
                    "duration_ms": duration,
                    "success": success,
                    "error": error_text,
                    "request_id": record.get("request_id"),
                    "user_id": record.get("user_id"),
                    "details": details,
                }
            )

            stats = event_stats.setdefault(
                event,
                {
                    "event": event,
                    "count": 0,
                    "success": 0,
                    "failures": 0,
                    "duration_total": 0.0,
                    "duration_samples": 0,
                    "duration_max": None,
                    "last_ts": None,
                },
            )
            stats["count"] += 1
            if success is False or error_text:
                stats["failures"] += 1
            else:
                stats["success"] += 1
            if duration is not None:
                stats["duration_total"] += duration
                stats["duration_samples"] += 1
                stats["duration_max"] = (
                    duration
                    if stats["duration_max"] is None
                    else max(stats["duration_max"], duration)
                )
            if ts_obj and (stats["last_ts"] is None or ts_obj > stats["last_ts"]):
                stats["last_ts"] = ts_obj

    summaries: list[dict[str, Any]] = []
    for stats in event_stats.values():
        duration_avg = None
        if stats["duration_samples"]:
            duration_avg = stats["duration_total"] / stats["duration_samples"]
        count = stats["count"] or 1
        success_rate = stats["success"] / count
        summaries.append(
            {
                "event": stats["event"],
                "count": stats["count"],
                "success": stats["success"],
                "failures": stats["failures"],
                "success_rate": success_rate,
                "success_pct": success_rate * 100,
                "avg_duration": duration_avg,
                "max_duration": stats["duration_max"],
                "last_ts": stats["last_ts"],
            }
        )
    summaries.sort(key=lambda item: item["count"], reverse=True)

    context = {
        "entries": entries,
        "summaries": summaries,
        "available_events": sorted(available_events),
        "active_event": active_event,
        "limit": limit,
        "limit_options": LIMIT_CHOICES,
        "default_limit": DEFAULT_LIMIT,
        "file_exists": METRICS_PATH.exists(),
        "has_more": total_lines > buffer_len if METRICS_PATH.exists() else False,
        "total_lines": total_lines,
        "loaded_lines": buffer_len,
    }
    return render(request, "trading/observability.html", context)
