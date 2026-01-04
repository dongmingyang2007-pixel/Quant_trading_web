from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional
import uuid
import json

from django.contrib.auth import get_user_model
from django.conf import settings
from django.db import IntegrityError, OperationalError, transaction
import time
import logging

from .backtest_logger import BacktestLogEntry, append_log

from .models import BacktestRecord as BacktestRecordModel

LOGGER = logging.getLogger(__name__)
FALLBACK_HISTORY_PATH = settings.DATA_CACHE_DIR / "history_fallback.json"
FALLBACK_HISTORY_LIMIT = 80


@dataclass(slots=True)
class BacktestRecord:
    timestamp: str
    ticker: str
    benchmark: str
    engine: str
    start_date: str
    end_date: str
    metrics: list[dict[str, Any]]
    stats: dict[str, Any]
    params: dict[str, Any]
    warnings: list[str]
    snapshot: dict[str, Any]
    title: str
    tags: list[str]
    notes: str
    starred: bool
    user_id: Optional[str] = None
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, user_id: Optional[str] = None) -> "BacktestRecord":
        try:
            tzinfo = datetime.UTC  # py311+
        except Exception:
            from datetime import timezone

            tzinfo = timezone.utc
        timestamp = datetime.now(tzinfo).isoformat(timespec="seconds")
        ticker = payload.get("ticker", "UNKNOWN")
        benchmark = payload.get("benchmark_ticker", "")
        engine = payload.get("engine", "")
        start_date = payload.get("start_date", "")
        end_date = payload.get("end_date", "")
        metrics = _coerce_json(payload.get("metrics", []), default=[])
        if not isinstance(metrics, list):
            metrics = []
        stats = _coerce_json(payload.get("stats", {}), default={})
        if not isinstance(stats, dict):
            stats = {}
        warnings = _coerce_json(payload.get("warnings", []), default=[])
        if not isinstance(warnings, list):
            warnings = []
        params = _coerce_json(payload.get("params") or {}, default={})
        if not isinstance(params, dict):
            params = {}
        params = {k: v for k, v in params.items() if v is not None}
        snapshot = _coerce_json(payload.get("snapshot") or payload, default={})
        if not isinstance(snapshot, dict):
            snapshot = {}
        title = str(payload.get("title") or "").strip()
        if not title:
            title = f"{ticker} · {engine or 'Strategy'}"
        tags = _normalize_tags(payload.get("tags"))
        notes = str(payload.get("notes") or "").strip()
        starred = bool(payload.get("starred") or False)
        return cls(
            timestamp=timestamp,
            ticker=ticker,
            benchmark=benchmark,
            engine=engine,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            stats=stats,
            params=params,
            warnings=warnings,
            snapshot=snapshot,
            title=title,
            tags=tags,
            notes=notes,
            starred=starred,
            user_id=user_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_history(limit: int = 25, *, user_id: Optional[str] = None) -> list[dict[str, Any]]:
    if not user_id:
        return []
    qs = BacktestRecordModel.objects.filter(user_id=user_id).order_by("-timestamp")
    db_entries = [
        {
            "record_id": obj.record_id,
            "timestamp": obj.timestamp.isoformat(),
            "ticker": obj.ticker,
            "benchmark": obj.benchmark,
            "engine": obj.engine,
            "start_date": obj.start_date,
            "end_date": obj.end_date,
            "metrics": obj.metrics,
            "stats": obj.stats,
            "params": obj.params,
            "warnings": obj.warnings,
            "snapshot": obj.snapshot,
            "title": obj.title,
            "tags": obj.tags,
            "notes": obj.notes,
            "starred": obj.starred,
            "user_id": str(obj.user_id),
        }
        for obj in qs
    ]
    fallback_entries = _load_fallback_history(user_id=user_id)
    merged: dict[str, dict[str, Any]] = {entry["record_id"]: entry for entry in fallback_entries}
    for entry in db_entries:
        merged[entry["record_id"]] = entry
    combined = list(merged.values())
    combined.sort(key=lambda item: _parse_history_timestamp(item.get("timestamp")), reverse=True)
    return combined[:limit]


def _build_log_entry(record: BacktestRecord) -> BacktestLogEntry:
    stats = record.stats or {}
    return BacktestLogEntry(
        record_id=record.record_id,
        timestamp=record.timestamp,
        ticker=record.ticker,
        engine=record.engine or "unknown",
        sharpe=float(stats.get("sharpe", 0.0) or 0.0),
        total_return=float(stats.get("total_return", 0.0) or 0.0),
        max_drawdown=float(stats.get("max_drawdown", 0.0) or 0.0),
        validation_sharpe=stats.get("validation_penalized_sharpe")
        or (stats.get("validation_report") or {}).get("penalized_sharpe"),
        execution_cost=float(stats.get("execution_cost_total", 0.0) or 0.0),
        notes=list(record.warnings or []),
        request_id=record.params.get("request_id") if isinstance(record.params, dict) else None,
        user_id=record.user_id,
        model_version=record.params.get("model_version") if isinstance(record.params, dict) else None,
        data_version=record.params.get("data_version") if isinstance(record.params, dict) else None,
        latency_ms=stats.get("exec_latency_ms"),
        seeds=stats.get("seeds"),
        versions=stats.get("environment"),
    )


def append_history(record: BacktestRecord) -> bool:
    """
    Persist backtest history. If DB 未写入，则回退到本地日志文件，避免完全丢失。
    """
    fallback_logged = False
    if not record.user_id:
        append_log(_build_log_entry(record))
        return False

    user_model = get_user_model()
    try:
        user = user_model.objects.get(pk=record.user_id)
    except user_model.DoesNotExist:
        append_log(_build_log_entry(record))
        return False

    attempts = 6
    for attempt in range(attempts):
        try:
            with transaction.atomic():
                BacktestRecordModel.objects.update_or_create(
                    record_id=record.record_id,
                    defaults={
                        "user": user,
                        "timestamp": datetime.fromisoformat(record.timestamp.rstrip("Z")),
                        "ticker": record.ticker,
                        "benchmark": record.benchmark,
                        "engine": record.engine,
                        "start_date": record.start_date,
                        "end_date": record.end_date,
                        "metrics": record.metrics,
                        "stats": record.stats,
                        "params": record.params,
                        "warnings": record.warnings,
                        "snapshot": record.snapshot,
                        "title": record.title,
                        "tags": record.tags,
                        "notes": record.notes,
                        "starred": record.starred,
                    },
                )
            fallback_logged = False
            break
        except OperationalError as exc:
            if attempt < attempts - 1 and "locked" in str(exc).lower():
                time.sleep(0.2 * (2**attempt))
                continue
            LOGGER.warning("Backtest history persist failed: %s", exc)
            fallback_logged = True
            break
        except (ValueError, IntegrityError):
            LOGGER.warning("Backtest history persist failed: invalid payload", exc_info=True)
            fallback_logged = True
            break
        except Exception:
            LOGGER.warning("Backtest history persist failed: unexpected error", exc_info=True)
            fallback_logged = True
            break

    if fallback_logged:
        append_log(_build_log_entry(record))
        return False
    return True


def get_history_record(record_id: str, *, user_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    if not user_id:
        return None
    obj = BacktestRecordModel.objects.filter(record_id=record_id, user_id=user_id).first()
    if not obj:
        fallback = _get_fallback_record(record_id, user_id=user_id)
        return fallback
    return {
        "record_id": obj.record_id,
        "timestamp": obj.timestamp.isoformat(),
        "ticker": obj.ticker,
        "benchmark": obj.benchmark,
        "engine": obj.engine,
        "start_date": obj.start_date,
        "end_date": obj.end_date,
        "metrics": obj.metrics,
        "stats": obj.stats,
        "params": obj.params,
        "warnings": obj.warnings,
        "snapshot": obj.snapshot,
        "title": obj.title,
        "tags": obj.tags,
        "notes": obj.notes,
        "starred": obj.starred,
        "user_id": str(obj.user_id),
    }


def delete_history_record(record_id: str, *, user_id: Optional[str] = None) -> bool:
    if not user_id:
        return False
    deleted, _ = BacktestRecordModel.objects.filter(record_id=record_id, user_id=user_id).delete()
    fallback_deleted = _delete_fallback_record(record_id, user_id=user_id)
    return bool(deleted) or fallback_deleted


def update_history_meta(
    record_id: str,
    *,
    user_id: Optional[str],
    title: Optional[str] = None,
    tags: Optional[list[str]] = None,
    notes: Optional[str] = None,
    starred: Optional[bool] = None,
) -> Optional[dict[str, Any]]:
    if not user_id:
        return None
    obj = BacktestRecordModel.objects.filter(record_id=record_id, user_id=user_id).first()
    if not obj:
        return _update_fallback_meta(
            record_id,
            user_id=user_id,
            title=title,
            tags=tags,
            notes=notes,
            starred=starred,
        )
    update_fields = []
    if title is not None:
        obj.title = title
        update_fields.append("title")
    if tags is not None:
        obj.tags = tags
        update_fields.append("tags")
    if notes is not None:
        obj.notes = notes
        update_fields.append("notes")
    if starred is not None:
        obj.starred = bool(starred)
        update_fields.append("starred")
    if update_fields:
        obj.save(update_fields=update_fields)
    return {
        "record_id": obj.record_id,
        "title": obj.title,
        "tags": obj.tags,
        "notes": obj.notes,
        "starred": obj.starred,
    }


def _normalize_tags(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        values = [item.strip() for item in raw.replace("，", ",").split(",")]
        return [item for item in values if item]
    if isinstance(raw, (list, tuple, set)):
        values = [str(item).strip() for item in raw if str(item).strip()]
        return values
    return []


def _coerce_json(value: Any, *, default: Any) -> Any:
    try:
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        return json.loads(serialized)
    except Exception:
        return default


_COMPACT_SNAPSHOT_KEYS = {
    "ticker",
    "start_date",
    "end_date",
    "metrics",
    "stats",
    "params",
    "warnings",
    "benchmark_ticker",
    "benchmark_metrics",
    "engine",
    "engine_label",
    "risk_profile",
    "capital",
    "metadata",
    "repro",
    "market_context",
    "return_series",
    "overview_series",
    "recent_rows",
    "interactive_chart",
    "recommendations",
    "target_portfolio",
    "trade_list",
    "risk_dashboard",
}


def compact_history_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    compacted = {key: payload.get(key) for key in _COMPACT_SNAPSHOT_KEYS if key in payload}
    return _coerce_json(compacted, default={})


def _save_fallback_history(entries: list[dict[str, Any]]) -> None:
    FALLBACK_HISTORY_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def append_fallback_history(record: BacktestRecord) -> bool:
    try:
        entry = _coerce_json(record.to_dict(), default={})
        if not isinstance(entry, dict):
            return False
        entries = _load_fallback_history()
        entries = [
            item
            for item in entries
            if isinstance(item, dict) and item.get("record_id") != record.record_id
        ]
        entries.insert(0, entry)
        entries = entries[:FALLBACK_HISTORY_LIMIT]
        _save_fallback_history(entries)
        return True
    except Exception:
        return False


def _compact_interactive_chart(value: Any, *, max_points: int = 800) -> dict[str, Any]:
    """Keep only the latest points from the interactive chart for history snapshots."""
    if not isinstance(value, dict):
        return {}
    compacted: dict[str, Any] = {}
    for key, series in value.items():
        if isinstance(series, list):
            compacted[key] = series[-max_points:]
        else:
            compacted[key] = series
    return compacted


def sanitize_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Convert backtest result payload into JSON-serializable dict.
    Removes heavy binary charts, keeps trimmed interactive payloads, and coerces unsupported objects via str().
    """
    if not isinstance(payload, dict):
        return {}
    filtered: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "charts":
            continue
        if key == "interactive_chart":
            filtered[key] = _compact_interactive_chart(value)
            continue
        filtered[key] = value
    filtered = _prune_heavy_fields(filtered)
    try:
        serialized = json.dumps(filtered, ensure_ascii=False, default=str)
        return json.loads(serialized)
    except Exception:
        return {}


def _normalize_user_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_history_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.min


def _normalize_history_entry(entry: dict[str, Any], *, user_id: Optional[str]) -> Optional[dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    record_id = entry.get("record_id")
    if not record_id:
        return None
    if user_id is not None:
        entry_user = _normalize_user_id(entry.get("user_id"))
        if entry_user != _normalize_user_id(user_id):
            return None
    return {
        "record_id": record_id,
        "timestamp": entry.get("timestamp") or "",
        "ticker": entry.get("ticker") or "",
        "benchmark": entry.get("benchmark") or entry.get("benchmark_ticker") or "",
        "engine": entry.get("engine") or "",
        "start_date": entry.get("start_date") or "",
        "end_date": entry.get("end_date") or "",
        "metrics": entry.get("metrics") or [],
        "stats": entry.get("stats") or {},
        "params": entry.get("params") or {},
        "warnings": entry.get("warnings") or [],
        "snapshot": entry.get("snapshot") or {},
        "title": entry.get("title") or "",
        "tags": entry.get("tags") or [],
        "notes": entry.get("notes") or "",
        "starred": bool(entry.get("starred") or False),
        "user_id": _normalize_user_id(entry.get("user_id")) or "",
    }


def _load_fallback_history(user_id: Optional[str] = None) -> list[dict[str, Any]]:
    if not FALLBACK_HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(FALLBACK_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    data = [entry for entry in data if isinstance(entry, dict)]
    normalized: list[dict[str, Any]] = []
    for entry in data:
        normalized_entry = _normalize_history_entry(entry, user_id=user_id)
        if normalized_entry:
            normalized.append(normalized_entry)
    return normalized if user_id is not None else data


def _get_fallback_record(record_id: str, *, user_id: Optional[str]) -> Optional[dict[str, Any]]:
    for entry in _load_fallback_history(user_id=user_id):
        if entry.get("record_id") == record_id:
            return entry
    return None


def _delete_fallback_record(record_id: str, *, user_id: Optional[str]) -> bool:
    entries = _load_fallback_history()
    if not entries:
        return False
    target_user = _normalize_user_id(user_id)
    kept: list[dict[str, Any]] = []
    removed = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_user = _normalize_user_id(entry.get("user_id"))
        if entry.get("record_id") == record_id and entry_user == target_user:
            removed = True
            continue
        kept.append(entry)
    if removed:
        _save_fallback_history(kept)
    return removed


def _update_fallback_meta(
    record_id: str,
    *,
    user_id: Optional[str],
    title: Optional[str],
    tags: Optional[list[str]],
    notes: Optional[str],
    starred: Optional[bool],
) -> Optional[dict[str, Any]]:
    entries = _load_fallback_history()
    if not entries:
        return None
    target_user = _normalize_user_id(user_id)
    updated_entry: dict[str, Any] | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_user = _normalize_user_id(entry.get("user_id"))
        if entry.get("record_id") != record_id or entry_user != target_user:
            continue
        if title is not None:
            entry["title"] = title
        if tags is not None:
            entry["tags"] = tags
        if notes is not None:
            entry["notes"] = notes
        if starred is not None:
            entry["starred"] = bool(starred)
        updated_entry = entry
        break
    if updated_entry is None:
        return None
    _save_fallback_history(entries)
    return {
        "record_id": updated_entry.get("record_id"),
        "title": updated_entry.get("title") or "",
        "tags": updated_entry.get("tags") or [],
        "notes": updated_entry.get("notes") or "",
        "starred": bool(updated_entry.get("starred") or False),
    }


def _prune_heavy_fields(value: Any, *, key: str | None = None) -> Any:
    drop_keys = {"shap_img", "calibration_plot", "threshold_heatmap"}
    image_like_keys = {"chart", "img"}
    if key in drop_keys:
        return None
    if key in image_like_keys and isinstance(value, str) and len(value) > 2000:
        return None
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for child_key, child_value in value.items():
            pruned = _prune_heavy_fields(child_value, key=str(child_key))
            if pruned is None and child_key in drop_keys:
                continue
            if pruned is None and child_key in image_like_keys and isinstance(child_value, str) and len(child_value) > 2000:
                continue
            cleaned[child_key] = pruned
        return cleaned
    if isinstance(value, list):
        return [_prune_heavy_fields(item) for item in value]
    return value
