from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, List, Optional
import uuid
import json

from django.apps import apps
from django.contrib.auth import get_user_model
from django.db import IntegrityError

from .backtest_logger import BacktestLogEntry, append_log

from .models import BacktestRecord as BacktestRecordModel


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
        metrics = payload.get("metrics", [])
        stats = payload.get("stats", {})
        warnings = payload.get("warnings", [])
        params = {k: v for k, v in (payload.get("params") or {}).items() if v is not None}
        snapshot = payload.get("snapshot") or payload
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
            user_id=user_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_history(limit: int = 25, *, user_id: Optional[str] = None) -> list[dict[str, Any]]:
    if not user_id:
        return []
    qs = BacktestRecordModel.objects.filter(user_id=user_id).order_by("-timestamp")[:limit]
    return [
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
            "user_id": str(obj.user_id),
        }
        for obj in qs
    ]


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


def append_history(record: BacktestRecord) -> None:
    """
    Persist backtest history. If DB 未写入，则回退到本地日志文件，避免完全丢失。
    """
    fallback_logged = False
    if not record.user_id:
        append_log(_build_log_entry(record))
        return

    user_model = get_user_model()
    try:
        user = user_model.objects.get(pk=record.user_id)
    except user_model.DoesNotExist:
        append_log(BacktestLogEntry(**record.to_dict()))
        return

    try:
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
            },
        )
    except (ValueError, IntegrityError):
        fallback_logged = True
    except Exception:
        fallback_logged = True

    if fallback_logged:
        append_log(_build_log_entry(record))


def get_history_record(record_id: str, *, user_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    if not user_id:
        return None
    obj = BacktestRecordModel.objects.filter(record_id=record_id, user_id=user_id).first()
    if not obj:
        return None
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
        "user_id": str(obj.user_id),
    }


def delete_history_record(record_id: str, *, user_id: Optional[str] = None) -> bool:
    if not user_id:
        return False
    deleted, _ = BacktestRecordModel.objects.filter(record_id=record_id, user_id=user_id).delete()
    return bool(deleted)


def sanitize_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Convert backtest result payload into JSON-serializable dict.
    Removes large fields like charts and coerces unsupported objects via str().
    """
    if not isinstance(payload, dict):
        return {}
    # Drop heavy / non-essential keys before serialization
    filtered = {}
    for key, value in payload.items():
        if key == "charts":
            continue
        filtered[key] = value
    try:
        serialized = json.dumps(filtered, ensure_ascii=False, default=str)
        return json.loads(serialized)
    except Exception:
        return {}
