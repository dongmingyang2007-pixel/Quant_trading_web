from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Callable, Dict
import logging

from django.conf import settings

try:
    from celery import states
    from celery.result import AsyncResult
except Exception:  # pragma: no cover - celery always installed in prod
    states = None  # type: ignore
    AsyncResult = None  # type: ignore

from .observability import record_metric
from .tasks import (
    execute_backtest,
    execute_rl_job,
    execute_training_job,
    run_backtest_task,
    run_rl_task,
    run_training_task,
)


class SyncResult:
    def __init__(self, result: Dict[str, Any]):
        self.id = f"sync-{uuid.uuid4().hex}"
        self._result = result

    def ready(self) -> bool:
        return True

    @property
    def state(self) -> str:
        return "SUCCESS"

    @property
    def result(self) -> Dict[str, Any]:
        return self._result

def _should_use_async() -> bool:
    """
    Decide whether to dispatch to Celery.
    Default: try async (Celery worker) unless CELERY_TASK_ALWAYS_EAGER is explicitly True.
    If async fails (no broker/worker), we fall back to sync in _submit_task.
    """
    if getattr(settings, "CELERY_TASK_ALWAYS_EAGER", False):
        return False
    return True


def _serialize_payload(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serialize_payload(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_payload(item) for item in value]
    return value


def _submit_task(payload: Dict[str, Any], async_task: Any, sync_callable: Callable[[Dict[str, Any]], Dict[str, Any]]):
    serialized = _serialize_payload(payload)
    using_async = _should_use_async()
    if using_async:
        try:
            job = async_task.delay(serialized)
            record_metric("task_queue.dispatch", mode="async", task=getattr(async_task, "__name__", ""), state="submitted")
            return job
        except Exception as exc:
            logging.getLogger(__name__).warning("Async dispatch failed, falling back to sync: %s", exc)
            record_metric(
                "task_queue.dispatch_fallback",
                mode="sync",
                task=getattr(async_task, "__name__", ""),
                reason="async_failed",
                error=str(exc),
            )
    result = sync_callable(serialized)
    record_metric(
        "task_queue.dispatch",
        mode="sync",
        task=getattr(async_task, "__name__", ""),
        state="executed",
        reason="eager" if not using_async else "fallback",
    )
    return SyncResult(result)


def submit_backtest_task(payload: Dict[str, Any]) -> Any:
    return _submit_task(payload, run_backtest_task, execute_backtest)


def submit_training_task(payload: Dict[str, Any]) -> Any:
    return _submit_task(payload, run_training_task, execute_training_job)


def submit_rl_task(payload: Dict[str, Any]) -> Any:
    return _submit_task(payload, run_rl_task, execute_rl_job)


def get_task_status(task_id: str) -> Dict[str, Any]:
    if task_id.startswith("sync-"):
        return {"task_id": task_id, "state": "SUCCESS"}
    if AsyncResult is None:
        return {"task_id": task_id, "state": "UNKNOWN"}
    result = AsyncResult(task_id)
    payload: Dict[str, Any] = {"task_id": task_id, "state": result.state}
    if result.ready():
        try:
            payload["result"] = result.get(propagate=False)
        except Exception as exc:  # pragma: no cover
            payload["error"] = str(exc)
    return payload
