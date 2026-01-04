from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Callable, Dict
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

try:
    from celery import states
    from celery.result import AsyncResult
except Exception:  # pragma: no cover - celery always installed in prod
    states = None  # type: ignore
    AsyncResult = None  # type: ignore

from .observability import record_metric
from .tasks import (
    execute_backtest,
    execute_robustness_job,
    execute_rl_job,
    execute_training_job,
    run_backtest_task,
    run_robustness_task,
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


class LocalAsyncResult:
    def __init__(self, task_id: str):
        self.id = task_id
        self._state = "PENDING"

    @property
    def state(self) -> str:
        return self._state


LOCAL_MAX_WORKERS = int(os.environ.get("LOCAL_TASK_WORKERS", "2") or 2)
LOCAL_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, LOCAL_MAX_WORKERS))


def _normalize_user_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _local_meta(progress: int, stage: str) -> dict[str, Any]:
    return {"progress": progress, "stage": stage, "updated_at": timezone.now().isoformat()}


def _update_task_execution(
    task_id: str,
    *,
    state: str,
    meta: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> None:
    from .models import TaskExecution

    updates: dict[str, Any] = {"state": state}
    if meta is not None:
        updates["meta"] = meta
    if result is not None:
        updates["result"] = result
    if error is not None:
        updates["error"] = error
    if started_at is not None:
        updates["started_at"] = started_at
    if finished_at is not None:
        updates["finished_at"] = finished_at
    TaskExecution.objects.filter(task_id=task_id).update(**updates)


def _cancel_requested(task_id: str) -> bool:
    from .models import TaskExecution

    return TaskExecution.objects.filter(task_id=task_id, cancel_requested=True).exists()


def _should_use_async() -> bool:
    """
    Decide whether to dispatch to Celery.
    Default: try async (Celery worker) unless CELERY_TASK_ALWAYS_EAGER is explicitly True.
    If async fails (no broker/worker), we fall back to sync in _submit_task.
    """
    if getattr(settings, "CELERY_TASK_ALWAYS_EAGER", False):
        return False
    broker = getattr(settings, "CELERY_BROKER_URL", "") or ""
    if broker.startswith("memory://"):
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


def _run_local_backtest(task_id: str, payload: Dict[str, Any]) -> None:
    close_old_connections()
    started_at = timezone.now()
    try:
        if _cancel_requested(task_id):
            _update_task_execution(task_id, state="REVOKED", meta=_local_meta(0, "cancelled"), finished_at=timezone.now())
            return
        _update_task_execution(task_id, state="PROGRESS", meta=_local_meta(10, "bootstrap"), started_at=started_at)
        _update_task_execution(task_id, state="PROGRESS", meta=_local_meta(50, "running_backtest"))
        result = execute_backtest(payload)
        if _cancel_requested(task_id):
            _update_task_execution(task_id, state="REVOKED", meta=_local_meta(90, "cancelled"), finished_at=timezone.now())
            return
        _update_task_execution(
            task_id,
            state="SUCCESS",
            meta=_local_meta(100, "finalizing"),
            result=result,
            finished_at=timezone.now(),
        )
    except Exception as exc:  # pragma: no cover - safety guard
        _update_task_execution(
            task_id,
            state="FAILURE",
            meta=_local_meta(0, "failed"),
            error=str(exc),
            finished_at=timezone.now(),
        )
    finally:
        close_old_connections()


def _submit_local_backtest(payload: Dict[str, Any]) -> LocalAsyncResult:
    from .models import TaskExecution

    task_id = f"local-{uuid.uuid4().hex}"
    user_id = _normalize_user_id(payload.get("user_id"))
    TaskExecution.objects.create(
        task_id=task_id,
        user_id=user_id,
        kind="backtest",
        state="PENDING",
        meta=_local_meta(0, "queued"),
    )
    LOCAL_EXECUTOR.submit(_run_local_backtest, task_id, payload)
    record_metric("task_queue.dispatch", mode="local", task="run_backtest_task", state="submitted")
    return LocalAsyncResult(task_id)


def _run_local_robustness(task_id: str, payload: Dict[str, Any]) -> None:
    close_old_connections()
    started_at = timezone.now()
    try:
        if _cancel_requested(task_id):
            _update_task_execution(task_id, state="REVOKED", meta=_local_meta(0, "cancelled"), finished_at=timezone.now())
            return
        _update_task_execution(task_id, state="PROGRESS", meta=_local_meta(10, "bootstrap"), started_at=started_at)
        _update_task_execution(task_id, state="PROGRESS", meta=_local_meta(50, "running_robustness"))
        result = execute_robustness_job(payload)
        if _cancel_requested(task_id):
            _update_task_execution(task_id, state="REVOKED", meta=_local_meta(90, "cancelled"), finished_at=timezone.now())
            return
        _update_task_execution(
            task_id,
            state="SUCCESS",
            meta=_local_meta(100, "finalizing"),
            result=result,
            finished_at=timezone.now(),
        )
    except Exception as exc:  # pragma: no cover - safety guard
        _update_task_execution(
            task_id,
            state="FAILURE",
            meta=_local_meta(0, "failed"),
            error=str(exc),
            finished_at=timezone.now(),
        )
    finally:
        close_old_connections()


def _submit_local_robustness(payload: Dict[str, Any]) -> LocalAsyncResult:
    from .models import TaskExecution

    task_id = f"local-{uuid.uuid4().hex}"
    user_id = _normalize_user_id(payload.get("user_id"))
    TaskExecution.objects.create(
        task_id=task_id,
        user_id=user_id,
        kind="robustness",
        state="PENDING",
        meta=_local_meta(0, "queued"),
    )
    LOCAL_EXECUTOR.submit(_run_local_robustness, task_id, payload)
    record_metric("task_queue.dispatch", mode="local", task="run_robustness_task", state="submitted")
    return LocalAsyncResult(task_id)


def submit_backtest_task(payload: Dict[str, Any]) -> Any:
    serialized = _serialize_payload(payload)
    if _should_use_async():
        try:
            job = run_backtest_task.delay(serialized)
            record_metric("task_queue.dispatch", mode="async", task="run_backtest_task", state="submitted")
            return job
        except Exception as exc:
            logging.getLogger(__name__).warning("Async dispatch failed, falling back to local runner: %s", exc)
            record_metric(
                "task_queue.dispatch_fallback",
                mode="local",
                task="run_backtest_task",
                reason="async_failed",
                error=str(exc),
            )
    return _submit_local_backtest(serialized)


def submit_robustness_task(payload: Dict[str, Any]) -> Any:
    serialized = _serialize_payload(payload)
    if _should_use_async():
        try:
            job = run_robustness_task.delay(serialized)
            record_metric("task_queue.dispatch", mode="async", task="run_robustness_task", state="submitted")
            return job
        except Exception as exc:
            logging.getLogger(__name__).warning("Async dispatch failed, falling back to local runner: %s", exc)
            record_metric(
                "task_queue.dispatch_fallback",
                mode="local",
                task="run_robustness_task",
                reason="async_failed",
                error=str(exc),
            )
    return _submit_local_robustness(serialized)


def submit_training_task(payload: Dict[str, Any]) -> Any:
    return _submit_task(payload, run_training_task, execute_training_job)


def submit_rl_task(payload: Dict[str, Any]) -> Any:
    return _submit_task(payload, run_rl_task, execute_rl_job)


def get_task_status(task_id: str) -> Dict[str, Any]:
    if task_id.startswith("pending-"):
        return {
            "task_id": task_id,
            "state": "FAILURE",
            "error": "Task submission failed before a job id was created.",
        }
    if task_id.startswith("local-"):
        from .models import TaskExecution

        execution = TaskExecution.objects.filter(task_id=task_id).first()
        if not execution:
            return {"task_id": task_id, "state": "UNKNOWN", "error": "Task not found."}
        payload: Dict[str, Any] = {"task_id": task_id, "state": execution.state}
        if execution.meta:
            payload["meta"] = execution.meta
        if execution.result:
            payload["result"] = execution.result
        if execution.error:
            payload["error"] = execution.error
        return payload
    if task_id.startswith("sync-"):
        return {"task_id": task_id, "state": "SUCCESS", "meta": {"progress": 100}}
    if getattr(settings, "CELERY_TASK_ALWAYS_EAGER", False) and not getattr(settings, "CELERY_TASK_STORE_EAGER_RESULT", False):
        return {
            "task_id": task_id,
            "state": "FAILURE",
            "error": "Task results are not stored when eager mode is enabled.",
        }
    if AsyncResult is None:
        return {"task_id": task_id, "state": "UNKNOWN"}
    result = AsyncResult(task_id)
    payload: Dict[str, Any] = {"task_id": task_id, "state": result.state}
    if result.info:
        if isinstance(result.info, dict):
            payload["meta"] = result.info
        else:
            payload["meta"] = {"detail": str(result.info)}
    if result.ready():
        try:
            payload["result"] = result.get(propagate=False)
        except Exception as exc:  # pragma: no cover
            payload["error"] = str(exc)
    return payload


def cancel_task(task_id: str) -> Dict[str, Any]:
    if task_id.startswith("pending-"):
        return {"task_id": task_id, "state": "FAILURE", "error": "Task was never submitted."}
    if task_id.startswith("local-"):
        from .models import TaskExecution

        execution = TaskExecution.objects.filter(task_id=task_id).first()
        if not execution:
            return {"task_id": task_id, "state": "UNKNOWN", "error": "Task not found."}
        if execution.state in {"SUCCESS", "FAILURE", "REVOKED"}:
            return {"task_id": task_id, "state": execution.state}
        progress = 0
        if isinstance(execution.meta, dict):
            progress = int(execution.meta.get("progress") or 0)
        TaskExecution.objects.filter(task_id=task_id).update(
            cancel_requested=True,
            state="REVOKED",
            meta=_local_meta(progress, "cancelled"),
            finished_at=timezone.now(),
        )
        return {"task_id": task_id, "state": "REVOKED"}
    if task_id.startswith("sync-"):
        return {"task_id": task_id, "state": "SUCCESS", "detail": "Sync tasks cannot be cancelled."}
    if AsyncResult is None:
        return {"task_id": task_id, "state": "UNKNOWN", "error": "Celery is not available."}
    result = AsyncResult(task_id)
    if result.state in {"SUCCESS", "FAILURE", "REVOKED"}:
        return {"task_id": task_id, "state": result.state}
    terminate = result.state in {"STARTED", "PROGRESS"}
    result.revoke(terminate=terminate)
    return {"task_id": task_id, "state": "REVOKED"}
