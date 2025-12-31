from __future__ import annotations

from dataclasses import fields
from datetime import date, datetime
from typing import Any, Dict, List, get_args, get_origin, get_type_hints

from celery import shared_task
from django.conf import settings

from .strategies import StrategyInput, run_quant_pipeline
from .history import BacktestRecord, append_history, sanitize_snapshot
from .train_ml import run_engine_benchmark
from paper.engine import run_pending_sessions


_STRATEGY_INPUT_TYPES = get_type_hints(StrategyInput)


def _is_date_type(type_hint: object | None) -> bool:
    if type_hint is date:
        return True
    origin = get_origin(type_hint)
    if origin is None:
        return False
    return date in get_args(type_hint)


def _parse_date(value: str) -> date | None:
    text = value.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            return None


def _deserialize_strategy_input(payload: Dict[str, Any]) -> StrategyInput:
    converted: Dict[str, Any] = {}
    for field in fields(StrategyInput):
        name = field.name
        if name not in payload:
            continue
        value = payload[name]
        if value is None:
            continue
        type_hint = _STRATEGY_INPUT_TYPES.get(name)
        if _is_date_type(type_hint) and isinstance(value, str):
            parsed = _parse_date(value)
            if parsed is not None:
                converted[name] = parsed
                continue
        converted[name] = value
    return StrategyInput(**converted)


def _persist_history(result: Dict[str, Any], user_id: str | None) -> str | None:
    if not user_id:
        return None
    snapshot = sanitize_snapshot(result)
    payload = dict(result)
    payload["snapshot"] = snapshot
    try:
        record = BacktestRecord.from_payload(payload, user_id=user_id)
        append_history(record)
        return record.record_id
    except Exception:
        return None


def execute_backtest(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = _deserialize_strategy_input(payload)
    result = run_quant_pipeline(params)
    history_id = _persist_history(result, getattr(params, "user_id", None))
    if getattr(settings, "TASK_RETURN_SNAPSHOT", False):
        safe_result = sanitize_snapshot(result)
        return {
            "history_id": history_id,
            "result": safe_result,
        }
    return {"history_id": history_id}


def execute_training_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Benchmark candidate ML engines asynchronously."""

    base_params = payload.get("base_params") or {}
    params = _deserialize_strategy_input(base_params)
    tickers: List[str] = [
        symbol.strip().upper()
        for symbol in payload.get("tickers", [])
        if isinstance(symbol, str) and symbol.strip()
    ]
    engines_payload = payload.get("engines") or []
    engines: List[str] | None = None
    if isinstance(engines_payload, list):
        engines = [str(item).strip() for item in engines_payload if str(item).strip()]
    result = run_engine_benchmark(
        tickers or [params.ticker],
        params.start_date,
        params.end_date,
        params,
        engines=engines,
    )
    return {"history_id": None, "result": result}


def execute_rl_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = _deserialize_strategy_input(payload)
    result = run_quant_pipeline(params)
    history_id = _persist_history(result, getattr(params, "user_id", None))
    rl_summary = {
        "playbook": result.get("rl_playbook"),
        "stats": result.get("stats"),
        "warnings": result.get("warnings", []),
    }
    return {"history_id": history_id, "result": sanitize_snapshot(rl_summary)}


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 3},
    default_retry_delay=10,
    ignore_result=False,
    queue="backtests",
)
def run_backtest_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    return execute_backtest(payload)


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 3},
    default_retry_delay=10,
    ignore_result=False,
    queue="training",
)
def run_training_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    return execute_training_job(payload)


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 3},
    default_retry_delay=10,
    ignore_result=False,
    queue="rl",
)
def run_rl_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    return execute_rl_job(payload)


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 2},
    default_retry_delay=10,
    ignore_result=False,
    queue="paper_trading",
)
def run_paper_trading_heartbeat(limit: int = 20) -> list[dict[str, Any]]:
    """定时刷新正在运行的模拟实盘会话。"""
    return run_pending_sessions(limit=limit)
