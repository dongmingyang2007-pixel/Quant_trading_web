from __future__ import annotations

from dataclasses import fields, replace
from datetime import date, datetime
from itertools import product
import logging
from typing import Any, Dict, List, get_args, get_origin, get_type_hints

from celery import shared_task
from django.conf import settings
from django.utils import timezone

from .strategies import StrategyInput, run_quant_pipeline
from .history import (
    BacktestRecord,
    append_fallback_history,
    append_history,
    compact_history_snapshot,
    sanitize_snapshot,
)
from .train_ml import run_engine_benchmark
from paper.engine import run_pending_sessions


_STRATEGY_INPUT_TYPES = get_type_hints(StrategyInput)
LOGGER = logging.getLogger(__name__)


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


def _persist_history(result: Dict[str, Any], user_id_override: str | None = None) -> str | None:
    resolved_user_id = user_id_override
    if resolved_user_id is None:
        params = result.get("params") if isinstance(result, dict) else None
        if isinstance(params, dict):
            resolved_user_id = params.get("user_id")
    if not resolved_user_id:
        return None
    snapshot = sanitize_snapshot(result)
    payload = {
        "ticker": result.get("ticker"),
        "benchmark_ticker": result.get("benchmark_ticker") or "",
        "engine": result.get("engine") or "",
        "start_date": result.get("start_date") or "",
        "end_date": result.get("end_date") or "",
        "metrics": result.get("metrics") or [],
        "stats": result.get("stats") or {},
        "params": result.get("params") or {},
        "warnings": result.get("warnings") or [],
        "snapshot": snapshot,
        "title": result.get("title") or "",
        "tags": result.get("tags") or [],
    }
    safe_payload = sanitize_snapshot(payload)
    record = BacktestRecord.from_payload(safe_payload, user_id=str(resolved_user_id))
    try:
        if append_history(record):
            return record.record_id
        compact_snapshot = compact_history_snapshot(snapshot)
        if compact_snapshot:
            record.snapshot = compact_snapshot
        if append_history(record):
            return record.record_id
        if not append_fallback_history(record):
            LOGGER.warning("Backtest history fallback write failed for %s", record.record_id)
        return record.record_id
    except Exception:
        LOGGER.exception("Backtest history persist failed, fallback to file for %s", record.record_id)
        append_fallback_history(record)
        return record.record_id


def execute_backtest(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = _deserialize_strategy_input(payload)
    result = run_quant_pipeline(params)
    user_override = getattr(params, "user_id", None)
    history_id = _persist_history(result, user_id_override=str(user_override) if user_override else None)
    if getattr(settings, "TASK_RETURN_SNAPSHOT", False):
        safe_result = sanitize_snapshot(result)
        return {
            "history_id": history_id,
            "result": safe_result,
        }
    return {"history_id": history_id}


def _coerce_float_list(values: Any) -> list[float]:
    if not values:
        return []
    if isinstance(values, (int, float, str)):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    result: list[float] = []
    for item in values:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            continue
    return result


def _clamp_value(value: float, *, low: float, high: float) -> float:
    return max(low, min(high, value))


def _split_cost_bps(total_bps: float, params: StrategyInput) -> tuple[float, float]:
    total_bps = max(0.0, total_bps)
    base_total = float((params.transaction_cost_bps or 0.0) + (params.slippage_bps or 0.0))
    if base_total > 0:
        txn_ratio = (params.transaction_cost_bps or 0.0) / base_total
        slip_ratio = (params.slippage_bps or 0.0) / base_total
        return total_bps * txn_ratio, total_bps * slip_ratio
    return total_bps, 0.0


def _default_robustness_grid(params: StrategyInput) -> dict[str, list[float]]:
    base_cost_rate = float((params.transaction_cost_bps + params.slippage_bps) / 10000.0)
    if base_cost_rate <= 0:
        cost_rates = [0.0004, 0.0008]
    else:
        cost_rates = [base_cost_rate * 0.6, base_cost_rate * 1.2]
    base_adv = float(params.max_adv_participation or 0.1)
    adv_values = [base_adv * 0.7, base_adv * 1.3]
    base_entry = float(params.entry_threshold or 0.55)
    thresholds = [base_entry - 0.03, base_entry, base_entry + 0.03]
    return {
        "cost_rates": [round(_clamp_value(rate, low=0.0, high=0.01), 6) for rate in cost_rates],
        "adv_participation": [round(_clamp_value(rate, low=0.02, high=0.3), 4) for rate in adv_values],
        "thresholds": [round(_clamp_value(value, low=0.05, high=0.95), 4) for value in thresholds],
    }


def _compute_robustness_score(metrics: dict[str, Any]) -> float:
    sharpe = float(metrics.get("sharpe") or 0.0)
    max_dd = abs(float(metrics.get("max_drawdown") or 0.0))
    coverage = float(metrics.get("avg_coverage") or 0.0)
    cost_ratio = float(metrics.get("cost_ratio") or 0.0)
    return sharpe - max_dd * 2.0 - (1.0 - coverage) * 0.6 - cost_ratio * 1.5


def execute_robustness_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = _deserialize_strategy_input(payload)
    grid_config = payload.get("robustness") or {}
    if not isinstance(grid_config, dict):
        grid_config = {}
    defaults = _default_robustness_grid(params)
    cost_rates = _coerce_float_list(grid_config.get("cost_rates")) or defaults["cost_rates"]
    adv_values = _coerce_float_list(grid_config.get("adv_participation")) or defaults["adv_participation"]
    thresholds = _coerce_float_list(grid_config.get("thresholds")) or defaults["thresholds"]
    max_runs = int(grid_config.get("max_runs") or 12)
    max_runs = max(4, min(24, max_runs))

    cost_rates = sorted({round(_clamp_value(rate, low=0.0, high=0.01), 6) for rate in cost_rates})
    adv_values = sorted({round(_clamp_value(rate, low=0.02, high=0.3), 4) for rate in adv_values})
    thresholds = sorted({round(_clamp_value(value, low=0.05, high=0.95), 4) for value in thresholds})

    base_entry = float(params.entry_threshold or 0.55)
    base_exit = float(params.exit_threshold or 0.45)
    combos: list[tuple[float, float, float, float]] = []
    for cost_rate, adv, entry in product(cost_rates, adv_values, thresholds):
        exit_thr = base_exit + (entry - base_entry)
        if exit_thr <= 0 or exit_thr >= 1:
            continue
        if entry <= exit_thr:
            continue
        combos.append((cost_rate, adv, entry, round(exit_thr, 4)))
    if len(combos) > max_runs:
        combos = combos[:max_runs]

    cells: list[dict[str, Any]] = []
    for idx, (cost_rate, adv, entry, exit_thr) in enumerate(combos, start=1):
        total_bps = cost_rate * 10000.0
        txn_bps, slip_bps = _split_cost_bps(total_bps, params)
        variant = replace(
            params,
            transaction_cost_bps=txn_bps,
            slippage_bps=slip_bps,
            max_adv_participation=adv,
            entry_threshold=entry,
            exit_threshold=exit_thr,
            optimize_thresholds=False,
            include_plots=False,
            show_ai_thoughts=False,
        )
        if variant.request_id:
            variant.request_id = f"{variant.request_id}-robust-{idx}"
        try:
            result = run_quant_pipeline(variant)
            stats = result.get("stats") if isinstance(result, dict) else {}
            exec_stats = stats.get("execution_stats") if isinstance(stats, dict) else {}
            metrics = {
                "sharpe": stats.get("sharpe") if isinstance(stats, dict) else None,
                "max_drawdown": stats.get("max_drawdown") if isinstance(stats, dict) else None,
                "avg_coverage": exec_stats.get("avg_coverage") if isinstance(exec_stats, dict) else None,
                "total_return": stats.get("total_return") if isinstance(stats, dict) else None,
                "cost_ratio": stats.get("cost_ratio") if isinstance(stats, dict) else None,
            }
            score = _compute_robustness_score(metrics)
            cells.append(
                {
                    "cost_rate": cost_rate,
                    "adv_participation": adv,
                    "entry_threshold": entry,
                    "exit_threshold": exit_thr,
                    "metrics": metrics,
                    "score": round(score, 4),
                    "ok": True,
                }
            )
        except Exception as exc:
            cells.append(
                {
                    "cost_rate": cost_rate,
                    "adv_participation": adv,
                    "entry_threshold": entry,
                    "exit_threshold": exit_thr,
                    "metrics": {},
                    "score": None,
                    "ok": False,
                    "error": str(exc),
                }
            )

    scored = [cell for cell in cells if cell.get("ok") and cell.get("score") is not None]
    scored_sorted = sorted(scored, key=lambda item: item["score"], reverse=True)
    best = scored_sorted[0] if scored_sorted else None
    top_samples = scored_sorted[: max(1, min(3, len(scored_sorted)))]
    if top_samples:
        cost_range = [min(item["cost_rate"] for item in top_samples), max(item["cost_rate"] for item in top_samples)]
        adv_range = [
            min(item["adv_participation"] for item in top_samples),
            max(item["adv_participation"] for item in top_samples),
        ]
        threshold_range = [
            min(item["entry_threshold"] for item in top_samples),
            max(item["entry_threshold"] for item in top_samples),
        ]
    else:
        cost_range = adv_range = threshold_range = []

    grid_payload = {
        "cost_rates": cost_rates,
        "adv_participation": adv_values,
        "thresholds": thresholds,
        "cells": cells,
        "best": best,
        "recommendations": {
            "cost_rate_range": cost_range,
            "adv_participation_range": adv_range,
            "threshold_range": threshold_range,
        },
    }
    return {"grid": grid_payload}


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
    user_override = getattr(params, "user_id", None)
    history_id = _persist_history(result, user_id_override=str(user_override) if user_override else None)
    rl_summary = {
        "playbook": result.get("rl_playbook"),
        "stats": result.get("stats"),
        "warnings": result.get("warnings", []),
    }
    return {"history_id": history_id, "result": sanitize_snapshot(rl_summary)}


def _progress_meta(progress: int, stage: str) -> Dict[str, Any]:
    return {"progress": progress, "stage": stage, "updated_at": timezone.now().isoformat()}


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 3},
    default_retry_delay=10,
    ignore_result=False,
    queue="backtests",
)
def run_backtest_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    self.update_state(state="PROGRESS", meta=_progress_meta(10, "bootstrap"))
    self.update_state(state="PROGRESS", meta=_progress_meta(50, "running_backtest"))
    result = execute_backtest(payload)
    self.update_state(state="PROGRESS", meta=_progress_meta(90, "finalizing"))
    return result


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 3},
    default_retry_delay=10,
    ignore_result=False,
    queue="training",
)
def run_training_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    self.update_state(state="PROGRESS", meta=_progress_meta(15, "benchmark"))
    self.update_state(state="PROGRESS", meta=_progress_meta(50, "running_training"))
    result = execute_training_job(payload)
    self.update_state(state="PROGRESS", meta=_progress_meta(90, "finalizing"))
    return result


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 3},
    default_retry_delay=10,
    ignore_result=False,
    queue="rl",
)
def run_rl_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    self.update_state(state="PROGRESS", meta=_progress_meta(10, "rl_pipeline"))
    self.update_state(state="PROGRESS", meta=_progress_meta(50, "running_rl"))
    result = execute_rl_job(payload)
    self.update_state(state="PROGRESS", meta=_progress_meta(90, "finalizing"))
    return result


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 2},
    default_retry_delay=10,
    ignore_result=False,
    queue="backtests",
)
def run_robustness_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    self.update_state(state="PROGRESS", meta=_progress_meta(10, "bootstrap"))
    self.update_state(state="PROGRESS", meta=_progress_meta(50, "running_robustness"))
    result = execute_robustness_job(payload)
    self.update_state(state="PROGRESS", meta=_progress_meta(90, "finalizing"))
    return result


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
