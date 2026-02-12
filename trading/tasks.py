from __future__ import annotations

from dataclasses import fields, replace
from datetime import date, datetime
from itertools import product
import logging
import threading
import time
from typing import Any, Dict, List, get_args, get_origin, get_type_hints

from celery import shared_task
from django.conf import settings
from django.contrib.auth import get_user_model
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
from .llm import LLMIntegrationError, generate_ai_commentary
from paper.engine import run_pending_sessions


_STRATEGY_INPUT_TYPES = get_type_hints(StrategyInput)
LOGGER = logging.getLogger(__name__)
_SNAPSHOT_TRIGGER_LOCK = threading.Lock()
_SNAPSHOT_TRIGGER_LAST_TS = 0.0
_SNAPSHOT_TRIGGER_THREAD: threading.Thread | None = None
_SNAPSHOT_LOOP_TIMER_LOCK = threading.Lock()
_SNAPSHOT_LOOP_TIMER: threading.Timer | None = None


def _use_local_snapshot_loop() -> bool:
    broker = str(getattr(settings, "CELERY_BROKER_URL", "") or "").strip().lower()
    always_eager = bool(getattr(settings, "CELERY_TASK_ALWAYS_EAGER", False))
    return broker.startswith("memory://") and not always_eager


def _schedule_snapshot_refresh_timer(*, user_id: str | None, delay: int) -> None:
    global _SNAPSHOT_LOOP_TIMER
    safe_delay = max(1, int(delay or 1))
    with _SNAPSHOT_LOOP_TIMER_LOCK:
        if _SNAPSHOT_LOOP_TIMER is not None and _SNAPSHOT_LOOP_TIMER.is_alive():
            return

        def _runner() -> None:
            try:
                trigger_market_snapshot_refresh(user_id=user_id)
            except Exception:
                LOGGER.exception("Local snapshot loop trigger failed")

        timer = threading.Timer(safe_delay, _runner)
        timer.daemon = True
        _SNAPSHOT_LOOP_TIMER = timer
        timer.start()


def _resolve_snapshot_user_id() -> str | None:
    try:
        from .models import RealtimeProfile, UserProfile
        from .market_provider import has_market_data_credentials, resolve_market_provider
    except Exception:
        return None

    def _user_has_source_credentials(user_id: int | str | None) -> bool:
        if not user_id:
            return False
        normalized = str(user_id)
        try:
            provider = resolve_market_provider(user_id=normalized)
            return has_market_data_credentials(user_id=normalized, provider=provider)
        except Exception:
            return False

    try:
        active_profile = RealtimeProfile.objects.filter(is_active=True).order_by("-updated_at").first()
        if active_profile and active_profile.user_id and _user_has_source_credentials(active_profile.user_id):
            return str(active_profile.user_id)
    except Exception:
        pass

    try:
        preferred_provider = str(getattr(settings, "MARKET_DATA_PROVIDER", "") or "").strip().lower()
        fallback_user: str | None = None
        for profile in UserProfile.objects.order_by("-updated_at")[:20]:
            uid = str(profile.user_id)
            if not _user_has_source_credentials(uid):
                continue
            try:
                provider = resolve_market_provider(user_id=uid)
            except Exception:
                provider = ""
            if preferred_provider and provider == preferred_provider:
                return uid
            if fallback_user is None:
                fallback_user = uid
        if fallback_user:
            return fallback_user
    except Exception:
        pass

    return None


def trigger_market_snapshot_refresh(
    *,
    user_id: str | None = None,
    prefer_thread: bool = False,
) -> dict[str, Any]:
    """Best-effort async trigger for snapshot rankings refresh.

    Prefers Celery scheduling, and falls back to an in-process daemon thread
    when no worker/broker is available. This function is non-blocking.
    """

    normalized_user_id = str(user_id).strip() if user_id is not None else None
    if normalized_user_id == "":
        normalized_user_id = None

    effective_user_id = normalized_user_id or _resolve_snapshot_user_id()
    force_thread_fallback = False
    try:
        from .market_provider import resolve_market_provider
        from .views import market as market_views

        provider = resolve_market_provider(user_id=effective_user_id)
        progress_payload = market_views.read_state(market_views.SNAPSHOT_RANKINGS_PROGRESS_STATE, default={})  # type: ignore[attr-defined]
        if isinstance(progress_payload, dict):
            progress_status = str(progress_payload.get("status") or "").strip().lower()
            if progress_status == "error":
                force_thread_fallback = True
        for timeframe_key in ("1d", *tuple(getattr(market_views, "MARKET_RANKINGS_TIMEFRAME_KEYS", ()))):
            payload = market_views._resolve_building_snapshot_payload(timeframe_key, provider=provider)  # type: ignore[attr-defined]
            if isinstance(payload, dict) and str(payload.get("status") or "").strip().lower() in {"stalled", "error"}:
                force_thread_fallback = True
                break
    except Exception:
        force_thread_fallback = False
    if prefer_thread:
        force_thread_fallback = True

    if not force_thread_fallback:
        try:
            refresh_market_snapshot_rankings.apply_async(kwargs={"user_id": effective_user_id})
            if _use_local_snapshot_loop():
                _schedule_snapshot_refresh_timer(user_id=effective_user_id, delay=1)
            return {"status": "queued", "via": "celery"}
        except Exception:
            LOGGER.debug("Celery enqueue for market snapshot refresh failed; fallback to thread", exc_info=True)

    min_interval = max(3, int(getattr(settings, "MARKET_RANKINGS_LOOP_DELAY_SECONDS", 3) or 3))
    now = time.time()
    with _SNAPSHOT_TRIGGER_LOCK:
        global _SNAPSHOT_TRIGGER_LAST_TS, _SNAPSHOT_TRIGGER_THREAD
        active_thread = _SNAPSHOT_TRIGGER_THREAD
        if active_thread is not None and active_thread.is_alive():
            return {"status": "skipped", "via": "thread", "reason": "already_running"}
        if now - _SNAPSHOT_TRIGGER_LAST_TS < min_interval:
            return {"status": "skipped", "via": "thread", "reason": "cooldown"}

        def _runner() -> None:
            try:
                refresh_market_snapshot_rankings(user_id=effective_user_id)
            except Exception:
                LOGGER.exception("In-process market snapshot refresh failed")

        thread = threading.Thread(target=_runner, name="market-snapshot-refresh", daemon=True)
        _SNAPSHOT_TRIGGER_THREAD = thread
        _SNAPSHOT_TRIGGER_LAST_TS = now
        thread.start()
    return {"status": "queued", "via": "thread"}


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


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 2},
    default_retry_delay=10,
    ignore_result=False,
    queue="paper_trading",
)
def refresh_market_snapshot_rankings(user_id: str | None = None) -> dict[str, Any]:
    from .views import market as market_views

    effective_user_id = str(user_id).strip() if user_id is not None else None
    if not effective_user_id:
        effective_user_id = _resolve_snapshot_user_id()
    result = market_views.refresh_snapshot_rankings(user_id=effective_user_id)

    if getattr(settings, "MARKET_RANKINGS_CONTINUOUS_LOOP", False) and not getattr(settings, "CELERY_TASK_ALWAYS_EAGER", False):
        status = str((result or {}).get("status") or "").strip().lower()
        default_delay = int(getattr(settings, "MARKET_RANKINGS_LOOP_DELAY_SECONDS", 3) or 3)
        if status == "locked":
            delay = max(1, min(default_delay, 5))
        elif status in {"missing_credentials", "error"}:
            delay = max(
                30,
                min(
                    int(getattr(settings, "MARKET_RANKINGS_REFRESH_SECONDS", 900) or 900),
                    300,
                ),
            )
        else:
            delay = max(1, default_delay)
        try:
            refresh_market_snapshot_rankings.apply_async(countdown=delay)
            if _use_local_snapshot_loop():
                _schedule_snapshot_refresh_timer(user_id=effective_user_id, delay=delay)
        except Exception:
            LOGGER.exception("Failed to schedule next market rankings refresh loop; falling back to local timer")
            _schedule_snapshot_refresh_timer(user_id=effective_user_id, delay=delay)

    return result


def _normalize_ai_history(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    max_chars = int(getattr(settings, "AI_CHAT_MAX_MESSAGE_CHARS", 4000))
    normalized: list[dict[str, str]] = []
    for entry in raw[-int(getattr(settings, "AI_CHAT_MAX_HISTORY", 20)) :]:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "user")[:20]
        content = str(entry.get("content") or "")[:max_chars]
        normalized.append({"role": role or "user", "content": content})
    return normalized


def execute_ai_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"error": "invalid_payload"}
    context = payload.get("context")
    if not isinstance(context, dict):
        return {"error": "missing_context"}
    message = payload.get("message") or ""
    if not isinstance(message, str):
        message = str(message)
    max_chars = int(getattr(settings, "AI_CHAT_MAX_MESSAGE_CHARS", 4000))
    if len(message) > max_chars:
        message = message[:max_chars]

    web_query = payload.get("web_query")
    if not isinstance(web_query, str):
        web_query = None
    try:
        web_max_results = int(payload.get("web_max_results")) if payload.get("web_max_results") is not None else None
    except (TypeError, ValueError):
        web_max_results = None
    if web_max_results is not None:
        web_max_results = max(1, min(web_max_results, 12))

    response_schema = payload.get("response_schema")
    response_format = payload.get("response_format")
    if not isinstance(response_schema, dict):
        response_schema = None
    if not isinstance(response_format, dict):
        response_format = None

    rag_query = payload.get("rag_query")
    if not isinstance(rag_query, str):
        rag_query = None
    rag_context = payload.get("rag_context")
    if not isinstance(rag_context, str):
        rag_context = None
    if rag_context and len(rag_context) > 2000:
        rag_context = rag_context[:2000]
    try:
        rag_top_k = int(payload.get("rag_top_k")) if payload.get("rag_top_k") is not None else None
    except (TypeError, ValueError):
        rag_top_k = None
    if rag_top_k is not None:
        rag_top_k = max(1, min(rag_top_k, 20))

    tools = payload.get("tools")
    if isinstance(tools, (list, tuple, set)):
        tools = [str(item).strip() for item in tools if str(item).strip()][:12]
    elif isinstance(tools, str):
        tools = tools.strip()
    elif isinstance(tools, bool):
        tools = tools
    else:
        tools = None

    tool_choice = payload.get("tool_choice")
    if not isinstance(tool_choice, (str, dict)):
        tool_choice = None

    image_list: list[str] = []
    images = payload.get("images")
    if isinstance(images, (list, tuple)):
        for item in images:
            if item:
                image_list.append(str(item))
    images = image_list[:4]

    extra_params = payload.get("extra_params")
    if not isinstance(extra_params, dict):
        extra_params = None

    user_obj = None
    user_id = payload.get("user_id")
    if user_id:
        try:
            user_obj = get_user_model().objects.filter(pk=user_id).first()
        except Exception:
            user_obj = None

    try:
        result = generate_ai_commentary(
            context,
            show_thoughts=bool(payload.get("show_thoughts", True)),
            user_message=message,
            history=_normalize_ai_history(payload.get("history")),
            enable_web=bool(payload.get("enable_web", False)),
            web_query=web_query,
            web_max_results=web_max_results,
            profile=True,
            model_name=payload.get("model"),
            tools=tools,
            tool_choice=tool_choice,
            response_schema=response_schema,
            response_format=response_format,
            rag_query=rag_query,
            rag_top_k=rag_top_k,
            rag_context=rag_context,
            images=images,
            extra_params=extra_params,
            user=user_obj,
        )
    except LLMIntegrationError as exc:
        return {"error": str(exc)}
    return result


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=10,
    retry_kwargs={"max_retries": 2},
    default_retry_delay=10,
    ignore_result=False,
    queue="ai",
)
def run_ai_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    self.update_state(state="PROGRESS", meta=_progress_meta(10, "ai_bootstrap"))
    self.update_state(state="PROGRESS", meta=_progress_meta(50, "ai_running"))
    result = execute_ai_job(payload)
    self.update_state(state="PROGRESS", meta=_progress_meta(90, "ai_finalizing"))
    return result
