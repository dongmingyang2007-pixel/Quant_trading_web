from __future__ import annotations

import json
import os
import queue
import re
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
import threading
import time

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_POST
from typing import Any
from django.utils.translation import gettext as _

from .. import screener
from ..forms import QuantStrategyForm
from ..llm import LLMIntegrationError, generate_ai_commentary
from ..observability import ensure_request_id, record_metric, track_latency
from ..task_queue import (
    SyncResult,
    get_task_status,
    submit_backtest_task,
    submit_rl_task,
    submit_training_task,
)
from ..train_ml import available_engines
from ..rate_limit import check_rate_limit, rate_limit_key
from .dashboard import build_strategy_input

SCREENER_PAGE_SIZE = int(os.environ.get("SCREENER_PAGE_SIZE", "50") or 50)
SCREENER_MAX_PAGE_SIZE = int(os.environ.get("SCREENER_MAX_PAGE_SIZE", "100") or 100)
SCREENER_MAX_OFFSET = int(os.environ.get("SCREENER_MAX_OFFSET", str(SCREENER_MAX_PAGE_SIZE * 40)))
REQUEST_MAX_TIMEOUT = float(os.environ.get("API_MAX_TIMEOUT_SECONDS", "25") or 25.0)

AI_EXECUTOR = ThreadPoolExecutor(max_workers=getattr(settings, "AI_CHAT_MAX_WORKERS", 4))
AI_MAX_IN_FLIGHT = max(1, getattr(settings, "AI_CHAT_MAX_IN_FLIGHT", 8))
AI_GUARD_WAIT = max(0.05, float(getattr(settings, "AI_CHAT_GUARD_WAIT_SECONDS", 0.75)))
AI_INFLIGHT_GUARD = threading.BoundedSemaphore(value=AI_MAX_IN_FLIGHT)
AI_RATE_WINDOW = max(5, getattr(settings, "AI_CHAT_RATE_WINDOW_SECONDS", 60))
AI_RATE_MAX_CALLS = max(1, getattr(settings, "AI_CHAT_RATE_MAX_CALLS", 30))
AI_RATE_CACHE_ALIAS = getattr(settings, "AI_CHAT_RATE_CACHE_ALIAS", "default")


def _rate_key(request):
    return rate_limit_key(request)


def _is_rate_limited(request):
    state = check_rate_limit(
        cache_alias=AI_RATE_CACHE_ALIAS,
        key=f"ai-chat-rate:{_rate_key(request)}",
        window=AI_RATE_WINDOW,
        max_calls=AI_RATE_MAX_CALLS,
    )
    return state.limited, state.retry_after


def _resolve_page_size(raw: str | None) -> tuple[int, bool]:
    requested = SCREENER_PAGE_SIZE
    try:
        requested = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        requested = SCREENER_PAGE_SIZE
    value = max(5, requested)
    value = min(value, SCREENER_MAX_PAGE_SIZE)
    return value, value != requested


def _resolve_offset(raw: str | None) -> tuple[int, bool]:
    requested = 0
    try:
        requested = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        requested = 0
    value = max(0, requested)
    value = min(value, SCREENER_MAX_OFFSET)
    return value, value != requested


def _clamp_timeout(raw: Any) -> tuple[float, bool]:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = REQUEST_MAX_TIMEOUT
    clamped = min(max(1.0, val), REQUEST_MAX_TIMEOUT)
    return clamped, clamped != val


def _sanitize_short_text(value: str | None, *, max_length: int = 32) -> str:
    if not value:
        return ""
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in {"-", "_", " ", "/"})
    return cleaned[:max_length]


def _sse_message(event: str, data: dict[str, object]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _build_screener_snapshot(*, user, params, request_id: str) -> tuple[dict[str, Any], int]:
    if hasattr(params, "dict"):
        params = params.dict()
    elif not isinstance(params, dict):
        params = {}
    offset, offset_clamped = _resolve_offset(params.get("offset"))
    limit, limit_clamped = _resolve_page_size(params.get("limit"))
    sector_param = params.get("sector") or params.get("sector_slug")
    sector_label, sector_slug = screener.resolve_sector(sector_param)
    industry = _sanitize_short_text(params.get("industry")) or None
    market = screener.sanitize_market(params.get("market"))
    market_meta = screener.ALLOWED_MARKETS.get(market, {})
    user_id = user.id if getattr(user, "is_authenticated", False) else None

    try:
        with track_latency(
            "api.screener.fetch",
            user_id=user_id,
            request_id=request_id,
            market=market,
            limit=limit,
            offset=offset,
        ):
            page_data = screener.fetch_page(
                offset=offset,
                size=limit,
                sector=sector_label,
                industry=industry,
                market=market,
            )
    except Exception as exc:
        record_metric(
            "api.screener.error",
            request_id=request_id,
            user_id=user_id,
            error=str(exc),
        )
        return ({"error": str(exc), "rows": [], "has_more": False}, 503)

    loaded = page_data.get("offset", offset) + len(page_data.get("rows", []))
    payload = {
        "rows": page_data.get("rows", []),
        "offset": page_data.get("offset", offset),
        "size": page_data.get("size", limit),
        "has_more": page_data.get("has_more", False),
        "sector": sector_label,
        "sector_slug": sector_slug,
        "industry": industry,
        "market": market,
        "market_label": market_meta.get("label", market.upper()),
        "loaded": loaded,
        "next_offset": loaded,
        "total": page_data.get("total", 0),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "limit_clamped": limit_clamped,
        "offset_clamped": offset_clamped,
    }
    record_metric(
        "api.screener.response",
        request_id=request_id,
        user_id=user_id,
        rows=len(payload["rows"]),
        has_more=payload["has_more"],
        market=market,
    )
    return payload, 200


def _normalize_symbol_list(value: Any, *, fallback: str | None = None, limit: int = 20) -> list[str]:
    symbols: list[str] = []
    items: list[str] = []
    if isinstance(value, str):
        items = [chunk.strip() for chunk in re.split(r"[\s,;/]+", value) if chunk.strip()]
    elif isinstance(value, (list, tuple, set)):
        for entry in value:
            if isinstance(entry, str) and entry.strip():
                items.append(entry.strip())
    if not items and fallback:
        items = [fallback]
    seen = set()
    for item in items:
        token = item.upper()
        if token and token not in seen:
            seen.add(token)
            symbols.append(token)
        if len(symbols) >= limit:
            break
    return symbols


def _normalize_engines(value: Any) -> list[str]:
    allowed = {engine.lower(): engine for engine in available_engines()}
    engines: list[str] = []
    if isinstance(value, str):
        candidates = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    elif isinstance(value, (list, tuple, set)):
        candidates = [str(chunk).strip() for chunk in value if str(chunk).strip()]
    else:
        candidates = []
    for item in candidates:
        key = item.lower()
        normalized = allowed.get(key)
        if normalized and normalized not in engines:
            engines.append(normalized)
    return engines


@login_required
def screener_snapshot_api(request):
    request_id = ensure_request_id(request)
    payload, status_code = _build_screener_snapshot(user=request.user, params=request.GET, request_id=request_id)
    payload["request_id"] = request_id
    return JsonResponse(payload, json_dumps_params={"ensure_ascii": False})


@require_POST
@login_required
def ai_chat(request):
    request_id = ensure_request_id(request)
    limited, retry_after = _is_rate_limited(request)
    if limited:
        record_metric(
            "ai.chat.rate_limited",
            request_id=request_id,
            user_id=request.user.id,
            retry_after=retry_after,
        )
        payload = {"error": _("请求过于频繁，请稍后再试。"), "rate_limited": True, "request_id": request_id}
        if retry_after:
            payload["retry_after_seconds"] = retry_after
        return JsonResponse(payload, status=429)
    guard_acquired = AI_INFLIGHT_GUARD.acquire(timeout=AI_GUARD_WAIT)
    if not guard_acquired:
        record_metric(
            "ai.chat.busy",
            request_id=request_id,
            user_id=request.user.id,
        )
        return JsonResponse(
            {"error": _("AI 当前请求较多，请稍后再试。"), "busy": True, "request_id": request_id},
            status=503,
        )
    future = None
    try:
        raw_body = request.body or b""
        if raw_body and len(raw_body) > settings.AI_CHAT_MAX_PAYLOAD_BYTES:
            return JsonResponse({"error": _("请求内容过大，请缩短问题或清空历史。"), "request_id": request_id}, status=400)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return JsonResponse({"error": _("无效的请求体"), "request_id": request_id}, status=400)

        context = payload.get("context")
        message = payload.get("message")
        history = payload.get("history") or []
        show_thoughts = payload.get("show_thoughts", True)
        enable_web = bool(payload.get("enable_web", False))
        model_name = payload.get("model")

        if not isinstance(context, dict):
            return JsonResponse({"error": _("缺少上下文数据")}, status=400)

        if not isinstance(history, list):
            history = []
        history = history[-settings.AI_CHAT_MAX_HISTORY:]
        trimmed_history: list[dict[str, str]] = []
        for entry in history:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "user")[:20]
            content = str(entry.get("content") or "")[: settings.AI_CHAT_MAX_MESSAGE_CHARS]
            trimmed_history.append({"role": role or "user", "content": content})
        history = trimmed_history

        if message is None:
            message = ""
        if not isinstance(message, str):
            message = str(message)
        if len(message) > settings.AI_CHAT_MAX_MESSAGE_CHARS:
            message = message[: settings.AI_CHAT_MAX_MESSAGE_CHARS]

        if not message and not history:
            return JsonResponse({"error": _("请先输入问题或提供历史上下文。"), "request_id": request_id}, status=400)

        def _invoke_ai():
            with track_latency(
                "ai.chat.invoke",
                user_id=request.user.id,
                request_id=request_id,
                model=model_name or "",
                enable_web=enable_web,
            ):
                return generate_ai_commentary(
                    context,
                    show_thoughts=bool(show_thoughts),
                    user_message=message,
                    history=history,
                    enable_web=enable_web,
                    profile=True,
                    model_name=model_name,
                )

        future = AI_EXECUTOR.submit(_invoke_ai)
        ai_payload = future.result(timeout=settings.AI_CHAT_TIMEOUT_SECONDS)
    except FuturesTimeout:
        if future:
            future.cancel()
        record_metric(
            "ai.chat.timeout",
            request_id=request_id,
            user_id=request.user.id,
            model=model_name or "",
        )
        return JsonResponse({"error": _("AI 分析超时，请稍后重试。"), "request_id": request_id}, status=504)
    except LLMIntegrationError as exc:
        record_metric(
            "ai.chat.error",
            request_id=request_id,
            user_id=request.user.id,
            model=model_name or "",
            error=str(exc),
        )
        return JsonResponse({"error": str(exc), "request_id": request_id}, status=500)
    finally:
        AI_INFLIGHT_GUARD.release()

    ai_payload["history"] = history + [
        {
            "role": "assistant",
            "content": ai_payload.get("answer", ""),
            "web_used": ai_payload.get("web_used", enable_web),
            "web_results": ai_payload.get("web_results", []),
        }
    ]
    ai_payload["request_id"] = request_id
    record_metric(
        "ai.chat.response",
        request_id=request_id,
        user_id=request.user.id,
        model=model_name or "",
        enable_web=enable_web,
    )
    return JsonResponse(ai_payload)


@require_POST
@login_required
def ai_chat_stream(request):
    """Server-sent events endpoint that pushes AI进度 + 最终答案。"""
    request_id = ensure_request_id(request)
    limited, retry_after = _is_rate_limited(request)
    if limited:
        payload = {"error": _("请求过于频繁，请稍后再试。"), "rate_limited": True, "request_id": request_id}
        if retry_after:
            payload["retry_after_seconds"] = retry_after
        return JsonResponse(payload, status=429)
    limited, retry_after = _is_rate_limited(request)
    if limited:
        payload = {"error": _("请求过于频繁，请稍后再试。"), "rate_limited": True, "request_id": request_id}
        if retry_after:
            payload["retry_after_seconds"] = retry_after
        return JsonResponse(payload, status=429)
    guard_acquired = AI_INFLIGHT_GUARD.acquire(timeout=AI_GUARD_WAIT)
    if not guard_acquired:
        return JsonResponse(
            {"error": _("AI 当前请求较多，请稍后再试。"), "busy": True, "request_id": request_id},
            status=503,
        )

    guard_released = False

    def _release_guard():
        nonlocal guard_released
        if not guard_released:
            AI_INFLIGHT_GUARD.release()
            guard_released = True

    try:
        raw_body = request.body or b""
        if raw_body and len(raw_body) > settings.AI_CHAT_MAX_PAYLOAD_BYTES:
            _release_guard()
            return JsonResponse({"error": _("请求内容过大，请缩短问题或清空历史。"), "request_id": request_id}, status=400)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            _release_guard()
            return JsonResponse({"error": _("无效的请求体"), "request_id": request_id}, status=400)

        context = payload.get("context")
        message = payload.get("message")
        history = payload.get("history") or []
        show_thoughts = payload.get("show_thoughts", True)
        enable_web = bool(payload.get("enable_web", False))
        model_name = payload.get("model")

        if not isinstance(context, dict):
            _release_guard()
            return JsonResponse({"error": "缺少上下文数据", "request_id": request_id}, status=400)

        if not isinstance(history, list):
            history = []
        history = history[-settings.AI_CHAT_MAX_HISTORY:]
        trimmed_history: list[dict[str, str]] = []
        for entry in history:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "user")[:20]
            content = str(entry.get("content") or "")[: settings.AI_CHAT_MAX_MESSAGE_CHARS]
            trimmed_history.append({"role": role or "user", "content": content})
        history = trimmed_history

        if message is None:
            message = ""
        if not isinstance(message, str):
            message = str(message)
        if len(message) > settings.AI_CHAT_MAX_MESSAGE_CHARS:
            message = message[: settings.AI_CHAT_MAX_MESSAGE_CHARS]

        if not message and not history:
            _release_guard()
            return JsonResponse({"error": _("请先输入问题或提供历史上下文。"), "request_id": request_id}, status=400)

        event_queue: "queue.Queue[tuple[str, dict[str, Any]]]" = queue.Queue()

        def _emit(stage: str, payload: dict[str, Any]) -> None:
            event_queue.put(("progress", {"stage": stage, **payload, "request_id": request_id}))

        def _worker() -> None:
            try:
                def _progress(stage: str, extra: dict[str, Any]) -> None:
                    _emit(stage, extra)

                result = generate_ai_commentary(
                    context,
                    show_thoughts=bool(show_thoughts),
                    user_message=message,
                    history=history,
                    enable_web=enable_web,
                    profile=True,
                    model_name=model_name,
                    progress_callback=_progress,
                )
                event_queue.put(("message", {**result, "request_id": request_id}))
                record_metric(
                    "ai.chat.stream.response",
                    request_id=request_id,
                    user_id=request.user.id,
                    model=result.get("selected_model"),
                )
            except LLMIntegrationError as exc:
                event_queue.put(("error", {"error": str(exc), "request_id": request_id}))
            except Exception as exc:  # pragma: no cover - safety net
                event_queue.put(("error", {"error": f"{exc}", "request_id": request_id}))
            finally:
                event_queue.put(("end", {"request_id": request_id}))
                _release_guard()

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        def event_stream():
            yield _sse_message("progress", {"message": _("AI 已开始处理"), "request_id": request_id})
            while True:
                event, data = event_queue.get()
                yield _sse_message(event, data)
                if event == "end":
                    break

        record_metric(
            "ai.chat.stream.connect",
            request_id=request_id,
            user_id=request.user.id,
            model=model_name or "",
        )
        response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response
    except Exception:
        _release_guard()
        raise


@login_required
@require_POST
def enqueue_backtest_task(request):
    request_id = ensure_request_id(request)
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": _("请求体解析失败。"), "request_id": request_id}, status=400)
    form_data = payload.get("params") or {}
    form = QuantStrategyForm(form_data, language=getattr(request, "LANGUAGE_CODE", None))
    if not form.is_valid():
        return JsonResponse(
            {"error": _("提交内容有误，请检查后重试。"), "details": form.errors, "request_id": request_id},
            status=400,
            json_dumps_params={"ensure_ascii": False},
        )
    strategy_input, _ = build_strategy_input(form.cleaned_data, request_id=request_id, user=request.user)
    job = submit_backtest_task(asdict(strategy_input))
    response = {
        "task_id": getattr(job, "id", ""),
        "state": getattr(job, "state", "PENDING"),
        "request_id": request_id,
    }
    if isinstance(job, SyncResult):
        response["result"] = job.result
    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})


@login_required
def _task_status_response(request, task_id: str):
    status = get_task_status(task_id)
    status["request_id"] = ensure_request_id(request)
    return JsonResponse(status, json_dumps_params={"ensure_ascii": False})


@login_required
def backtest_task_status(request, task_id: str):
    return _task_status_response(request, task_id)


@login_required
@require_POST
def enqueue_training_task(request):
    request_id = ensure_request_id(request)
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": _("请求体解析失败。"), "request_id": request_id}, status=400)
    form_data = payload.get("params") or {}
    form = QuantStrategyForm(form_data, language=getattr(request, "LANGUAGE_CODE", None))
    if not form.is_valid():
        return JsonResponse(
            {"error": _("提交内容有误，请检查后重试。"), "details": form.errors, "request_id": request_id},
            status=400,
            json_dumps_params={"ensure_ascii": False},
        )
    strategy_input, _ = build_strategy_input(form.cleaned_data, request_id=request_id, user=request.user)
    raw_tickers = payload.get("tickers") or form_data.get("tickers")
    tickers = _normalize_symbol_list(raw_tickers, fallback=str(form.cleaned_data.get("ticker", "")).upper())
    if not tickers:
        return JsonResponse({"error": _("需要至少一个有效的股票代码。"), "request_id": request_id}, status=400)
    engines = _normalize_engines(payload.get("engines") or form_data.get("engines"))
    job_payload: dict[str, Any] = {
        "base_params": asdict(strategy_input),
        "tickers": tickers,
    }
    if engines:
        job_payload["engines"] = engines
    job = submit_training_task(job_payload)
    response = {
        "task_id": getattr(job, "id", ""),
        "state": getattr(job, "state", "PENDING"),
        "request_id": request_id,
    }
    if isinstance(job, SyncResult):
        response["result"] = job.result
    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})


@login_required
def training_task_status(request, task_id: str):
    return _task_status_response(request, task_id)


@login_required
@require_POST
def enqueue_rl_task(request):
    request_id = ensure_request_id(request)
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": _("请求体解析失败。"), "request_id": request_id}, status=400)
    form_data = payload.get("params") or {}
    form = QuantStrategyForm(form_data, language=getattr(request, "LANGUAGE_CODE", None))
    if not form.is_valid():
        return JsonResponse(
            {"error": _("提交内容有误，请检查后重试。"), "details": form.errors, "request_id": request_id},
            status=400,
            json_dumps_params={"ensure_ascii": False},
        )
    strategy_input, _ = build_strategy_input(form.cleaned_data, request_id=request_id, user=request.user)
    job = submit_rl_task(asdict(strategy_input))
    response = {
        "task_id": getattr(job, "id", ""),
        "state": getattr(job, "state", "PENDING"),
        "request_id": request_id,
    }
    if isinstance(job, SyncResult):
        response["result"] = job.result
    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})


@login_required
def rl_task_status(request, task_id: str):
    return _task_status_response(request, task_id)
