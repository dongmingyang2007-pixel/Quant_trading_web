from __future__ import annotations

import io
import json
import os
import queue
import re
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timezone
import threading
from itertools import combinations

import pandas as pd

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
    submit_ai_task,
    submit_backtest_task,
    submit_rl_task,
    submit_training_task,
)
from ..train_ml import available_engines
from ..rate_limit import check_rate_limit, rate_limit_key
from ..history import get_history_record
from ..portfolio import portfolio_stats
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
AI_RAG_MAX_FILES = int(os.environ.get("AI_RAG_MAX_FILES", "4") or 4)
AI_RAG_MAX_BYTES = int(os.environ.get("AI_RAG_MAX_BYTES", str(2_000_000)) or 2_000_000)


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


def _read_rag_upload(upload) -> tuple[str, str | None, str]:
    name = getattr(upload, "name", "") or "upload"
    size = getattr(upload, "size", None)
    if isinstance(size, int) and size > AI_RAG_MAX_BYTES:
        return "", "file_too_large", name
    try:
        data = upload.read(AI_RAG_MAX_BYTES + 1)
    except Exception:
        return "", "read_failed", name
    if len(data) > AI_RAG_MAX_BYTES:
        return "", "file_too_large", name
    ext = os.path.splitext(name)[1].lower()
    if ext == ".pdf":
        try:
            import pdfplumber  # type: ignore
        except Exception:
            return "", "pdf_unavailable", name
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                pages = [(page.extract_text() or "") for page in pdf.pages]
            text = "\n".join([p for p in pages if p.strip()])
        except Exception:
            return "", "pdf_parse_failed", name
        return text, None, name
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="ignore")
    return text, None, name


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
                user_id=str(user.id) if getattr(user, "is_authenticated", False) else None,
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
        web_query = payload.get("web_query")
        web_max_results = payload.get("web_max_results")
        tools = payload.get("tools")
        tool_choice = payload.get("tool_choice")
        response_schema = payload.get("response_schema")
        response_format = payload.get("response_format")
        rag_query = payload.get("rag_query")
        rag_top_k = payload.get("rag_top_k")
        rag_context = payload.get("rag_context")
        images = payload.get("images") or []
        extra_params = payload.get("extra_params")
        web_query = payload.get("web_query")
        web_max_results = payload.get("web_max_results")
        tools = payload.get("tools")
        tool_choice = payload.get("tool_choice")
        response_schema = payload.get("response_schema")
        response_format = payload.get("response_format")
        rag_query = payload.get("rag_query")
        rag_top_k = payload.get("rag_top_k")
        rag_context = payload.get("rag_context")
        images = payload.get("images") or []
        extra_params = payload.get("extra_params")

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

        if not isinstance(web_query, str):
            web_query = None
        try:
            web_max_results = int(web_max_results) if web_max_results is not None else None
        except (TypeError, ValueError):
            web_max_results = None
        if web_max_results is not None:
            web_max_results = max(1, min(web_max_results, 12))

        if not isinstance(response_schema, dict):
            response_schema = None
        if not isinstance(response_format, dict):
            response_format = None
        if not isinstance(rag_query, str):
            rag_query = None
        try:
            rag_top_k = int(rag_top_k) if rag_top_k is not None else None
        except (TypeError, ValueError):
            rag_top_k = None
        if rag_top_k is not None:
            rag_top_k = max(1, min(rag_top_k, 20))
        if not isinstance(rag_context, str):
            rag_context = None
        if rag_context and len(rag_context) > 2000:
            rag_context = rag_context[:2000]

        if isinstance(tools, (list, tuple, set)):
            tools = [str(item).strip() for item in tools if str(item).strip()][:12]
        elif isinstance(tools, str):
            tools = tools.strip()
        elif isinstance(tools, bool):
            tools = tools
        else:
            tools = None

        if not isinstance(tool_choice, (str, dict)):
            tool_choice = None

        image_list: list[str] = []
        if isinstance(images, (list, tuple)):
            for item in images:
                if item:
                    image_list.append(str(item))
        images = image_list[:4]

        if not isinstance(extra_params, dict):
            extra_params = None

        if not isinstance(web_query, str):
            web_query = None
        try:
            web_max_results = int(web_max_results) if web_max_results is not None else None
        except (TypeError, ValueError):
            web_max_results = None
        if web_max_results is not None:
            web_max_results = max(1, min(web_max_results, 12))

        if not isinstance(response_schema, dict):
            response_schema = None
        if not isinstance(response_format, dict):
            response_format = None
        if not isinstance(rag_query, str):
            rag_query = None
        try:
            rag_top_k = int(rag_top_k) if rag_top_k is not None else None
        except (TypeError, ValueError):
            rag_top_k = None
        if rag_top_k is not None:
            rag_top_k = max(1, min(rag_top_k, 20))
        if not isinstance(rag_context, str):
            rag_context = None
        if rag_context and len(rag_context) > 2000:
            rag_context = rag_context[:2000]

        if isinstance(tools, (list, tuple, set)):
            tools = [str(item).strip() for item in tools if str(item).strip()][:12]
        elif isinstance(tools, str):
            tools = tools.strip()
        elif isinstance(tools, bool):
            tools = tools
        else:
            tools = None

        if not isinstance(tool_choice, (str, dict)):
            tool_choice = None

        image_list: list[str] = []
        if isinstance(images, (list, tuple)):
            for item in images:
                if item:
                    image_list.append(str(item))
        images = image_list[:4]

        if not isinstance(extra_params, dict):
            extra_params = None

        if message is None:
            message = ""
        if not isinstance(message, str):
            message = str(message)
        if len(message) > settings.AI_CHAT_MAX_MESSAGE_CHARS:
            message = message[: settings.AI_CHAT_MAX_MESSAGE_CHARS]

        if not message and not history and not images:
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
                    web_query=web_query,
                    web_max_results=web_max_results,
                    profile=True,
                    model_name=model_name,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_schema=response_schema,
                    response_format=response_format,
                    rag_query=rag_query,
                    rag_top_k=rag_top_k,
                    rag_context=rag_context,
                    images=images,
                    extra_params=extra_params,
                    user=request.user,
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
        timings_ms=ai_payload.get("timings_ms"),
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

        if not message and not history and not images:
            _release_guard()
            return JsonResponse({"error": _("请先输入问题或提供历史上下文。"), "request_id": request_id}, status=400)

        event_queue: "queue.Queue[tuple[str, dict[str, Any]]]" = queue.Queue()

        def _emit(stage: str, payload: dict[str, Any]) -> None:
            event_name = "delta" if stage == "delta" else "progress"
            event_queue.put((event_name, {"stage": stage, **payload, "request_id": request_id}))

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
                    web_query=web_query,
                    web_max_results=web_max_results,
                    profile=True,
                    model_name=model_name,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_schema=response_schema,
                    response_format=response_format,
                    rag_query=rag_query,
                    rag_top_k=rag_top_k,
                    rag_context=rag_context,
                    images=images,
                    extra_params=extra_params,
                    progress_callback=_progress,
                    user=request.user,
                )
                # 推送成本提示（如有）
                if result.get("profile"):
                    cost_payload = {"profile": result["profile"], "request_id": request_id}
                    _emit("progress", {"stage": "cost", **cost_payload})
                event_queue.put(("message", {**result, "request_id": request_id}))
                record_metric(
                    "ai.chat.stream.response",
                    request_id=request_id,
                    user_id=request.user.id,
                    model=result.get("selected_model"),
                    timings_ms=result.get("timings_ms"),
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
def rag_ingest(request):
    request_id = ensure_request_id(request)
    texts: list[str] = []
    errors: list[str] = []
    metadata: dict[str, Any] = {}
    chunk_size = None
    overlap = None

    if request.FILES:
        uploads = list(request.FILES.values())[: max(1, AI_RAG_MAX_FILES)]
        for upload in uploads:
            text, err, name = _read_rag_upload(upload)
            if err:
                errors.append(f"{name}:{err}")
                continue
            if text and text.strip():
                texts.append(text)
    else:
        try:
            payload = json.loads((request.body or b"{}").decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return JsonResponse({"error": _("无效的请求体"), "request_id": request_id}, status=400)
        if not isinstance(payload, dict):
            return JsonResponse({"error": _("无效的请求体"), "request_id": request_id}, status=400)
        raw_texts = payload.get("texts") or []
        if isinstance(payload.get("text"), str):
            raw_texts = [payload.get("text")] + (raw_texts if isinstance(raw_texts, list) else [])
        if isinstance(raw_texts, str):
            raw_texts = [raw_texts]
        if isinstance(raw_texts, list):
            for item in raw_texts[:20]:
                if item:
                    texts.append(str(item))
        if isinstance(payload.get("metadata"), dict):
            metadata = payload.get("metadata")
        chunk_size = payload.get("chunk_size")
        overlap = payload.get("overlap")

    if not texts:
        return JsonResponse(
            {"error": _("未提供可导入的文本。"), "request_id": request_id, "errors": errors},
            status=400,
        )

    try:
        chunk_size = int(chunk_size) if chunk_size is not None else None
    except (TypeError, ValueError):
        chunk_size = None
    if chunk_size is not None:
        chunk_size = max(120, min(chunk_size, 1200))
    try:
        overlap = int(overlap) if overlap is not None else None
    except (TypeError, ValueError):
        overlap = None
    if overlap is not None:
        overlap = max(0, min(overlap, 400))

    try:
        from ..ai_rag import ingest_texts

        result = ingest_texts(
            texts,
            user_id=str(request.user.id),
            metadata=metadata,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except Exception as exc:
        return JsonResponse({"error": str(exc), "request_id": request_id}, status=500)

    payload = {"request_id": request_id, "errors": errors}
    if isinstance(result, dict):
        payload.update(result)
    return JsonResponse(payload, json_dumps_params={"ensure_ascii": False})


@login_required
@require_POST
def rag_query_api(request):
    request_id = ensure_request_id(request)
    try:
        payload = json.loads((request.body or b"{}").decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": _("无效的请求体"), "request_id": request_id}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({"error": _("无效的请求体"), "request_id": request_id}, status=400)
    query = payload.get("query") or payload.get("text") or ""
    if not isinstance(query, str) or not query.strip():
        return JsonResponse({"error": _("缺少检索问题。"), "request_id": request_id}, status=400)
    try:
        top_k = int(payload.get("top_k") or 5)
    except (TypeError, ValueError):
        top_k = 5
    top_k = max(1, min(top_k, 20))
    try:
        from ..ai_rag import query as rag_query

        results = rag_query(query.strip(), user_id=str(request.user.id), top_k=top_k)
    except Exception as exc:
        return JsonResponse({"error": str(exc), "request_id": request_id}, status=500)
    return JsonResponse({"results": results, "request_id": request_id}, json_dumps_params={"ensure_ascii": False})


@login_required
@require_POST
def enqueue_ai_task(request):
    request_id = ensure_request_id(request)
    raw_body = request.body or b""
    if raw_body and len(raw_body) > settings.AI_CHAT_MAX_PAYLOAD_BYTES:
        return JsonResponse({"error": _("请求内容过大，请缩短问题或清空历史。"), "request_id": request_id}, status=400)
    try:
        payload = json.loads(raw_body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": _("无效的请求体"), "request_id": request_id}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({"error": _("无效的请求体"), "request_id": request_id}, status=400)

    context = payload.get("context")
    if not isinstance(context, dict):
        return JsonResponse({"error": _("缺少上下文数据"), "request_id": request_id}, status=400)

    message = payload.get("message") or ""
    if not isinstance(message, str):
        message = str(message)
    if len(message) > settings.AI_CHAT_MAX_MESSAGE_CHARS:
        message = message[: settings.AI_CHAT_MAX_MESSAGE_CHARS]

    history = payload.get("history") or []
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

    raw_images = payload.get("images")
    if not message and not trimmed_history and not raw_images:
        return JsonResponse({"error": _("请先输入问题或提供历史上下文。"), "request_id": request_id}, status=400)

    task_payload = {
        "context": context,
        "message": message,
        "history": trimmed_history,
        "show_thoughts": bool(payload.get("show_thoughts", True)),
        "enable_web": bool(payload.get("enable_web", False)),
        "web_query": payload.get("web_query"),
        "web_max_results": payload.get("web_max_results"),
        "model": payload.get("model"),
        "tools": payload.get("tools"),
        "tool_choice": payload.get("tool_choice"),
        "response_schema": payload.get("response_schema"),
        "response_format": payload.get("response_format"),
        "rag_query": payload.get("rag_query"),
        "rag_top_k": payload.get("rag_top_k"),
        "rag_context": payload.get("rag_context"),
        "images": raw_images,
        "extra_params": payload.get("extra_params"),
        "user_id": request.user.id,
    }
    job = submit_ai_task(task_payload)
    response = {
        "task_id": getattr(job, "id", ""),
        "state": getattr(job, "state", "PENDING"),
        "request_id": request_id,
    }
    if isinstance(job, SyncResult):
        response["result"] = job.result
    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})


@login_required
def ai_task_status(request, task_id: str):
    status = get_task_status(task_id)
    status["request_id"] = ensure_request_id(request)
    return JsonResponse(status, json_dumps_params={"ensure_ascii": False})


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
    strategy_input, _warnings = build_strategy_input(form.cleaned_data, request_id=request_id, user=request.user)
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
    strategy_input, _warnings = build_strategy_input(form.cleaned_data, request_id=request_id, user=request.user)
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
def build_portfolio_api(request):
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": _("Invalid payload.")}, status=400)

    components = payload.get("components")
    if not isinstance(components, list):
        return JsonResponse({"error": _("Please choose at least one strategy.")}, status=400)

    aggregated: dict[str, float] = {}
    for entry in components:
        if not isinstance(entry, dict):
            continue
        record_id = str(entry.get("record_id") or "").strip()
        try:
            weight = float(entry.get("weight") or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        if record_id and weight > 0:
            aggregated[record_id] = aggregated.get(record_id, 0.0) + weight

    try:
        cash_weight = max(float(payload.get("cash_weight") or 0.0), 0.0)
    except (TypeError, ValueError):
        cash_weight = 0.0

    if not aggregated:
        return JsonResponse({"error": _("Select at least one strategy with a positive weight.")}, status=400)

    series_map: dict[str, pd.Series] = {}
    info_map: dict[str, dict[str, Any]] = {}
    for record_id, raw_weight in aggregated.items():
        record = get_history_record(record_id, user_id=str(request.user.id))
        if not record:
            continue
        snapshot = record.get("snapshot") or {}
        return_rows = snapshot.get("return_series")
        if not isinstance(return_rows, list) or not return_rows:
            return_rows = snapshot.get("recent_rows") or []
        dates = []
        returns = []
        for row in return_rows:
            date_value = row.get("date") or row.get("timestamp")
            if not date_value:
                continue
            try:
                dates.append(pd.to_datetime(date_value))
            except Exception:
                continue
            daily_val = row.get("daily_return")
            if daily_val is None:
                daily_val = row.get("strategy_return")
            try:
                returns.append(float(daily_val or 0.0))
            except (TypeError, ValueError):
                returns.append(0.0)
        if len(dates) < 5:
            continue
        label = f"{record.get('ticker', 'Strategy')} · {record.get('engine', '')}"
        suffix = 1
        unique = label
        while unique in series_map:
            suffix += 1
            unique = f"{label} #{suffix}"
        series_map[unique] = pd.Series(returns, index=dates).sort_index().fillna(0.0)
        info_map[unique] = {"record_id": record_id, "weight": max(raw_weight, 0.0)}

    if not series_map:
        return JsonResponse({"error": _("No valid historical runs were found.")}, status=400)

    total_weight = sum(item["weight"] for item in info_map.values()) + cash_weight
    if total_weight <= 0:
        return JsonResponse({"error": _("All weights are zero.")}, status=400)

    for label in info_map:
        info_map[label]["weight"] = info_map[label]["weight"] / total_weight
    cash_norm = cash_weight / total_weight if cash_weight > 0 else 0.0

    aligned = pd.DataFrame(series_map).sort_index().fillna(0.0)
    if cash_norm > 0:
        aligned["现金"] = 0.0

    weight_lookup = {label: info["weight"] for label, info in info_map.items()}
    if cash_norm > 0:
        weight_lookup["现金"] = cash_norm
    weight_series = pd.Series({col: weight_lookup.get(col, 0.0) for col in aligned.columns})
    combined_returns = aligned.mul(weight_series, axis=1).sum(axis=1)
    cumulative = (1 + combined_returns).cumprod()
    metrics = portfolio_stats(combined_returns)

    curve_points = [
        {"time": idx.strftime("%Y-%m-%d"), "value": round(float(val), 6)}
        for idx, val in cumulative.dropna().items()
    ]

    component_curves = []
    for label, series in series_map.items():
        curve = (1 + series).cumprod()
        component_curves.append(
            {
                "label": label,
                "record_id": info_map[label]["record_id"],
                "weight": info_map[label]["weight"],
                "points": [
                    {"time": idx.strftime("%Y-%m-%d"), "value": round(float(val), 6)}
                    for idx, val in curve.dropna().items()
                ],
            }
        )

    correlation_pairs: list[dict[str, Any]] = []
    if len(series_map) >= 2:
        corr_frame = aligned[[col for col in aligned.columns if col in series_map]]
        corr_matrix = corr_frame.corr().replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        for a, b in combinations(corr_matrix.columns, 2):
            correlation_pairs.append({"a": a, "b": b, "value": float(corr_matrix.at[a, b])})

    response = {
        "weights": weight_lookup,
        "metrics": metrics,
        "curve": curve_points,
        "components": component_curves,
        "correlation": correlation_pairs,
    }
    return JsonResponse(response)


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
    strategy_input, _warnings = build_strategy_input(form.cleaned_data, request_id=request_id, user=request.user)
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
