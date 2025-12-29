from __future__ import annotations

import json

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from ..observability import ensure_request_id, record_metric, track_latency
from ..rate_limit import check_rate_limit, rate_limit_key
from ..screen_patterns import analyze_screen_frame, PATTERN_KEYS
from ..screen_training import save_sample, train_model, load_samples


@login_required
@ensure_csrf_cookie
def screen_analyzer(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "trading/screen_analyzer.html",
        {
            "analyzer_api": "trading:screen_analyzer_api",
            "analysis_interval": getattr(settings, "SCREEN_ANALYZER_INTERVAL_MS", 1200),
            "ocr_enabled": getattr(settings, "SCREEN_ANALYZER_OCR_ENABLED", True),
        },
    )


@login_required
@require_POST
def screen_analyzer_api(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    rate_state = check_rate_limit(
        cache_alias=getattr(settings, "SCREEN_ANALYZER_RATE_CACHE_ALIAS", "default"),
        key=f"screen-analyzer:{rate_limit_key(request)}",
        window=getattr(settings, "SCREEN_ANALYZER_RATE_WINDOW_SECONDS", 20),
        max_calls=getattr(settings, "SCREEN_ANALYZER_RATE_MAX_CALLS", 30),
    )
    if rate_state.limited:
        return JsonResponse(
            {
                "error": "rate_limited",
                "retry_after_seconds": rate_state.retry_after,
                "request_id": request_id,
            },
            status=429,
        )

    max_body = int(getattr(settings, "SCREEN_ANALYZER_MAX_REQUEST_BYTES", 1_500_000))
    if request.body and len(request.body) > max_body:
        return JsonResponse(
            {"error": "payload_too_large", "request_id": request_id},
            status=413,
        )

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse(
            {"error": "invalid_payload", "request_id": request_id},
            status=400,
        )
    if not isinstance(payload, dict):
        payload = {}
    image_data = payload.get("image")
    header_image = payload.get("header_image")
    mode = payload.get("mode") or "auto"
    calibration = payload.get("calibration") if isinstance(payload.get("calibration"), dict) else None
    ocr_enabled = payload.get("ocr_enabled", True)
    ocr_enabled = bool(ocr_enabled) and getattr(settings, "SCREEN_ANALYZER_OCR_ENABLED", True)
    include_waves = payload.get("include_waves", getattr(settings, "SCREEN_ANALYZER_INCLUDE_WAVES", True))
    include_fusion = payload.get("include_fusion", getattr(settings, "SCREEN_ANALYZER_INCLUDE_FUSION", True))
    include_timings = payload.get("include_timings", getattr(settings, "SCREEN_ANALYZER_INCLUDE_TIMINGS", False))
    overlay_layers = payload.get("overlay_layers")
    session_id = payload.get("session_id")
    if not image_data:
        return JsonResponse(
            {"error": "missing_image", "request_id": request_id},
            status=400,
        )

    try:
        with track_latency("screen_analyzer.frame", request_id=request_id, user_id=request.user.id):
            result = analyze_screen_frame(
                str(image_data),
                mode=str(mode),
                calibration=calibration,
                header_image=str(header_image) if header_image else None,
                enable_ocr=bool(ocr_enabled),
                include_waves=bool(include_waves),
                include_fusion=bool(include_fusion),
                include_timings=bool(include_timings),
                overlay_layers=overlay_layers if isinstance(overlay_layers, list) else None,
                session_id=str(session_id) if session_id else None,
            )
    except Exception as exc:  # pragma: no cover - defensive
        record_metric(
            "screen_analyzer.error",
            request_id=request_id,
            user_id=request.user.id,
            error=str(exc),
        )
        return JsonResponse(
            {"error": "analysis_failed", "request_id": request_id},
            status=500,
        )

    if "error" in result:
        response = {"error": result["error"], "request_id": request_id}
        if result.get("next_action"):
            response["next_action"] = result["next_action"]
        if result.get("diagnostics"):
            response["diagnostics"] = result["diagnostics"]
        if result.get("timings_ms"):
            response["timings_ms"] = result["timings_ms"]
        return JsonResponse(response, status=400)

    response = {**result, "request_id": request_id}
    record_metric(
        "screen_analyzer.response",
        request_id=request_id,
        user_id=request.user.id,
        pattern=response.get("pattern_key"),
        confidence=response.get("confidence"),
    )
    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})


@login_required
@require_POST
def screen_analyzer_sample_api(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": "invalid_payload", "request_id": request_id}, status=400)
    if not isinstance(payload, dict):
        payload = {}
    image_data = payload.get("image")
    label = payload.get("label")
    mode = payload.get("mode") or "auto"
    calibration = payload.get("calibration") if isinstance(payload.get("calibration"), dict) else None
    if not image_data or not label:
        return JsonResponse({"error": "missing_fields", "request_id": request_id}, status=400)
    if label not in PATTERN_KEYS:
        return JsonResponse({"error": "invalid_label", "request_id": request_id}, status=400)

    try:
        analysis = analyze_screen_frame(
            str(image_data),
            mode=str(mode),
            calibration=calibration,
            include_features=True,
            enable_ocr=False,
            include_waves=False,
            include_fusion=False,
            include_timings=False,
        )
    except Exception as exc:
        record_metric(
            "screen_analyzer.sample_error",
            request_id=request_id,
            user_id=request.user.id,
            error=str(exc),
        )
        return JsonResponse({"error": "analysis_failed", "request_id": request_id}, status=500)

    if analysis.get("error"):
        return JsonResponse({"error": analysis.get("error"), "request_id": request_id}, status=400)
    features = analysis.get("features")
    if not isinstance(features, list) or not features:
        return JsonResponse({"error": "feature_extract_failed", "request_id": request_id}, status=400)
    try:
        save_sample(features, label=str(label), meta={"mode": mode, "user": request.user.id})
    except Exception as exc:
        return JsonResponse({"error": str(exc), "request_id": request_id}, status=400)

    total = len(load_samples())
    return JsonResponse(
        {"status": "saved", "total_samples": total, "request_id": request_id},
        json_dumps_params={"ensure_ascii": False},
    )


@login_required
@require_POST
def screen_analyzer_train_api(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    try:
        min_samples = int(getattr(settings, "SCREEN_ANALYZER_TRAIN_MIN_SAMPLES", 18))
        metrics = train_model(min_samples=min_samples)
    except RuntimeError as exc:
        return JsonResponse({"error": str(exc), "request_id": request_id}, status=400)
    except Exception as exc:
        record_metric(
            "screen_analyzer.train_error",
            request_id=request_id,
            user_id=request.user.id,
            error=str(exc),
        )
        return JsonResponse({"error": "train_failed", "request_id": request_id}, status=500)
    response = {
        "status": "trained",
        "total_samples": metrics.total_samples,
        "classes": metrics.classes,
        "accuracy": metrics.accuracy,
        "test_size": metrics.test_size,
        "request_id": request_id,
    }
    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})


__all__ = [
    "screen_analyzer",
    "screen_analyzer_api",
    "screen_analyzer_sample_api",
    "screen_analyzer_train_api",
]
