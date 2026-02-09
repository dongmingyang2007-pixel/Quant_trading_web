from __future__ import annotations

import logging
from typing import Any

from django.http import JsonResponse

from .observability import record_metric

LOGGER = logging.getLogger(__name__)


def _normalize_message(message: str | None, fallback: str = "Request failed.") -> str:
    text = (message or "").strip()
    return text or fallback


def _build_payload(
    *,
    error_code: str,
    message: str,
    request_id: str | None = None,
    include_legacy_error: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error_code": error_code,
        "message": message,
    }
    if include_legacy_error:
        payload["error"] = message
    if request_id:
        payload["request_id"] = request_id
    if extra:
        payload.update(extra)
    return payload


def _record_sanitized_error(
    *,
    error_code: str,
    status_code: int,
    request_id: str | None = None,
    user_id: str | int | None = None,
    endpoint: str | None = None,
) -> None:
    record_metric(
        "api.error_sanitized",
        error_code=error_code,
        status_code=status_code,
        request_id=request_id,
        user_id=str(user_id) if user_id is not None else None,
        endpoint=endpoint,
    )


def log_sanitized_exception(
    *,
    context: str,
    exc: Exception,
    error_code: str,
    request_id: str | None = None,
    user_id: str | int | None = None,
    endpoint: str | None = None,
    status_code: int = 500,
) -> None:
    LOGGER.exception("%s [error_code=%s request_id=%s]", context, error_code, request_id)
    _record_sanitized_error(
        error_code=error_code,
        status_code=status_code,
        request_id=request_id,
        user_id=user_id,
        endpoint=endpoint,
    )


def json_error(
    *,
    error_code: str,
    message: str,
    status_code: int,
    request_id: str | None = None,
    user_id: str | int | None = None,
    endpoint: str | None = None,
    include_legacy_error: bool = True,
    extra: dict[str, Any] | None = None,
) -> JsonResponse:
    safe_message = _normalize_message(message)
    _record_sanitized_error(
        error_code=error_code,
        status_code=status_code,
        request_id=request_id,
        user_id=user_id,
        endpoint=endpoint,
    )
    payload = _build_payload(
        error_code=error_code,
        message=safe_message,
        request_id=request_id,
        include_legacy_error=include_legacy_error,
        extra=extra,
    )
    return JsonResponse(payload, status=status_code)


def drf_error(
    *,
    error_code: str,
    message: str,
    status_code: int,
    request_id: str | None = None,
    user_id: str | int | None = None,
    endpoint: str | None = None,
    include_legacy_error: bool = True,
    extra: dict[str, Any] | None = None,
):
    from rest_framework.response import Response

    safe_message = _normalize_message(message)
    _record_sanitized_error(
        error_code=error_code,
        status_code=status_code,
        request_id=request_id,
        user_id=user_id,
        endpoint=endpoint,
    )
    payload = _build_payload(
        error_code=error_code,
        message=safe_message,
        request_id=request_id,
        include_legacy_error=include_legacy_error,
        extra=extra,
    )
    return Response(payload, status=status_code)
