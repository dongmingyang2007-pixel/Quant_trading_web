from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.utils.translation import gettext as _
from django.views.decorators.http import require_POST

from ..error_contract import json_error
from ..observability import ensure_request_id

_MIGRATED_TARGET = "/market/?view=chart&tool=wave"


def _migrated_error(request: HttpRequest, endpoint: str) -> JsonResponse:
    request_id = ensure_request_id(request)
    return json_error(
        error_code="screen_analyzer_migrated",
        message=_("屏幕波型已迁移到股市信息图表分析。"),
        status_code=410,
        request_id=request_id,
        user_id=getattr(getattr(request, "user", None), "id", None),
        endpoint=endpoint,
        extra={"redirect_to": _MIGRATED_TARGET},
    )


@login_required
def screen_analyzer(request: HttpRequest) -> HttpResponse:
    return redirect(_MIGRATED_TARGET)


@login_required
@require_POST
def screen_analyzer_api(request: HttpRequest) -> JsonResponse:
    return _migrated_error(request, "api.screen.analyze")


@login_required
@require_POST
def screen_analyzer_sample_api(request: HttpRequest) -> JsonResponse:
    return _migrated_error(request, "api.screen.sample")


@login_required
@require_POST
def screen_analyzer_train_api(request: HttpRequest) -> JsonResponse:
    return _migrated_error(request, "api.screen.train")


__all__ = [
    "screen_analyzer",
    "screen_analyzer_api",
    "screen_analyzer_sample_api",
    "screen_analyzer_train_api",
]

