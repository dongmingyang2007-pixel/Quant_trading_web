from __future__ import annotations

from django.conf import settings
from django.http import Http404, HttpRequest, JsonResponse


def market_auth_debug(request: HttpRequest) -> JsonResponse:
    user = getattr(request, "user", None)
    if not settings.DEBUG or not getattr(user, "is_authenticated", False) or not getattr(user, "is_staff", False):
        raise Http404("Not found")

    is_authenticated = bool(getattr(user, "is_authenticated", False))
    session = getattr(request, "session", None)
    session_key = getattr(session, "session_key", None) if session is not None else None
    payload = {
        "is_authenticated": is_authenticated,
        "user_id": getattr(user, "id", None) if is_authenticated else None,
        "session_key_present": bool(session_key),
        "has_session_cookie": "sessionid" in request.COOKIES,
        "cookie_names": sorted(request.COOKIES.keys()),
        "secure_cookie_required": bool(getattr(settings, "SESSION_COOKIE_SECURE", False)),
        "scheme": request.scheme,
        "host": request.get_host(),
        "path": request.path,
        "referer": request.META.get("HTTP_REFERER", ""),
        "origin": request.META.get("HTTP_ORIGIN", ""),
    }
    return JsonResponse(payload, json_dumps_params={"ensure_ascii": False})


__all__ = ["market_auth_debug"]
