from __future__ import annotations

import json
import uuid
from typing import Any

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods, require_POST

from paper.models import PaperTradingSession

from ..history import delete_history_record, update_history_meta as update_backtest_meta


@require_POST
@login_required
def delete_history(request, record_id: str):
    success = delete_history_record(record_id, user_id=str(request.user.id))
    status_code = 200 if success else 404
    return JsonResponse({"success": success}, status=status_code)


def _parse_json_body(request) -> dict[str, Any] | None:
    if request.body:
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    if request.POST:
        return request.POST.dict()
    return {}


def _normalize_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [item.strip() for item in raw.replace("，", ",").split(",")]
    elif isinstance(raw, (list, tuple, set)):
        items = [str(item).strip() for item in raw]
    else:
        raise ValueError("invalid_tags")
    cleaned: list[str] = []
    for item in items:
        if item and item not in cleaned:
            cleaned.append(item)
    return cleaned


def _coerce_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(raw, (int, float)):
        return bool(raw)
    raise ValueError("invalid_bool")


@require_http_methods(["POST", "PATCH"])
def update_history_meta(request, record_id: str):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "auth_required"}, status=401)

    payload = _parse_json_body(request)
    if payload is None:
        return JsonResponse({"error": "invalid_json"}, status=400)

    payload_id = payload.get("id") or payload.get("record_id")
    if payload_id and str(payload_id) != str(record_id):
        return JsonResponse({"error": "id_mismatch"}, status=400)

    title = None
    if "title" in payload or "name" in payload:
        raw_title = payload.get("title") if "title" in payload else payload.get("name")
        title = "" if raw_title is None else str(raw_title)

    notes = None
    if "notes" in payload or "note" in payload:
        raw_notes = payload.get("notes") if "notes" in payload else payload.get("note")
        notes = "" if raw_notes is None else str(raw_notes)

    tags_provided = "tags" in payload or "tag" in payload
    tags = None
    if tags_provided:
        raw_tags = payload.get("tags") if "tags" in payload else payload.get("tag")
        try:
            tags = _normalize_tags(raw_tags)
        except ValueError:
            return JsonResponse({"error": "invalid_tags"}, status=400)

    starred = None
    if "starred" in payload or "is_favorite" in payload:
        raw_starred = payload.get("starred") if "starred" in payload else payload.get("is_favorite")
        try:
            starred = _coerce_bool(raw_starred)
        except ValueError:
            return JsonResponse({"error": "invalid_starred"}, status=400)

    if title is None and notes is None and not tags_provided and starred is None:
        return JsonResponse({"error": "empty_payload"}, status=400)

    updated = update_backtest_meta(
        record_id,
        user_id=str(request.user.id),
        title=title,
        tags=tags if tags_provided else None,
        notes=notes,
        starred=starred,
    )
    if updated:
        return JsonResponse({"status": "success", **updated})

    session = None
    try:
        session_uuid = uuid.UUID(str(record_id))
    except (TypeError, ValueError):
        session_uuid = None
    if session_uuid:
        session = PaperTradingSession.objects.filter(session_id=session_uuid, user=request.user).first()

    if not session:
        return JsonResponse({"error": "not_found"}, status=404)

    update_fields: list[str] = []
    if title is not None:
        session.name = title
        update_fields.append("name")

    config = session.config if isinstance(session.config, dict) else {}
    meta = config.get("meta") if isinstance(config.get("meta"), dict) else {}
    meta_updated = False
    if notes is not None:
        meta["notes"] = notes
        meta_updated = True
    if tags_provided:
        meta["tags"] = tags or []
        meta_updated = True
    if starred is not None:
        meta["starred"] = starred
        meta_updated = True
    if meta_updated:
        config = dict(config)
        config["meta"] = meta
        session.config = config
        update_fields.append("config")

    if update_fields:
        session.save(update_fields=update_fields)

    response = {"status": "success", "session_id": str(session.session_id)}
    if title is not None:
        response["name"] = session.name
    if meta_updated:
        response["meta"] = meta
    return JsonResponse(response)
