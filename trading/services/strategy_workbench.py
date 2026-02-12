from __future__ import annotations

from typing import Any

from django.utils import timezone

from ..history import load_history
from ..market_provider import resolve_market_context
from ..models import RealtimeProfile, TaskExecution
from ..realtime.config import DEFAULT_CONFIG_NAME, load_realtime_config_from_payload
from ..realtime.presets import build_retail_short_term_template, list_realtime_templates
from ..realtime.schema import RealtimePayloadError, validate_realtime_payload
from ..realtime.storage import read_state, write_state

WORKSPACE_CHOICES = {"trade", "backtest", "review"}
SHORTTERM_WORKBENCH_MIGRATION = "/api/v1/strategy/workbench/"
SHORTTERM_TRADING_MODE_MIGRATION = "/api/v1/strategy/workbench/trading-mode/"


def _parse_unix_ts(value: Any) -> float | None:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    return ts


def _safe_rows(payload: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw = payload.get(key)
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def ensure_active_realtime_profile(user) -> tuple[RealtimeProfile, dict[str, Any]]:
    profile = RealtimeProfile.objects.filter(user=user, is_active=True).order_by("-updated_at").first()
    if profile is None:
        payload = validate_realtime_payload({})
        profile = RealtimeProfile.objects.create(
            user=user,
            name="Realtime Profile",
            description="",
            payload=payload,
            is_active=True,
        )
        write_state(DEFAULT_CONFIG_NAME, payload)
        return profile, payload
    payload = profile.payload if isinstance(profile.payload, dict) else {}
    return profile, payload


def _build_trade_payload(user, *, request_id: str) -> dict[str, Any]:
    context = resolve_market_context(user=user)
    market_data_source = str(context.get("market_data_source") or "alpaca")
    execution_source = str(context.get("execution_source") or "alpaca")

    profile, payload = ensure_active_realtime_profile(user)
    config = load_realtime_config_from_payload(payload)

    universe_state = read_state("universe_state.json", default={})
    universe_ranked = read_state("universe_ranked.json", default={})
    focus_state = read_state("focus_state.json", default={})
    focus_summary = read_state("focus_summary.json", default={})
    bars_latest = read_state("bars_latest.json", default={})
    stream_state = read_state("stream_state.json", default={})
    signals_latest = read_state("signals_latest.json", default={})
    trading_state = read_state("trading_state.json", default={})
    trade_signals_latest = read_state("trade_signals_latest.json", default={})
    trade_orders_latest = read_state("trade_orders_latest.json", default={})

    now_ts = timezone.now().timestamp()
    last_update_ts = max(
        (
            ts
            for ts in (
                _parse_unix_ts(universe_state.get("updated_at") if isinstance(universe_state, dict) else None),
                _parse_unix_ts(focus_summary.get("updated_at") if isinstance(focus_summary, dict) else None),
                _parse_unix_ts(bars_latest.get("updated_at") if isinstance(bars_latest, dict) else None),
            )
            if ts is not None
        ),
        default=0.0,
    )
    stale_after = max(30, int(config.engine.focus_refresh_seconds) * 2)
    engine_online = bool(last_update_ts and (now_ts - last_update_ts) <= stale_after)

    ranked_entries = _safe_rows(universe_ranked, "entries")[:30]
    focus_entries = _safe_rows(focus_state, "symbols")[:40]
    bars_rows = _safe_rows(bars_latest, "bars")[:30]
    signals_rows = _safe_rows(signals_latest, "signals")[:30]
    trade_signal_rows = _safe_rows(trade_signals_latest, "signals")[:20]
    trade_order_rows = _safe_rows(trade_orders_latest, "orders")[:20]

    return {
        "source": market_data_source,
        "market_data_source": market_data_source,
        "execution_source": execution_source,
        "active_profile": {
            "profile_id": profile.profile_id,
            "name": profile.name,
        },
        "templates": list_realtime_templates(),
        "engine": {
            "online": engine_online,
            "stream_status": stream_state.get("status") if isinstance(stream_state, dict) else None,
            "stream_detail": stream_state.get("detail") if isinstance(stream_state, dict) else None,
            "stale_after": stale_after,
            "updated_at": last_update_ts or None,
        },
        "trading": {
            "mode": config.trading.mode,
            "trading_enabled": bool(config.trading.enabled),
            "execution_enabled": bool(config.trading.execution.enabled),
            "state": trading_state if isinstance(trading_state, dict) else {},
            "risk_guard": {
                "ok": not bool(
                    ((trading_state if isinstance(trading_state, dict) else {}).get("risk_guard") or {}).get("reason")
                ),
                "reason": ((trading_state if isinstance(trading_state, dict) else {}).get("risk_guard") or {}).get(
                    "reason", ""
                ),
            },
        },
        "summary": {
            "universe_count": int(universe_state.get("count") or 0) if isinstance(universe_state, dict) else 0,
            "focus_count": len(focus_entries),
            "bars_count": len(bars_rows),
            "signals_count": len(signals_rows),
            "orders_count": len(trade_order_rows),
        },
        "universe": ranked_entries,
        "focus": focus_entries,
        "bars": bars_rows,
        "signals": signals_rows,
        "trade_signals": trade_signal_rows,
        "orders": trade_order_rows,
        "request_id": request_id,
    }


def _build_backtest_payload(user) -> dict[str, Any]:
    user_id = str(getattr(user, "id", "") or "")
    history_runs = load_history(limit=20, user_id=user_id) if user_id else []
    latest = history_runs[0] if history_runs else {}
    latest_snapshot = latest.get("snapshot") if isinstance(latest, dict) else {}
    latest_params = latest.get("params") if isinstance(latest, dict) else {}

    latest_run = {
        "history_id": latest.get("record_id") if isinstance(latest, dict) else "",
        "ticker": latest.get("ticker") if isinstance(latest, dict) else "",
        "status": "ready" if latest else "empty",
        "timestamp": latest.get("timestamp") if isinstance(latest, dict) else "",
        "engine": latest.get("engine") if isinstance(latest, dict) else "",
        "window": {
            "start": latest.get("start_date") if isinstance(latest, dict) else "",
            "end": latest.get("end_date") if isinstance(latest, dict) else "",
        },
    }
    if isinstance(latest_snapshot, dict):
        latest_run["metrics"] = (latest_snapshot.get("metrics") or [])[:3]
    if isinstance(latest_params, dict):
        latest_run["risk_profile"] = latest_params.get("risk_profile")
        latest_run["strategy_engine"] = latest_params.get("strategy_engine")

    task_rows = (
        TaskExecution.objects.filter(user=user, kind__in=["backtest", "training", "rl"])
        .order_by("-created_at")
        .values("task_id", "kind", "state", "created_at", "updated_at")
    )[:20]
    tasks = [
        {
            "task_id": row["task_id"],
            "kind": row["kind"],
            "state": row["state"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in task_rows
    ]

    history_briefs = []
    for item in history_runs[:10]:
        if not isinstance(item, dict):
            continue
        history_briefs.append(
            {
                "history_id": item.get("record_id", ""),
                "ticker": item.get("ticker", ""),
                "engine": item.get("engine", ""),
                "period": {
                    "start": item.get("start_date", ""),
                    "end": item.get("end_date", ""),
                },
                "timestamp": item.get("timestamp", ""),
            }
        )

    return {
        "latest_run": latest_run,
        "tasks": tasks,
        "history_briefs": history_briefs,
    }


def _build_review_payload(backtest_payload: dict[str, Any]) -> dict[str, Any]:
    latest = (backtest_payload.get("history_briefs") or [{}])[0] if backtest_payload.get("history_briefs") else {}
    latest_id = latest.get("history_id")
    execution_diagnostics: list[dict[str, Any]] = []
    risk_diagnostics: list[dict[str, Any]] = []
    ai_briefs: list[dict[str, Any]] = []

    if latest_id:
        # Keep this lightweight and safe: review is a summary layer, not a data-heavy panel.
        execution_diagnostics.append({"label": "Latest Run", "value": latest.get("ticker") or "N/A"})
        execution_diagnostics.append({"label": "Engine", "value": latest.get("engine") or "N/A"})
        risk_diagnostics.append(
            {
                "label": "Window",
                "value": f"{(latest.get('period') or {}).get('start', '-')}"
                f" -> {(latest.get('period') or {}).get('end', '-')}",
            }
        )
        ai_briefs.append({"title": "Run Summary", "body": "Use latest run details in Backtest workspace for deep analysis."})

    return {
        "execution_diagnostics": execution_diagnostics,
        "risk_diagnostics": risk_diagnostics,
        "ai_briefs": ai_briefs,
    }


def trade_source_unavailable(trade_payload: dict[str, Any] | None) -> bool:
    if not isinstance(trade_payload, dict):
        return True
    engine = trade_payload.get("engine")
    summary = trade_payload.get("summary")
    if not isinstance(engine, dict) or not isinstance(summary, dict):
        return True
    if engine.get("online"):
        return False
    return int(summary.get("focus_count") or 0) == 0 and int(summary.get("signals_count") or 0) == 0


def build_strategy_workbench_payload(user, *, request_id: str, workspace: str) -> dict[str, Any]:
    workspace_key = str(workspace or "").strip().lower() or "trade"
    if workspace_key not in WORKSPACE_CHOICES:
        raise ValueError("invalid_workspace")

    trade_payload = _build_trade_payload(user, request_id=request_id)
    backtest_payload = _build_backtest_payload(user)
    review_payload = _build_review_payload(backtest_payload)
    context = resolve_market_context(user=user)
    market_data_source = str(context.get("market_data_source") or "alpaca")
    execution_source = str(context.get("execution_source") or "alpaca")

    return {
        "request_id": request_id,
        "workspace": workspace_key,
        "source": market_data_source,
        "market_data_source": market_data_source,
        "execution_source": execution_source,
        "deprecated": False,
        "trade": trade_payload,
        "backtest": backtest_payload,
        "review": review_payload,
    }


def apply_strategy_trading_mode(user, body: dict[str, Any], *, request_id: str) -> tuple[int, dict[str, Any]]:
    profile, payload = ensure_active_realtime_profile(user)
    data = body if isinstance(body, dict) else {}

    template_key = str(data.get("template_key") or "").strip().lower()
    applied_template = ""
    if template_key:
        if template_key not in {"retail_minute", "retail_second"}:
            return (
                400,
                {
                    "error_code": "invalid_template_key",
                    "message": "Invalid short-term template.",
                    "request_id": request_id,
                },
            )
        payload = build_retail_short_term_template(template_key)
        applied_template = template_key

    mode = str(data.get("mode") or "").strip().lower()
    if not mode:
        mode = str((payload.get("trading") or {}).get("mode") or "paper").strip().lower()
    if mode not in {"paper", "live"}:
        return (
            400,
            {
                "error_code": "invalid_trading_mode",
                "message": "Trading mode must be paper or live.",
                "request_id": request_id,
            },
        )

    if mode == "live":
        confirm_live = bool(data.get("confirm_live"))
        confirm_phrase = str(data.get("confirm_phrase") or "").strip().upper()
        if not confirm_live or confirm_phrase != "LIVE":
            return (
                400,
                {
                    "error_code": "live_confirmation_required",
                    "message": "Live mode requires explicit confirmation.",
                    "request_id": request_id,
                },
            )

    trading_payload = payload.get("trading")
    if not isinstance(trading_payload, dict):
        trading_payload = {}
    payload["trading"] = trading_payload
    execution_payload = trading_payload.get("execution")
    if not isinstance(execution_payload, dict):
        execution_payload = {}
    trading_payload["execution"] = execution_payload

    trading_payload["enabled"] = True
    trading_payload["mode"] = mode
    execution_payload["enabled"] = True
    execution_payload["dry_run"] = False

    try:
        normalized = validate_realtime_payload(payload)
    except RealtimePayloadError:
        return (
            400,
            {
                "error_code": "invalid_realtime_config",
                "message": "Realtime trading config validation failed.",
                "request_id": request_id,
            },
        )

    if not profile.is_active:
        RealtimeProfile.objects.filter(user=user).update(is_active=False)
        profile.is_active = True
    profile.payload = normalized
    profile.save(update_fields=["payload", "is_active", "updated_at"])
    write_state(DEFAULT_CONFIG_NAME, normalized)
    config = load_realtime_config_from_payload(normalized)

    response = {
        "ok": True,
        "mode": config.trading.mode,
        "trading_enabled": bool(config.trading.enabled),
        "execution_enabled": bool(config.trading.execution.enabled),
        "request_id": request_id,
    }
    if applied_template:
        response["applied_template"] = applied_template
    return 200, response


def describe_strategy_trading_mode(user, *, request_id: str) -> dict[str, Any]:
    context = resolve_market_context(user=user)
    market_data_source = str(context.get("market_data_source") or "alpaca")
    execution_source = str(context.get("execution_source") or "alpaca")

    profile, payload = ensure_active_realtime_profile(user)
    config = load_realtime_config_from_payload(payload)
    template_items = list_realtime_templates()
    allowed_templates = [str(item.get("key") or "") for item in template_items if isinstance(item, dict)]

    return {
        "request_id": request_id,
        "source": market_data_source,
        "market_data_source": market_data_source,
        "execution_source": execution_source,
        "deprecated": False,
        "active_profile": {
            "profile_id": profile.profile_id,
            "name": profile.name,
        },
        "mode": config.trading.mode,
        "trading_enabled": bool(config.trading.enabled),
        "execution_enabled": bool(config.trading.execution.enabled),
        "allowed_modes": ["paper", "live"],
        "templates": template_items,
        "allowed_templates": [item for item in allowed_templates if item],
        "confirmation_rules": {
            "live": {
                "confirm_live_required": True,
                "confirm_phrase": "LIVE",
            }
        },
    }
