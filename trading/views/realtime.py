from __future__ import annotations

import json
import time
from datetime import datetime, timezone as dt_timezone
from typing import Any

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from ..models import RealtimeProfile
from ..realtime.manual_orders import submit_manual_order
from ..realtime.config import DEFAULT_CONFIG_NAME, load_realtime_config_from_payload
from ..realtime.schema import RealtimePayloadError, validate_realtime_payload
from ..realtime.storage import read_state, write_state


def _parse_unix_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    dt = datetime.fromtimestamp(ts, tz=dt_timezone.utc)
    return timezone.localtime(dt)


def _parse_iso_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if timezone.is_naive(parsed):
        parsed = parsed.replace(tzinfo=dt_timezone.utc)
    return timezone.localtime(parsed)


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


@login_required
@require_http_methods(["GET", "POST"])
def realtime_settings(request):
    success_message = None
    error_message = None

    profiles = list(RealtimeProfile.objects.filter(user=request.user).order_by("-updated_at"))
    active_profile = next((profile for profile in profiles if profile.is_active), None)

    selected_profile = None
    requested_profile_id = (request.GET.get("profile") or "").strip()
    if requested_profile_id and requested_profile_id != "new":
        selected_profile = next(
            (profile for profile in profiles if str(profile.profile_id) == requested_profile_id),
            None,
        )
    if selected_profile is None:
        selected_profile = active_profile or (profiles[0] if profiles else None)

    payload_text = None
    if request.method == "POST":
        action = (request.POST.get("action") or "").strip()
        if action in {"save", "activate", "delete"}:
            profile_id = (request.POST.get("profile_id") or "").strip()
            if action == "save":
                name = (request.POST.get("name") or "").strip() or "Realtime Profile"
                description = (request.POST.get("description") or "").strip()
                raw_payload = (request.POST.get("payload") or "").strip()
                is_active = bool(request.POST.get("is_active"))
                payload_text = raw_payload
                payload: dict[str, Any] = {}
                if raw_payload:
                    try:
                        payload = json.loads(raw_payload)
                    except json.JSONDecodeError:
                        error_message = "Invalid JSON payload."
                    else:
                        if not isinstance(payload, dict):
                            error_message = "Payload must be a JSON object."
                if error_message is None:
                    try:
                        normalized = validate_realtime_payload(payload)
                    except RealtimePayloadError as exc:
                        error_message = str(exc)
                    target = None
                    if profile_id:
                        target = RealtimeProfile.objects.filter(user=request.user, profile_id=profile_id).first()
                        if not target:
                            error_message = "Profile not found."
                    if error_message is None:
                        if target:
                            target.name = name
                            target.description = description
                            target.payload = normalized
                            target.is_active = is_active
                            target.save(update_fields=["name", "description", "payload", "is_active", "updated_at"])
                        else:
                            target = RealtimeProfile.objects.create(
                                user=request.user,
                                name=name,
                                description=description,
                                payload=normalized,
                                is_active=is_active,
                            )
                        if is_active:
                            RealtimeProfile.objects.filter(user=request.user).exclude(
                                profile_id=target.profile_id
                            ).update(is_active=False)
                            write_state(DEFAULT_CONFIG_NAME, normalized)
                        success_message = "Realtime profile saved."
                        selected_profile = target
            elif action == "activate":
                target = RealtimeProfile.objects.filter(user=request.user, profile_id=profile_id).first()
                if not target:
                    error_message = "Profile not found."
                else:
                    RealtimeProfile.objects.filter(user=request.user).update(is_active=False)
                    target.is_active = True
                    target.save(update_fields=["is_active", "updated_at"])
                    if isinstance(target.payload, dict):
                        write_state(DEFAULT_CONFIG_NAME, target.payload)
                    selected_profile = target
                    success_message = "Realtime profile activated."
            elif action == "delete":
                target = RealtimeProfile.objects.filter(user=request.user, profile_id=profile_id).first()
                if not target:
                    error_message = "Profile not found."
                else:
                    target.delete()
                    success_message = "Realtime profile deleted."
                    selected_profile = None

        profiles = list(RealtimeProfile.objects.filter(user=request.user).order_by("-updated_at"))
        active_profile = next((profile for profile in profiles if profile.is_active), None)
        if selected_profile and selected_profile not in profiles:
            selected_profile = active_profile or (profiles[0] if profiles else None)

    if payload_text is None:
        if selected_profile:
            payload_text = json.dumps(selected_profile.payload or {}, indent=2, sort_keys=True)
        else:
            payload_text = json.dumps(validate_realtime_payload({}), indent=2, sort_keys=True)

    context = {
        "profiles": profiles,
        "active_profile": active_profile,
        "selected_profile": selected_profile,
        "payload_text": payload_text,
        "success_message": success_message,
        "error_message": error_message,
    }
    return render(request, "trading/realtime_settings.html", context)


@login_required
@require_http_methods(["GET", "POST"])
def realtime_monitor(request):
    profiles = list(RealtimeProfile.objects.filter(user=request.user).order_by("-updated_at"))
    active_profile = next((profile for profile in profiles if profile.is_active), None)

    if request.method == "POST":
        action = (request.POST.get("action") or "").strip()
        if not active_profile:
            messages.error(request, "尚未激活实时引擎配置档案。")
            return redirect("trading:realtime_monitor")

        payload = active_profile.payload if isinstance(active_profile.payload, dict) else {}
        trading_payload = payload.get("trading")
        if not isinstance(trading_payload, dict):
            trading_payload = {}
        payload["trading"] = trading_payload
        execution_payload = trading_payload.get("execution")
        if not isinstance(execution_payload, dict):
            execution_payload = {}
        trading_payload["execution"] = execution_payload

        if action in {"toggle_trading", "toggle_execution"}:
            value = _parse_bool(request.POST.get("value"))
            if action == "toggle_trading":
                trading_payload["enabled"] = value
            elif action == "toggle_execution":
                execution_payload["enabled"] = value
            try:
                normalized = validate_realtime_payload(payload)
            except RealtimePayloadError as exc:
                messages.error(request, f"配置保存失败：{exc}")
                return redirect("trading:realtime_monitor")
            active_profile.payload = normalized
            active_profile.save(update_fields=["payload", "updated_at"])
            write_state(DEFAULT_CONFIG_NAME, normalized)
            messages.success(request, "实时交易配置已更新。")
            return redirect("trading:realtime_monitor")
        if action == "set_mode":
            mode = (request.POST.get("value") or "").strip().lower()
            if mode not in {"paper", "live"}:
                messages.error(request, "交易模式无效。")
                return redirect("trading:realtime_monitor")
            trading_payload["mode"] = mode
            try:
                normalized = validate_realtime_payload(payload)
            except RealtimePayloadError as exc:
                messages.error(request, f"配置保存失败：{exc}")
                return redirect("trading:realtime_monitor")
            active_profile.payload = normalized
            active_profile.save(update_fields=["payload", "updated_at"])
            write_state(DEFAULT_CONFIG_NAME, normalized)
            messages.success(request, "交易模式已更新。")
            return redirect("trading:realtime_monitor")
        if action in {"enable_live_trading", "disable_live_trading"}:
            trading_payload["enabled"] = True
            execution_payload["enabled"] = True
            if action == "enable_live_trading":
                trading_payload["mode"] = "live"
                execution_payload["dry_run"] = False
            else:
                trading_payload["mode"] = "paper"
                execution_payload["dry_run"] = False
            try:
                normalized = validate_realtime_payload(payload)
            except RealtimePayloadError as exc:
                messages.error(request, f"配置保存失败：{exc}")
                return redirect("trading:realtime_monitor")
            active_profile.payload = normalized
            active_profile.save(update_fields=["payload", "updated_at"])
            write_state(DEFAULT_CONFIG_NAME, normalized)
            if action == "enable_live_trading":
                messages.success(request, "已启用实盘下单（Live）。")
            else:
                messages.success(request, "已退出实盘，下单将使用模拟模式。")
            return redirect("trading:realtime_monitor")

        if action == "manual_order":
            symbol = (request.POST.get("symbol") or "").strip()
            side = (request.POST.get("side") or "").strip()
            notional_raw = request.POST.get("notional")
            try:
                notional = float(notional_raw) if notional_raw is not None else 0.0
            except (TypeError, ValueError):
                notional = 0.0
            result = submit_manual_order(user=request.user, symbol=symbol, side=side, notional=notional)
            if result.get("ok"):
                messages.success(request, result.get("message") or "订单已提交。")
            else:
                messages.error(request, result.get("message") or "订单提交失败。")
            return redirect("trading:realtime_monitor")

        messages.error(request, "未知操作。")
        return redirect("trading:realtime_monitor")
    config = load_realtime_config_from_payload(active_profile.payload if active_profile else {})

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

    now = time.time()
    universe_updated = _parse_unix_ts(universe_state.get("updated_at"))
    focus_updated = _parse_unix_ts(focus_summary.get("updated_at"))
    bars_updated = _parse_unix_ts(bars_latest.get("updated_at"))
    stream_updated = _parse_unix_ts(stream_state.get("updated_at")) if isinstance(stream_state, dict) else None

    last_update_ts = max(
        [value for value in [universe_state.get("updated_at"), focus_summary.get("updated_at"), bars_latest.get("updated_at")] if value],
        default=0,
    )
    stale_after = max(30, config.engine.focus_refresh_seconds * 2)
    engine_online = bool(last_update_ts and (now - last_update_ts) <= stale_after)

    focus_entries = focus_state.get("symbols") if isinstance(focus_state, dict) else None
    if not isinstance(focus_entries, list):
        focus_entries = []
    focus_payload = []
    for entry in focus_entries:
        if not isinstance(entry, dict):
            continue
        symbol = str(entry.get("symbol") or "").upper()
        if not symbol:
            continue
        since_ts = entry.get("since_ts")
        focus_payload.append(
            {
                "symbol": symbol,
                "since": _parse_unix_ts(since_ts),
            }
        )

    ranked_entries = []
    if isinstance(universe_ranked, dict):
        raw_entries = universe_ranked.get("entries")
        if isinstance(raw_entries, list):
            for entry in raw_entries[:30]:
                if isinstance(entry, dict):
                    ranked_entries.append(entry)

    bars_rows = []
    if isinstance(bars_latest, dict):
        rows = bars_latest.get("bars")
        if isinstance(rows, list):
            bars_rows = [row for row in rows if isinstance(row, dict)]

    signals_rows = []
    if isinstance(signals_latest, dict):
        rows = signals_latest.get("signals")
        if isinstance(rows, list):
            signals_rows = [row for row in rows if isinstance(row, dict)]

    trade_signal_rows = []
    if isinstance(trade_signals_latest, dict):
        rows = trade_signals_latest.get("signals")
        if isinstance(rows, list):
            trade_signal_rows = [row for row in rows if isinstance(row, dict)]

    trade_order_rows = []
    if isinstance(trade_orders_latest, dict):
        rows = trade_orders_latest.get("orders")
        if isinstance(rows, list):
            trade_order_rows = [row for row in rows if isinstance(row, dict)]

    last_tick = None
    for row in bars_rows:
        ts = _parse_iso_ts(row.get("timestamp"))
        if ts and (last_tick is None or ts > last_tick):
            last_tick = ts

    context = {
        "active_profile": active_profile,
        "engine_online": engine_online,
        "stream_status": stream_state.get("status") if isinstance(stream_state, dict) else None,
        "stream_detail": stream_state.get("detail") if isinstance(stream_state, dict) else None,
        "stream_updated": stream_updated,
        "stale_after": stale_after,
        "universe_updated": universe_updated,
        "focus_updated": focus_updated,
        "bars_updated": bars_updated,
        "last_tick": last_tick,
        "focus_entries": focus_payload,
        "focus_count": len(focus_payload),
        "universe_count": int(universe_state.get("count") or 0) if isinstance(universe_state, dict) else 0,
        "ranked_entries": ranked_entries,
        "bars_rows": bars_rows[:12],
        "signals_rows": signals_rows,
        "trading_state": trading_state if isinstance(trading_state, dict) else {},
        "trade_signal_rows": trade_signal_rows,
        "trade_order_rows": trade_order_rows,
        "config": config,
    }
    return render(request, "trading/realtime_monitor.html", context)
