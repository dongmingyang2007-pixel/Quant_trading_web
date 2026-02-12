from __future__ import annotations

import time
from typing import Any

from django.core.management.base import BaseCommand

from ...market_provider import resolve_market_provider
from ...realtime.storage import read_state, write_state
from ...tasks import trigger_market_snapshot_refresh
from ...views import market as market_views


class Command(BaseCommand):
    help = "Repair stale/stuck market rankings state files and trigger a non-blocking rebuild."

    def add_arguments(self, parser):
        parser.add_argument(
            "--user-id",
            type=str,
            default=None,
            help="User id used to resolve provider credentials and trigger rebuild.",
        )
        parser.add_argument(
            "--provider",
            type=str,
            choices=["alpaca", "massive"],
            default=None,
            help="Optional provider override. Defaults to resolved provider for the user.",
        )
        parser.add_argument(
            "--no-trigger",
            action="store_true",
            help="Only repair state files without triggering refresh.",
        )

    def _coerce_ts(self, payload: dict[str, Any]) -> float | None:
        ts = market_views._coerce_timestamp(payload.get("updated_ts"))
        if ts is None:
            ts = market_views._coerce_timestamp(payload.get("updated_at"))
        if ts is None:
            ts = market_views._coerce_timestamp(payload.get("started_ts"))
        if ts is None:
            ts = market_views._coerce_timestamp(payload.get("started_at"))
        return ts

    def _mark_stalled(self, state_name: str, payload: dict[str, Any], now_ts: float) -> None:
        stalled_payload = dict(payload)
        stalled_payload["status"] = "stalled"
        stalled_payload["stalled"] = True
        stalled_payload["stalled_at"] = now_ts
        stalled_payload["stalled_ts"] = now_ts
        stalled_payload["updated_at"] = now_ts
        stalled_payload["updated_ts"] = now_ts
        stalled_payload.setdefault("error", "stalled_timeout")
        write_state(state_name, stalled_payload)

    def _build_state_name(self, timeframe_key: str) -> str:
        if timeframe_key == "1d":
            return market_views.SNAPSHOT_RANKINGS_BUILDING_STATE
        return market_views._snapshot_timeframe_building_state_name(timeframe_key)

    def handle(self, *args, **options):
        user_id_option = options.get("user_id")
        user_id = str(user_id_option).strip() if user_id_option is not None else None
        if user_id == "":
            user_id = None
        provider_option = options.get("provider")
        provider = str(provider_option).strip().lower() if provider_option else None
        if not provider:
            provider = resolve_market_provider(user_id=user_id)

        now_ts = time.time()
        stalled_threshold = max(60, int(getattr(market_views, "MARKET_RANKINGS_STALLED_SECONDS", 120) or 120))
        repaired_states: list[str] = []
        invalid_snapshots: list[str] = []

        timeframe_keys = ["1d", *list(getattr(market_views, "MARKET_RANKINGS_TIMEFRAME_KEYS", ()))]

        for timeframe_key in timeframe_keys:
            state_name = self._build_state_name(timeframe_key)
            payload = read_state(state_name, default={})
            if not isinstance(payload, dict) or not payload:
                continue
            source = str(payload.get("source") or "").strip().lower()
            if source and provider and source != provider:
                continue
            status = str(payload.get("status") or "").strip().lower()
            if status != "running":
                continue
            updated_ts = self._coerce_ts(payload)
            if updated_ts is None:
                continue
            if now_ts - float(updated_ts) <= stalled_threshold:
                continue
            self._mark_stalled(state_name, payload, now_ts)
            repaired_states.append(f"{timeframe_key}:{state_name}")

        for timeframe_key in timeframe_keys:
            payload = market_views._resolve_active_snapshot_payload(timeframe_key, provider=provider)
            if not isinstance(payload, dict):
                continue
            rows = payload.get("rows")
            if not isinstance(rows, list) or not rows:
                continue
            invalid_lists = []
            for list_type in market_views.LIST_TYPES:
                if not market_views._snapshot_rows_schema_valid_for_list(rows, list_type=list_type):
                    invalid_lists.append(list_type)
            if not invalid_lists:
                continue
            invalid_snapshots.append(f"{timeframe_key}:{','.join(invalid_lists)}")
            build_state_name = self._build_state_name(timeframe_key)
            write_state(
                build_state_name,
                {
                    "status": "error",
                    "source": provider,
                    "timeframe": timeframe_key,
                    "error": "invalid_snapshot_schema",
                    "updated_at": now_ts,
                    "updated_ts": now_ts,
                },
            )

        trigger_result: dict[str, Any] | None = None
        if not options.get("no_trigger"):
            trigger_result = trigger_market_snapshot_refresh(user_id=user_id, prefer_thread=True)

        self.stdout.write(
            self.style.SUCCESS(
                "Repaired market rankings state"
                f" | provider={provider}"
                f" | stalled_fixed={len(repaired_states)}"
                f" | invalid_snapshots={len(invalid_snapshots)}"
            )
        )
        if repaired_states:
            self.stdout.write("stalled entries: " + ", ".join(repaired_states))
        if invalid_snapshots:
            self.stdout.write("invalid snapshots: " + ", ".join(invalid_snapshots))
        if trigger_result is not None:
            self.stdout.write("trigger: " + str(trigger_result))
