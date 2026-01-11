from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..observability import record_metric
from .config import FocusConfig
from .storage import read_state, write_state


@dataclass(slots=True)
class FocusEntry:
    symbol: str
    since_ts: float

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "since_ts": self.since_ts}


def _load_state() -> dict[str, float]:
    payload = read_state("focus_state.json", default={})
    entries = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {}
    output: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        symbol = str(entry.get("symbol") or "").upper()
        if not symbol:
            continue
        try:
            since_ts = float(entry.get("since_ts") or 0)
        except (TypeError, ValueError):
            since_ts = 0.0
        if since_ts <= 0:
            since_ts = time.time()
        output[symbol] = since_ts
    return output


def update_focus(
    universe_symbols: list[str],
    config: FocusConfig,
) -> list[FocusEntry]:
    now = time.time()
    previous = _load_state()
    locked = [
        symbol
        for symbol, since_ts in previous.items()
        if (now - since_ts) < config.min_residence_seconds
    ]
    locked = locked[: config.size]

    remaining_slots = max(0, config.size - len(locked))
    max_new = max(0, min(config.max_churn_per_refresh, remaining_slots))

    existing_candidates = [sym for sym in previous.keys() if sym not in locked]
    keep_existing = existing_candidates[: max(0, remaining_slots - max_new)]

    new_candidates = [sym for sym in universe_symbols if sym not in locked and sym not in keep_existing]
    new_selected = new_candidates[:max_new]

    focus_symbols = locked + keep_existing + new_selected
    focus_symbols = focus_symbols[: config.size]

    entries: list[FocusEntry] = []
    for symbol in focus_symbols:
        since_ts = previous.get(symbol, now)
        entries.append(FocusEntry(symbol=symbol, since_ts=since_ts))

    write_state(
        "focus_state.json",
        {
            "updated_at": now,
            "symbols": [entry.to_dict() for entry in entries],
        },
    )
    record_metric(
        "realtime.focus.updated",
        total=len(entries),
        locked=len(locked),
        new=len(new_selected),
    )
    return entries
