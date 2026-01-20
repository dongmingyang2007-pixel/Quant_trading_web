from __future__ import annotations

import time
from typing import Iterable

from ..file_utils import update_json_file
from .storage import read_state, state_path

EXTRA_SYMBOLS_STATE = "market_subscriptions.json"
EXTRA_SYMBOLS_MAX = 50


def _clean_symbols(symbols: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        symbol = str(raw or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        cleaned.append(symbol)
        if len(cleaned) >= EXTRA_SYMBOLS_MAX:
            break
    return cleaned


def read_subscription_state() -> tuple[list[str], float]:
    payload = read_state(EXTRA_SYMBOLS_STATE, default={})
    symbols = _clean_symbols(payload.get("symbols") or [])
    try:
        updated_at = float(payload.get("updated_at") or 0.0)
    except (TypeError, ValueError):
        updated_at = 0.0
    return symbols, updated_at


def update_subscription_state(symbols: Iterable[str], *, replace: bool = False) -> list[str]:
    cleaned = _clean_symbols(symbols)
    if not cleaned:
        return read_subscription_state()[0]

    def _update(current: dict) -> dict:
        existing = _clean_symbols(current.get("symbols") or [])
        if replace:
            merged = cleaned
        else:
            merged: list[str] = []
            seen: set[str] = set()
            for sym in cleaned + existing:
                if sym in seen:
                    continue
                seen.add(sym)
                merged.append(sym)
                if len(merged) >= EXTRA_SYMBOLS_MAX:
                    break
        return {"updated_at": time.time(), "symbols": merged}

    updated = update_json_file(
        state_path(EXTRA_SYMBOLS_STATE),
        default={"updated_at": 0.0, "symbols": []},
        update_fn=_update,
    )
    return _clean_symbols(updated.get("symbols") or [])
