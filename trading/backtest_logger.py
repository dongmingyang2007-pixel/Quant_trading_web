from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
import json

from django.conf import settings

LOG_DIR = settings.DATA_CACHE_DIR / "reports"
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class BacktestLogEntry:
    record_id: str
    timestamp: str
    ticker: str
    engine: str
    sharpe: float
    total_return: float
    max_drawdown: float
    validation_sharpe: float | None
    execution_cost: float
    notes: List[str]
    request_id: str | None = None
    user_id: str | None = None
    model_version: str | None = None
    data_version: str | None = None
    latency_ms: float | None = None
    seeds: dict[str, object] | None = None
    versions: dict[str, object] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _log_file_for(ticker: str) -> Path:
    safe = ticker.replace("/", "_")
    return LOG_DIR / f"{safe}_benchmarks.json"


def append_log(entry: BacktestLogEntry) -> None:
    path = _log_file_for(entry.ticker)
    history: List[Dict[str, Any]] = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            history = []
    payload = entry.to_dict()
    payload = {k: v for k, v in payload.items() if v not in (None, [])}
    if entry.notes:
        payload["notes"] = entry.notes
    history.append(payload)
    history = sorted(history, key=lambda x: x["timestamp"], reverse=True)[:200]
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def top_runs(ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
    path = _log_file_for(ticker)
    if not path.exists():
        return []
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return sorted(history, key=lambda x: (x.get("sharpe", 0.0), x.get("total_return", 0.0)), reverse=True)[:limit]
