from __future__ import annotations

from decimal import Decimal
from datetime import datetime

from trading.history import sanitize_snapshot


def test_sanitize_snapshot_removes_charts_and_serializes_types():
    payload = {
        "ticker": "AAPL",
        "charts": [{"figure": "base64data"}],
        "metrics": [{"label": "Return", "value": Decimal("0.12")}],
        "stats": {"generated_at": datetime(2024, 1, 1, 12, 30)},
    }
    sanitized = sanitize_snapshot(payload)
    assert "charts" not in sanitized
    assert sanitized["metrics"][0]["value"] == "0.12"
    assert sanitized["stats"]["generated_at"].startswith("2024-01-01")
