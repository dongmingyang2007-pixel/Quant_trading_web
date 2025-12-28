from __future__ import annotations

from datetime import date

import pandas as pd
import pandas.testing as pdt

from trading.strategies import core


def _make_prices() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=8, freq="B")
    prices = pd.DataFrame(
        {
            "adj close": [100.0 + i for i in range(len(idx))],
            "close": [100.0 + i for i in range(len(idx))],
            "open": [100.5 + i for i in range(len(idx))],
            "volume": [1_000_000.0] * len(idx),
        },
        index=idx,
    )
    return prices


def test_apply_listing_window_trims_to_ipo_and_delist(monkeypatch):
    listing = pd.DataFrame(
        {
            "symbol": ["TEST"],
            "ipodate": ["2024-01-03"],
            "delistingdate": ["2024-01-07"],
            "status": ["Delisted"],
        }
    )
    monkeypatch.setattr(core, "_load_listing_status", lambda: listing)

    warnings: list[str] = []
    data = _make_prices()
    trimmed, eff_start, eff_end = core._apply_listing_window(
        data,
        "TEST",
        date(2024, 1, 1),
        date(2024, 1, 10),
        warnings,
    )

    assert eff_start == date(2024, 1, 3)
    assert eff_end == date(2024, 1, 7)
    assert trimmed.index.min().date() == date(2024, 1, 3)
    assert trimmed.index.max().date() <= date(2024, 1, 7)
    assert len(warnings) >= 2


def test_apply_listing_window_warns_when_missing(monkeypatch):
    listing = pd.DataFrame({"symbol": ["OTHER"]})
    monkeypatch.setattr(core, "_load_listing_status", lambda: listing)

    warnings: list[str] = []
    data = _make_prices()
    trimmed, eff_start, eff_end = core._apply_listing_window(
        data,
        "TEST",
        date(2024, 1, 1),
        date(2024, 1, 10),
        warnings,
    )

    pdt.assert_frame_equal(trimmed, data)
    assert eff_start == date(2024, 1, 1)
    assert eff_end == date(2024, 1, 10)
    assert any("TEST" in item for item in warnings)
