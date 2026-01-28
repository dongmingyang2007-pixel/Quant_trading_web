from trading.market_aggregation import (
    aggregate_trades_to_tick_bars,
    aggregate_trades_to_time_bars,
)


def test_aggregate_trades_to_tick_bars() -> None:
    trades = [
        {"t": 1000.0, "p": 10, "s": 1},
        {"t": 1000.5, "p": 12, "s": 2},
        {"t": 1001.0, "p": 11, "s": 1},
        {"t": 1001.1, "p": 9, "s": 1},
    ]
    bars = aggregate_trades_to_tick_bars(trades, ticks_per_bar=2, max_bars=10)
    assert len(bars) == 2
    first = bars[0]
    assert first["open"] == 10
    assert first["high"] == 12
    assert first["low"] == 10
    assert first["close"] == 12
    assert first["volume"] == 3
    assert first["trade_count"] == 2
    second = bars[1]
    assert second["open"] == 11
    assert second["high"] == 11
    assert second["low"] == 9
    assert second["close"] == 9
    assert second["volume"] == 2
    assert second["trade_count"] == 2
    assert second["time"] > first["time"]


def test_aggregate_trades_to_time_bars() -> None:
    trades = [
        {"t": 0.0, "p": 10, "s": 1},
        {"t": 30.0, "p": 12, "s": 2},
        {"t": 70.0, "p": 11, "s": 1},
        {"t": 119.0, "p": 13, "s": 1},
    ]
    bars = aggregate_trades_to_time_bars(trades, interval_seconds=60, max_bars=10)
    assert len(bars) == 2
    first = bars[0]
    assert first["open"] == 10
    assert first["high"] == 12
    assert first["low"] == 10
    assert first["close"] == 12
    assert first["volume"] == 3
    assert first["trade_count"] == 2
    second = bars[1]
    assert second["open"] == 11
    assert second["high"] == 13
    assert second["low"] == 11
    assert second["close"] == 13
    assert second["volume"] == 2
    assert second["trade_count"] == 2
    assert second["time"] > first["time"]
