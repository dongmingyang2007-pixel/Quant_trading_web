from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .signals import Signal, SignalAction


@dataclass(slots=True)
class StrategyContext:
    symbol: str
    positions: dict[str, float] = field(default_factory=dict)
    account_equity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy:
    def __init__(self, name: str, *, universe: list[str] | None = None) -> None:
        self.name = name
        self.universe = universe or []
        self.state: dict[str, Any] = {}

    def generate_signal(self, market_data: dict[str, Any], context: StrategyContext) -> Signal:
        """Compute trading signal given latest market data."""
        raise NotImplementedError

    def on_market_data(self, market_data: dict[str, Any], context: StrategyContext) -> Signal:
        """Default hook for incoming market data."""
        return self.generate_signal(market_data, context)

    def _build_hold(self, symbol: str) -> Signal:
        return Signal(symbol=symbol, action=SignalAction.HOLD, weight=0.0)
