from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SignalAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(slots=True)
class Signal:
    symbol: str
    action: SignalAction | str
    weight: float
    confidence: float | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def direction(self) -> float:
        if self.action in {SignalAction.BUY, "buy"}:
            return 1.0
        if self.action in {SignalAction.SELL, "sell"}:
            return -1.0
        if self.weight < 0:
            return -1.0
        if self.weight > 0:
            return 1.0
        return 0.0


@dataclass(slots=True)
class CombinedSignal:
    symbol: str
    weight: float
    action: SignalAction
    confidence: float | None = None
    timestamp: datetime | None = None
    components: list[Signal] = field(default_factory=list)
