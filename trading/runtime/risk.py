from __future__ import annotations

from dataclasses import dataclass

from .signals import CombinedSignal


@dataclass(slots=True)
class RiskLimits:
    max_position_weight: float = 0.3
    max_leverage: float = 1.5
    min_confidence: float | None = None


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def apply_risk_controls(self, signal: CombinedSignal) -> CombinedSignal:
        if signal is None:
            return signal
        max_w = abs(self.limits.max_position_weight)
        if max_w > 0 and abs(signal.weight) > max_w:
            signal.weight = max_w if signal.weight > 0 else -max_w
        if self.limits.min_confidence is not None and signal.confidence is not None:
            if signal.confidence < self.limits.min_confidence:
                signal.weight = 0.0
        if abs(signal.weight) == 0:
            signal.action = signal.action.HOLD
        return signal
