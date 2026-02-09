from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .signals import CombinedSignal, Signal, SignalAction


@dataclass(slots=True)
class StrategyStats:
    sharpe: float | None = None
    ic: float | None = None
    weight: float | None = None


class StrategyCombiner:
    def __init__(
        self,
        *,
        method: str = "weighted_avg",
        weights: dict[str, float] | None = None,
        stats: dict[str, StrategyStats] | None = None,
    ) -> None:
        self.method = method
        self.weights = weights or {}
        self.stats = stats or {}

    def combine_signals(
        self,
        signals: list[Signal],
        *,
        timestamp=None,
    ) -> list[CombinedSignal]:
        grouped: dict[str, list[Signal]] = defaultdict(list)
        for sig in signals:
            if sig and sig.symbol:
                grouped[str(sig.symbol).upper()].append(sig)
        combined: list[CombinedSignal] = []
        for symbol, bucket in grouped.items():
            combined.append(self._combine_symbol(symbol, bucket, timestamp=timestamp))
        return combined

    def _combine_symbol(
        self,
        symbol: str,
        bucket: list[Signal],
        *,
        timestamp=None,
    ) -> CombinedSignal:
        method = (self.method or "weighted_avg").lower()
        if method in {"vote", "voting"}:
            weight, confidence = self._vote(bucket)
        else:
            weight, confidence = self._weighted_avg(bucket, method=method)
        action = SignalAction.HOLD
        if weight > 0:
            action = SignalAction.BUY
        elif weight < 0:
            action = SignalAction.SELL
        return CombinedSignal(
            symbol=symbol,
            weight=float(weight),
            action=action,
            confidence=confidence,
            timestamp=timestamp,
            components=bucket,
        )

    def _weighted_avg(self, bucket: list[Signal], *, method: str) -> tuple[float, float | None]:
        weights: list[float] = []
        raw: list[float] = []
        confidences: list[float] = []
        for sig in bucket:
            name = (sig.metadata or {}).get("strategy") or ""
            weight = self.weights.get(name)
            if weight is None:
                stats = self.stats.get(name)
                if stats:
                    if method in {"sharpe", "sharpe_weight"} and stats.sharpe is not None:
                        weight = stats.sharpe
                    elif method in {"ic", "ic_weight"} and stats.ic is not None:
                        weight = stats.ic
                    elif stats.weight is not None:
                        weight = stats.weight
            if weight is None:
                weight = 1.0
            weights.append(float(weight))
            raw.append(float(sig.weight))
            if sig.confidence is not None:
                confidences.append(float(sig.confidence))
        if not raw:
            return 0.0, None
        total_weight = sum(abs(w) for w in weights) or 1.0
        combined = sum(r * w for r, w in zip(raw, weights)) / total_weight
        confidence = sum(confidences) / len(confidences) if confidences else None
        return combined, confidence

    def _vote(self, bucket: list[Signal]) -> tuple[float, float | None]:
        votes = 0.0
        total = 0.0
        confidences: list[float] = []
        for sig in bucket:
            name = (sig.metadata or {}).get("strategy") or ""
            weight = self.weights.get(name, 1.0)
            direction = sig.direction()
            votes += direction * weight
            total += abs(weight)
            if sig.confidence is not None:
                confidences.append(float(sig.confidence))
        if total == 0:
            return 0.0, None
        combined = votes / total
        confidence = sum(confidences) / len(confidences) if confidences else None
        return combined, confidence
