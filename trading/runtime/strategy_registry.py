from __future__ import annotations

from collections import deque
from typing import Any, Iterable

from .signals import Signal, SignalAction
from .strategy import BaseStrategy, StrategyContext


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class MomentumStrategy(BaseStrategy):
    def __init__(
        self,
        name: str = "momentum",
        *,
        lookback_bars: int = 3,
        entry_threshold: float = 0.002,
        exit_threshold: float = 0.001,
        max_weight: float = 1.0,
        universe: list[str] | None = None,
    ) -> None:
        super().__init__(name=name, universe=universe)
        self.lookback_bars = max(2, int(lookback_bars))
        self.entry_threshold = max(0.0, float(entry_threshold))
        self.exit_threshold = max(0.0, float(exit_threshold))
        self.max_weight = max(0.0, float(max_weight))
        self._history: dict[str, deque[float]] = {}
        self._positions: dict[str, float] = {}

    def generate_signal(self, market_data: dict[str, Any], context: StrategyContext) -> Signal:
        symbol = (market_data.get("symbol") or context.symbol or "").upper()
        close = market_data.get("close")
        if close is None:
            close = market_data.get("price")
        if not symbol or close is None:
            return self._build_hold(symbol)
        try:
            close_val = float(close)
        except (TypeError, ValueError):
            return self._build_hold(symbol)
        history = self._history.setdefault(symbol, deque(maxlen=self.lookback_bars))
        history.append(close_val)
        if len(history) < self.lookback_bars:
            return self._build_hold(symbol)
        base = history[0]
        if not base:
            return self._build_hold(symbol)
        momentum = (history[-1] / base) - 1.0
        previous = self._positions.get(symbol, 0.0)
        weight = 0.0
        if abs(momentum) >= self.entry_threshold:
            weight = self.max_weight if momentum > 0 else -self.max_weight
        elif previous and abs(momentum) >= self.exit_threshold:
            weight = previous
        self._positions[symbol] = weight
        action = SignalAction.HOLD
        if weight > 0:
            action = SignalAction.BUY
        elif weight < 0:
            action = SignalAction.SELL
        confidence = None
        if self.entry_threshold > 0:
            confidence = _clamp(abs(momentum) / self.entry_threshold, 0.0, 1.0)
        return Signal(
            symbol=symbol,
            action=action,
            weight=weight,
            confidence=confidence,
            metadata={"momentum": momentum},
        )


class RSIMeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        name: str = "mean_reversion",
        *,
        rsi_period: int = 14,
        upper: float = 70.0,
        lower: float = 30.0,
        max_weight: float = 1.0,
        universe: list[str] | None = None,
    ) -> None:
        super().__init__(name=name, universe=universe)
        self.rsi_period = max(2, int(rsi_period))
        self.upper = float(upper)
        self.lower = float(lower)
        self.max_weight = max(0.0, float(max_weight))
        self._history: dict[str, deque[float]] = {}

    def _compute_rsi(self, values: Iterable[float]) -> float | None:
        closes = list(values)
        if len(closes) <= self.rsi_period:
            return None
        gains = 0.0
        losses = 0.0
        for prev, curr in zip(closes[-self.rsi_period - 1 : -1], closes[-self.rsi_period :]):
            delta = curr - prev
            if delta >= 0:
                gains += delta
            else:
                losses -= delta
        if gains == 0 and losses == 0:
            return 50.0
        if losses == 0:
            return 100.0
        rs = gains / losses
        return 100.0 - (100.0 / (1.0 + rs))

    def generate_signal(self, market_data: dict[str, Any], context: StrategyContext) -> Signal:
        symbol = (market_data.get("symbol") or context.symbol or "").upper()
        close = market_data.get("close")
        if close is None:
            close = market_data.get("price")
        if not symbol or close is None:
            return self._build_hold(symbol)
        try:
            close_val = float(close)
        except (TypeError, ValueError):
            return self._build_hold(symbol)
        history = self._history.setdefault(symbol, deque(maxlen=self.rsi_period + 1))
        history.append(close_val)
        rsi = self._compute_rsi(history)
        if rsi is None:
            return self._build_hold(symbol)
        weight = 0.0
        if rsi >= self.upper:
            weight = -self.max_weight
        elif rsi <= self.lower:
            weight = self.max_weight
        action = SignalAction.HOLD
        if weight > 0:
            action = SignalAction.BUY
        elif weight < 0:
            action = SignalAction.SELL
        confidence = None
        if weight:
            anchor = self.upper if weight < 0 else self.lower
            confidence = _clamp(abs(rsi - anchor) / 50.0, 0.0, 1.0)
        return Signal(
            symbol=symbol,
            action=action,
            weight=weight,
            confidence=confidence,
            metadata={"rsi": rsi},
        )


class SMACrossStrategy(BaseStrategy):
    def __init__(
        self,
        name: str = "sma_cross",
        *,
        short_window: int = 10,
        long_window: int = 30,
        max_weight: float = 1.0,
        universe: list[str] | None = None,
    ) -> None:
        super().__init__(name=name, universe=universe)
        self.short_window = max(2, int(short_window))
        self.long_window = max(self.short_window + 1, int(long_window))
        self.max_weight = max(0.0, float(max_weight))
        self._history: dict[str, deque[float]] = {}

    def generate_signal(self, market_data: dict[str, Any], context: StrategyContext) -> Signal:
        symbol = (market_data.get("symbol") or context.symbol or "").upper()
        close = market_data.get("close")
        if close is None:
            close = market_data.get("price")
        if not symbol or close is None:
            return self._build_hold(symbol)
        try:
            close_val = float(close)
        except (TypeError, ValueError):
            return self._build_hold(symbol)
        history = self._history.setdefault(symbol, deque(maxlen=self.long_window))
        history.append(close_val)
        if len(history) < self.long_window:
            return self._build_hold(symbol)
        short_ma = sum(list(history)[-self.short_window :]) / self.short_window
        long_ma = sum(history) / self.long_window
        weight = 0.0
        if short_ma > long_ma:
            weight = self.max_weight
        elif short_ma < long_ma:
            weight = -self.max_weight
        action = SignalAction.HOLD
        if weight > 0:
            action = SignalAction.BUY
        elif weight < 0:
            action = SignalAction.SELL
        confidence = _clamp(abs(short_ma - long_ma) / max(long_ma, 1e-6), 0.0, 1.0)
        return Signal(
            symbol=symbol,
            action=action,
            weight=weight,
            confidence=confidence,
            metadata={"short_ma": short_ma, "long_ma": long_ma},
        )


STRATEGY_REGISTRY = {
    "momentum": MomentumStrategy,
    "mean_reversion": RSIMeanReversionStrategy,
    "rsi": RSIMeanReversionStrategy,
    "sma": SMACrossStrategy,
    "sma_cross": SMACrossStrategy,
    "ma_cross": SMACrossStrategy,
}


def build_strategies(specs: Iterable[Any], *, universe: list[str] | None = None) -> list[BaseStrategy]:
    strategies: list[BaseStrategy] = []
    for spec in specs:
        if isinstance(spec, dict):
            raw = spec
        else:
            raw = {
                "name": getattr(spec, "name", ""),
                "enabled": getattr(spec, "enabled", True),
                "params": getattr(spec, "params", {}),
            }
            if hasattr(spec, "weight"):
                raw["weight"] = getattr(spec, "weight")
        name = str(raw.get("name") or raw.get("strategy") or "").strip().lower()
        if not name:
            continue
        enabled = bool(raw.get("enabled", True))
        if not enabled:
            continue
        params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        strategy_cls = STRATEGY_REGISTRY.get(name)
        if not strategy_cls:
            continue
        strategies.append(strategy_cls(name=name, universe=universe, **params))
    return strategies
