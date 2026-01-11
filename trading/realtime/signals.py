from __future__ import annotations

from collections import defaultdict, deque
import statistics
import time
from typing import Any

from ..file_utils import update_json_file
from ..observability import record_metric
from .config import SignalConfig
from .features import compute_features
from .storage import append_ndjson, state_path


class SignalEngine:
    def __init__(self, config: SignalConfig, *, latest_limit: int = 20) -> None:
        self.config = config
        self.latest_limit = max(1, int(latest_limit))
        self._history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.lookback_bars + 1)
        )
        self._ret_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.lookback_bars)
        )

    def on_bar(self, bar: dict[str, Any]) -> None:
        symbol = str(bar.get("symbol") or "").upper()
        close = bar.get("close")
        if not symbol or close is None:
            return
        if bar.get("stale"):
            skip_reason = "stale"
        else:
            skip_reason = None

        history = self._history[symbol]
        prev_close = history[-1] if history else None
        try:
            history.append(float(close))
        except (TypeError, ValueError):
            return
        if len(history) <= self.config.lookback_bars:
            return
        base_close = history[0]
        momentum = None
        try:
            momentum = (history[-1] / base_close) - 1.0
        except (TypeError, ValueError, ZeroDivisionError):
            momentum = None

        features = compute_features(bar, prev_close=prev_close)
        ret_val = features.get("ret_5s")
        if ret_val is not None:
            self._ret_history[symbol].append(float(ret_val))
        realized_vol = None
        ret_series = self._ret_history.get(symbol)
        if ret_series and len(ret_series) >= 2:
            realized_vol = statistics.pstdev(ret_series)
        features["realized_vol"] = realized_vol
        if skip_reason is None:
            if ret_val is None:
                skip_reason = "no_return"
        if skip_reason is None:
            volume = features.get("volume")
            if volume is not None and volume < self.config.min_volume:
                skip_reason = "min_volume"
            spread_bps = features.get("spread_bps")
            if spread_bps is not None and spread_bps > self.config.max_spread_bps:
                skip_reason = "spread"

        signal = "hold"
        if momentum is not None and skip_reason is None:
            if abs(momentum) >= self.config.entry_threshold:
                signal = "long" if momentum > 0 else "short"
        payload = {
            "timestamp": bar.get("timestamp"),
            "symbol": symbol,
            "signal": signal,
            "score": momentum,
            "skip_reason": skip_reason,
            "bar": {
                "timeframe": bar.get("timeframe"),
                "open": bar.get("open"),
                "high": bar.get("high"),
                "low": bar.get("low"),
                "close": bar.get("close"),
                "volume": bar.get("volume"),
            },
            "features": features,
        }
        stamp = time.strftime("%Y%m%d", time.gmtime())
        append_ndjson(f"signals_{stamp}.ndjson", [payload])
        self._update_latest(payload)
        if signal != "hold" and skip_reason is None:
            record_metric("realtime.signal.fired", symbol=symbol, signal=signal, score=momentum)

    def _update_latest(self, payload: dict[str, Any]) -> None:
        def updater(current: Any) -> Any:
            if not isinstance(current, dict):
                current = {}
            signals = current.get("signals")
            if not isinstance(signals, list):
                signals = []
            signals.append(payload)
            current["signals"] = signals[-self.latest_limit :]
            current["updated_at"] = time.time()
            return current

        update_json_file(state_path("signals_latest.json"), default={"signals": []}, update_fn=updater)
