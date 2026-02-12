from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time

from .signals import CombinedSignal


@dataclass(slots=True)
class RiskLimits:
    max_position_weight: float = 0.3
    max_leverage: float = 1.5
    min_confidence: float | None = None
    max_daily_loss_pct: float = 0.03
    kill_switch_cooldown_seconds: int = 900


class RiskManager:
    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self._daily_anchor_date = None
        self._daily_anchor_equity = None
        self._blocked_until_ts = None
        self._last_guard_reason = None
        self._last_drawdown_pct = 0.0

    def _ensure_daily_anchor(self, account_equity: float, now_ts: float) -> None:
        utc_day = datetime.fromtimestamp(now_ts, tz=timezone.utc).date()
        if self._daily_anchor_date != utc_day or not self._daily_anchor_equity:
            self._daily_anchor_date = utc_day
            self._daily_anchor_equity = max(float(account_equity or 0.0), 0.0)
            self._blocked_until_ts = None
            self._last_guard_reason = None
            self._last_drawdown_pct = 0.0

    def check_kill_switch(self, account_equity: float | None, *, now_ts: float | None = None) -> tuple[bool, str | None]:
        if self.limits.max_daily_loss_pct <= 0:
            return False, None
        try:
            equity = float(account_equity or 0.0)
        except (TypeError, ValueError):
            equity = 0.0
        if equity <= 0:
            return False, None
        if now_ts is None:
            now_ts = time.time()

        self._ensure_daily_anchor(equity, now_ts)
        if self._blocked_until_ts is not None and now_ts < self._blocked_until_ts:
            self._last_guard_reason = "daily_loss_cooldown"
            return True, self._last_guard_reason

        anchor = float(self._daily_anchor_equity or 0.0)
        if anchor <= 0:
            return False, None
        drawdown = max(0.0, (anchor - equity) / anchor)
        self._last_drawdown_pct = drawdown
        if drawdown >= self.limits.max_daily_loss_pct:
            cooldown = max(int(self.limits.kill_switch_cooldown_seconds or 0), 0)
            self._blocked_until_ts = now_ts + cooldown if cooldown else now_ts
            self._last_guard_reason = "max_daily_loss"
            return True, self._last_guard_reason
        self._last_guard_reason = None
        return False, None

    def snapshot(self) -> dict[str, float | str | None]:
        return {
            "anchor_equity": self._daily_anchor_equity,
            "drawdown_pct": self._last_drawdown_pct,
            "blocked_until_ts": self._blocked_until_ts,
            "reason": self._last_guard_reason,
        }

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
