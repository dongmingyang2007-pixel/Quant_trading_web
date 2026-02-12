from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any
import time

from .account import AccountManager, AccountState
from .combiner import StrategyCombiner
from .execution import AlpacaExecutionClient, OrderRequest, OrderResult
from .risk import RiskManager
from .signals import CombinedSignal, Signal
from .strategy import BaseStrategy, StrategyContext


@dataclass(slots=True)
class LiveTradingPipeline:
    strategies: list[BaseStrategy]
    combiner: StrategyCombiner
    risk_manager: RiskManager
    execution: AlpacaExecutionClient
    account: AccountManager
    latest_account: AccountState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_config: dict[str, Any] = field(default_factory=dict)
    _order_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def refresh_account(self) -> AccountState:
        self.latest_account = self.account.get_account_state()
        return self.latest_account

    def on_market_data(self, market_data: dict[str, Any]) -> list[CombinedSignal]:
        symbol = str(market_data.get("symbol") or "").upper()
        account_state = self.latest_account or self.refresh_account()
        context = StrategyContext(
            symbol=symbol,
            positions={pos.symbol: pos.qty for pos in self.account.list_positions()},
            account_equity=account_state.equity,
        )
        raw_signals: list[Signal] = []
        for strat in self.strategies:
            signal = strat.on_market_data(market_data, context)
            if signal:
                signal.metadata = {**(signal.metadata or {}), "strategy": strat.name}
                raw_signals.append(signal)
        combined = self.combiner.combine_signals(raw_signals, timestamp=market_data.get("timestamp"))
        final_signals: list[CombinedSignal] = []
        for sig in combined:
            final_signals.append(self.risk_manager.apply_risk_controls(sig))
        return final_signals

    def execute_signals(self, signals: list[CombinedSignal]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        account_state = self.latest_account or self.refresh_account()
        equity = max(account_state.equity, 0.0)
        guard_blocked = False
        guard_reason = None
        check_guard = getattr(self.risk_manager, "check_kill_switch", None)
        if callable(check_guard):
            try:
                guard_blocked, guard_reason = check_guard(account_state.equity)
            except Exception:
                guard_blocked, guard_reason = False, None
        snapshot_fn = getattr(self.risk_manager, "snapshot", None)
        if callable(snapshot_fn):
            try:
                self.metadata["risk_guard"] = snapshot_fn()
            except Exception:
                self.metadata["risk_guard"] = {}
        order_type = str(self.execution_config.get("order_type", "market") or "market").lower()
        time_in_force = str(self.execution_config.get("time_in_force", "day") or "day").lower()
        max_orders_per_minute = int(self.execution_config.get("max_orders_per_minute", 60) or 60)
        now_ts = None
        if guard_blocked:
            status = f"risk_blocked:{guard_reason or 'guard'}"
            for signal in signals:
                if not signal.symbol or signal.weight == 0:
                    continue
                results.append({"signal": signal, "order": OrderResult(order_id=None, status=status)})
            return results
        for signal in signals:
            if not signal.symbol or signal.weight == 0:
                continue
            if max_orders_per_minute > 0:
                if now_ts is None:
                    now_ts = time.time()
                while self._order_timestamps and now_ts - self._order_timestamps[0] > 60:
                    self._order_timestamps.popleft()
                if len(self._order_timestamps) >= max_orders_per_minute:
                    results.append({"signal": signal, "order": OrderResult(order_id=None, status="throttled")})
                    continue
            notional = abs(signal.weight) * equity
            side = "buy" if signal.weight > 0 else "sell"
            order = OrderRequest(
                symbol=signal.symbol,
                notional=notional,
                side=side,
                order_type=order_type,
                time_in_force=time_in_force,
            )
            result = self.execution.submit_order(order)
            if now_ts is None:
                now_ts = time.time()
            self._order_timestamps.append(now_ts)
            results.append({"signal": signal, "order": result})
        return results
