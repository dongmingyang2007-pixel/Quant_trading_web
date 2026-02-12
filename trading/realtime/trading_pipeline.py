from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..alpaca_data import resolve_alpaca_trading_mode
from ..runtime import (
    AccountManager,
    AlpacaExecutionClient,
    LiveTradingPipeline,
    RiskLimits,
    RiskManager,
    StrategyCombiner,
)
from ..runtime.strategy_registry import build_strategies
from .config import RealtimeConfig


def _resolve_mode(config_mode: str, *, user_id: str | None) -> str:
    normalized = (config_mode or "").strip().lower()
    if normalized in {"paper", "live"}:
        return normalized
    return resolve_alpaca_trading_mode(user_id=user_id)


def build_trading_pipeline(config: RealtimeConfig, *, user_id: str | None) -> LiveTradingPipeline | None:
    trading = config.trading
    if not trading.enabled:
        return None
    strategies = build_strategies([asdict(spec) for spec in trading.strategies])
    if not strategies:
        return None
    weights = dict(trading.combiner.weights)
    if not weights:
        weights = {spec.name: spec.weight for spec in trading.strategies if spec.enabled}
    combiner = StrategyCombiner(method=trading.combiner.method, weights=weights)
    risk_limits = RiskLimits(
        max_position_weight=trading.risk.max_position_weight,
        max_leverage=trading.risk.max_leverage,
        min_confidence=trading.risk.min_confidence,
        max_daily_loss_pct=trading.risk.max_daily_loss_pct,
        kill_switch_cooldown_seconds=trading.risk.kill_switch_cooldown_seconds,
    )
    risk_manager = RiskManager(risk_limits)
    mode = _resolve_mode(trading.mode, user_id=user_id)
    execution = AlpacaExecutionClient(user_id=user_id, mode=mode)
    account = AccountManager(execution)
    execution_config: dict[str, Any] = {
        "order_type": trading.execution.order_type,
        "time_in_force": trading.execution.time_in_force,
        "max_orders_per_minute": trading.execution.max_orders_per_minute,
    }
    pipeline = LiveTradingPipeline(
        strategies=strategies,
        combiner=combiner,
        risk_manager=risk_manager,
        execution=execution,
        account=account,
        execution_config=execution_config,
        metadata={
            "mode": mode,
            "combiner": trading.combiner.method,
            "strategy_count": len(strategies),
        },
    )
    return pipeline
