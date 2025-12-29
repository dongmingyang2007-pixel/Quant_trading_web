from __future__ import annotations


import numpy as np
import pandas as pd

from .config import StrategyInput


def apply_execution_model(
    exposure: pd.Series,
    prices: pd.Series,
    *,
    volume: pd.Series | None = None,
    adv: pd.Series | None = None,
    params: StrategyInput,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, dict[str, float]]:
    """
    Unified execution model for all strategies.

    - Enforces participation limits vs ADV (dollar volume).
    - Applies partial fills when impact is high.
    - Blocks fills on halted days (zero volume) and optional limit-move days.
    """
    base_exposure = exposure.fillna(0.0)
    price = prices.reindex(base_exposure.index).ffill().bfill()
    volume_series = volume.reindex(base_exposure.index).fillna(0.0) if volume is not None else pd.Series(0.0, index=base_exposure.index)

    if adv is not None:
        adv_series = adv.reindex(base_exposure.index).ffill().fillna(0.0)
    else:
        adv_series = (volume_series * price).rolling(20, min_periods=5).mean().fillna(0.0)

    participation = max(0.01, min(0.5, params.max_adv_participation or 0.1))
    liquidity_buffer = max(0.01, min(0.5, params.execution_liquidity_buffer or 0.05))
    effective_participation = min(participation, liquidity_buffer)

    turnover = base_exposure.diff().abs().fillna(base_exposure.abs())
    capital = max(float(getattr(params, "capital", 0.0) or 0.0), 1.0)
    turnover_notional = turnover * capital
    capacity = adv_series * effective_participation
    impact = turnover_notional / capacity.replace(0, np.nan)
    impact = impact.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fill_prob = np.exp(-impact.clip(0, 10))
    halt_mask = (volume_series <= 0) | price.isna() if volume is not None else price.isna()
    limit_threshold = getattr(params, "limit_move_threshold", None)
    limit_mask = pd.Series(False, index=base_exposure.index)
    if limit_threshold is not None and limit_threshold > 0:
        limit_mask = price.pct_change().abs().fillna(0.0) >= float(limit_threshold)
    fill_prob = fill_prob.where(~(halt_mask | limit_mask), 0.0)

    filled_turnover = turnover * fill_prob
    unfilled = turnover - filled_turnover

    atr_pct = price.pct_change().abs().rolling(14, min_periods=5).mean().fillna(0.0)
    base_spread = (atr_pct * 1e4).clip(lower=2.0, upper=50.0)
    spread_bps = base_spread + float(getattr(params, "slippage_bps", 0.0) or 0.0)
    slip_rate = (spread_bps / 1e4) + (impact * (params.execution_penalty_bps or 5) / 1e4)
    execution_cost = filled_turnover * slip_rate
    transaction_cost = filled_turnover * (params.transaction_cost_bps / 10000.0)

    adjusted_exposure = base_exposure.copy()
    if filled_turnover.notna().any():
        deltas = adjusted_exposure.diff().fillna(adjusted_exposure)
        scaled_deltas = deltas * fill_prob
        adjusted_exposure = scaled_deltas.cumsum()

    coverage = filled_turnover / turnover.replace(0, np.nan)
    coverage = coverage.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    stats: dict[str, float] = {
        "avg_coverage": float(coverage.mean()) if not coverage.empty else 1.0,
        "median_coverage": float(coverage.median()) if not coverage.empty else 1.0,
        "unfilled_ratio": float(unfilled.sum() / turnover.sum()) if turnover.sum() else 0.0,
        "avg_impact": float(impact.mean()) if not impact.empty else 0.0,
        "avg_spread_bps": float(spread_bps.mean()) if not spread_bps.empty else 0.0,
        "halt_days": float(halt_mask.sum()),
        "limit_days": float(limit_mask.sum()),
        "participation": float(participation),
        "liquidity_buffer": float(liquidity_buffer),
        "effective_participation": float(effective_participation),
    }

    return adjusted_exposure, transaction_cost, execution_cost, coverage, stats
