from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import StrategyInput
from .execution import apply_execution_model
from .indicators import _normalized_open_prices


def _compute_return_components(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    close = frame.get("adj close")
    if close is None:
        close = frame.get("close")
    if close is None:
        return pd.Series(0.0, index=frame.index), pd.Series(0.0, index=frame.index)
    if "open" in frame:
        open_price = _normalized_open_prices(frame)
    else:
        open_price = close
    open_price = open_price.reindex(close.index).ffill().bfill()
    close = close.reindex(open_price.index).ffill().bfill()
    overnight = open_price / close.shift(1) - 1
    intraday = close / open_price - 1
    return overnight.fillna(0.0), intraday.fillna(0.0)


def compute_realized_returns(frame: pd.DataFrame, params: StrategyInput) -> pd.Series:
    """Return realized return series aligned with event-driven backtest."""
    overnight_ret, intraday_ret = _compute_return_components(frame)
    if params.return_path == "open_to_close":
        return intraday_ret
    if params.return_path == "close_to_open":
        return overnight_ret
    return (1 + overnight_ret) * (1 + intraday_ret) - 1


def run_event_backtest(
    frame: pd.DataFrame,
    target_exposure: pd.Series,
    params: StrategyInput,
    *,
    leverage: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Event-style backtest using daily bars with execution delay and fill constraints."""
    backtest = frame.copy()
    target = target_exposure.reindex(backtest.index).fillna(0.0)
    delay = max(0, int(getattr(params, "execution_delay_days", 1) or 0))
    if delay:
        target = target.shift(delay).fillna(0.0)

    if not getattr(params, "allow_short", True):
        target = target.clip(lower=0.0)
    max_weight = getattr(params, "max_weight", None)
    if max_weight is not None:
        max_weight = abs(float(max_weight))
        target = target.clip(lower=-max_weight, upper=max_weight)
    min_weight = getattr(params, "min_weight", None)
    if min_weight is not None and min_weight > 0:
        min_weight = float(min_weight)
        target = target.where(target.abs() >= min_weight, 0.0)

    adv_hits = 0
    if "adv" in backtest and backtest["adv"].notna().any():
        adv_series = backtest["adv"].fillna(0.0)
        adv_participation = max(0.0, min(1.0, params.max_adv_participation or 0.1))
        capital = max(float(params.capital or 0.0), 1.0)
        adv_limit_weight = (adv_series * adv_participation) / capital
        mask = target.abs() > adv_limit_weight
        adv_hits = int(mask.sum())
        target = target.where(~mask, adv_limit_weight * np.sign(target))

    adjusted, txn_cost, exec_cost, coverage, exec_stats = apply_execution_model(
        target,
        backtest["adj close"] if "adj close" in backtest else backtest["close"],
        volume=backtest.get("volume"),
        adv=backtest.get("adv"),
        params=params,
    )
    exec_stats = dict(exec_stats or {})
    exec_stats["adv_hard_cap_hits"] = adv_hits

    overnight_ret, intraday_ret = _compute_return_components(backtest)
    asset_return = compute_realized_returns(backtest, params)

    prev_exposure = adjusted.shift(1).fillna(0.0)
    if params.return_path == "open_to_close":
        gross = adjusted * intraday_ret
    elif params.return_path == "close_to_open":
        gross = prev_exposure * overnight_ret
    else:
        gross = prev_exposure * overnight_ret + adjusted * intraday_ret

    long_daily = float(params.long_borrow_cost_bps or params.borrow_cost_bps) / 10000.0 / 252.0
    short_daily = float(params.short_borrow_cost_bps or params.borrow_cost_bps) / 10000.0 / 252.0
    borrow_cost = adjusted.clip(lower=0.0) * long_daily + (-adjusted.clip(upper=0.0)) * short_daily

    backtest["exposure"] = adjusted
    if leverage is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            adj_position = adjusted / leverage.replace(0, np.nan)
        backtest["position"] = adj_position.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        backtest["leverage"] = leverage.reindex(backtest.index).fillna(0.0)
    else:
        backtest["position"] = adjusted
        backtest["leverage"] = 1.0

    backtest["transaction_cost"] = txn_cost
    backtest["execution_cost"] = exec_cost
    backtest["fill_coverage"] = coverage
    backtest["borrow_cost"] = borrow_cost
    backtest["strategy_return_gross"] = gross
    backtest["strategy_return"] = gross - txn_cost - exec_cost - borrow_cost
    backtest["asset_return"] = asset_return
    backtest["cum_strategy"] = (1 + backtest["strategy_return"].fillna(0.0)).cumprod()
    backtest["cum_buy_hold"] = (1 + asset_return.fillna(0.0)).cumprod()

    events: list[str] = []
    if adv_hits:
        events.append(f"目标仓位触发 ADV 硬上限 {adv_hits} 次，已按容量压缩。")
    return backtest, exec_stats, events
