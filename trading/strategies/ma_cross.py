from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import StrategyInput
from .indicators import _compute_asset_returns
from .risk import (
    calculate_target_leverage,
    enforce_min_holding,
    enforce_risk_limits,
)

# 类型提示用的回调，避免循环导入
SummarizeFn = Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]]
OOSFn = Callable[[pd.DataFrame, StrategyInput], dict[str, Any] | None]


def apply_execution_model(
    backtest: pd.DataFrame,
    price_source: pd.DataFrame,
    params: StrategyInput,
) -> tuple[pd.DataFrame, list[str]]:
    """根据成交量估算执行冲击成本。"""
    if "volume" not in price_source.columns:
        return backtest, []
    price = backtest.get("adj close")
    if price is None or price.empty:
        return backtest, []
    raw_volume = price_source["volume"].reindex(backtest.index).ffill().bfill()
    dollar_volume = (raw_volume * price).rolling(20).mean()
    exposure_change = backtest["exposure"].diff().abs().fillna(backtest["exposure"].abs())
    turnover_value = exposure_change * params.capital
    liquidity_buffer = max(params.execution_liquidity_buffer, 0.01)
    liquidity_capacity = dollar_volume * liquidity_buffer
    if liquidity_capacity.isna().all() or liquidity_capacity.fillna(0.0).sum() == 0:
        return backtest, ["执行模型：缺少成交量数据，已跳过撮合成本估计。"]
    impact = turnover_value / liquidity_capacity.replace(0, np.nan)
    impact = impact.clip(lower=0.0, upper=5.0).fillna(0.0)
    if params.execution_mode == "limit":
        fill_prob = np.exp(-impact.clip(0, 5))
        penalty = (1 - fill_prob) * np.abs(backtest["strategy_return_gross"]) + impact * (params.execution_penalty_bps / 10000.0)
    else:
        penalty = impact * (params.execution_penalty_bps / 10000.0)
    backtest["execution_cost"] = penalty
    backtest["strategy_return"] = backtest["strategy_return"] - penalty
    events = []
    if penalty.sum() > 0:
        avg_impact = float(impact.replace([np.inf, -np.inf], np.nan).mean())
        events.append(f"执行撮合模型：平均冲击 {avg_impact:.2f}×ADV，额外成本 {penalty.sum():.4f}。")
    return backtest, events


def backtest_sma_strategy(
    prices: pd.DataFrame,
    params: StrategyInput,
    *,
    summarize_backtest_fn: SummarizeFn | None = None,
    compute_oos_report: OOSFn | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """
    均线交叉策略回测，聚焦趋势跟踪＋基础风控。
    """
    backtest = prices.dropna(subset=["sma_short", "sma_long"]).copy()
    backtest["signal"] = np.where(backtest["sma_short"] > backtest["sma_long"], 1.0, 0.0)
    backtest["position"] = enforce_min_holding(
        pd.Series(backtest["signal"], index=backtest.index), params.min_holding_days
    )

    asset_returns = _compute_asset_returns(backtest, params)
    backtest["asset_return"] = asset_returns
    backtest["volatility"] = asset_returns.rolling(window=20).std().fillna(0.0) * np.sqrt(252)
    backtest["leverage"] = calculate_target_leverage(
        backtest["position"], backtest["volatility"], params.volatility_target, params.max_leverage
    )

    exposure_series, overlay_events = enforce_risk_limits(
        backtest["position"],
        backtest["leverage"],
        asset_returns,
        params,
    )
    backtest["exposure"] = exposure_series
    with np.errstate(divide="ignore", invalid="ignore"):
        adj_position = backtest["exposure"] / backtest["leverage"].replace(0, np.nan)
    backtest["position"] = adj_position.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    exposure_change = backtest["exposure"].diff().abs().fillna(backtest["exposure"].abs())
    cost_rate = (params.transaction_cost_bps + params.slippage_bps) / 10000.0
    # ADV 参与率约束
    adv_hits = 0
    if "adv" in backtest and backtest["adv"].notna().any():
        max_part = max(0.0, min(1.0, params.max_adv_participation or 0.1))
        adv_limit = backtest["adv"].fillna(0.0) * max_part
        mask = backtest["exposure"].abs() > adv_limit
        adv_hits = int(mask.sum())
        capped = backtest["exposure"].where(~mask, 0.0)
        if adv_hits > 0:
            backtest["exposure"] = capped
            exposure_change = backtest["exposure"].diff().abs().fillna(backtest["exposure"].abs())
    backtest["transaction_cost"] = exposure_change * cost_rate

    shifted_exposure = (
        backtest["exposure"]
        if params.return_path in {"close_to_open", "open_to_close"}
        else backtest["exposure"].shift(fill_value=0)
    )
    backtest["strategy_return_gross"] = asset_returns * shifted_exposure
    long_daily = float(params.long_borrow_cost_bps or params.borrow_cost_bps) / 10000.0 / 252.0
    short_daily = float(params.short_borrow_cost_bps or params.borrow_cost_bps) / 10000.0 / 252.0
    borrow_cost = (
        backtest["exposure"].clip(lower=0.0) * long_daily
        + (-backtest["exposure"].clip(upper=0.0)) * short_daily
    )
    backtest["borrow_cost"] = borrow_cost
    backtest["strategy_return"] = backtest["strategy_return_gross"] - backtest["transaction_cost"] - borrow_cost
    backtest, execution_events = apply_execution_model(backtest, prices, params)
    backtest["cum_strategy"] = (1 + backtest["strategy_return"]).cumprod()
    backtest["cum_buy_hold"] = (1 + asset_returns).cumprod()

    if summarize_backtest_fn is None or compute_oos_report is None:
        # 延迟导入以避免循环依赖
        from .core import summarize_backtest as _summarize  # type: ignore
        from .core import _compute_oos_from_backtest as _oos  # type: ignore
        summarize_backtest_fn = summarize_backtest_fn or _summarize
        compute_oos_report = compute_oos_report or _oos

    metrics, stats = summarize_backtest_fn(
        backtest,
        params,
        include_prediction=False,
        include_auc=False,
        feature_columns=[
            "sma_short",
            "sma_long",
            "rsi",
            "volatility",
            "position",
        ],
    )
    stats["cost_assumptions"] = {
        "slippage_model": params.slippage_model,
        "cost_rate": cost_rate,
        "long_borrow_bps": params.long_borrow_cost_bps or params.borrow_cost_bps,
        "short_borrow_bps": params.short_borrow_cost_bps or params.borrow_cost_bps,
        "adv_participation": params.max_adv_participation,
        "execution_mode": params.execution_mode,
    }
    aggregate_events = []
    if overlay_events:
        aggregate_events.extend(overlay_events)
    if adv_hits > 0:
        aggregate_events.append(f"因 ADV 参与率上限({params.max_adv_participation:.0%}) 清零 {adv_hits} 次仓位，避免不可成交。")
    aggregate_events.extend(execution_events)
    oos_report = compute_oos_report(backtest, params) if compute_oos_report else None
    if oos_report:
        stats["validation_report_detected"] = "sma_pfws"
        stats["validation_oos_summary"] = oos_report.get("summary")
        stats["validation_oos_folds"] = oos_report.get("folds")
        stats["validation_penalized_sharpe"] = oos_report.get("penalized_sharpe")
        stats["validation_train_window"] = oos_report.get("train_window")
        stats["validation_test_window"] = oos_report.get("test_window")
        stats["validation_embargo"] = oos_report.get("embargo")
    if not oos_report:
        aggregate_events.append("提示：此策略样本外 PFWS 指标生成失败，当前仅展示全量回测结果。")
    if aggregate_events:
        stats["risk_events"] = aggregate_events
    return backtest, metrics, stats


def format_table(backtest: pd.DataFrame) -> list[dict[str, Any]]:
    """准备表格数据，便于前端展示最近信号/仓位。"""
    columns = [
        "signal",
        "position",
        "adj close",
        "sma_short",
        "sma_long",
        "rsi",
        "strategy_return",
        "leverage",
        "cum_strategy",
        "cum_buy_hold",
    ]
    if "probability" in backtest.columns:
        columns.insert(3, "probability")
    if "transaction_cost" in backtest.columns:
        columns.append("transaction_cost")

    subset = backtest[columns].tail(30)
    subset.index = subset.index.date
    return [
        {
            "date": idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx),
            "position": int(row["position"]),
            "signal": int(row["signal"]),
            "adj_close": round(float(row["adj close"]), 2),
            "sma_short": round(float(row["sma_short"]), 2),
            "sma_long": round(float(row["sma_long"]), 2),
            "rsi": round(float(row["rsi"]), 2),
            "daily_return": round(float(row["strategy_return"]), 4),
            "leverage": round(float(row["leverage"]), 2),
            "cum_strategy": round(float(row["cum_strategy"]), 4),
            "cum_buy_hold": round(float(row["cum_buy_hold"]), 4),
            "probability": round(float(row["probability"]), 3) if "probability" in subset.columns else None,
            "transaction_cost": round(float(row["transaction_cost"]), 5)
            if "transaction_cost" in subset.columns
            else None,
        }
        for idx, row in subset.iterrows()
    ]


__all__ = [
    "apply_execution_model",
    "backtest_sma_strategy",
    "format_table",
]
