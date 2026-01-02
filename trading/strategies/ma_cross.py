from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import StrategyInput
from .execution import apply_execution_model as execute_orders
from .event_engine import compute_realized_returns, run_event_backtest
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
    """根据成交量与参与率估算执行冲击成本（统一模型）。"""
    price = backtest.get("adj close")
    if price is None or price.empty:
        return backtest, []
    volume = price_source["volume"].reindex(backtest.index).ffill().bfill() if "volume" in price_source.columns else None
    adjusted, txn_cost, exec_cost, coverage, stats = execute_orders(
        backtest["exposure"],
        price,
        volume=volume,
        adv=backtest.get("adv"),
        params=params,
    )
    backtest["exposure"] = adjusted
    if "leverage" in backtest:
        with np.errstate(divide="ignore", invalid="ignore"):
            adj_position = adjusted / backtest["leverage"].replace(0, np.nan)
        backtest["position"] = adj_position.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    backtest["transaction_cost"] = txn_cost
    backtest["execution_cost"] = exec_cost
    backtest["strategy_return"] = backtest["strategy_return_gross"] - txn_cost - exec_cost - backtest.get("borrow_cost", 0.0)
    events = []
    if stats.get("avg_coverage") is not None and stats.get("avg_coverage", 1.0) < 0.95:
        events.append(f"执行撮合模型：成交覆盖率 {stats['avg_coverage']:.0%}，存在部分未成交。")
    if stats.get("halt_days", 0.0) > 0:
        events.append(f"执行撮合模型：检测到 {int(stats['halt_days'])} 天停牌/无成交，未执行交易。")
    if stats.get("limit_days", 0.0) > 0:
        events.append(f"执行撮合模型：检测到 {int(stats['limit_days'])} 天触及涨跌幅限制，未执行交易。")
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

    # Skip trading on ultra-illiquid days (e.g., near-zero ADV or zero volume)
    liquidity_blocks = 0
    try:
        adv_series = backtest["adv"].fillna(0.0) if "adv" in backtest else pd.Series(0.0, index=backtest.index)
        vol_series = backtest["volume"].fillna(0.0) if "volume" in backtest else pd.Series(0.0, index=backtest.index)
        adv_median = float(adv_series.median())
        if adv_median > 0:
            floor = adv_median * 0.1
            illiquid_mask = (adv_series < floor) | (vol_series <= 0)
            liquidity_blocks = int(illiquid_mask.sum())
            if liquidity_blocks:
                backtest.loc[illiquid_mask, ["signal", "position"]] = 0.0
    except Exception:
        liquidity_blocks = 0

    asset_returns = compute_realized_returns(backtest, params)
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
    cost_rate = (params.transaction_cost_bps + params.slippage_bps) / 10000.0
    backtest, exec_stats, execution_events = run_event_backtest(
        backtest,
        exposure_series,
        params,
        leverage=backtest["leverage"],
    )

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
    adv_hits = int(exec_stats.get("adv_hard_cap_hits") or 0)
    stats["execution_stats"] = {
        "avg_coverage": exec_stats.get("avg_coverage"),
        "unfilled_ratio": exec_stats.get("unfilled_ratio"),
        "avg_spread_bps": exec_stats.get("avg_spread_bps"),
        "halt_days": exec_stats.get("halt_days"),
        "limit_days": exec_stats.get("limit_days"),
        "participation": exec_stats.get("participation"),
        "effective_participation": exec_stats.get("effective_participation"),
        "adv_hard_cap_hits": adv_hits,
    }
    stats["cost_assumptions"] = {
        "slippage_model": params.slippage_model,
        "cost_rate": cost_rate,
        "long_borrow_bps": params.long_borrow_cost_bps or params.borrow_cost_bps,
        "short_borrow_bps": params.short_borrow_cost_bps or params.borrow_cost_bps,
        "adv_participation": params.max_adv_participation,
        "execution_mode": params.execution_mode,
    }
    stats["liquidity_blocks"] = liquidity_blocks
    aggregate_events = []
    if overlay_events:
        aggregate_events.extend(overlay_events)
    if adv_hits > 0:
        participation = params.max_adv_participation or 0.1
        aggregate_events.append(f"因 ADV 参与率上限({participation:.0%}) 压缩 {adv_hits} 次仓位，避免不可成交。")
    if liquidity_blocks > 0:
        aggregate_events.append(f"因成交额过低/停牌跳过 {liquidity_blocks} 个交易日的信号。")
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
