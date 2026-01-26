from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import StrategyInput
from .event_engine import compute_realized_returns, run_event_backtest
from .risk import calculate_target_leverage, enforce_min_holding, enforce_risk_limits

# 类型提示用的回调，避免循环导入
SummarizeFn = Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]]
OOSFn = Callable[[pd.DataFrame, StrategyInput], dict[str, Any] | None]


def _ensure_rsi(prices: pd.DataFrame, params: StrategyInput) -> pd.Series:
    if "rsi" in prices.columns and prices["rsi"].notna().any():
        return prices["rsi"]
    rsi_period = max(int(params.rsi_period or 14), 1)
    delta = prices["adj close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    alpha = 1 / rsi_period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(0.0)


def backtest_mean_reversion_strategy(
    prices: pd.DataFrame,
    params: StrategyInput,
    *,
    summarize_backtest_fn: SummarizeFn | None = None,
    compute_oos_report: OOSFn | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """
    RSI 均值回归策略回测：超买做空，超卖做多。
    """
    backtest = prices.copy()
    backtest["rsi"] = _ensure_rsi(backtest, params)
    backtest = backtest.dropna(subset=["rsi"]).copy()

    backtest["signal"] = np.where(
        backtest["rsi"] > 70,
        -1.0,
        np.where(backtest["rsi"] < 30, 1.0, 0.0),
    )
    backtest["position"] = enforce_min_holding(
        pd.Series(backtest["signal"], index=backtest.index),
        params.min_holding_days,
    )

    # Skip trading on ultra-illiquid days (align with SMA strategy)
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
        backtest["position"],
        backtest["volatility"],
        params.volatility_target,
        params.max_leverage,
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
        stats["validation_report_detected"] = "mean_reversion_pfws"
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


__all__ = ["backtest_mean_reversion_strategy"]
