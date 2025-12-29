from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pandas as pd

from .risk import calculate_max_drawdown

# 指标描述与通用格式化/统计函数

METRIC_DESCRIPTIONS = {
    "策略累计收益率": "策略净值相对起点的总涨幅，衡量绝对收益表现。",
    "买入持有收益率": "直接持有标的（不做调仓）的回报，可用于衡量策略超额收益。",
    "夏普比率": "单位波动获得的超额收益，>1 表示风险调整后表现良好。",
    "最大回撤": "净值从高点到低点的最大跌幅，越小越稳健。",
    "索提诺比率": "仅惩罚下跌波动的收益风险比，更关注下行风险。",
    "年化波动率": "收益波动程度，反映风险水平。",
    "年化复合收益率": "复利视角的年化收益，更贴近日常收益口径。",
    "卡玛比率": "年化收益与最大回撤之比，衡量单位回撤创造收益能力。",
    "胜率": "盈利交易日占比，可反映策略稳定性。",
    "单日平均盈亏": "平均上涨日与下跌日的收益率，衡量盈亏对称度。",
    "平均持仓比例": "在市场中的平均资金暴露程度。",
    "平均杠杆（波动率目标）": "为达到目标波动率所需的平均杠杆倍数。",
    "日度95%VaR": "在 95% 情况下，单日最大可能亏损的保守估计。",
    "日度95%CVaR": "落在 VaR 置信区间外的平均亏损，衡量尾部风险。",
    "交易日数量": "回测样本的交易日数量。",
    "基准累计收益率": "选择的对比指数或资产在回测期内的收益。",
    "基准年化波动率": "基准资产的波动水平。",
    "基准夏普比率": "基准资产的风险调整后收益表现。",
    "策略相对基准α": "策略相对基准的年化超额收益（CAPM α）。",
    "β系数": "策略对基准波动的敏感度，>1 表示更敏感。",
    "与基准相关系数": "策略与基准收益的同步度，接近 0 更有分散度。",
    "信息比率": "单位跟踪误差的超额收益，越高代表超额收益稳定。",
    "跟踪误差": "策略相对基准收益差的波动度。",
    "预测胜率": "模型预测方向与实际收益方向一致的比例。",
    "ROC-AUC": "预测概率区分上涨与下跌的能力，0.5 为随机水准。",
    "年化换手率": "一年内的仓位变动次数衡量交易频率。",
    "平均持仓天数": "单次开仓持有的平均持续时间。",
    "成本占收益比": "交易成本与策略总收益的比例，用于衡量成本侵蚀。",
}


def build_metric(label: str, value: str) -> dict[str, str]:
    return {
        "label": label,
        "value": value,
        "explain": METRIC_DESCRIPTIONS.get(label, ""),
    }


def format_percentage(value: float) -> str:
    """Format a decimal return as percentage string."""
    if value is None or math.isnan(value):
        return "N/A"
    return f"{value:.2%}"


def format_currency(value: float) -> str:
    try:
        return f"{value:,.0f}"
    except (TypeError, ValueError):
        return str(value)


def fig_to_base64(fig) -> str:
    import base64
    import io
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def calculate_sharpe(returns: pd.Series, trading_days: int, risk_free_rate: float = 0.0) -> float:
    excess_returns = returns - risk_free_rate / trading_days
    std = returns.std()
    if std == 0:
        return 0.0
    return np.sqrt(trading_days) * excess_returns.mean() / std


def calculate_sortino(returns: pd.Series, trading_days: int, risk_free_rate: float = 0.0) -> float:
    downside = returns.copy()
    downside[downside > 0] = 0
    downside_std = np.sqrt((downside**2).mean())
    if downside_std == 0:
        return 0.0
    avg_excess = returns.mean() - risk_free_rate / trading_days
    return np.sqrt(trading_days) * avg_excess / downside_std


def get_risk_free_rate_annual() -> float:
    try:
        return float(os.environ.get("RISK_FREE_RATE_ANNUAL", "0.0"))
    except ValueError:
        return 0.0


def calculate_cagr(cumulative_returns: pd.Series, trading_days: int) -> float:
    if cumulative_returns.empty:
        return 0.0
    total_return = cumulative_returns.iloc[-1]
    periods = cumulative_returns.shape[0]
    if periods <= 1 or total_return <= 0:
        return 0.0
    return total_return ** (trading_days / periods) - 1


def calculate_calmar(cagr: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return 0.0
    return cagr / abs(max_drawdown)


def calculate_hit_ratio(returns: pd.Series) -> float:
    positive = (returns > 0).sum()
    total = (returns != 0).sum()
    if total == 0:
        return 0.0
    return positive / total


def calculate_avg_gain_loss(returns: pd.Series) -> tuple[float, float]:
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    avg_gain = positive.mean() if not positive.empty else 0.0
    avg_loss = negative.mean() if not negative.empty else 0.0
    return avg_gain, avg_loss


def calculate_holding_periods(position: pd.Series) -> float:
    """平均持仓天数（交易日）。"""
    if position.empty:
        return 0.0
    clean = position.fillna(0).round().astype(int)
    durations: list[int] = []
    current = clean.iloc[0]
    length = 1 if current != 0 else 0

    for value in clean.iloc[1:]:
        if value == current and value != 0:
            length += 1
        else:
            if current != 0 and length > 0:
                durations.append(length)
            current = value
            length = 1 if current != 0 else 0
    if current != 0 and length > 0:
        durations.append(length)
    return float(np.mean(durations)) if durations else 0.0


def calculate_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    variance = benchmark_returns.var()
    if variance == 0:
        return 0.0
    covariance = strategy_returns.cov(benchmark_returns)
    return covariance / variance


def calculate_var_cvar(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    """Compute daily Value at Risk and Conditional VaR at given confidence."""
    if returns.empty:
        return 0.0, 0.0
    quantile = returns.quantile(1 - alpha)
    var = min(quantile, 0.0)
    tail_losses = returns[returns <= quantile]
    cvar = tail_losses.mean() if not tail_losses.empty else 0.0
    return var, cvar


def build_core_metrics(
    stats: dict[str, Any],
    *,
    include_prediction: bool = False,
    include_auc: bool = False,
) -> list[dict[str, str]]:
    """渲染核心指标卡片。"""

    def _float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    metrics = [
        build_metric("策略累计收益率", format_percentage(_float(stats.get("total_return")))),
        build_metric("买入持有收益率", format_percentage(_float(stats.get("buy_hold_return")))),
        build_metric("夏普比率", f"{_float(stats.get('sharpe')):.2f}"),
        build_metric(
            "夏普置信区间",
            f"[{_float(stats.get('sharpe_ci_lower')):.2f}, {_float(stats.get('sharpe_ci_upper')):.2f}]"
            if stats.get("sharpe_ci_lower") is not None and stats.get("sharpe_ci_upper") is not None
            else "N/A",
        ),
        build_metric(
            "Deflated Sharpe",
            f"{_float(stats.get('deflated_sharpe')):.2f}" if stats.get("deflated_sharpe") is not None else "N/A",
        ),
        build_metric("最大回撤", format_percentage(_float(stats.get("max_drawdown")))),
        build_metric("索提诺比率", f"{_float(stats.get('sortino')):.2f}"),
        build_metric("年化波动率", format_percentage(_float(stats.get("volatility")))),
        build_metric("年化复合收益率", format_percentage(_float(stats.get("cagr")))),
        build_metric("卡玛比率", f"{_float(stats.get('calmar')):.2f}"),
        build_metric("胜率", format_percentage(_float(stats.get("hit_ratio")))),
        build_metric(
            "单日平均盈亏",
            f"{format_percentage(_float(stats.get('avg_gain')))} / {format_percentage(_float(stats.get('avg_loss')))}",
        ),
        build_metric("平均持仓比例", format_percentage(_float(stats.get("avg_exposure")))),
        build_metric("平均杠杆（波动率目标）", f"{_float(stats.get('avg_leverage')):.2f}x"),
        build_metric("日度95%VaR", format_percentage(-_float(stats.get("var_95")))),
        build_metric("日度95%CVaR", format_percentage(-_float(stats.get("cvar_95")))),
        build_metric("最长回撤恢复期(TWR)", f"{_float(stats.get('twr_days')):.0f} 天" if stats.get("twr_days") is not None else "N/A"),
        build_metric("持续亏损天数", str(int(_float(stats.get("loss_streak"))))),
        build_metric("最长恢复期", f"{_float(stats.get('recovery_days')):.0f} 天" if stats.get("recovery_days") is not None else "N/A"),
        build_metric(
            "White RC 调整p值",
            f"{_float(stats.get('sharpe_pvalue_adjusted')):.3f}" if stats.get("sharpe_pvalue_adjusted") is not None else "N/A",
        ),
    ]

    validation_summary = (
        stats.get("validation_summary_compact")
        or stats.get("validation_oos_summary")
        or {}
    )
    sharpe_oos = validation_summary.get("sharpe") if isinstance(validation_summary, dict) else None
    if isinstance(sharpe_oos, dict) and sharpe_oos:
        metrics.append(
            build_metric(
                "OOS夏普(均值±std)",
                f"{_float(sharpe_oos.get('mean')):.2f} ± {_float(sharpe_oos.get('std')):.2f}",
            )
        )
        metrics.append(
            build_metric(
                "OOS夏普IQR",
                f"{_float(sharpe_oos.get('iqr')):.2f}",
            )
        )

    if include_prediction and stats.get("prediction_accuracy") is not None:
        metrics.append(build_metric("预测胜率", format_percentage(_float(stats.get("prediction_accuracy")))))
    if include_auc and stats.get("auc") is not None:
        auc = _float(stats.get("auc"), float("nan"))
        value = "N/A" if math.isnan(auc) else f"{auc:.2f}"
        metrics.append(build_metric("ROC-AUC", value))
    if stats.get("calibration"):
        calib = stats["calibration"]
        brier = _float(calib.get("brier"), float("nan"))
        brier_val = "N/A" if math.isnan(brier) else f"{brier:.4f}"
        metrics.append(build_metric("Brier Score", brier_val))

    metrics.extend(
        [
            build_metric("年化换手率", format_percentage(_float(stats.get("annual_turnover")))),
            build_metric("平均持仓天数", f"{_float(stats.get('average_holding_days')):.1f}"),
            build_metric("成本占收益比", format_percentage(_float(stats.get("cost_ratio")))),
            build_metric("交易日数量", str(int(_float(stats.get("trading_days"), 0.0)))),
        ]
    )
    return metrics


def compute_validation_metrics(pnl: pd.Series) -> dict[str, float]:
    """基于收益序列的基础评估指标（用于 OOS/PFWS、RL 等）。"""
    pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()
    if pnl.empty:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "hit_ratio": 0.0}
    cumulative = (1 + pnl).cumprod()
    return {
        "sharpe": float(calculate_sharpe(pnl, 252)),
        "cagr": float(calculate_cagr(cumulative, 252)),
        "max_drawdown": float(calculate_max_drawdown(cumulative)),
        "hit_ratio": float(calculate_hit_ratio(pnl)),
    }


def aggregate_oos_metrics(slices: list[dict[str, Any]]) -> dict[str, Any]:
    """对样本外折的指标做均值/std/IQR 汇总。"""
    summary: dict[str, Any] = {}
    if not slices:
        return summary
    for key in ("sharpe", "cagr", "max_drawdown", "hit_ratio"):
        values = [float(entry.get(key, 0.0)) for entry in slices if entry.get(key) is not None]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "iqr": float(q3 - q1),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }
    return summary


def build_oos_boxplot(distributions: dict[str, list[float]], title: str) -> str | None:
    """基于样本外分布绘制箱线图，返回 base64."""
    try:
        import matplotlib.pyplot as plt

        metrics = []
        labels = []
        for key, values in distributions.items():
            if not values:
                continue
            metrics.append(values)
            labels.append(key.upper())
        if not metrics:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(metrics, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#93c5fd", alpha=0.7))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        return fig_to_base64(fig)
    except Exception:
        return None


__all__ = [
    "METRIC_DESCRIPTIONS",
    "build_core_metrics",
    "build_metric",
    "calculate_avg_gain_loss",
    "calculate_beta",
    "calculate_cagr",
    "calculate_calmar",
    "calculate_holding_periods",
    "calculate_hit_ratio",
    "calculate_sharpe",
    "calculate_sortino",
    "calculate_var_cvar",
    "format_currency",
    "format_percentage",
    "fig_to_base64",
    "get_risk_free_rate_annual",
    "compute_validation_metrics",
    "aggregate_oos_metrics",
    "build_oos_boxplot",
]
