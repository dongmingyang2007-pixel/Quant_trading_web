from __future__ import annotations

from typing import Any, Optional
import math
import textwrap
import re

import numpy as np
import pandas as pd
from django.utils.translation import gettext_lazy as _

from .config import StrategyInput
from .metrics import (
    format_currency,
    format_percentage,
)
from .risk import calculate_max_drawdown
from .. import screener

try:  # optional baselines
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ARIMA = None  # type: ignore

try:  # optional baselines
    from statsmodels.tsa.vector_ar.var_model import VAR  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    VAR = None  # type: ignore

try:  # optional data source
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yf = None  # type: ignore

try:  # optional graph analytics
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nx = None  # type: ignore

try:  # optional sentiment analyzer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None  # type: ignore

try:  # optional ML utilities
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional dependency
    MLPClassifier = None  # type: ignore
    StandardScaler = None  # type: ignore

try:  # optional deep learning backend
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore

RISK_PROFILE_LABELS = {
    "conservative": _("保守"),
    "balanced": _("平衡"),
    "aggressive": _("进取"),
}


def normal_cdf(value: float) -> float:
    """Standard normal CDF with safe fallback."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(x):
        return 0.5
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def generate_recommendations(
    stats: dict[str, Any],
    benchmark_stats: Optional[dict[str, Any]],
    params: StrategyInput,
    market_context: dict[str, Any],
) -> list[dict[str, Any]]:
    risk = (params.risk_profile or "balanced").lower()
    risk_label = RISK_PROFILE_LABELS.get(risk, risk)
    base_allocations = {
        "conservative": 0.4,
        "balanced": 0.6,
        "aggressive": 0.75,
    }
    core_weight = base_allocations.get(risk, 0.6)
    hedge_weight = max(0.0, 1 - core_weight)
    overlay_weight = 0.0

    sharpe = stats.get("sharpe", 0.0) or 0.0
    cagr = stats.get("cagr", 0.0) or 0.0
    max_drawdown = stats.get("max_drawdown", 0.0) or 0.0
    volatility = stats.get("volatility", 0.0) or 0.0
    annual_factor = stats.get("annual_factor", 252)
    trading_days = max(1, stats.get("trading_days", 1))
    capital = params.capital or 0.0

    benchmark_total_return = 0.0
    benchmark_vol = 0.0
    benchmark_correlation = 0.0
    benchmark_mu = 0.0

    if benchmark_stats:
        benchmark_total_return = benchmark_stats.get("total_return", 0.0) or 0.0
        benchmark_vol = benchmark_stats.get("volatility", 0.0) or 0.0
        benchmark_correlation = benchmark_stats.get("correlation", 0.0) or 0.0
        benchmark_mu = (1 + benchmark_total_return) ** (annual_factor / trading_days) - 1 if trading_days else 0.0

    def fmt_pct(value: float) -> str:
        return format_percentage(value)

    def breakdown(weights: list[tuple[str, float]]) -> list[dict[str, str]]:
        breakdown_rows: list[dict[str, str]] = []
        for label, pct in weights:
            amount = capital * pct if capital else 0.0
            breakdown_rows.append(
                {
                    "label": label,
                    "percent": fmt_pct(pct),
                    "amount": format_currency(amount) if capital else "—",
                }
            )
        return breakdown_rows

    def project_return(weight_strategy: float, weight_benchmark: float) -> tuple[str, float]:
        strategy_mu = cagr
        strategy_vol = volatility
        combined_mu = weight_strategy * strategy_mu + weight_benchmark * benchmark_mu
        combined_var = (weight_strategy * strategy_vol) ** 2 + (weight_benchmark * benchmark_vol) ** 2
        combined_var += 2 * weight_strategy * weight_benchmark * strategy_vol * benchmark_vol * benchmark_correlation
        combined_var = max(combined_var, 0.0)
        combined_vol = math.sqrt(combined_var)
        if combined_vol == 0:
            success_prob = 1.0 if combined_mu > 0 else 0.0
        else:
            success_prob = normal_cdf(combined_mu / combined_vol)
        expected_final = capital * (1 + combined_mu)
        projection = (
            f"投入 {format_currency(capital)} 约一年后期望价值 {format_currency(expected_final)}"
            if capital
            else "未提供资金规模，建议输入可支配资金以估算收益。"
        )
        if capital:
            projection += f"，获得正收益的概率约 {success_prob * 100:.1f}%。"
        return projection, success_prob

    def build_timeline(plan_name: str, success_prob: Optional[float]) -> list[dict[str, str]]:
        best_prob = f"{(success_prob or 0) * 100:.1f}%" if success_prob is not None else "—"
        base_prob = f"{format_percentage(cagr)} / {format_percentage(volatility)}"
        return [
            {
                "phase": "起步（0-1个月）",
                "base": f"确认信号并分批建仓，执行预设风控：滚动回撤 {format_percentage(stats.get('avg_exposure', 0.0) or 0.0)} 以内保持仓位。",
                "best": f"若信号强劲、成交量放大，可加速完成建仓；预计正收益概率 {best_prob}。",
                "worst": "若建仓阶段市场高波动，建议降低仓位 10%-20%，并辅以指数或期权对冲。",
            },
            {
                "phase": "持有管理（1-6个月）",
                "base": f"按月再平衡，保持策略/防御仓位结构；关注夏普 {stats.get('sharpe', 0.0):.2f} 与回撤走势。",
                "best": "若策略净值新高且回撤受控，可将盈利部分滚入同主题增强资产或分红。",
                "worst": f"若出现连续三周胜率 <40% 或回撤触发 {format_percentage(stats.get('var_95', 0.0) or 0.0)} VaR，执行止损并回归现金。",
            },
            {
                "phase": "评估与再配置（6-12个月）",
                "base": f"滚动检视收益/波动比（当前 {base_prob}），根据宏观与行业信号决定是否延续策略。",
                "best": "若策略跑赢基准 α 保持正数，可逐步提高目标仓位或扩充至相关行业篮子。",
                "worst": "若趋势逆转或基本面恶化，建议转换到低波策略/债券资产，保留核心盈利。",
            },
        ]

    recommendations: list[dict[str, Any]] = []

    projection, base_success_prob = project_return(core_weight, 0.0 if not benchmark_stats else hedge_weight * 0.0)
    recommendations.append(
        {
            "title": "核心动量组合",
            "subtitle": f"以策略信号为主资产，适用于{risk_label}投资者维护收益与波动的平衡。",
            "allocation": [
                f"{fmt_pct(core_weight)} 配置本策略（可通过 ETF/股票篮子复制）",
                f"{fmt_pct(hedge_weight)} 留作现金或货币基金以缓冲波动" if hedge_weight > 0 else "如风险承受度允许，可短期保持满仓策略",
            ],
            "breakdown": breakdown(
                [
                    (f"{params.ticker.upper()} 策略组合", core_weight),
                    ("现金 / 货币基金", hedge_weight),
                ]
            ),
            "projection": projection,
            "actions": textwrap.dedent(
                f"""
                当前夏普比率 {sharpe:.2f}，单位风险补偿依旧健康；最大回撤 {fmt_pct(max_drawdown)}，建议设置
                20% 的动态仓位缓冲（回撤触发时逐步减仓）。可采用“月度再平衡 + 10% 回撤止损”框架来维持纪律。
                """
            ).strip(),
            "success_probability": base_success_prob * 100 if base_success_prob is not None else None,
            "timeline": build_timeline("核心动量组合", base_success_prob),
        }
    )

    if benchmark_stats:
        hedge_allocation = fmt_pct(hedge_weight) if hedge_weight > 0 else "适量"
        projection, success_prob = project_return(core_weight, hedge_weight)
        recommendations.append(
            {
                "title": "防御性对冲组合",
                "subtitle": f"将策略与基准 {params.benchmark_ticker.upper()} 组合，削弱单一资产风险。",
                "allocation": [
                    f"{fmt_pct(core_weight)} 策略多元资产/动量篮子",
                    f"{hedge_allocation} {params.benchmark_ticker.upper()} 指数基金或防御资产",
                ],
                "breakdown": breakdown(
                    [
                        (f"{params.ticker.upper()} 策略组合", core_weight),
                        (f"{params.benchmark_ticker.upper()} 指数/ETF", hedge_weight),
                    ]
                ),
                "projection": projection,
                "actions": textwrap.dedent(
                    f"""
                    基准波动率 {fmt_pct(benchmark_vol)}，与策略的相关系数 {benchmark_correlation:.2f}。
                    建议使用 60-90 天滚动 β 做风险预算，β>1.2 时提高指数/国债权重；β<0.8 时可增配策略。
                    当前组合预估正收益概率约 {success_prob * 100:.1f}% ，相较单一资产更平滑。
                    """
                ).strip(),
                "success_probability": success_prob * 100 if success_prob is not None else None,
                "timeline": build_timeline("防御性对冲组合", success_prob),
            }
        )
    else:
        projection, success_prob = project_return(core_weight, 0.0)
        recommendations.append(
            {
                "title": "防御性现金垫层",
                "subtitle": "无可用基准时，以现金/短久期债券为缓冲区。",
                "allocation": [
                    f"{fmt_pct(core_weight)} 策略组合",
                    f"{fmt_pct(hedge_weight)} 现金、短久期债券或货币基金",
                ],
                "breakdown": breakdown(
                    [
                        (f"{params.ticker.upper()} 策略组合", core_weight),
                        ("现金 / 短久期债券", hedge_weight),
                    ]
                ),
                "projection": projection,
                "actions": textwrap.dedent(
                    f"""
                    建议保留至少 {fmt_pct(hedge_weight)} 的随时可用资金。当策略回撤超过 {fmt_pct(volatility)} 时，
                    采用分批补仓方式（例如每下降 5% 加仓 10%）提升均值回归收益，同时保留止损纪律。
                    """
                ).strip(),
                "success_probability": success_prob * 100 if success_prob is not None else None,
                "timeline": build_timeline("防御性现金垫层", success_prob),
            }
        )

    if risk == "aggressive" and sharpe > 1:
        overlay_weight = min(0.3, max(0.0, cagr * 1.5))
        effective_weight = min(0.9, core_weight + overlay_weight)
        projection, success_prob = project_return(effective_weight, hedge_weight)
        recommendations.append(
            {
                "title": "进取型增强策略",
                "subtitle": "适合具备衍生品经验的投资者，在波动可控时提升收益目标。",
                "allocation": [
                    f"{fmt_pct(effective_weight)} 策略组合或杠杆 ETF",
                    f"{fmt_pct(max(0.0, 1 - effective_weight))} 现金/保证金仓位",
                    "信号强劲时可使用股指期货、牛市价差或看涨期权做 1.2x~1.4x 杠杆",
                ],
                "breakdown": breakdown(
                    [
                        (f"{params.ticker.upper()} 策略组合/杠杆 ETF", effective_weight),
                        ("现金 / 保证金仓位", max(0.0, 1 - effective_weight)),
                    ]
                ),
                "projection": projection,
                "actions": textwrap.dedent(
                    f"""
                    建议设定两条风控线：一是策略滚动年化波动率超过 25% 或 VaR 超过 {fmt_pct(-stats.get('var_95', 0.0) or 0.0)} 时降杠杆；
                    二是 RSI 连续高于 70 且价量背离时锁定利润，将仓位回落至基础权重。
                    预估正收益概率约 {success_prob * 100:.1f}%。
                    """
                ).strip(),
                "success_probability": success_prob * 100 if success_prob is not None else None,
                "timeline": build_timeline("进取型增强策略", success_prob),
            }
        )
    else:
        projection, success_prob = project_return(core_weight * 0.8, hedge_weight)
        recommendations.append(
            {
                "title": "再平衡与风险预算",
                "subtitle": "通过趋势过滤与资产再分配稳健提升风险收益比。",
                "allocation": [
                    "核心策略：信号向上时保持 80% 仓位，信号疲弱或 RSI>70 时减至 40%-50%",
                    "防御资产：股指回撤或宏观数据恶化时，增配国债/黄金 ETF 各 10%",
                ],
                "breakdown": breakdown(
                    [
                        (f"{params.ticker.upper()} 策略组合（动态）", core_weight * 0.8),
                        ("现金 / 防御资产", max(0.0, 1 - core_weight * 0.8)),
                    ]
                ),
                "projection": projection,
                "actions": textwrap.dedent(
                    f"""
                    建议“日内不操作、周度检查、月度再平衡”，并设定胜率阈值（连续三周 <40% 暂停交易）。
                    当前方案预估正收益概率约 {success_prob * 100:.1f}% ，适合希望稳健增值的{risk_label}投资者。
                    """
                ).strip(),
                "success_probability": success_prob * 100 if success_prob is not None else None,
                "timeline": build_timeline("再平衡与风险预算", success_prob),
            }
        )

    return recommendations


def build_related_portfolios(
    params: StrategyInput, market_context: dict[str, Any], capital: float
) -> list[dict[str, Any]]:
    primary = params.ticker.upper()
    tickers = market_context.get("tickers", []) if market_context else []
    top_related = tickers[:4]
    related: list[dict[str, Any]] = []

    if top_related:
        allocation = 1 / len(top_related) if top_related else 0
        related.append(
            {
                "title": "行业主题增强组合",
                "description": (
                    f"根据最新资讯筛选出与 {primary} 同行业或供应链关系紧密的标的，"
                    "采用等权配置获取主题溢价，同时在组合层面分散单一公司的特有风险。"
                ),
                "tickers": top_related,
                "breakdown": [
                    {
                        "label": symbol,
                        "percent": format_percentage(allocation),
                        "amount": format_currency(capital * allocation) if capital else "—",
                    }
                    for symbol in top_related
                ],
                "rationale": "适合卫星仓位，观察行业共振趋势时逐步放大配置。",
            }
        )

    defensive_bundle = ["GLD", "TLT", "BIL"]
    related.append(
        {
            "title": "稳健对冲组合",
            "description": (
                "通过贵金属（GLD）、长期国债（TLT）与短期现金替代（BIL）构建的低相关性篮子，"
                "在系统性风险上升时可缓冲权益投资的波动，并提供流动性来源。"
            ),
            "tickers": defensive_bundle,
            "breakdown": [
                {
                    "label": "GLD 黄金 ETF",
                    "percent": "30.00%",
                    "amount": format_currency(capital * 0.3) if capital else "—",
                },
                {
                    "label": "TLT 20年期美债 ETF",
                    "percent": "40.00%",
                    "amount": format_currency(capital * 0.4) if capital else "—",
                },
                {
                    "label": "BIL 超短期国库券 ETF",
                    "percent": "30.00%",
                    "amount": format_currency(capital * 0.3) if capital else "—",
                },
            ],
            "rationale": "用于在市场突发回撤时快速对冲或作为现金管理池。",
        }
    )

    global_bundle = ["ACWI", "QQQ", "SPY"]
    related.append(
        {
            "title": "全球资产配置组合",
            "description": (
                "聚焦全球多元资产：ACWI 提供全球宽基敞口，QQQ 捕捉科技成长，SPY 代表美股核心仓位，"
                "适合作为策略之外的长期资产基石。"
            ),
            "tickers": global_bundle,
            "breakdown": [
                {
                    "label": "ACWI 全球宽基",
                    "percent": "40.00%",
                    "amount": format_currency(capital * 0.4) if capital else "—",
                },
                {
                    "label": "QQQ 纳斯达克100",
                    "percent": "30.00%",
                    "amount": format_currency(capital * 0.3) if capital else "—",
                },
                {
                    "label": "SPY 标普500",
                    "percent": "30.00%",
                    "amount": format_currency(capital * 0.3) if capital else "—",
                },
            ],
            "rationale": "用于建立核心长期仓位，与策略组合形成‘核心+卫星’配置框架。",
        }
    )

    return related


def build_statistical_baselines(prices: pd.DataFrame, params: StrategyInput) -> dict[str, Any]:
    """Generate ARIMA/VAR baselines to benchmark ML策略."""
    def _with_inferred_freq(series: pd.Series) -> pd.Series:
        if series.empty or not isinstance(series.index, pd.DatetimeIndex):
            return series
        freq = series.index.freq or pd.infer_freq(series.index)
        if not freq:
            freq = "B"
        try:
            series = series.asfreq(freq)
        except ValueError:
            series = series.asfreq("B")
        return series.ffill()

    closes = _with_inferred_freq(prices.get("adj close", pd.Series(dtype=float))).dropna()
    volumes = _with_inferred_freq(prices.get("volume", pd.Series(dtype=float))).dropna()
    baseline: dict[str, Any] = {"arima": None, "var": None, "diagnostics": []}

    if ARIMA is not None and closes.shape[0] >= 60:
        try:
            ideal_order = (1, 1, 1)
            model = ARIMA(closes, order=ideal_order)
            res = model.fit()
            forecast = res.forecast(steps=5)
            conf_int = res.get_forecast(steps=5).conf_int(alpha=0.1)
            baseline["arima"] = {
                "order": ideal_order,
                "aic": float(res.aic),
                "bic": float(res.bic),
                "forecast": [float(v) for v in forecast],
                "conf_int": conf_int.round(4).values.tolist(),
                "summary": f"ARIMA{ideal_order} AIC={res.aic:.1f}",
            }
        except Exception as exc:  # pragma: no cover - statsmodels optional
            baseline["diagnostics"].append(f"ARIMA 预测失败：{exc}")
    elif ARIMA is None:
        baseline["diagnostics"].append("缺少 statsmodels，无法生成 ARIMA 基线。")
    else:
        baseline["diagnostics"].append("数据量不足，ARIMA 需要至少 60 条有效价格。")

    if VAR is not None and closes.shape[0] >= 80 and volumes.shape[0] >= 80:
        try:
            aligned = pd.concat(
                [closes.pct_change().dropna().rename("return"), volumes.pct_change().dropna().rename("volume")],
                axis=1,
            ).dropna()
            model = VAR(aligned)
            res = model.fit(maxlags=4, ic="aic")
            forecast = res.forecast(aligned.values[-res.k_ar :], steps=5)
            baseline["var"] = {
                "lags": res.k_ar,
                "aic": float(res.aic),
                "forecast": forecast.tolist(),
                "summary": f"VAR(lag={res.k_ar}) 捕捉价格-量的互动",
            }
        except Exception as exc:  # pragma: no cover
            baseline["diagnostics"].append(f"VAR 拟合失败：{exc}")
    elif VAR is None:
        baseline["diagnostics"].append("缺少 statsmodels，无法生成 VAR 基线。")
    else:
        baseline["diagnostics"].append("数据量不足，VAR 需要至少 80 条有效样本。")

    return baseline


def _safe_get(info: dict[str, Any], *keys: str) -> Any:
    value: Any = info
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def build_factor_snapshot(params: StrategyInput) -> dict[str, Any]:
    fundamentals: dict[str, Any] = {}
    try:
        ticker = yf.Ticker(params.ticker)
        fundamentals = ticker.info or {}
    except Exception:
        fundamentals = {}

    factor_rows: list[dict[str, Any]] = []
    mappings = [
        ("市盈率 (PE)", ("trailingPE",)),
        ("预期PE", ("forwardPE",)),
        ("市净率 (PB)", ("priceToBook",)),
        ("净利率", ("profitMargins",)),
        ("ROE", ("returnOnEquity",)),
        ("收入同比增长", ("revenueGrowth",)),
        ("自由现金流/收入", ("freeCashflow",)),
        ("Beta", ("beta",)),
    ]
    for label, path in mappings:
        value = _safe_get(fundamentals, *path)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            if abs(value) < 1:
                formatted = f"{value*100:.2f}%"
            else:
                formatted = f"{value:.2f}"
        else:
            formatted = str(value)
        factor_rows.append({"label": label, "value": formatted})

    sector = fundamentals.get("sector") or ""
    industry = fundamentals.get("industry") or ""
    cap = fundamentals.get("marketCap")
    cap_display = f"{cap/1e9:.2f}B" if isinstance(cap, (int, float)) else "—"

    return {
        "meta": {
            "sector": sector,
            "industry": industry,
            "market_cap": cap_display,
            "summary": fundamentals.get("longBusinessSummary") or "",
        },
        "factors": factor_rows,
    }


def build_sentiment_snapshot(market_context: dict[str, Any]) -> dict[str, Any]:
    analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
    sentiments: list[dict[str, Any]] = []
    aggregate = 0.0
    if analyzer and market_context and market_context.get("news"):
        for item in market_context["news"]:
            title = item.get("title") or ""
            snippet = item.get("snippet") or ""
            joined = f"{title} {snippet}".strip()
            if not joined:
                continue
            score = analyzer.polarity_scores(joined)["compound"]
            aggregate += score
            sentiments.append(
                {
                    "title": title,
                    "score": score,
                    "label": "利好" if score > 0.1 else "利空" if score < -0.1 else "中性",
                    "url": item.get("url"),
                }
            )
    reason = ""
    avg_score = 0.0
    if sentiments:
        avg_score = aggregate / len(sentiments)
        if avg_score > 0.15:
            reason = "整体情绪偏正面，短期动能有望延续。"
        elif avg_score < -0.15:
            reason = "新闻舆情偏负面，需关注回撤风险。"
        else:
            reason = "情绪中性，可重点观察基本面与技术信号。"
    elif analyzer is None:
        reason = "缺少 vaderSentiment，推荐 pip install vaderSentiment 以启用情绪分析。"
    else:
        reason = "近期无足够新闻供情绪分析参考。"

    return {
        "available": bool(sentiments),
        "average": avg_score,
        "insight": reason,
        "items": sentiments[:6],
    }


def build_multimodal_bundle(
    params: StrategyInput,
    feature_dataset: pd.DataFrame | None,
    market_context: dict[str, Any],
    fundamentals_override: dict[str, Any] | None = None,
    macro_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = None
    if fundamentals_override:
        market_cap = fundamentals_override.get("marketCap")
        if isinstance(market_cap, (int, float)):
            cap_display = f"{market_cap/1e9:.2f}B"
        else:
            cap_display = market_cap
        meta = {
            "sector": fundamentals_override.get("sector"),
            "industry": fundamentals_override.get("industry"),
            "market_cap": cap_display or "—",
            "summary": fundamentals_override.get("summary", ""),
        }
        factor_map = {
            "利润率": fundamentals_override.get("profitMargins"),
            "营业利润率": fundamentals_override.get("operatingMargins"),
            "净资产收益率": fundamentals_override.get("returnOnEquity"),
            "资产回报率": fundamentals_override.get("returnOnAssets"),
            "收入增长": fundamentals_override.get("revenueGrowth"),
            "季度盈利增长": fundamentals_override.get("earningsQuarterlyGrowth"),
        }
        factors = [
            {"label": label, "value": f"{value*100:.2f}%"} for label, value in factor_map.items() if isinstance(value, (int, float))
        ]
        snapshot = {"meta": meta, "factors": factors}

    if not snapshot:
        snapshot = build_factor_snapshot(params)
    sentiment = build_sentiment_snapshot(market_context)

    technical_momentums: dict[str, Any] | None = None
    if feature_dataset is not None and not feature_dataset.empty:
        latest = feature_dataset.iloc[-1]
        technical_momentums = {
            "short_return": format_percentage(float(latest.get("return_5d", np.nan))),
            "mid_return": format_percentage(float(latest.get("return_21d", np.nan))),
            "momentum_short": float(latest.get("momentum_short", 0.0)),
            "momentum_long": float(latest.get("momentum_long", 0.0)),
            "rsi": float(latest.get("rsi", np.nan)),
        }

    return {
        "fundamentals": snapshot,
        "sentiment": sentiment,
        "technicals": technical_momentums,
        "macro": macro_bundle,
    }


def run_deep_signal_model(dataset: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any] | None:
    if dataset is None or dataset.empty or len(feature_columns) < 3:
        return None

    df = dataset.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns + ["target", "future_return"])
    if df.shape[0] < 160:
        return None

    values = df[feature_columns].astype(float).values
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0) + 1e-12
    normalized = (values - mean) / std

    y = df["target"].astype(float).values
    window = min(48, max(12, len(feature_columns)))
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    for idx in range(window, len(normalized)):
        seq = normalized[idx - window : idx]
        if np.isnan(seq).any():
            continue
        sequences.append(seq)
        labels.append(y[idx])

    if len(sequences) < 150:
        return None

    X_np = np.stack(sequences)
    y_np = np.array(labels)
    split = int(len(X_np) * 0.8)
    if split <= 0 or split >= len(X_np) - 10:
        return None

    X_train = X_np[:split]
    y_train = y_np[:split]
    X_test = X_np[split:]
    y_test = y_np[split:]

    reports: list[dict[str, Any]] = []

    if torch is not None and nn is not None:
        device = torch.device("cpu")

        def train_sequence_model(name: str, model: nn.Module, epochs: int = 20, lr: float = 0.003) -> dict[str, Any] | None:
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
            y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
            X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
            try:
                for epoch in range(epochs):
                    perm = torch.randperm(X_train_t.size(0))
                    X_train_epoch = X_train_t[perm]
                    y_train_epoch = y_train_t[perm]
                    for start in range(0, X_train_epoch.size(0), 32):
                        end = start + 32
                        xb = X_train_epoch[start:end]
                        yb = y_train_epoch[start:end]
                        optimizer.zero_grad()
                        pred = model(xb)
                        loss = criterion(pred, yb)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
                        optimizer.step()
                model.eval()
                with torch.no_grad():
                    probs = model(X_test_t).cpu().numpy().flatten()
                    preds = (probs >= 0.5).astype(int)
                    accuracy = float((preds == y_test.astype(int)).mean())
                    confidence = float(probs[-1]) if len(probs) else None
                return {
                    "name": name,
                    "accuracy": accuracy,
                    "confidence": confidence,
                    "sample": int(len(X_test)),
                }
            except Exception:
                return None

        class LSTMHead(nn.Module):
            def __init__(self, input_size: int):
                super().__init__()
                hidden = min(96, max(24, input_size * 2))
                self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden // 2, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                logits = self.head(last)
                return torch.sigmoid(logits)

        class TransformerHead(nn.Module):
            def __init__(self, input_size: int, seq_len: int):
                super().__init__()
                d_model = min(96, max(32, input_size * 2))
                self.seq_len = seq_len
                self.input_proj = nn.Linear(input_size, d_model)
                self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
                nhead = max(1, d_model // 16)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=0.2,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model // 2, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                seq_len = x.size(1)
                if seq_len < self.seq_len:
                    pad = self.seq_len - seq_len
                    x = torch.nn.functional.pad(x, (0, 0, pad, 0))
                    seq_len = self.seq_len
                elif seq_len > self.seq_len:
                    x = x[:, -self.seq_len :, :]
                    seq_len = self.seq_len
                h = self.input_proj(x) + self.pos_embed[:, :seq_len, :]
                enc = self.encoder(h)
                pooled = enc.mean(dim=1)
                logits = self.head(pooled)
                return torch.sigmoid(logits)

        lstm_model = LSTMHead(len(feature_columns)).to(device)
        lstm_report = train_sequence_model(
            name=f"LSTM (window={window})",
            model=lstm_model,
            epochs=22,
            lr=0.003,
        )
        if lstm_report:
            reports.append(lstm_report)

        transformer_model = TransformerHead(len(feature_columns), seq_len=window).to(device)
        transformer_report = train_sequence_model(
            name=f"Transformer (window={window})",
            model=transformer_model,
            epochs=18,
            lr=0.0025,
        )
        if transformer_report:
            reports.append(transformer_report)

    if MLPClassifier is not None:
        split_idx = int(len(df) * 0.8)
        if 0 < split_idx < len(df) - 10:
            X_train_df = df[feature_columns].iloc[:split_idx]
            X_test_df = df[feature_columns].iloc[split_idx:]
            y_train_df = df["target"].iloc[:split_idx]
            y_test_df = df["target"].iloc[split_idx:]

            scaler = StandardScaler() if StandardScaler else None
            if scaler is not None:
                X_train_df = pd.DataFrame(
                    scaler.fit_transform(X_train_df),
                    index=X_train_df.index,
                    columns=X_train_df.columns,
                )
                X_test_df = pd.DataFrame(
                    scaler.transform(X_test_df),
                    index=X_test_df.index,
                    columns=X_test_df.columns,
                )

            try:
                clf = MLPClassifier(
                    hidden_layer_sizes=(128, 64, 16),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=400,
                    random_state=42,
                )
                clf.fit(X_train_df, y_train_df)
                score = float(clf.score(X_test_df, y_test_df))
                proba = clf.predict_proba(X_test_df)[:, 1] if hasattr(clf, "predict_proba") else None
                confidence = float(proba[-1]) if proba is not None and len(proba) else None
                reports.append(
                    {
                        "name": "MLP Baseline",
                        "accuracy": score,
                        "confidence": confidence,
                        "sample": len(X_test_df),
                    }
                )
            except Exception:
                pass

    if not reports:
        return None

    reports_sorted = sorted(reports, key=lambda item: item.get("accuracy", 0.0), reverse=True)
    best = reports_sorted[0]
    return {
        "accuracy": best.get("accuracy"),
        "confidence": best.get("confidence"),
        "sample": best.get("sample"),
        "best_model": best.get("name"),
        "models": reports_sorted,
    }


def build_model_ensemble_view(
    statistical_bundle: dict[str, Any] | None,
    ml_stats: dict[str, Any],
    deep_bundle: dict[str, Any] | None,
    graph_bundle: dict[str, Any] | None = None,
    factor_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: list[str] = []
    confidence_score = 0.0
    anchors = []

    if statistical_bundle and statistical_bundle.get("arima"):
        stats = statistical_bundle["arima"]
        forecast_value = stats.get("forecast", [0])[0]
        summary.append(f"ARIMA 基线预测 5 日变动 {forecast_value:.2f}，用于短线对照。")
        confidence_score += 0.18
        anchors.append("统计基线")
    if deep_bundle:
        acc = deep_bundle.get("accuracy")
        note = deep_bundle.get("best_model", "深度信号")
        if acc is not None:
            summary.append(f"{note} 准确率 {acc*100:.1f}% ，样本 {deep_bundle.get('sample')}。")
            confidence_score += min(0.3, max(0.12, acc))
        else:
            summary.append(note)
            confidence_score += 0.12
        anchors.append("深度信号")
        model_details = deep_bundle.get("models")
        if model_details and len(model_details) > 1:
            best_second = model_details[1]
            summary.append(
                f"备选模型 {best_second['name']} 准确率 {best_second['accuracy']*100:.1f}%。"
            )
    if ml_stats:
        sharpe = ml_stats.get("sharpe", 0.0) or 0.0
        win_rate = format_percentage(ml_stats.get("hit_ratio", 0.0) or 0.0)
        summary.append(f"机器学习引擎夏普 {sharpe:.2f}，胜率 {win_rate}。")
        confidence_score += min(0.32, max(0.1, sharpe / 3))
        anchors.append("机器学习")

    if graph_bundle and graph_bundle.get("available"):
        density = graph_bundle["stats"].get("density", 0.0)
        risk_score = graph_bundle.get("risk_score")
        summary.append(f"图谱密度 {density:.3f}，网络风险评分 {risk_score:.2f}。")
        confidence_score += min(0.12, max(-0.1, 0.15 - density))
        anchors.append("图谱")

    if factor_bundle and factor_bundle.get("available"):
        composite = factor_bundle.get("composite", 0.0)
        summary.append(f"多因子综合 Z 分 {composite:.2f}，Top 信号：{', '.join(item['name'] for item in factor_bundle.get('top_factors', [])[:3])}.")
        confidence_score += min(0.18, max(0.0, abs(composite)))
        anchors.append("因子")

    blended_comment = "、".join(anchors) + " 多模型组合" if anchors else "组合策略"
    recommendation = (
        "多模型观点一致，可维持当前仓位并关注风险阈值。"
        if confidence_score >= 0.65
        else "模型观点分歧，建议降低仓位或等待信号共识。"
    )
    return {
        "summary": summary,
        "confidence": round(confidence_score, 2),
        "recommendation": recommendation,
        "title": blended_comment,
    }


def analyze_factor_effectiveness(
    dataset: pd.DataFrame | None,
    feature_columns: list[str] | None,
) -> dict[str, Any]:
    if dataset is None or dataset.empty or not feature_columns:
        return {"available": False, "message": "缺少特征矩阵，无法评估因子表现。"}

    df = dataset.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns + ["future_return"])
    if df.shape[0] < 120:
        return {"available": False, "message": "样本不足，无法计算因子统计。"}

    future_return = df["future_return"].astype(float)
    if future_return.std() == 0:
        return {"available": False, "message": "收益序列缺乏波动，无法计算因子表现。"}

    factors: list[dict[str, Any]] = []
    for col in feature_columns:
        series = df[col].astype(float)
        if series.nunique() < 10 or series.std() == 0:
            continue
        z = (series - series.mean()) / (series.std() + 1e-12)
        ic = float(np.corrcoef(z, future_return)[0, 1])
        ranks = z.rank(pct=True)
        high = future_return[ranks >= 0.8]
        low = future_return[ranks <= 0.2]
        long_short = float((high.mean() - low.mean()) * 100) if not high.empty and not low.empty else 0.0
        factors.append(
            {
                "name": col,
                "ic": round(ic, 3),
                "long_short": round(long_short, 3),
                "strength": round(abs(ic) + abs(long_short) / 100, 3),
            }
        )

    if not factors:
        return {"available": False, "message": "无有效因子信号。"}

    factors_sorted = sorted(factors, key=lambda item: item["strength"], reverse=True)
    composite = round(float(np.mean([item["ic"] for item in factors_sorted[:5]])), 3)
    return {
        "available": True,
        "factors": factors_sorted,
        "top_factors": factors_sorted[:5],
        "composite": composite,
        "message": "IC 与长短组合收益评估因子质量，可结合实盘进一步验证。",
    }


def build_knowledge_graph_bundle(
    params: StrategyInput,
    market_context: dict[str, Any],
    feature_dataset: pd.DataFrame | None,
) -> dict[str, Any]:
    ticker = params.ticker.upper()
    if not market_context or not market_context.get("news"):
        return {"available": False, "message": "近期缺少可用于构建图谱的资讯。"}
    if nx is None:
        return {"available": False, "message": "未安装 networkx，无法生成知识图谱。"}

    G = nx.Graph()
    G.add_node(ticker, type="ticker")

    sector = None
    industry = None
    try:
        info = yf.Ticker(params.ticker).info or {}
        sector = info.get("sector")
        industry = info.get("industry")
    except Exception:
        pass
    if sector:
        G.add_node(f"行业:{sector}", type="sector")
        G.add_edge(ticker, f"行业:{sector}", weight=1.0)
    if industry:
        G.add_node(f"子行业:{industry}", type="industry")
        G.add_edge(ticker, f"子行业:{industry}", weight=1.0)

    token_pattern = re.compile(r"\b[A-Z]{2,6}\b")
    for item in market_context.get("news", []):
        title = item.get("title") or ""
        snippet = item.get("snippet") or ""
        raw = f"{title} {snippet}"
        tokens = {tok for tok in token_pattern.findall(raw) if not tok.isdigit()}
        if not tokens:
            continue
        weight = 1.0 + float(item.get("score", 0) or 0)
        for tok in tokens:
            if tok not in G:
                G.add_node(tok, type="ticker")
            G.add_edge(ticker, tok, weight=G.get_edge_data(ticker, tok, {"weight": 0}).get("weight", 0) + weight)
        if len(tokens) > 1:
            listed = sorted(tokens)
            for i in range(len(listed) - 1):
                a, b = listed[i], listed[i + 1]
                if a == b:
                    continue
                G.add_edge(a, b, weight=G.get_edge_data(a, b, {"weight": 0}).get("weight", 0) + 0.5)

    if feature_dataset is not None and "momentum_long" in feature_dataset.columns:
        latest = feature_dataset.tail(1).iloc[0]
        momentum = float(latest.get("momentum_long", 0.0))
        G.nodes[ticker]["momentum"] = momentum

    centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(
        ((node, data, centrality.get(node, 0.0)) for node, data in G.nodes(data=True) if node != ticker),
        key=lambda item: item[2],
        reverse=True,
    )[:6]
    highlights = [
        {
            "node": node,
            "centrality": round(score, 3),
            "type": data.get("type", "ticker"),
        }
        for node, data, score in sorted_nodes
    ]
    stats = {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "density": round(nx.density(G), 4) if G.number_of_nodes() > 1 else 0.0,
    }
    centrality_values = [score for _, _, score in sorted_nodes] or [0.0]
    avg_centrality = float(np.mean(centrality_values))
    risk_score = round(float(avg_centrality + stats["density"]), 3)
    insight = (
        "网络集中度较高，需关注行业联动风险。"
        if risk_score > 1.2
        else "网络分散度适中，事件传导风险可控。"
    )
    return {
        "available": True,
        "stats": stats,
        "highlights": highlights,
        "risk_score": risk_score,
        "message": insight,
    }


def build_factor_scorecard(
    prices: pd.DataFrame,
    feature_dataset: pd.DataFrame | None,
    fundamentals: dict[str, Any] | None,
) -> dict[str, Any]:
    if prices is None or prices.empty:
        return {"available": False, "message": "缺少价格数据，无法生成因子得分。"}

    closes = prices.get("adj close", pd.Series(dtype=float)).dropna()
    if closes.shape[0] < 120:
        return {"available": False, "message": "样本不足，无法生成稳定因子。"}

    returns = closes.pct_change().dropna()
    momentum_21 = (1 + returns).rolling(21).apply(lambda x: np.prod(1 + x) - 1).dropna()
    momentum_score = float(momentum_21.tail(1).iloc[0]) if not momentum_21.empty else 0.0
    volatility = float(returns.tail(60).std() * np.sqrt(252)) if returns.shape[0] >= 60 else float(returns.std())
    drawdown_series = (1 + returns).cumprod()
    drawdown = float(calculate_max_drawdown(drawdown_series))

    fundamentals = fundamentals or {}
    factors_raw = [
        ("动量", momentum_score),
        ("波动率", -volatility),
        ("最大回撤", -abs(drawdown)),
    ]
    for label, key in [
        ("盈利能力 (ROE)", "returnOnEquity"),
        ("利润率", "profitMargins"),
        ("估值 (1/PE)", "trailingPE"),
    ]:
        value = fundamentals.get(key)
        if value is None:
            continue
        val = float(value)
        if label.startswith("估值"):
            val = 1 / val if val not in (0, np.nan) else 0.0
        factors_raw.append((label, val))

    if feature_dataset is not None and "volume_z" in feature_dataset.columns:
        liquidity = float(feature_dataset["volume_z"].tail(1).iloc[0])
        factors_raw.append(("成交活跃度", liquidity))

    scores = []
    values = np.array([item[1] for item in factors_raw], dtype=float)
    if values.size:
        mean = np.mean(values)
        std = np.std(values) + 1e-9
        zscores = (values - mean) / std
        for (label, raw), z in zip(factors_raw, zscores):
            scores.append({"factor": label, "raw": round(raw, 4), "score": round(float(z), 3)})

    composite = round(float(np.mean([entry["score"] for entry in scores])) if scores else 0.0, 3)
    return {
        "available": True,
        "scores": scores,
        "composite": composite,
        "message": "Z-score 标准化后的多因子得分，可用于参考仓位调整。",
    }


def compute_model_weights(
    statistical_bundle: dict[str, Any] | None,
    ml_stats: dict[str, Any] | None,
    deep_bundle: dict[str, Any] | None,
    knowledge_bundle: dict[str, Any] | None,
    factor_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    def _add(name: str, score: float, rationale: str) -> None:
        if score > 0:
            candidates.append({"name": name, "score": score, "rationale": rationale})

    if ml_stats:
        sharpe = ml_stats.get("sharpe", 0.0) or 0.0
        max_dd = abs(ml_stats.get("max_drawdown", 0.0) or 0.0)
        score = max(0.25, min(0.7, 0.45 + sharpe / 5 - max_dd))
        rationale = f"夏普 {sharpe:.2f}、最大回撤 {format_percentage(max_dd)}"
        _add("机器学习主策略", score, rationale)

    if deep_bundle and deep_bundle.get("accuracy") is not None:
        acc = float(deep_bundle["accuracy"])
        confidence = float(deep_bundle.get("confidence") or 0.5)
        score = max(0.1, min(0.35, acc * 0.8 + confidence * 0.2))
        best = deep_bundle.get("best_model", "深度模型")
        rationale = f"{best} 准确率 {acc*100:.1f}%、置信度 {confidence:.2f}"
        _add("深度信号", score, rationale)

    if statistical_bundle and statistical_bundle.get("arima"):
        arima = statistical_bundle["arima"]
        aic = float(arima.get("aic", 0.0))
        score = max(0.05, min(0.2, 0.2 - aic / 6000))
        rationale = f"ARIMA AIC {aic:.1f}，提供短线基准"
        _add("统计基线", score, rationale)

    if knowledge_bundle and knowledge_bundle.get("available"):
        risk_score = float(knowledge_bundle.get("risk_score", 0.0))
        adjustment = max(-0.15, min(0.1, 0.12 - risk_score / 4))
        rationale = f"网络风险评分 {risk_score:.2f}"
        if adjustment > 0:
            _add("图谱风控调节", adjustment, rationale)

    if factor_bundle and factor_bundle.get("available"):
        composite = float(factor_bundle.get("composite", 0.0))
        score = max(0.05, min(0.25, abs(composite)))
        top_names = ", ".join(item["name"] for item in factor_bundle.get("top_factors", [])[:3])
        direction = "正向" if composite >= 0 else "反向"
        rationale = f"{direction}信号 · Top 因子：{top_names}"
        _add("因子信号", score, rationale)

    if not candidates:
        return {"available": False, "allocations": [], "message": "缺少可用于分配权重的模型评分。"}

    total = sum(item["score"] for item in candidates)
    if total <= 0:
        return {"available": False, "allocations": [], "message": "模型评分为零，无法生成权重。"}

    allocations = [
        {
            "name": item["name"],
            "weight": round(item["score"] / total, 3),
            "rationale": item["rationale"],
        }
        for item in candidates
    ]
    return {"available": True, "allocations": allocations}


def build_risk_dashboard(
    stats: dict[str, Any],
    benchmark_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = []
    var_value = stats.get("var_95")
    cvar_value = stats.get("cvar_95")
    metrics.append(
        {
            "label": "最大回撤",
            "value": format_percentage(stats.get("max_drawdown", 0.0) or 0.0),
            "comment": "历史净值从高到低的最大跌幅，衡量情绪压力。",
        }
    )
    metrics.append(
        {
            "label": "波动率",
            "value": format_percentage(stats.get("volatility", 0.0) or 0.0),
            "comment": "年化标准差，反映收益的波动程度。",
        }
    )
    metrics.append(
        {
            "label": "夏普比",
            "value": f"{stats.get('sharpe', 0.0):.2f}",
            "comment": "单位风险获取的超额收益，>1 代表风险调整后表现良好。",
        }
    )
    if var_value is not None:
        metrics.append(
            {
                "label": "日度 VaR 95%",
                "value": format_percentage(-var_value),
                "comment": "在正常市场条件下，日内损失超过该值的概率约为 5%。",
            }
        )
    if cvar_value is not None:
        metrics.append(
            {
                "label": "日度 CVaR 95%",
                "value": format_percentage(-cvar_value),
                "comment": "在极端情况下的平均损失（最差 5% 状况的均值）。",
            }
        )
    if stats.get("annual_turnover") is not None:
        metrics.append(
            {
                "label": "年化换手率",
                "value": format_percentage(stats.get("annual_turnover", 0.0) or 0.0),
                "comment": "仓位调换频率，可用于评估交易成本敏感性。",
            }
        )
    benchmark_delta = None
    if benchmark_stats:
        benchmark_delta = float(stats.get("cagr", 0.0) or 0.0) - float(benchmark_stats.get("total_return", 0.0) or 0.0)
    risk_level = "高" if stats.get("max_drawdown", 0.0) <= -0.2 or stats.get("volatility", 0.0) >= 0.35 else "中"
    if stats.get("volatility", 0.0) <= 0.15 and (stats.get("max_drawdown", 0.0) or 0.0) > -0.1:
        risk_level = "低"
    insight = (
        "策略风险偏高，需严格控制仓位并结合对冲。"
        if risk_level == "高"
        else "策略风险处于可接受水平，建议保持纪律性调仓。"
    )
    return {
        "metrics": metrics,
        "risk_level": risk_level,
        "benchmark_delta": benchmark_delta,
        "insight": insight,
    }


def build_mlops_report(params: StrategyInput, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "auto_retrain": params.auto_apply_best_config,
        "train_window": params.train_window,
        "test_window": params.test_window,
        "last_trading_days": stats.get("trading_days"),
        "recent_sharpe": stats.get("recent_sharpe_60d"),
        "notes": "建议每个训练窗口或当滚动夏普 < 0 时重新训练模型，并记录指标漂移情况。",
    }


def build_scenario_simulation(
    backtest: pd.DataFrame,
    stats: dict[str, Any],
    horizon_days: int = 21,
    simulations: int = 400,
) -> dict[str, Any]:
    returns = backtest.get("strategy_return")
    if returns is None or returns.dropna().shape[0] < max(60, horizon_days):
        return {
            "available": False,
            "message": "历史样本不足，暂无法生成情景模拟。",
        }
    series = returns.dropna().astype(float)
    series = series[np.isfinite(series)]
    if series.empty:
        return {
            "available": False,
            "message": "收益序列无有效值。",
        }
    rng = np.random.default_rng(seed=42)
    sample = rng.choice(series.values, size=(simulations, horizon_days), replace=True)
    cumulative = np.prod(1 + sample, axis=1) - 1
    optimistic, base, defensive = np.quantile(cumulative, [0.85, 0.55, 0.15])
    expected = float(np.mean(cumulative))

    def scenario(label: str, value: float, description: str, allocation_hint: str) -> dict[str, Any]:
        return {
            "label": label,
            "return": format_percentage(float(value)),
            "description": description,
            "allocation_hint": allocation_hint,
        }

    volatility = stats.get("volatility", 0.0) or 0.0
    max_drawdown = stats.get("max_drawdown", 0.0) or 0.0
    scenarios = [
        scenario(
            "乐观冲刺",
            optimistic,
            "市场情绪向好且信号同步，适合分批加仓并预留止盈。",
            "考虑将盈利滚入进取型资产，同时保留 10% 现金缓冲。",
        ),
        scenario(
            "稳健基线",
            base,
            "沿用当前策略与仓位，按周再平衡即可。",
            "保持模型权重不变，关注交易成本与胜率。",
        ),
        scenario(
            "防守下行",
            defensive,
            "若市场震荡或突发利空，需提前设置回撤阈值。",
            "跌幅扩大时将仓位降至 40%-60%，并结合防御性组合。",
        ),
    ]
    return {
        "available": True,
        "horizon_days": horizon_days,
        "scenarios": scenarios,
        "volatility": format_percentage(volatility),
        "max_drawdown": format_percentage(max_drawdown),
        "expected_return": format_percentage(expected),
        "insight": (
            f"若维持历史节奏，未来 {horizon_days} 日平均回报约 {format_percentage(expected)}；"
            f"基线情形 {format_percentage(base)}，需重点防守回撤 {format_percentage(defensive)}。"
        ),
        "notes": "基于历史收益的自助抽样模拟，实际表现将受到宏观与流动性影响。",
    }


def build_opportunity_radar(
    params: StrategyInput,
    factor_effectiveness: dict[str, Any] | None,
    knowledge_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        page = screener.fetch_page(size=30)
    except Exception as exc:
        return {
            "available": False,
            "message": f"获取行情失败：{exc}",
        }
    rows = page.get("rows", []) if isinstance(page, dict) else []
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        price = row.get("price")
        change_pct = row.get("change_pct")
        if price is None or change_pct is None:
            continue
        name = row.get("name") or row.get("ticker") or "未命名"
        clean_rows.append(
            {
                "ticker": row.get("ticker"),
                "name": name,
                "price": float(price),
                "change_pct": float(change_pct),
            }
        )
    if not clean_rows:
        return {
            "available": False,
            "message": "暂无实时行情数据。",
        }
    leaders = sorted(clean_rows, key=lambda x: x["change_pct"], reverse=True)[:4]
    laggards = sorted(clean_rows, key=lambda x: x["change_pct"])[:4]
    steady = sorted(clean_rows, key=lambda x: abs(x["change_pct"]))[:4]
    factor_hint = ""
    if factor_effectiveness and factor_effectiveness.get("available"):
        top_factors = factor_effectiveness.get("top_factors", [])
        if top_factors:
            factor_hint = top_factors[0]["name"]
    knowledge_hint = ""
    if knowledge_bundle and knowledge_bundle.get("available"):
        highlights = knowledge_bundle.get("highlights") or []
        if highlights:
            knowledge_hint = highlights[0]["node"]
    summary = "结合胜率和因子信号，关注涨幅领先的高景气板块，同时在回调名单寻找低吸机会。"
    if factor_hint:
        summary += f" 当前主导因子：{factor_hint}。"
    if knowledge_hint:
        summary += f" 图谱热点：{knowledge_hint}。"
    return {
        "available": True,
        "leaders": leaders,
        "laggards": laggards,
        "steady": steady,
        "summary": summary,
        "insight": (
            f"涨幅榜关注：{', '.join(item['ticker'] for item in leaders[:3]) or '暂无'}；"
            f"回调区：{', '.join(item['ticker'] for item in laggards[:3]) or '暂无'}；"
            f"防守仓观察：{', '.join(item['ticker'] for item in steady[:2]) or '暂无'}。"
        ),
    }


def summarize_macro_highlight(macro_bundle: dict[str, Any]) -> str:
    if not macro_bundle:
        return "暂无宏观提示"
    insights: list[str] = []
    for entry in macro_bundle.values():
        if not isinstance(entry, dict):
            continue
        if not entry.get("available"):
            continue
        label = entry.get("short") or entry.get("label")
        trend = entry.get("trend")
        change_21 = entry.get("change_21d")
        if isinstance(change_21, (int, float)):
            change_text = f"{change_21:+.2f}%"
        else:
            change_text = "—"
        insights.append(f"{label}: {trend}（21日 {change_text}）")
    return "；".join(insights) if insights else "暂无宏观提示"


def build_executive_briefing(
    params: StrategyInput,
    ensemble_bundle: dict[str, Any] | None,
    model_weights: dict[str, Any] | None,
    risk_dashboard: dict[str, Any] | None,
    knowledge_bundle: dict[str, Any] | None,
    factor_effectiveness: dict[str, Any] | None,
    multimodal_bundle: dict[str, Any] | None,
    deep_bundle: dict[str, Any] | None,
    scenario_bundle: dict[str, Any] | None = None,
    opportunity_bundle: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []

    if ensemble_bundle:
        confidence = ensemble_bundle.get("confidence")
        status = f"{confidence:.2f}" if confidence is not None else "—"
        cards.append(
            {
                "title": "模型共识",
                "status": status,
                "body": (
                    "; ".join(ensemble_bundle.get("summary", [])[:2])
                    or "多模型评估完成，可查看详细信心来源。"
                ),
                "cta_label": "查看详情",
                "cta_href": "#advisor-pane",
            }
        )

    if model_weights and model_weights.get("available"):
        allocations = model_weights.get("allocations", [])
        top = allocations[0] if allocations else {}
        formatted = f"{top.get('name')} {top.get('weight', 0):.2f}" if top else "—"
        cards.append(
            {
                "title": "策略权重",
                "status": formatted,
                "body": allocations[0].get("rationale", "") if allocations else "暂无可用权重信息。",
                "cta_label": "组合建议",
                "cta_href": "#model-weights-title",
            }
        )

    if scenario_bundle and scenario_bundle.get("available"):
        scenarios = scenario_bundle.get("scenarios", [])
        base_return = scenarios[1]["return"] if len(scenarios) >= 2 else "—"
        optimistic = scenarios[0]["return"] if scenarios else "—"
        defensive = scenarios[-1]["return"] if scenarios else "—"
        cards.append(
            {
                "title": "情景模拟",
                "status": f"基线 {base_return}",
                "body": f"乐观 {optimistic} / 防守 {defensive}，配合仓位节奏执行。",
                "cta_label": "查看情景",
                "cta_href": "#scenario-board",
            }
        )

    if risk_dashboard:
        cards.append(
            {
                "title": "风险雷达",
                "status": risk_dashboard.get("risk_level", "—"),
                "body": (risk_dashboard.get("insight") or "")[:160],
                "cta_label": "风险指标",
                "cta_href": "#risk-dashboard-title",
            }
        )

    if knowledge_bundle and knowledge_bundle.get("available"):
        insight = knowledge_bundle.get("message", "")
        cards.append(
            {
                "title": "知识图谱",
                "status": f"风险 {knowledge_bundle.get('risk_score', 0):.2f}",
                "body": insight[:160],
                "cta_label": "图谱洞察",
                "cta_href": "#knowledge-graph-title",
            }
        )

    if factor_effectiveness and factor_effectiveness.get("available"):
        composite = factor_effectiveness.get("composite")
        cards.append(
            {
                "title": "因子信号",
                "status": f"IC {composite:.2f}" if composite is not None else "—",
                "body": factor_effectiveness.get("message", "")[:160],
                "cta_label": "查看因子",
                "cta_href": "#factor-effectiveness-title",
            }
        )

    if multimodal_bundle:
        sentiment = multimodal_bundle.get("sentiment", {})
        sentiment_avg = sentiment.get("average")
        macro_text = ""
        macro_bundle = multimodal_bundle.get("macro") or {}
        if isinstance(macro_bundle, dict):
            available = [entry for entry in macro_bundle.values() if isinstance(entry, dict) and entry.get("available")]
            if available:
                first = available[0]
                macro_text = f"；宏观：{first.get('label')} {first.get('trend')}"
        body_text = (sentiment.get("insight", "") or "情绪中性")[:160]
        if macro_text:
            body_text = f"{body_text}{macro_text}"
        cards.append(
            {
                "title": "多模态情绪",
                "status": f"{sentiment_avg:.2f}" if sentiment_avg is not None else "—",
                "body": body_text,
                "cta_label": "市场信息",
                "cta_href": "#macro-dashboard-title",
            }
        )

    if deep_bundle:
        status = deep_bundle.get("best_model", "深度信号")
        cards.append(
            {
                "title": "深度评估",
                "status": status,
                "body": f"准确率 {deep_bundle.get('accuracy', 0.0)*100:.1f}% ，样本 {deep_bundle.get('sample', '-')}",
                "cta_label": "AI 研判",
                "cta_href": "#ai-panel",
            }
        )

    if opportunity_bundle and opportunity_bundle.get("available"):
        leaders = opportunity_bundle.get("leaders", [])
        highlight = ", ".join(item["ticker"] for item in leaders[:2]) if leaders else "暂无"
        cards.append(
            {
                "title": "机会雷达",
                "status": highlight,
                "body": opportunity_bundle.get("summary", "")[:160],
                "cta_label": "机会详情",
                "cta_href": "#opportunity-radar",
            }
        )

    # Remove potential duplicates if same href
    unique_cards: dict[str, dict[str, Any]] = {}
    for card in cards:
        key = card.get("title")
        if key and key not in unique_cards:
            unique_cards[key] = card
    return list(unique_cards.values())


def build_user_questions(
    stats: dict[str, Any],
    recommendations: list[dict[str, Any]],
    risk_dashboard: dict[str, Any] | None,
    model_weights: dict[str, Any] | None,
    ensemble_bundle: dict[str, Any] | None,
    scenario_bundle: dict[str, Any] | None,
    opportunity_bundle: dict[str, Any] | None,
) -> list[dict[str, str]]:
    answers: list[dict[str, str]] = []

    cagr = stats.get("cagr", 0.0) or 0.0
    sharpe = stats.get("sharpe", 0.0) or 0.0
    answers.append(
        {
            "question": "能赚多少？",
            "answer": (
                f"历史回测显示年化复合收益率 {format_percentage(cagr)}，夏普比 {sharpe:.2f}。"
                " 以过去表现估算，10 万本金若维持同样节奏，一年期望收益约 "
                f"{format_currency(100000 * (1 + cagr))}（不含交易费用与滑点）。"
            ),
        }
    )

    if recommendations:
        first_plan = recommendations[0]
        answers.append(
            {
                "question": "怎么投才能赚？",
                "answer": (
                    f"金融顾问建议：{first_plan['title']}。"
                    f" 核心执行：{first_plan['actions']}"
                ),
            }
        )

    risk_text = "风险等级：中，维持纪律性调仓。"
    if risk_dashboard:
        risk_text = f"风险等级：{risk_dashboard.get('risk_level', '—')}。{risk_dashboard.get('insight', '')}"
    answers.append(
        {
            "question": "风险如何控制？",
            "answer": risk_text,
        }
    )

    if model_weights and model_weights.get("available"):
        weight_desc = "; ".join(
            f"{item['name']} {item['weight']:.2f}"
            for item in model_weights.get("allocations", [])[:3]
        )
        answers.append(
            {
                "question": "有没有组合投资建议？",
                "answer": (
                    f"根据模型评分，推荐的信号组合权重：{weight_desc}。"
                    " 可在组合策略页查看详细分配。"
                ),
            }
        )

    if ensemble_bundle:
        answers.append(
            {
                "question": "模型观点一致吗？",
                "answer": (
                    f"综合置信度 {ensemble_bundle.get('confidence', 0.0):.2f}，"
                    "整体观点：" + "；".join(ensemble_bundle.get("summary", [])[:2])
                ),
            }
        )

    if scenario_bundle and scenario_bundle.get("available"):
        scenarios = scenario_bundle.get("scenarios", [])
        base = scenarios[1]["return"] if len(scenarios) >= 2 else "—"
        defensive = scenarios[-1]["return"] if scenarios else "—"
        answers.append(
            {
                "question": "最坏情况会怎样？",
                "answer": (
                    f"情景模拟显示：基线 {base}，若出现突发回撤约为 {defensive}。"
                    " 建议提前设定回撤阈值并结合防御资产调仓。"
                ),
            }
        )

    if opportunity_bundle and opportunity_bundle.get("available"):
        leaders = opportunity_bundle.get("leaders", [])
        highlight = ", ".join(item["ticker"] for item in leaders[:3]) if leaders else "暂无"
        answers.append(
            {
                "question": "市场上有什么机会？",
                "answer": (
                    f"机会雷达聚焦：{highlight}，可结合策略权重逐步建仓。"
                    " 同时关注回调名单寻找低风险切入点。"
                ),
            }
        )

    return answers


def build_advisor_playbook(
    stats: dict[str, Any],
    user_guidance: dict[str, Any],
    recommendations: list[dict[str, Any]],
    scenario_bundle: dict[str, Any] | None,
    risk_dashboard: dict[str, Any] | None,
    opportunity_bundle: dict[str, Any] | None,
    macro_highlight: str | None,
) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []

    def shorten(text: str, width: int = 92) -> str:
        return textwrap.shorten(text, width=width, placeholder="…")

    def add_section(title: str, points: list[str], tag: str | None = None) -> None:
        clean = [point for point in points if point]
        if not clean:
            return
        entry: dict[str, Any] = {"title": title, "points": clean[:4]}
        if tag:
            entry["tag"] = tag
        sections.append(entry)

    cagr = stats.get("cagr", 0.0) or 0.0
    total_return = stats.get("total_return", 0.0) or 0.0
    sharpe = stats.get("sharpe", 0.0) or 0.0
    profit_points = [
        f"历史年化 {format_percentage(cagr)} · 累计收益 {format_percentage(total_return)}",
        f"风险调整后夏普比 {sharpe:.2f}",
    ]
    if scenario_bundle and scenario_bundle.get("available"):
        scenarios = scenario_bundle.get("scenarios", [])
        horizon = scenario_bundle.get("horizon_days")
        if scenarios:
            base_entry = scenarios[1] if len(scenarios) >= 2 else scenarios[0]
            defensive_entry = scenarios[-1]
            base = base_entry.get("return", "—")
            defensive = defensive_entry.get("return", "—")
        else:
            base = "—"
            defensive = "—"
        profit_points.append(
            f"未来 {horizon} 日基线 {base}，防守情形约 {defensive}"
        )
        insight = scenario_bundle.get("insight")
        if insight:
            profit_points.append(shorten(insight))
    add_section("盈利空间", profit_points, tag="收益概览")

    action_points: list[str] = []
    if recommendations:
        first_plan = recommendations[0]
        action_points.append(shorten(f"{first_plan['title']}：{first_plan['actions']}"))
    for step in (user_guidance.get("action_plan") or [])[:2]:
        action_points.append(shorten(f"{step['title']}：{step['detail']}"))
    add_section("操作路线", action_points, tag="执行计划")

    risk_points: list[str] = []
    risk_alerts = user_guidance.get("risk_alerts") or []
    risk_points.extend(risk_alerts[:2])
    if risk_dashboard:
        risk_points.append(
            f"风险等级：{risk_dashboard.get('risk_level', '—')} · {shorten(risk_dashboard.get('insight', ''))}"
        )
        metrics = risk_dashboard.get("metrics") or []
        if metrics:
            first_metric = metrics[0]
            risk_points.append(
                shorten(f"{first_metric['label']}：{first_metric['value']}（{first_metric['comment']}）")
            )
    add_section("风险防守", risk_points, tag="风控")

    opportunity_points: list[str] = []
    if opportunity_bundle and opportunity_bundle.get("available"):
        opportunity_points.append(shorten(opportunity_bundle.get("summary", "")))
        opportunity_points.append(shorten(opportunity_bundle.get("insight", "")))
    if macro_highlight:
        opportunity_points.append(shorten(f"宏观提示：{macro_highlight}"))
    interest_keywords = user_guidance.get("primary_goal_label")
    if interest_keywords:
        opportunity_points.append(shorten(f"目标定位：{interest_keywords}"))
    add_section("机会与提醒", opportunity_points, tag="市场观察")

    return {
        "available": bool(sections),
        "sections": sections,
    }


def _compute_horizon_statistics(backtest: pd.DataFrame, window: int) -> dict[str, Any] | None:
    if backtest.empty or backtest.shape[0] < max(5, window):
        return None
    window_data = backtest.tail(window)
    if window_data.empty:
        return None
    returns = window_data.get("strategy_return")
    if returns is None or returns.empty:
        return None
    cumulative = (1 + returns).cumprod()
    horizon_return = float(cumulative.iloc[-1] - 1)
    horizon_vol = float(returns.std() * np.sqrt(252)) if returns.std() is not None else 0.0
    horizon_sharpe = float(np.sqrt(252) * returns.mean() / (returns.std() + 1e-12))
    hit_ratio = float((returns > 0).mean()) if not returns.empty else 0.0
    latest_prob = float(window_data.get("probability", pd.Series(dtype=float)).dropna().iloc[-1]) if "probability" in window_data.columns and not window_data["probability"].dropna().empty else None
    confidence = normal_cdf(horizon_return / (horizon_vol + 1e-9)) if horizon_vol else 0.5
    drawdown = float(calculate_max_drawdown(cumulative)) if not cumulative.empty else 0.0
    return {
        "window": window,
        "return": horizon_return,
        "volatility": horizon_vol,
        "sharpe": horizon_sharpe,
        "drawdown": drawdown,
        "hit_ratio": hit_ratio,
        "latest_probability": latest_prob,
        "confidence": confidence,
    }


def build_flagship_research_bundle(
    params: StrategyInput,
    prices: pd.DataFrame,
    backtest: pd.DataFrame,
    stats: dict[str, Any],
    benchmark_stats: dict[str, Any] | None,
    market_context: dict[str, Any],
    combo_details: list[dict[str, Any]],
) -> dict[str, Any]:
    horizons_spec = [
        ("短线（日内~5日）", 5),
        ("中线（20~60日）", 60),
        ("长线（120~252日）", 252),
    ]
    horizon_rows: list[dict[str, Any]] = []
    for label, window in horizons_spec:
        metrics = _compute_horizon_statistics(backtest, window)
        if not metrics:
            continue
        horizon_rows.append(
            {
                "label": label,
                "window": window,
                "return": format_percentage(metrics["return"]),
                "volatility": format_percentage(metrics["volatility"]),
                "sharpe": f"{metrics['sharpe']:.2f}",
                "drawdown": format_percentage(metrics["drawdown"]),
                "hit_ratio": format_percentage(metrics["hit_ratio"]),
                "confidence": f"{metrics['confidence'] * 100:.1f}%",
                "latest_probability": metrics["latest_probability"],
            }
        )

    data_sources = []
    if not prices.empty:
        coverage = f"{prices.index.min().date()} → {prices.index.max().date()}"
        data_sources.append(
            {
                "name": "行情数据 (OHLCV)",
                "type": "结构化",
                "status": "已接入",
                "coverage": coverage,
                "notes": "由 yfinance 提供的日线级行情，可扩展至分钟级。",
            }
        )
    data_sources.extend(
        [
            {
                "name": "技术指标/特征库",
                "type": "特征工程",
                "status": "已构建",
                "coverage": f"当前使用 {stats.get('feature_count', 0)} 个核心特征",
                "notes": "支持指标扩展与复用型 Feature Store。",
            },
            {
                "name": "宏观与财务因子",
                "type": "结构化",
                "status": "集成中",
                "notes": "预留接入 FRED/财报 API，需进行频率对齐与因子暴露计算。",
            },
            {
                "name": "新闻/公告/社交媒体",
                "type": "非结构化",
                "status": "试运行" if market_context.get("news") else "待接入",
                "notes": "利用 DuckDuckGo/自建爬虫抽取情绪指标，当前结果用于资讯摘要。",
            },
            {
                "name": "金融知识图谱",
                "type": "关联数据",
                "status": "规划中",
                "notes": "准备将公司-行业-事件关系写入图数据库，结合 GNN 捕捉联动。",
            },
        ]
    )

    feature_columns = list(stats.get("feature_columns") or [])
    feature_library: dict[str, list[str]] = {
        "技术指标": [],
        "量价结构": [],
        "基本面/风格": [],
        "NLP/事件": [],
    }
    for col in feature_columns:
        name = col.lower()
        if any(tag in name for tag in ["sma", "ema", "rsi", "macd", "boll", "momentum"]):
            feature_library["技术指标"].append(col)
        elif any(tag in name for tag in ["volume", "turnover", "volatility", "vwap"]):
            feature_library["量价结构"].append(col)
        elif any(tag in name for tag in ["pe", "roe", "growth", "value", "quality"]):
            feature_library["基本面/风格"].append(col)
        elif any(tag in name for tag in ["sentiment", "news", "event", "text"]):
            feature_library["NLP/事件"].append(col)
        else:
            feature_library.setdefault("其他", []).append(col)

    model_stack: list[dict[str, Any]] = []
    primary_engine = params.strategy_engine
    if primary_engine == "ml_momentum":
        model_stack.append(
            {
                "layer": "短线动量",
                "models": ["Gradient Boosting / LightGBM", "概率校准 (Isotonic)", "阈值优化"],
                "objective": "预测未来 1-5 日方向与幅度，输出概率作为风控阈值。",
                "status": "在线",
            }
        )
        model_stack.append(
            {
                "layer": "中线 / 因子",
                "models": ["传统因子打分", "rolling Sharpe"],
                "objective": "提供组合锚点与风格暴露，辅助多策略集成。",
                "status": "在研",
            }
        )
    elif primary_engine == "multi_combo":
        for entry in combo_details:
            model_stack.append(
                {
                    "layer": entry["engine"],
                    "models": ["核心回测引擎"],
                    "objective": "与主策略并行计算，Stacking 组合信号。",
                    "status": "在线",
                }
            )
    else:
        model_stack.append(
            {
                "layer": "规则基线",
                "models": ["均线交叉", "波动率目标"],
                "objective": "提供稳健基线，便于评估机器学习增益。",
                "status": "在线",
            }
        )

    uncertainty = {
        "volatility": format_percentage(stats.get("volatility", 0.0)),
        "value_at_risk": format_percentage(-(stats.get("var_95", 0.0) or 0.0)),
        "conditional_var": format_percentage(-(stats.get("cvar_95", 0.0) or 0.0)),
        "recent_sharpe": f"{stats.get('recent_sharpe_60d', 0.0):.2f}"
        if "recent_sharpe_60d" in stats
        else "—",
        "notes": "VaR/CVaR 基于历史收益估计，后续将联动贝叶斯置信区间与 MC Dropout。",
    }

    training_days = stats.get("trading_days", len(backtest))
    mlops = {
        "sample_size": training_days,
        "train_window": params.train_window,
        "test_window": params.test_window,
        "embargo_days": params.embargo_days,
        "auto_retrain": params.auto_apply_best_config,
        "comment": "数据/模型版本已经持久化，支持 walk-forward 再训练与灰度验证。",
    }

    risk_controls = {
        "entry_threshold": params.entry_threshold,
        "exit_threshold": params.exit_threshold,
        "vol_target": params.volatility_target,
        "max_leverage": params.max_leverage,
        "stop_suggestion": "建议设定动态止损（最大回撤 × 0.6）与分级减仓阈值。",
    }

    knowledge_graph = {
        "status": "规划中",
        "next": "接入供应链/事件知识图谱，使用 GNN 建模联动风险与主题轮动。",
        "market_links": market_context.get("tickers", [])[:6],
    }

    benchmark = {
        "vol": format_percentage(benchmark_stats.get("volatility", 0.0)) if benchmark_stats else None,
        "sharpe": f"{benchmark_stats.get('sharpe', 0.0):.2f}" if benchmark_stats else None,
        "alpha": format_percentage(benchmark_stats.get("alpha", 0.0)) if benchmark_stats else None,
    }

    return {
        "multi_horizon": horizon_rows,
        "data_sources": data_sources,
        "feature_library": feature_library,
        "model_stack": model_stack,
        "uncertainty": uncertainty,
        "mlops": mlops,
        "risk_controls": risk_controls,
        "knowledge_graph": knowledge_graph,
        "benchmark_snapshot": benchmark,
    }


def build_key_takeaways(
    stats: dict[str, Any],
    benchmark_stats: Optional[dict[str, Any]],
    params: StrategyInput,
) -> list[str]:
    takeaways: list[str] = []
    risk_label = RISK_PROFILE_LABELS.get(params.risk_profile, params.risk_profile)
    takeaways.append(
        f"策略年化复合收益率为 {format_percentage(stats.get('cagr', 0.0) or 0.0)}，"
        f"夏普比率 {stats.get('sharpe', 0.0):.2f}。"
    )
    takeaways.append(
        f"最大回撤 {format_percentage(stats.get('max_drawdown', 0.0) or 0.0)}，平均杠杆 {stats.get('avg_leverage', 0.0):.2f}x，"
        "建议配合风控阈值执行仓位管理。"
    )
    if stats.get("prediction_accuracy") is not None:
        auc = stats.get("auc")
        auc_text = f"，ROC-AUC {auc:.2f}" if isinstance(auc, (int, float)) and not math.isnan(auc) else ""
        takeaways.append(
            f"模型方向预测胜率 {format_percentage(stats.get('prediction_accuracy', 0.0) or 0.0)}{auc_text}。"
        )
    turnover = stats.get("annual_turnover")
    if turnover is not None:
        takeaways.append(
            f"年化换手率约 {format_percentage(turnover or 0.0)}，平均持仓 {stats.get('average_holding_days', 0.0):.1f} 天，"
            f"交易成本占收益比 {format_percentage(stats.get('cost_ratio', 0.0) or 0.0)}。"
        )
    if benchmark_stats:
        takeaways.append(
            f"相对基准 {params.benchmark_ticker.upper()} 的 α 为 {format_percentage(benchmark_stats.get('alpha', 0.0) or 0.0)}，"
            f"β 为 {benchmark_stats.get('beta', 0.0):.2f}，信息比率 {benchmark_stats.get('info_ratio', 0.0):.2f}。"
        )
    else:
        takeaways.append("无可用基准数据，可考虑添加指数或行业 ETF 作为对照与对冲。")
    if params.capital:
        takeaways.append(
            f"以可支配资金 {format_currency(params.capital)} 计，若按核心配置执行，"
            f"一年期的期望收益约为 {format_currency(params.capital * (1 + stats.get('cagr', 0.0) or 0.0))}。"
        )
    takeaways.append(
        f"基于风险偏好（{risk_label}）建议组合中保留 {format_percentage(1 - stats.get('avg_exposure', 0.0) or 0.0)} 的缓冲资产以应对极端情况。"
    )
    return takeaways


def estimate_confidence(stats: dict[str, Any]) -> tuple[str, float]:
    sharpe = stats.get("sharpe", 0.0) or 0.0
    max_drawdown = abs(stats.get("max_drawdown", 0.0) or 0.0)
    cagr = stats.get("cagr", 0.0) or 0.0
    score = 0.0
    if sharpe >= 1.2:
        score += 0.4
    elif sharpe >= 0.8:
        score += 0.25
    elif sharpe >= 0.5:
        score += 0.15
    if cagr >= 0.15:
        score += 0.3
    elif cagr >= 0.08:
        score += 0.2
    elif cagr >= 0.04:
        score += 0.1
    if max_drawdown <= 0.15:
        score += 0.3
    elif max_drawdown <= 0.25:
        score += 0.2
    elif max_drawdown <= 0.35:
        score += 0.1
    score = max(0.0, min(score, 0.95))
    if score >= 0.7:
        label = "高"
    elif score >= 0.45:
        label = "中"
    else:
        label = "低"
    return label, score


def build_user_guidance(
    stats: dict[str, Any],
    benchmark_stats: Optional[dict[str, Any]],
    params: StrategyInput,
) -> dict[str, Any]:
    horizon_labels = {
        "short": "短期（0-6 个月）",
        "medium": "中期（6-24 个月）",
        "long": "长期（24 个月以上）",
    }
    experience_labels = {
        "novice": "新手",
        "intermediate": "进阶投资者",
        "advanced": "专业投资者",
    }
    goal_messages = {
        "growth": "以净值增长为主，侧重追踪趋势机会并控制回撤。",
        "income": "关注稳定现金流，可搭配分红/债券资产平滑收益。",
        "preserve": "以资金安全为先，策略信号用于增强收益但优先守住本金。",
    }

    cagr = stats.get("cagr", 0.0) or 0.0
    total_return = stats.get("total_return", 0.0) or 0.0
    max_drawdown = stats.get("max_drawdown", 0.0) or 0.0
    sharpe = stats.get("sharpe", 0.0) or 0.0
    volatility = stats.get("volatility", 0.0) or 0.0
    capital = params.capital or 0.0

    horizon_label = horizon_labels.get(params.investment_horizon, "中期（6-24 个月）")
    experience_label = experience_labels.get(params.experience_level, "新手")
    goal_text = goal_messages.get(params.primary_goal, goal_messages["growth"])

    benchmark_alpha = None
    if benchmark_stats:
        benchmark_alpha = benchmark_stats.get("alpha", 0.0)

    quick_summary: list[str] = []
    quick_summary.append(
        f"策略在回测期的累计收益约 {format_percentage(total_return)}，年化复合收益率 {format_percentage(cagr)}。"
    )
    quick_summary.append(
        f"最大回撤 {format_percentage(max_drawdown)}，夏普比 {sharpe:.2f}，波动率 {format_percentage(volatility)}。"
    )
    if benchmark_alpha is not None:
        quick_summary.append(
            f"相对基准的年化超额收益（α）约 {format_percentage(benchmark_alpha)}。"
        )
    quick_summary.append(f"目标定位：{goal_text} 建议持有周期：{horizon_label}。")

    action_plan: list[dict[str, str]] = []
    capital_text = (
        f"当前可支配资金 {format_currency(capital)}，建议保留 10%-20% 的现金备用。"
        if capital
        else "未填写资金规模，可结合自身预算分配仓位。"
    )
    action_plan.append(
        {
            "title": "准备阶段（本周内）",
            "detail": (
                f"{capital_text} 完成账户/券商准备，确认手续费与交易规则。"
                " 对照策略参数生成观察清单，并将基准指数纳入监控。"
            ),
            "priority": "高",
        }
    )
    action_plan.append(
        {
            "title": "执行阶段（持仓运行）",
            "detail": (
                f"依据模型信号和再平衡节奏：当短均线高于长均线且信号概率 >55% 时建仓，"
                f"回撤达到 {format_percentage(max_drawdown or 0.0)} 或 RSI>70 时分批降仓。"
                " 每周查看胜率与净值曲线，必要时将盈利锁定到现金或防御资产。"
            ),
            "priority": "高",
        }
    )
    action_plan.append(
        {
            "title": "复盘与升级（每月/季度）",
            "detail": (
                "记录实际收益与计划差异，关注夏普比和换手率是否符合预期；"
                "结合宏观/行业变化，适时调整关注关键词与对冲资产。"
            ),
            "priority": "中",
        }
    )

    risk_alerts: list[str] = []
    if max_drawdown <= -0.2:
        risk_alerts.append("历史最大回撤超过 20%，建仓时务必控制仓位并设置止损。")
    if volatility >= 0.3:
        risk_alerts.append("策略波动率偏高，适合少量试仓或搭配债券/货币基金分散风险。")
    if stats.get("cost_ratio", 0.0) and stats.get("cost_ratio", 0.0) > 0.25:
        risk_alerts.append("交易成本占收益比例较高，需关注滑点与换手率，避免频繁调仓。")
    recent_sharpe = stats.get("recent_sharpe_60d", 0.0) or 0.0
    if recent_sharpe < 0:
        risk_alerts.append("近 60 日滚动夏普为负，建议降仓或启用对冲，并在下个训练窗口重训模型。")
    if not risk_alerts:
        risk_alerts.append("请持续关注市场突发事件，必要时降低杠杆或暂停交易。")

    education_tips: list[str] = []
    if params.experience_level == "novice":
        education_tips.append("夏普比衡量单位风险获取的收益，>1 代表风险调整后表现健康。")
        education_tips.append("最大回撤表示净值从高点跌到低点的幅度，是评估情绪压力的重要指标。")
        education_tips.append("建议先使用模拟账户熟悉信号，再逐步投入真实资金。")
    elif params.experience_level == "intermediate":
        education_tips.append("定期跟踪策略在不同市场环境下的收益分布，评估是否需要多策略组合。")
        education_tips.append("通过滚动窗口重新训练模型，可避免过拟合特定时间段。")
    else:
        education_tips.append("可考虑将策略纳入多因子框架，与价值/质量因子组合，提升稳健性。")
        education_tips.append("建议监控信息比率与跟踪误差，衡量相对基准的超额收益质量。")

    confidence_label, confidence_score = estimate_confidence(stats)

    disclaimer = (
        "所有建议基于历史回测结果，不构成投资承诺。实际交易需结合个人风险承受能力、"
        "资金规划与市场环境，设置止损并谨慎使用杠杆。"
    )

    return {
        "quick_summary": quick_summary,
        "action_plan": action_plan,
        "risk_alerts": risk_alerts,
        "education_tips": education_tips,
        "confidence_label": confidence_label,
        "confidence_score": round(confidence_score, 3),
        "experience_label": experience_label,
        "investment_horizon_label": horizon_label,
        "primary_goal_label": goal_text,
        "disclaimer": disclaimer,
    }
