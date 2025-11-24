from __future__ import annotations

from typing import Any, Optional
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates

plt.switch_backend("Agg")

from .config import StrategyInput
from .metrics import fig_to_base64
from .risk import calculate_drawdown_series


def _format_chart_time(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, pd.DatetimeIndex)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if hasattr(value, "to_pydatetime"):
        try:
            return pd.Timestamp(value.to_pydatetime()).strftime("%Y-%m-%d")
        except Exception:
            pass
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def build_interactive_chart_payload(
    prices: pd.DataFrame, backtest: pd.DataFrame, params: StrategyInput, *, max_points: int = 1500
) -> dict[str, Any]:
    if prices.empty or backtest.empty:
        return {}
    display_prices = prices.sort_index().tail(max_points).copy()
    candle_cols = ["open", "high", "low", "close"]
    for col in candle_cols:
        if col not in display_prices.columns:
            display_prices[col] = display_prices.get("adj close")
    candles: list[dict[str, float | str]] = []
    for idx, row in display_prices.iterrows():
        try:
            open_val = float(row["open"])
            high_val = float(row["high"])
            low_val = float(row["low"])
            close_val = float(row["close"])
        except (TypeError, ValueError):
            continue
        if any(math.isnan(val) for val in (open_val, high_val, low_val, close_val)):
            continue
        candles.append(
            {
                "time": _format_chart_time(idx),
                "open": round(open_val, 4),
                "high": round(high_val, 4),
                "low": round(low_val, 4),
                "close": round(close_val, 4),
            }
        )

    def _series_from(column: str, precision: int = 4) -> list[dict[str, float | str]]:
        if column not in display_prices.columns:
            return []
        series: list[dict[str, float | str]] = []
        for idx, value in display_prices[column].dropna().items():
            try:
                num_val = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(num_val):
                continue
            series.append({"time": _format_chart_time(idx), "value": round(num_val, precision)})
        return series

    rsi_series = _series_from("rsi", precision=2)
    sma_short_series = _series_from("sma_short")
    sma_long_series = _series_from("sma_long")

    # Trade markers
    signals: list[dict[str, Any]] = []
    aligned_backtest = backtest.sort_index().loc[display_prices.index.min() : display_prices.index.max()]

    def _rounded(value: Any, precision: int = 4) -> float | None:
        try:
            num_val = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(num_val):
            return None
        return round(num_val, precision)

    def _grab(source: Any, key: str) -> Any:
        if source is None:
            return None
        if isinstance(source, dict):
            return source.get(key)
        try:
            return source.get(key)
        except AttributeError:
            try:
                return source[key]
            except Exception:
                return None

    for idx, row in aligned_backtest.iterrows():
        signal_value = row.get("signal")
        if signal_value is None:
            continue
        try:
            signal_int = int(signal_value)
        except (TypeError, ValueError):
            continue
        if signal_int == 0:
            continue
        trade_type = "buy" if signal_int > 0 else "sell"
        try:
            price_val = round(float(row.get("adj close", float("nan"))), 4)
        except (TypeError, ValueError):
            price_val = None
        try:
            cum_ret = round(float(row.get("cum_strategy", 0.0)), 4)
        except (TypeError, ValueError):
            cum_ret = None
        price_row = None
        try:
            price_row = display_prices.loc[idx]
        except KeyError:
            price_row = None
        context: dict[str, Any] = {}
        rsi_val = _rounded(_grab(price_row, "rsi") or _grab(row, "rsi"), 2)
        if rsi_val is not None:
            context["rsi"] = rsi_val
        sma_short_val = _rounded(_grab(price_row, "sma_short") or _grab(row, "sma_short"), 2)
        if sma_short_val is not None:
            context["sma_short"] = sma_short_val
        sma_long_val = _rounded(_grab(price_row, "sma_long") or _grab(row, "sma_long"), 2)
        if sma_long_val is not None:
            context["sma_long"] = sma_long_val
        probability_val = _rounded(_grab(row, "probability"), 3)
        if probability_val is not None:
            context["probability"] = probability_val
        leverage_val = _rounded(_grab(row, "leverage"), 2)
        if leverage_val is not None:
            context["leverage"] = leverage_val
        exposure_val = _rounded(_grab(row, "exposure"), 2)
        if exposure_val is not None:
            context["exposure"] = exposure_val
        strategy_ret_val = _rounded(_grab(row, "strategy_return"), 4)
        if strategy_ret_val is not None:
            context["strategy_return"] = strategy_ret_val
        position_val = _rounded(_grab(row, "position"), 2)
        if position_val is not None:
            context["position"] = position_val

        signal_entry = {
            "time": _format_chart_time(idx),
            "type": trade_type,
            "price": price_val,
            "position": row.get("position"),
            "cum_return": cum_ret,
            "daily_return": round(float(row.get("strategy_return", 0.0)), 4)
            if not pd.isna(row.get("strategy_return"))
            else None,
        }
        if context:
            signal_entry["context"] = context
        signals.append(signal_entry)

    if not candles:
        return {}
    return {
        "symbol": params.ticker.upper(),
        "candles": candles,
        "sma_short": sma_short_series,
        "sma_long": sma_long_series,
        "rsi": rsi_series,
        "signals": signals,
        "short_window": params.short_window,
        "long_window": params.long_window,
    }


def generate_charts(
    prices: pd.DataFrame,
    backtest: pd.DataFrame,
    benchmark_series: Optional[pd.DataFrame],
    params: StrategyInput,
) -> list[dict[str, str]]:
    charts: list[dict[str, str]] = []
    if prices.empty or backtest.empty:
        return charts

    date_formatter = mdates.DateFormatter("%Y-%m")
    benchmark_label = params.benchmark_ticker.upper() if params.benchmark_ticker else "基准"

    # 价格 + 均线 + RSI
    fig, (ax_price, ax_rsi) = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_price.plot(prices.index, prices["adj close"], color="#2563eb", label=f"{params.ticker.upper()} 收盘价")
    ax_price.plot(prices.index, prices["sma_short"], color="#10b981", label=f"短期均线({params.short_window})")
    ax_price.plot(prices.index, prices["sma_long"], color="#f59e0b", label=f"长期均线({params.long_window})")
    ax_price.set_ylabel("价格")
    ax_price.set_title("价格与均线信号")
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.2)

    ax_rsi.plot(prices.index, prices["rsi"], color="#7c3aed")
    ax_rsi.axhline(70, color="#ef4444", linestyle="--", linewidth=1, label="超买阈值 70")
    ax_rsi.axhline(30, color="#0ea5e9", linestyle="--", linewidth=1, label="超卖阈值 30")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.xaxis.set_major_formatter(date_formatter)
    ax_rsi.legend(loc="upper left")
    ax_rsi.grid(alpha=0.2)
    charts.append(
        {
            "title": "价格与技术指标",
            "img": fig_to_base64(fig),
        }
    )

    # 策略净值 vs 买入持有 vs 基准
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(backtest.index, backtest["cum_strategy"], label="策略净值", color="#2563eb")
    ax2.plot(backtest.index, backtest["cum_buy_hold"], label="买入持有", color="#9ca3af")
    if benchmark_series is not None and "benchmark_cum" in benchmark_series:
        ax2.plot(
            benchmark_series.index,
            benchmark_series["benchmark_cum"],
            label=f"基准净值（{benchmark_label}）",
            color="#f97316",
        )
    ax2.set_title("净值曲线对比")
    ax2.set_ylabel("累计收益倍数")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_formatter(date_formatter)
    charts.append({"title": "净值曲线对比", "img": fig_to_base64(fig2)})

    # 回撤曲线
    fig3, ax3 = plt.subplots(figsize=(11, 3.5))
    strategy_drawdown = calculate_drawdown_series(backtest["cum_strategy"])
    ax3.fill_between(strategy_drawdown.index, strategy_drawdown, color="#ef4444", alpha=0.35, step="pre", label="策略回撤")
    ax3.plot(strategy_drawdown.index, strategy_drawdown, color="#b91c1c", linewidth=1)
    if benchmark_series is not None and "benchmark_cum" in benchmark_series:
        benchmark_drawdown = calculate_drawdown_series(benchmark_series["benchmark_cum"])
        ax3.plot(
            benchmark_drawdown.index,
            benchmark_drawdown,
            color="#f59e0b",
            linewidth=1,
            linestyle="--",
            label="基准回撤",
        )
    ax3.set_title("最大回撤跟踪")
    ax3.set_ylabel("回撤比例")
    ax3.grid(alpha=0.2)
    ax3.legend(loc="lower left")
    ax3.xaxis.set_major_formatter(date_formatter)
    charts.append({"title": "回撤分析", "img": fig_to_base64(fig3)})

    # 日度收益分布
    fig4, ax4 = plt.subplots(figsize=(11, 4))
    ax4.hist(
        backtest["strategy_return"].dropna(),
        bins=50,
        alpha=0.6,
        label="策略日收益",
        color="#2563eb",
    )
    ax4.hist(
        backtest["adj close"].pct_change().dropna(),
        bins=50,
        alpha=0.4,
        label="标的日收益（买入持有）",
        color="#9ca3af",
    )
    if benchmark_series is not None:
        ax4.hist(
            benchmark_series["benchmark_return"].dropna(),
            bins=50,
            alpha=0.4,
            label="基准日收益",
            color="#f97316",
        )
    ax4.set_title("日度收益分布（风险对比）")
    ax4.set_xlabel("日收益率")
    ax4.set_ylabel("频次")
    ax4.legend(loc="upper right")
    ax4.grid(alpha=0.2)
    charts.append({"title": "收益分布", "img": fig_to_base64(fig4)})

    if "probability" in backtest.columns:
        fig5, ax5 = plt.subplots(figsize=(11, 3.5))
        ax5.plot(
            backtest.index,
            backtest["probability"].clip(0, 1),
            color="#0ea5e9",
            label="模型多头概率",
        )
        ax5.fill_between(
            backtest.index,
            backtest["probability"].clip(0, 1),
            color="#0ea5e9",
            alpha=0.15,
        )
        ax5.axhline(0.5, color="#9ca3af", linewidth=1, linestyle="--")
        ax5.set_ylim(0, 1)
        ax5.set_title("模型信号强度（多头概率）")
        ax5.set_ylabel("P(上涨)")
        ax5.grid(alpha=0.2)
        ax5.xaxis.set_major_formatter(date_formatter)
        ax5.legend(loc="upper left")
        charts.append({"title": "信号强度", "img": fig_to_base64(fig5)})

    # 60日滚动夏普与波动
    try:
        rwin = 60
        ret = backtest["strategy_return"].fillna(0)
        roll_mean = ret.rolling(rwin).mean()
        roll_std = ret.rolling(rwin).std()
        roll_sharpe = (roll_mean / roll_std).replace([np.inf, -np.inf], np.nan)
        roll_vol = roll_std * np.sqrt(252)
        fig6, ax6 = plt.subplots(figsize=(11, 3.5))
        ax6.plot(ret.index, roll_sharpe, color="#0f766e", label=f"滚动夏普({rwin}日)")
        ax6.set_ylabel("Sharpe")
        ax6_t = ax6.twinx()
        ax6_t.plot(ret.index, roll_vol, color="#a855f7", alpha=0.6, label=f"滚动波动({rwin}日)")
        ax6_t.set_ylabel("年化波动")
        ax6.set_title("滚动夏普与波动")
        ax6.grid(alpha=0.2)
        ax6.xaxis.set_major_formatter(date_formatter)
        charts.append({"title": "滚动夏普与波动", "img": fig_to_base64(fig6)})
    except Exception:
        pass

    # 60日滚动β（若有基准）
    if benchmark_series is not None and "benchmark_return" in benchmark_series:
        try:
            win = 60
            tmp = backtest[["strategy_return"]].join(benchmark_series[["benchmark_return"]], how="inner").dropna()
            cov = tmp["strategy_return"].rolling(win).cov(tmp["benchmark_return"])
            var = tmp["benchmark_return"].rolling(win).var()
            roll_beta = (cov / var).replace([np.inf, -np.inf], np.nan)
            fig7, ax7 = plt.subplots(figsize=(11, 3.5))
            ax7.plot(roll_beta.index, roll_beta, color="#f59e0b")
            ax7.axhline(1.0, color="#9ca3af", linestyle="--", linewidth=1)
            ax7.set_title(f"滚动β（{win}日）")
            ax7.grid(alpha=0.2)
            ax7.xaxis.set_major_formatter(date_formatter)
            charts.append({"title": "滚动β", "img": fig_to_base64(fig7)})
        except Exception:
            pass

    # 概率校准/提升图（仅 ML 策略可用）
    if "probability" in backtest.columns:
        try:
            outcome = (backtest["asset_return"].shift(-1) > 0).astype(int)
            dfc = pd.DataFrame({"p": backtest["probability"].clip(0, 1), "y": outcome}).dropna()
            bins = np.linspace(0, 1, 11)
            dfc["bin"] = np.digitize(dfc["p"], bins) - 1
            calib = dfc.groupby("bin").agg(p_mean=("p", "mean"), y_rate=("y", "mean"))
            lift = (
                dfc.assign(q=pd.qcut(dfc["p"], 10, duplicates="drop"))
                .groupby("q", observed=False)
                .agg(r=("y", "mean"))
            )
            fig8, ax8 = plt.subplots(1, 2, figsize=(12, 4))
            ax8[0].plot([0, 1], [0, 1], "--", color="#9ca3af")
            ax8[0].plot(calib["p_mean"], calib["y_rate"], marker="o", color="#2563eb")
            ax8[0].set_title("概率校准（可靠性图）")
            ax8[0].set_xlabel("预测概率")
            ax8[0].set_ylabel("实际上涨率")
            ax8[1].bar(range(len(lift)), lift["r"], color="#10b981")
            ax8[1].set_title("分位提升（高概率组应更高胜率）")
            ax8[1].set_xlabel("概率分位(低→高)")
            ax8[1].set_ylabel("上涨率")
            fig8.tight_layout()
            charts.append({"title": "校准与提升", "img": fig_to_base64(fig8)})
        except Exception:
            pass

    # 曝露/杠杆与换手
    try:
        expo = (backtest["position"] * backtest["leverage"]).fillna(0.0)
        turnover = expo.diff().abs().fillna(expo.abs())
        fig9, ax9 = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax9[0].plot(backtest.index, expo, color="#0ea5e9", label="曝露=仓位×杠杆")
        ax9[0].axhline(0, color="#9ca3af", linewidth=1)
        ax9[0].set_title("曝露与换手")
        ax9[0].legend(loc="upper left")
        ax9[0].grid(alpha=0.2)
        ax9[1].bar(backtest.index, turnover, color="#94a3b8")
        ax9[1].set_ylabel("|Δ曝露|")
        ax9[1].grid(alpha=0.2)
        ax9[1].xaxis.set_major_formatter(date_formatter)
        charts.append({"title": "曝露与换手", "img": fig_to_base64(fig9)})
    except Exception:
        pass

    # 未来情景预测（乐观/中性/悲观）
    try:
        hist = backtest["strategy_return"].dropna().tail(252)
        if not hist.empty:
            mu = float(hist.mean())
            sigma = float(hist.std())
            z = 0.84  # 约 80% 分位
            # 若有模型概率，用其对期望收益做自适应调制
            if "probability" in backtest.columns:
                p0 = float(backtest["probability"].dropna().iloc[-1])
                adj = (p0 - 0.5) * sigma * 2  # 信号越强，越偏离均值
            else:
                adj = 0.0
            r_mid = mu + adj
            r_opt = r_mid + z * sigma
            r_pes = r_mid - z * sigma
            horizon = 60  # 未来 60 个交易日
            last_date = backtest.index[-1]
            future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="B")
            base = float(backtest["cum_strategy"].iloc[-1])
            mid_path = base * np.cumprod(np.ones(horizon) * (1 + r_mid))
            opt_path = base * np.cumprod(np.ones(horizon) * (1 + r_opt))
            pes_path = base * np.cumprod(np.ones(horizon) * (1 + r_pes))

            # 画图：最近 60 日历史 + 未来乐观/悲观
            fig10, ax10 = plt.subplots(figsize=(11, 4))
            hist_idx = backtest.index[-60:]
            ax10.plot(hist_idx, backtest.loc[hist_idx, "cum_strategy"], color="#2563eb", label="历史净值")
            ax10.plot(future_idx, mid_path, color="#f59e0b", label="中性情形")
            ax10.plot(future_idx, opt_path, color="#10b981", label="乐观情形")
            ax10.plot(future_idx, pes_path, color="#ef4444", label="悲观情形")
            ax10.fill_between(future_idx, pes_path, opt_path, color="#e5e7eb", alpha=0.6, label="区间")
            ax10.set_title("未来 60 日情景预测（乐观/悲观）")
            ax10.set_ylabel("净值倍数")
            ax10.grid(alpha=0.2)
            ax10.legend(loc="upper left")
            ax10.xaxis.set_major_formatter(date_formatter)
            charts.append({"title": "情景预测（60日）", "img": fig_to_base64(fig10)})
    except Exception:
        pass

    return charts


__all__ = [
    "build_interactive_chart_payload",
    "generate_charts",
]
