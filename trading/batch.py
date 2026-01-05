from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Iterable, Sequence
import json
import logging

import pandas as pd

from .strategies import (
    StrategyInput,
    run_quant_pipeline,
    DATA_CACHE_DIR,
    calculate_sharpe,
    calculate_max_drawdown,
    calculate_cagr,
    calculate_sortino,
)
from .portfolio import combine, portfolio_stats

LOGGER = logging.getLogger(__name__)


def run_batch_backtests(
    tickers: Iterable[str],
    base_params: StrategyInput,
    engines: Sequence[str] | None = None,
    aggregate: bool = True,
    aggregate_scheme: str = "equal",
) -> pd.DataFrame:
    """
    批量运行回测，返回结果 DataFrame 并自动保存到 data_cache/batch_reports 下。

    Parameters
    ----------
    tickers : Iterable[str]
        需要回测的标的列表。
    base_params : StrategyInput
        基础参数（会为每个标的复制一份）。其中 ticker/strategy_engine 会被覆盖。
    engines : Sequence[str], optional
        要运行的策略引擎列表，默认仅运行 base_params.strategy_engine。

    Returns
    -------
    pd.DataFrame
        每个标的 / 策略的核心指标表。
    """

    tickers = [symbol.upper() for symbol in tickers]
    engines = list(engines) if engines else [base_params.strategy_engine]
    rows: list[dict[str, object]] = []

    pnl_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        for engine in engines:
            params = replace(base_params, ticker=ticker, strategy_engine=engine)
            try:
                result = run_quant_pipeline(params)
            except Exception as exc:
                message = str(exc).replace("\n", " ").strip()
                if len(message) > 200:
                    message = f"{message[:197]}..."
                LOGGER.exception("Batch backtest failed for %s/%s", ticker, engine)
                rows.append(
                    {
                        "ticker": ticker,
                        "engine": engine,
                        "start_date": str(base_params.start_date),
                        "end_date": str(base_params.end_date),
                        "total_return": None,
                        "cagr": None,
                        "sharpe": None,
                        "max_drawdown": None,
                        "volatility": None,
                        "confidence_score": None,
                        "confidence_label": None,
                        "error": message or "batch_backtest_failed",
                    }
                )
                continue
            stats = result.get("stats", {})
            pnl_series = pd.Series(dtype=float)
            return_rows = result.get("return_series")
            if not isinstance(return_rows, list) or not return_rows:
                return_rows = result.get("recent_rows")
            if isinstance(return_rows, list) and return_rows:
                try:
                    df_recent = pd.DataFrame(return_rows)
                    series_col = "daily_return" if "daily_return" in df_recent else "strategy_return"
                    if series_col in df_recent:
                        pnl_series = pd.Series(df_recent[series_col].astype(float).values, index=range(len(df_recent)))
                except Exception:
                    pnl_series = pd.Series(dtype=float)
            pnl_map[f"{ticker}-{engine}"] = pnl_series
            rows.append(
                {
                    "ticker": ticker,
                    "engine": engine,
                    "start_date": result.get("start_date"),
                    "end_date": result.get("end_date"),
                    "total_return": stats.get("total_return"),
                    "cagr": stats.get("cagr"),
                    "sharpe": stats.get("sharpe"),
                    "max_drawdown": stats.get("max_drawdown"),
                    "volatility": stats.get("volatility"),
                    "confidence_score": result.get("confidence_score"),
                    "confidence_label": result.get("confidence_label"),
                }
            )

    df = pd.DataFrame(rows)
    if aggregate and pnl_map:
        combined, weights = combine(pnl_map, scheme=aggregate_scheme)
        ann_factor = 252
        cumulative = (1 + combined.fillna(0.0)).cumprod() if not combined.empty else pd.Series(dtype=float)
        combo_total_return = float(cumulative.iloc[-1] - 1) if not cumulative.empty else None
        combo_sharpe = calculate_sharpe(combined.fillna(0.0), trading_days=ann_factor) if not combined.empty else None
        combo_sortino = calculate_sortino(combined.fillna(0.0), trading_days=ann_factor) if not combined.empty else None
        combo_max_dd = calculate_max_drawdown(cumulative) if not cumulative.empty else None
        combo_tail = portfolio_stats(combined.fillna(0.0))
        df.loc[len(df)] = {
            "ticker": "PORT",
            "engine": aggregate_scheme,
            "start_date": str(base_params.start_date),
            "end_date": str(base_params.end_date),
            "total_return": combo_total_return,
            "cagr": calculate_cagr(cumulative, trading_days=ann_factor) if not cumulative.empty else None,
            "sharpe": combo_sharpe,
            "max_drawdown": combo_max_dd,
            "volatility": float(combined.std() * (252 ** 0.5)) if not combined.empty else None,
            "confidence_score": None,
            "confidence_label": json.dumps(weights, ensure_ascii=False),
            "sortino": combo_sortino,
            "cvar_95": combo_tail.get("cvar_95") if combo_tail else None,
            "recovery_days": combo_tail.get("recovery_days") if combo_tail else None,
            "loss_streak": combo_tail.get("loss_streak") if combo_tail else None,
        }
    # 持久化权重与组合风险-供前端展示
    port_meta = {
        "weights": weights if aggregate and pnl_map else {},
        "combo_stats": combo_tail if aggregate and pnl_map else {},
    }
    reports_dir = DATA_CACHE_DIR / "batch_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    filename = reports_dir / f"batch_report_{datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    meta_path = filename.with_suffix(".json")
    try:
        meta_path.write_text(json.dumps(port_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return df
