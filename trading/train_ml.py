from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import json
from typing import Any, Iterable

import pandas as pd

from .strategies import StrategyInput, run_quant_pipeline, DATA_CACHE_DIR


def available_engines() -> list[str]:
    engines = ["sk_gbdt", "lightgbm", "catboost"]
    return engines


def engine_param_grid(engine: str) -> list[dict[str, Any]]:
    if engine == "lightgbm":
        return [
            {"learning_rate": 0.05, "n_estimators": 400, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.9},
            {"learning_rate": 0.03, "n_estimators": 600, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.9},
        ]
    if engine == "catboost":
        return [
            {"learning_rate": 0.05, "iterations": 500, "depth": 6, "subsample": 0.8},
            {"learning_rate": 0.03, "iterations": 700, "depth": 6, "subsample": 0.8},
        ]
    if engine == "sk_gbdt":
        return [
            {"learning_rate": 0.05, "n_estimators": 300, "max_depth": 3, "subsample": 0.8},
            {"learning_rate": 0.03, "n_estimators": 500, "max_depth": 3, "subsample": 0.8},
        ]
    return [{}]


def score_stats(stats: dict[str, Any]) -> float:
    sharpe = float(stats.get("sharpe") or 0.0)
    cagr = float(stats.get("cagr") or 0.0)
    mdd = float(stats.get("max_drawdown") or 0.0)
    return sharpe + 0.5 * cagr - 0.2 * abs(mdd)


def run_engine_benchmark(
    tickers: Iterable[str], start_date: date, end_date: date, base: StrategyInput, engines: list[str] | None = None
) -> dict[str, Any]:
    tickers = [t.upper() for t in tickers]
    engines = engines or available_engines()

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for engine in engines:
        for params in engine_param_grid(engine):
            perfs = []
            for ticker in tickers:
                si = replace(
                    base,
                    ticker=ticker,
                    strategy_engine="ml_momentum",
                    ml_model=engine,
                    ml_params=params,
                )
                res = run_quant_pipeline(si)
                stats = res.get("stats", {})
                perfs.append(
                    {
                        "ticker": ticker,
                        "engine": engine,
                        "params": params,
                        "sharpe": stats.get("sharpe"),
                        "cagr": stats.get("cagr"),
                        "max_drawdown": stats.get("max_drawdown"),
                        "score": score_stats(stats),
                    }
                )
            df = pd.DataFrame(perfs)
            agg = {
                "engine": engine,
                "params": params,
                "mean_sharpe": float(df["sharpe"].mean()),
                "mean_cagr": float(df["cagr"].mean()),
                "mean_mdd": float(df["max_drawdown"].mean()),
                "score": float(df["score"].mean()),
            }
            rows.append(agg)
            if best is None or agg["score"] > best["score"]:
                best = agg

    results = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "tickers": tickers,
        "candidates": rows,
        "best": best,
    }
    out_dir = DATA_CACHE_DIR / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "report.csv", index=False)
    with (out_dir / "best_ml_config.json").open("w", encoding="utf-8") as fh:
        json.dump(best, fh, ensure_ascii=False, indent=2)
    return results


def main() -> None:
    end = date.today()
    start = end - timedelta(days=365 * 5)
    base = StrategyInput(
        ticker="AAPL",
        benchmark_ticker="SPY",
        start_date=start,
        end_date=end,
        short_window=20,
        long_window=60,
        rsi_period=14,
        include_plots=False,
        show_ai_thoughts=False,
        risk_profile="balanced",
        capital=100000.0,
        volatility_target=0.15,
        transaction_cost_bps=8.0,
        slippage_bps=5.0,
        min_holding_days=3,
        optimize_thresholds=True,
        val_ratio=0.15,
        embargo_days=5,
        ml_model="sk_gbdt",
    )
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "SPY", "QQQ"]
    engines = ["lightgbm", "catboost", "sk_gbdt"]
    results = run_engine_benchmark(tickers, start, end, base, engines)
    print("Best:", results.get("best"))


if __name__ == "__main__":
    main()

