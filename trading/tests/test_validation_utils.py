from __future__ import annotations

from datetime import datetime, timedelta
from dataclasses import replace

import pandas as pd

from trading.validation import (
    build_purged_kfold_schedule,
    build_walk_forward_report,
    compute_tail_risk_summary,
)
from trading import strategies
from trading.strategies import (
    StrategyInput,
    _generate_validation_report,
    _tb_summary_from_dataset,
    _compute_asset_returns,
    build_core_metrics,
)


def _sample_returns(days: int = 400) -> pd.Series:
    idx = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(days)]
    # simple sine wave to ensure both gains/losses
    values = pd.Series([0.001 * ((i % 10) - 5) for i in range(days)], index=idx)
    return values


def test_walk_forward_report_generates_windows():
    report = build_walk_forward_report(_sample_returns(), window=120, step=60)
    assert report, "should produce walk-forward slices"
    assert "test_cagr" in report[0]


def test_tail_risk_summary_contains_var():
    summary = compute_tail_risk_summary(_sample_returns())
    assert "value_at_risk" in summary
    assert summary["confidence"] == 0.95


def test_purged_kfold_produces_schedule():
    idx = _sample_returns().index
    schedule = build_purged_kfold_schedule(idx, n_splits=4, embargo=3)
    assert schedule
    assert schedule[0]["train_size"] > schedule[0]["test_size"]


def test_generate_validation_report_penalized_sharpe():
    days = 260
    idx = pd.date_range("2023-01-01", periods=days, freq="B")
    df = pd.DataFrame(
        {
            "f1": [0.1 * (i % 5) for i in range(days)],
            "f2": [(-1) ** i * 0.02 for i in range(days)],
            "target": [0, 1] * (days // 2),
            "future_return": [0.0015 if i % 7 else -0.003 for i in range(days)],
        },
        index=idx,
    )
    params = StrategyInput(
        ticker="TEST",
        benchmark_ticker="SPY",
        start_date=idx[0].date(),
        end_date=idx[-1].date(),
        short_window=20,
        long_window=60,
        rsi_period=14,
        include_plots=False,
        show_ai_thoughts=False,
        risk_profile="balanced",
        capital=100000.0,
        validation_slices=3,
        train_window=120,
        test_window=20,
        embargo_days=5,
        entry_threshold=0.55,
        exit_threshold=0.45,
    )
    report = _generate_validation_report(df, ["f1", "f2"], params)
    assert report is not None
    assert report["summary"]["sharpe"]["mean"] is not None
    assert report.get("penalized_sharpe") is not None


def test_build_core_metrics_surfaces_oos_summary():
    stats = {
        "total_return": 0.12,
        "buy_hold_return": 0.08,
        "sharpe": 1.4,
        "max_drawdown": -0.1,
        "sortino": 1.2,
        "volatility": 0.2,
        "cagr": 0.11,
        "calmar": 1.1,
        "hit_ratio": 0.55,
        "avg_gain": 0.01,
        "avg_loss": -0.008,
        "avg_exposure": 0.6,
        "avg_leverage": 1.2,
        "var_95": -0.03,
        "cvar_95": -0.04,
        "twr_days": 15,
        "loss_streak": 4,
        "annual_turnover": 1.5,
        "average_holding_days": 6.0,
        "cost_ratio": 0.02,
        "trading_days": 250,
        "validation_summary_compact": {
            "sharpe": {"mean": 1.1, "std": 0.3, "iqr": 0.2},
        },
    }
    metrics = build_core_metrics(stats)
    labels = [m["label"] for m in metrics]
    assert any("OOS夏普" in label for label in labels), "should surface OOS PFWS summary in metrics"


def test_tb_summary_from_dataset():
    idx = pd.date_range("2023-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "tb_up_active": [0.01, 0.015, 0.02, 0.018, 0.017],
            "tb_down_active": [0.012, 0.013, 0.011, 0.014, 0.012],
        },
        index=idx,
    )
    summary = _tb_summary_from_dataset(df)
    assert "up" in summary and "down" in summary
    assert abs(summary["up"]["mean"] - 0.016) < 1e-6
    assert abs(summary["down"]["max"] - 0.014) < 1e-6


def test_compute_asset_returns_respects_return_path():
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    frame = pd.DataFrame(
        {
            "adj close": [100.0, 110.0, 121.0],
            "open": [101.0, 111.0, 122.0],
        },
        index=idx,
    )
    params_cc = StrategyInput(
        ticker="T",
        benchmark_ticker="SPY",
        start_date=idx[0].date(),
        end_date=idx[-1].date(),
        short_window=2,
        long_window=3,
        rsi_period=2,
        include_plots=False,
        show_ai_thoughts=False,
        risk_profile="balanced",
        capital=10000.0,
    )
    ret_cc = _compute_asset_returns(frame, params_cc)
    assert abs(ret_cc.iloc[1] - 0.10) < 1e-12

    params_co = replace(params_cc, return_path="close_to_open")
    ret_co = _compute_asset_returns(frame, params_co)
    # close->open uses next open vs current close: first element (111/100 -1) = 0.11
    assert abs(ret_co.iloc[0] - 0.11) < 1e-9
    assert ret_co.iloc[-1] == 0.0  # last filled to 0


def test_calibration_summary_added_to_stats():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    probs = pd.Series([0.1, 0.2, 0.8, 0.9, 0.7], index=idx)
    labels = pd.Series([0, 0, 1, 1, 1], index=idx)
    calib = strategies._calibration_summary(probs, labels)
    assert calib.get("brier") is not None
    assert calib.get("buckets")
