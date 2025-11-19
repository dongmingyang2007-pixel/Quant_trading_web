from __future__ import annotations

import math
import sys
import os
import subprocess
from datetime import datetime
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative / peak) - 1.0
    return float(drawdown.min())


def _annualized_return(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    total_return = float((1 + series).prod() - 1.0)
    years = len(series) / TRADING_DAYS
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def _annualized_vol(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.std() * math.sqrt(TRADING_DAYS))


def _sharpe(series: pd.Series, risk_free_rate: float = 0.0) -> float:
    if series.empty:
        return 0.0
    excess = series - risk_free_rate / TRADING_DAYS
    std = excess.std()
    if std == 0:
        return 0.0
    return float(math.sqrt(TRADING_DAYS) * excess.mean() / std)


def _hit_ratio(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    hits = (series > 0).sum()
    total = (series != 0).sum()
    return float(hits / total) if total else 0.0


def build_walk_forward_report(returns: pd.Series | None, *, window: int = 252, step: int = 63) -> list[dict[str, Any]]:
    if returns is None:
        return []
    clean = returns.dropna()
    if len(clean) < window + step:
        return []
    report: list[dict[str, Any]] = []
    for start in range(window, len(clean) - step + 1, step):
        train_slice = clean.iloc[start - window : start]
        test_slice = clean.iloc[start : start + step]
        entry = {
            "train_start": train_slice.index[0].strftime("%Y-%m-%d"),
            "train_end": train_slice.index[-1].strftime("%Y-%m-%d"),
            "test_start": test_slice.index[0].strftime("%Y-%m-%d"),
            "test_end": test_slice.index[-1].strftime("%Y-%m-%d"),
            "test_cagr": round(_annualized_return(test_slice), 4),
            "test_sharpe": round(_sharpe(test_slice), 2),
            "test_max_drawdown": round(_max_drawdown(test_slice), 4),
            "test_volatility": round(_annualized_vol(test_slice), 4),
            "test_hit_ratio": round(_hit_ratio(test_slice), 4),
        }
        report.append(entry)
    return report


def build_purged_kfold_schedule(index: Iterable[Any] | pd.Index | None, *, n_splits: int = 4, embargo: int = 5) -> list[dict[str, Any]]:
    if index is None:
        return []
    idx = pd.Index(index)
    total = len(idx)
    if total < n_splits * 2:
        return []
    fold_size = total // n_splits
    schedule: list[dict[str, Any]] = []
    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = test_start + fold_size
        if fold == n_splits - 1:
            test_end = total
        embargo_start = max(0, test_start - embargo)
        train_indices = list(range(0, embargo_start)) + list(range(test_end + embargo, total))
        if not train_indices:
            continue
        schedule.append(
            {
                "fold": fold + 1,
                "train_range": [
                    idx[train_indices[0]].strftime("%Y-%m-%d"),
                    idx[train_indices[-1]].strftime("%Y-%m-%d"),
                ],
                "test_range": [
                    idx[test_start].strftime("%Y-%m-%d"),
                    idx[min(test_end - 1, total - 1)].strftime("%Y-%m-%d"),
                ],
                "train_size": len(train_indices),
                "test_size": test_end - test_start,
                "embargo": embargo,
            }
        )
    return schedule


def compute_tail_risk_summary(returns: pd.Series | None, confidence: float = 0.95) -> dict[str, Any]:
    if returns is None:
        return {}
    clean = returns.dropna()
    if len(clean) < 30:
        return {}
    sorted_returns = np.sort(clean.to_numpy())
    var_idx = int((1 - confidence) * len(sorted_returns))
    var_value = float(sorted_returns[var_idx])
    cvar_value = float(sorted_returns[: var_idx + 1].mean())
    tail_ratio = float(abs(var_value) / max(abs(cvar_value), 1e-9))
    return {
        "confidence": confidence,
        "value_at_risk": var_value,
        "conditional_var": cvar_value,
        "tail_ratio": tail_ratio,
    }


def collect_repro_metadata(params) -> dict[str, Any]:
    """Capture seeds、版本、Git commit 等信息，写入快照。"""

    metadata: dict[str, Any] = {
        "ticker": params.ticker,
        "benchmark": params.benchmark_ticker,
        "start_date": params.start_date.isoformat(),
        "end_date": params.end_date.isoformat(),
        "model_version": getattr(params, "model_version", None),
        "data_version": getattr(params, "data_version", None),
        "requested_at": datetime.now(datetime.now().astimezone().tzinfo or datetime.utcnow().astimezone().tzinfo).isoformat(),
        "return_path": getattr(params, "return_path", "close_to_close"),
    }
    try:
        seed = int(getattr(params, "random_seed", os.environ.get("STRATEGY_SEED", "42")))
    except Exception:
        seed = int(os.environ.get("STRATEGY_SEED", "42"))
    metadata["seeds"] = {
        "python": seed,
        "numpy": seed,
        "torch": seed,
        "sb3": seed,
    }
    metadata["versions"] = {"python": sys.version}
    try:
        import numpy as _np
        metadata["versions"]["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import pandas as _pd
        metadata["versions"]["pandas"] = _pd.__version__
    except Exception:
        pass
    try:
        import sklearn as _sk
        metadata["versions"]["sklearn"] = _sk.__version__
    except Exception:
        pass
    try:
        import torch as _torch  # type: ignore
        metadata["versions"]["torch"] = getattr(_torch, "__version__", None)
    except Exception:
        pass
    try:
        import stable_baselines3 as _sb3  # type: ignore
        metadata["versions"]["stable_baselines3"] = getattr(_sb3, "__version__", None)
    except Exception:
        pass
    try:
        import yfinance as _yf  # type: ignore
        metadata["versions"]["yfinance"] = getattr(_yf, "__version__", None)
    except Exception:
        pass
    git_commit = None
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = None
    metadata["git_commit"] = git_commit
    metadata["exec_latency_ms"] = getattr(params, "exec_latency_ms", None)
    return metadata
