from __future__ import annotations

import math
import sys
import os
import subprocess
import hashlib
from datetime import datetime
from typing import Any, Iterable

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


def build_data_signature(
    frame: pd.DataFrame | None,
    *,
    columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    """为数据快照生成可复现签名（哈希/区间/行数/来源）。"""
    if frame is None or frame.empty:
        return {"rows": 0, "hash": None, "start": None, "end": None, "source": None}
    subset = frame
    if columns:
        cols = [col for col in columns if col in frame.columns]
        if cols:
            subset = frame[cols]
    hashed = pd.util.hash_pandas_object(subset, index=True).values  # type: ignore[attr-defined]
    digest = hashlib.sha256(hashed.tobytes()).hexdigest()
    idx = subset.index
    start = idx[0]
    end = idx[-1]
    source = getattr(frame, "attrs", {}).get("data_source")
    cache_path = getattr(frame, "attrs", {}).get("cache_path")
    return {
        "rows": int(len(subset)),
        "hash": digest,
        "start": str(start.date()) if hasattr(start, "date") else str(start),
        "end": str(end.date()) if hasattr(end, "date") else str(end),
        "source": source,
        "cache_path": cache_path,
    }


def assert_no_feature_leakage(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    label_columns: Iterable[str] | None = None,
) -> None:
    """防止训练特征包含未来/标签字段。"""
    labels = {
        "target",
        "target_multiclass",
        "forward_return",
        "future_return",
        "label",
    }
    if label_columns:
        labels.update({str(col) for col in label_columns})
    missing = [col for col in feature_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns missing from dataset: {missing}")
    leakage = []
    for col in feature_columns:
        lowered = col.lower()
        if col in labels:
            leakage.append(col)
            continue
        if lowered.startswith(("forward_", "future_", "target", "label", "tb_")):
            leakage.append(col)
    if leakage:
        raise ValueError(f"Feature leakage detected: {sorted(set(leakage))}")


def compute_cpcv_report(
    returns: pd.Series | None,
    *,
    n_groups: int = 5,
    repeats: int = 3,
) -> dict[str, Any]:
    """
    组合交叉验证 (CPCV) 报告：切分为 n_groups，重复多轮测试折，衡量稳健性。
    返回最差分位指标，供阈值/杠杆惩罚使用。
    """
    if returns is None:
        return {}
    clean = returns.dropna()
    if len(clean) < max(n_groups * 5, 60):
        return {}
    idx = clean.index
    indices = np.arange(len(clean))
    groups = np.array_split(indices, n_groups)
    rng = np.random.default_rng(seed=42)
    entries: list[dict[str, Any]] = []

    def _metrics(series: pd.Series) -> dict[str, float]:
        ser = series.fillna(0.0)
        if ser.empty:
            return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}
        sharpe = _sharpe(ser)
        cagr = _annualized_return(ser)
        mdd = _max_drawdown(ser)
        return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": mdd}

    for rep in range(repeats):
        perm = rng.permutation(n_groups)
        for test_gid in perm:
            test_idx = groups[test_gid]
            if len(test_idx) == 0:
                continue
            test_slice = clean.iloc[test_idx]
            m = _metrics(test_slice)
            entries.append(
                {
                    "repeat": rep + 1,
                    "group": int(test_gid) + 1,
                    "start": idx[test_idx[0]].strftime("%Y-%m-%d") if hasattr(idx[test_idx[0]], "strftime") else str(idx[test_idx[0]]),
                    "end": idx[test_idx[-1]].strftime("%Y-%m-%d") if hasattr(idx[test_idx[-1]], "strftime") else str(idx[test_idx[-1]]),
                    **m,
                }
            )

    if not entries:
        return {}
    sharpe_vals = [e["sharpe"] for e in entries]
    cagr_vals = [e["cagr"] for e in entries]
    mdd_vals = [e["max_drawdown"] for e in entries]
    summary = {
        "mean_sharpe": float(np.mean(sharpe_vals)),
        "p10_sharpe": float(np.percentile(sharpe_vals, 10)),
        "worst_sharpe": float(np.min(sharpe_vals)),
        "mean_cagr": float(np.mean(cagr_vals)),
        "p10_cagr": float(np.percentile(cagr_vals, 10)),
        "worst_mdd": float(np.min(mdd_vals)),
        "entries": entries,
        "groups": n_groups,
        "repeats": repeats,
    }
    return summary


def build_stress_report(
    returns: pd.Series | None,
    *,
    shocks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    压力回放：对策略收益施加跳空/放大波动等合成冲击，输出最差分位。
    默认包含：
      -5% 跳空、-10% 跳空、未来 10 日波动放大 2x。
    """
    if returns is None:
        return {}
    base = returns.dropna()
    if base.empty:
        return {}
    scenarios = shocks or [
        {"name": "gap_-5", "gap": -0.05, "vol_mult": 1.0, "window": 1},
        {"name": "gap_-10", "gap": -0.10, "vol_mult": 1.0, "window": 1},
        {"name": "vol_x2_10d", "gap": 0.0, "vol_mult": 2.0, "window": 10},
    ]
    results: list[dict[str, Any]] = []

    for scenario in scenarios:
        gap = float(scenario.get("gap", 0.0))
        vol_mult = float(scenario.get("vol_mult", 1.0))
        window = max(1, int(scenario.get("window", 1)))
        shocked = base.copy()
        if gap != 0.0:
            shocked.iloc[0] = shocked.iloc[0] + gap
        if vol_mult != 1.0:
            shocked.iloc[:window] = shocked.iloc[:window] * vol_mult
        shocked_for_mdd = pd.concat([pd.Series([0.0]), shocked.reset_index(drop=True)], ignore_index=True)
        metrics = {
            "total_return": float((1 + shocked).prod() - 1),
            "sharpe": _sharpe(shocked),
            "max_drawdown": _max_drawdown(shocked_for_mdd),
        }
        results.append(
            {
                "name": scenario.get("name", "stress"),
                **metrics,
            }
        )
    if not results:
        return {}
    worst_mdd = min(r["max_drawdown"] for r in results)
    worst_sharpe = min(r["sharpe"] for r in results)
    p10_mdd = float(np.percentile([r["max_drawdown"] for r in results], 10))
    report = {
        "scenarios": results,
        "worst_mdd": worst_mdd,
        "worst_sharpe": worst_sharpe,
        "p10_mdd": p10_mdd,
    }
    return report


def compute_psi(
    base: pd.Series | None,
    recent: pd.Series | None,
    *,
    bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI) 用于粗粒度分布漂移检测。
    返回 PSI 值，通常 >0.25 视为显著漂移。
    """
    if base is None or recent is None:
        return 0.0
    b = base.dropna()
    r = recent.dropna()
    if b.empty or r.empty:
        return 0.0
    # 使用分位数切分 base，统计 base/ recent 的频率
    try:
        quantiles = np.linspace(0, 1, bins + 1)
        edges = b.quantile(quantiles).to_numpy()
        edges[0] = -np.inf
        edges[-1] = np.inf
        b_counts, _ = np.histogram(b, bins=edges)
        r_counts, _ = np.histogram(r, bins=edges)
        b_pct = b_counts / max(b_counts.sum(), 1e-9)
        r_pct = r_counts / max(r_counts.sum(), 1e-9)
        psi = np.sum((r_pct - b_pct) * np.log((r_pct + 1e-9) / (b_pct + 1e-9)))
        return float(psi)
    except Exception:
        return 0.0
