from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None  # type: ignore

try:
    from catboost import CatBoostClassifier  # type: ignore
except Exception:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # optional parallel backend
    from joblib import Parallel, delayed  # type: ignore
except Exception:  # pragma: no cover
    Parallel = None  # type: ignore
    def delayed(func):  # type: ignore
        return func

from .config import StrategyInput, QuantStrategyError
from .indicators import (
    DEFAULT_FEATURE_COLUMNS,
    build_feature_frame,
    _normalized_open_prices,
    _select_forward_return,
)
from .event_engine import compute_realized_returns, run_event_backtest
from .store import DATA_CACHE_DIR, FEATURE_STORE
from .metrics import compute_validation_metrics, aggregate_oos_metrics, build_oos_boxplot, fig_to_base64
from ..optimization import PurgedWalkForwardSplit, _build_classifier
from ..ml_models import build_custom_sequence_model
from ..observability import record_metric
from ..validation import compute_cpcv_report, build_stress_report, compute_psi, assert_no_feature_leakage
from .risk import (
    apply_signal_filters,
    calculate_max_drawdown,
    calculate_target_leverage,
    enforce_risk_limits,
)
from .execution import apply_execution_model


def _calibration_summary(probs: pd.Series, labels: pd.Series, bins: int = 10) -> dict[str, Any]:
    """概率校准统计。"""
    clean = pd.DataFrame({"p": probs, "y": labels}).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {}
    clean["bucket"] = pd.qcut(clean["p"], q=min(bins, len(clean)), duplicates="drop")
    grouped = clean.groupby("bucket", observed=False)
    rows = []
    for _, grp in grouped:
        rows.append(
            {
                "mean_pred": float(grp["p"].mean()),
                "mean_true": float(grp["y"].mean()),
                "count": int(grp.shape[0]),
            }
        )
    brier = float(((clean["p"] - clean["y"]) ** 2).mean())
    return {"buckets": rows, "brier": brier}


def build_calibration_plot(calib: dict[str, Any]) -> str | None:
    """绘制概率校准曲线（base64）。"""
    try:
        import matplotlib.pyplot as plt
        buckets = calib.get("buckets") or []
        if not buckets:
            return None
        preds = [b["mean_pred"] for b in buckets]
        trues = [b["mean_true"] for b in buckets]
        counts = [b["count"] for b in buckets]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", label="理想校准")
        ax.plot(preds, trues, marker="o", color="#2563eb", label="实际")
        ax.grid(alpha=0.2)
        ax.set_xlabel("预测概率")
        ax.set_ylabel("真实命中率")
        ax.set_title("概率校准曲线")
        ax2 = ax.twinx()
        ax2.bar(preds, counts, alpha=0.15, color="#10b981", width=0.05, label="分桶样本数")
        ax2.set_ylabel("样本数")
        ax.legend(loc="upper left")
        return fig_to_base64(fig)
    except Exception:
        return None


def build_labels(frame: pd.DataFrame, params: StrategyInput) -> pd.DataFrame:
    """构建 direction / triple-barrier 标签并附加到行情框架。"""
    data = frame.copy()
    selected_return, label_path = _select_forward_return(data, params)
    data["forward_return"] = selected_return
    data["label_return_path"] = label_path
    if "forward_return" not in data and "adj close" in data:
        data["forward_return"] = data["adj close"].pct_change().shift(-1)
    if params.label_style == "triple_barrier":
        dynamic_up = params.tb_up
        dynamic_down = params.tb_down
        if params.tb_dynamic:
            try:
                vol = data["adj close"].pct_change().rolling(max(10, params.tb_vol_window)).std()
                dynamic_up = vol * params.tb_vol_multiplier
                dynamic_down = vol * params.tb_vol_multiplier
            except Exception:
                pass
        binary, multi = compute_triple_barrier_labels(
            data["adj close"],
            dynamic_up,
            dynamic_down,
            max(1, int(params.tb_max_holding)),
        )
        data["target"] = binary
        data["target_multiclass"] = multi
        if isinstance(dynamic_up, pd.Series):
            data["tb_up_active"] = dynamic_up.reindex(data.index)
        if isinstance(dynamic_down, pd.Series):
            data["tb_down_active"] = dynamic_down.reindex(data.index)
    else:
        data["target"] = (data["forward_return"] > 0).astype(int)
    return data


def compute_triple_barrier_labels(
    price: pd.Series,
    up: float | pd.Series,
    down: float | pd.Series,
    max_holding: int,
) -> tuple[pd.Series, pd.Series]:
    idx = price.index
    arr = price.to_numpy(dtype=float)
    n = len(arr)
    binary = np.full(n, np.nan)
    multi = np.zeros(n)
    if n <= 1:
        return pd.Series(binary, index=idx), pd.Series(multi, index=idx)
    max_holding = max(1, int(max_holding))

    if isinstance(up, pd.Series):
        up_series = up.reindex(idx)
        fill_val = float(up_series.mean()) if not up_series.dropna().empty else 0.0
        up_arr = up_series.ffill().bfill().fillna(fill_val).to_numpy(dtype=float)
    else:
        up_arr = np.full(n, float(up))
    if isinstance(down, pd.Series):
        down_series = down.reindex(idx)
        fill_val = float(down_series.mean()) if not down_series.dropna().empty else 0.0
        down_arr = down_series.ffill().bfill().fillna(fill_val).to_numpy(dtype=float)
    else:
        down_arr = np.full(n, float(down))

    pad = np.full(max_holding, np.nan, dtype=float)
    arr_pad = np.concatenate([arr, pad])
    window = max_holding + 1
    windows = np.lib.stride_tricks.sliding_window_view(arr_pad, window_shape=window)
    current = windows[:, 0]
    future = windows[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = future / current[:, None] - 1
    rets = np.where(np.isfinite(rets), rets, np.nan)

    steps = np.arange(max_holding)
    hit_up = rets >= up_arr[:, None]
    hit_down = rets <= -down_arr[:, None]
    t_up = np.min(np.where(hit_up, steps, np.inf), axis=1)
    t_down = np.min(np.where(hit_down, steps, np.inf), axis=1)

    remaining = (n - 1) - np.arange(n)
    horizon_len = np.clip(remaining, 0, max_holding)
    last_idx = np.clip(horizon_len - 1, 0, max_holding - 1).astype(int)
    last_ret = rets[np.arange(n), last_idx]
    last_ret = np.where(horizon_len > 0, last_ret, np.nan)

    valid = horizon_len > 0
    up_first = t_up < t_down
    down_first = t_down < t_up
    no_hit = ~(up_first | down_first)

    binary[valid & up_first] = 1
    binary[valid & down_first] = 0
    binary[valid & no_hit] = (last_ret > 0).astype(float)

    multi[valid & up_first] = 1
    multi[valid & down_first] = -1
    return pd.Series(binary, index=idx), pd.Series(multi, index=idx)


def _balanced_sample_weight(labels: pd.Series) -> np.ndarray | None:
    try:
        from sklearn.utils.class_weight import compute_sample_weight
    except Exception:
        return None
    try:
        return compute_sample_weight("balanced", labels)
    except Exception:
        return None


def _maybe_get_sample_weight(labels: pd.Series, params: StrategyInput) -> np.ndarray | None:
    default_mode = "balanced" if getattr(params, "label_style", "direction") == "triple_barrier" else "none"
    mode = str(getattr(params, "class_weight_mode", default_mode)).lower()
    if mode == "balanced":
        return _balanced_sample_weight(labels)
    if mode == "focal":
        try:
            gamma = float(getattr(params, "focal_gamma", 2.0))
        except Exception:
            gamma = 2.0
        counts = labels.value_counts()
        total = counts.sum()
        weights = labels.map(lambda x: (1 - counts.get(x, 0) / max(total, 1)) ** gamma)
        return weights.to_numpy(dtype=float)
    return None


def build_feature_matrix(prices: pd.DataFrame, params: StrategyInput) -> tuple[pd.DataFrame, list[str]]:
    """构建 ML 特征矩阵并对齐标签。"""
    cache_key: str | None = None
    if getattr(params, "use_feature_cache", True):
        try:
            cache_key = FEATURE_STORE.fingerprint(prices, params)
            cached = FEATURE_STORE.load(cache_key)
            if cached:
                dataset, feature_columns, _ = cached
                if "target" in dataset:
                    return dataset, feature_columns
        except Exception:
            cache_key = None

    dataset = prices.copy()
    required = {"sma_short", "sma_long", "boll_up", "boll_dn"}
    if not required.issubset(dataset.columns):
        dataset = build_feature_frame(
            dataset,
            params.short_window,
            params.long_window,
            params.rsi_period,
        )
    dataset["label_style"] = params.label_style
    dataset["return_path"] = getattr(params, "return_path", "close_to_close")
    if "forward_return_close" not in dataset:
        dataset["forward_return_close"] = dataset["forward_return"]
    if "forward_return_open" not in dataset:
        dataset["forward_return_open"] = dataset["forward_return_close"]
    if "forward_return_open_to_close" not in dataset:
        dataset["forward_return_open_to_close"] = (dataset["adj close"] / _normalized_open_prices(dataset).replace(0, np.nan)) - 1
    selected_returns, label_path = _select_forward_return(dataset, params)
    dataset["forward_return"] = selected_returns
    dataset["future_return"] = selected_returns
    dataset["return_sign"] = np.sign(selected_returns.fillna(0))
    dataset["return_path"] = getattr(params, "return_path", "close_to_close")
    dataset["label_return_path"] = label_path

    feature_columns = list(DEFAULT_FEATURE_COLUMNS)

    dataset["sma_diff"] = dataset["sma_short"] - dataset["sma_long"]
    dataset["sma_ratio"] = dataset["sma_short"] / dataset["sma_long"] - 1
    dataset["boll_bandwidth"] = (dataset["boll_up"] - dataset["boll_dn"]) / dataset["sma_long"]
    feature_columns.extend(["sma_diff", "sma_ratio", "boll_bandwidth"])

    try:
        assert_no_feature_leakage(dataset, feature_columns)
    except ValueError as exc:
        raise QuantStrategyError(f"Feature leakage guard failed: {exc}") from exc

    labeled = build_labels(dataset, params)
    if params.label_style == "triple_barrier":
        if "target_multiclass" not in labeled:
            raise QuantStrategyError("三重闸标签生成失败，缺少 target_multiclass。")
        dataset["target"] = labeled["target"]
        dataset["target_multiclass"] = labeled["target_multiclass"]
        dataset["tb_up_active"] = labeled.get("tb_up_active")
        dataset["tb_down_active"] = labeled.get("tb_down_active")
        dataset = dataset.dropna(subset=feature_columns + ["forward_return", "target"])
        dataset["future_return"] = dataset["forward_return"]
    else:
        dataset["target"] = labeled["target"]
        dataset = dataset.dropna(subset=feature_columns + ["forward_return", "target"])

    if cache_key:
        try:
            signature_parts = {
                "features": feature_columns,
                "label": params.label_style,
                "windows": [params.short_window, params.long_window, params.rsi_period],
                "return_path": getattr(params, "return_path", "close_to_close"),
                "label_return_path": label_path,
                "tb": {
                    "dynamic": params.tb_dynamic,
                    "up": params.tb_up,
                    "down": params.tb_down,
                    "vol_multiplier": params.tb_vol_multiplier,
                    "vol_window": params.tb_vol_window,
                    "max_holding": params.tb_max_holding,
                },
            }
            signature = json.dumps(signature_parts, ensure_ascii=False, sort_keys=True)
            FEATURE_STORE.save(cache_key, dataset, feature_columns, pipeline_signature=signature)
        except Exception:
            pass

    return dataset, feature_columns


def _infer_category(ticker: str) -> str:
    t = ticker.upper()
    mega = {"AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"}
    etf = {"SPY", "QQQ", "IWM", "TLT", "GLD"}
    sector = {"XLF", "XLK", "XLE", "XLV", "XLY"}
    if t in mega:
        return "mega"
    if t in etf:
        return "etf"
    if t in sector or (len(t) == 3 and t.startswith("XL")):
        return "sector"
    return "mega"


def load_best_ml_config(ticker: str) -> tuple[str | None, dict[str, Any] | None]:
    cfg_path = DATA_CACHE_DIR / "training" / "best_ml_config_overall.json"
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        cat = _infer_category(ticker)
        entry = (data or {}).get(cat)
        if not entry:
            return None, None
        engine = entry.get("engine")
        params = entry.get("params") or {}
        return engine, params
    except Exception:
        return None, None


def _simulate_expected_return(
    signal: pd.Series,
    returns: pd.Series,
    cost_rate: float,
) -> float:
    signal = signal.fillna(0.0)
    exposure = signal.shift(fill_value=0.0)
    pnl = returns.fillna(0.0) * exposure
    turnover = signal.diff().abs().fillna(signal.abs())
    cost = turnover * cost_rate
    total = (pnl - cost).sum()
    return float(total)


def tune_thresholds_on_validation(
    proba: pd.Series,
    returns: pd.Series,
    cost_rate: float,
    grid_long: list[float] | None = None,
    grid_short: list[float] | None = None,
    n_jobs: int = 1,
) -> tuple[float, float]:
    n = len(proba.dropna())
    step_long = 0.02 if n < 150 else 0.015
    step_short = 0.02 if n < 150 else 0.015
    grid_long = grid_long or [round(x, 3) for x in np.arange(0.54, 0.71, step_long)]
    grid_short = grid_short or [round(x, 3) for x in np.arange(0.29, 0.48, step_short)]
    candidates = [(e, x) for e in grid_long for x in grid_short if 0.0 < x < 0.5 < e < 1.0]
    if not candidates:
        return 0.55, 0.45
    if n_jobs and n_jobs > 1 and Parallel is not None:
        try:
            scores = Parallel(n_jobs=n_jobs)(
                delayed(_simulate_expected_return)(
                    pd.Series(np.where(proba >= e, 1.0, np.where(proba <= x, -1.0, 0.0)), index=proba.index),
                    returns,
                    cost_rate,
                )
                for e, x in candidates
            )
            best_idx = int(np.argmax(scores))
            return float(candidates[best_idx][0]), float(candidates[best_idx][1])
        except Exception:
            pass
    best = (-np.inf, 0.55, 0.45)
    for e, x in candidates:
        sig = pd.Series(np.where(proba >= e, 1.0, np.where(proba <= x, -1.0, 0.0)), index=proba.index)
        score = _simulate_expected_return(sig, returns, cost_rate)
        if score > best[0]:
            best = (score, e, x)
    return float(best[1]), float(best[2])


def _apply_execution_model(
    exposure: pd.Series,
    prices: pd.Series,
    adv: pd.Series | None,
    params: StrategyInput,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, dict[str, float]]:
    """Compatibility wrapper around unified execution model."""
    return apply_execution_model(exposure, prices, adv=adv, params=params)


def scan_threshold_stability(
    probabilities: pd.Series,
    future_returns: pd.Series,
    cost_rate: float,
    base_entry: float,
    base_exit: float,
) -> dict[str, Any]:
    proba = probabilities.dropna()
    if proba.empty or len(proba) < 40:
        return {}
    aligned_returns = future_returns.reindex(proba.index).fillna(0.0)
    entry_grid = np.linspace(max(0.5, base_entry - 0.05), min(0.95, base_entry + 0.05), 5)
    exit_grid = np.linspace(max(0.05, base_exit - 0.05), min(0.45, base_exit + 0.05), 5)
    entry_mesh, exit_mesh = np.meshgrid(entry_grid, exit_grid)
    entry_vec = entry_mesh.flatten()
    exit_vec = exit_mesh.flatten()
    valid_mask = exit_vec < (entry_vec - 0.02)
    entry_vec = entry_vec[valid_mask]
    exit_vec = exit_vec[valid_mask]
    if entry_vec.size == 0:
        return {}

    p_arr = proba.values[:, None]
    ret_arr = aligned_returns.values[:, None]
    entry_mat = entry_vec[None, :]
    exit_mat = exit_vec[None, :]

    signals = np.where(p_arr >= entry_mat, 1.0, np.where(p_arr <= exit_mat, -1.0, 0.0))
    exposure = np.vstack([np.zeros((1, signals.shape[1])), signals[:-1, :]])
    turnover = np.vstack([np.abs(exposure[0, :]), np.abs(np.diff(exposure, axis=0))])
    pnl = exposure * ret_arr
    linear_cost = turnover * cost_rate
    pnl_net = pnl - linear_cost

    mean = pnl_net.mean(axis=0)
    std = pnl_net.std(axis=0)
    ann_factor = 252.0
    sharpe = np.where(std > 0, np.sqrt(ann_factor) * mean / std, 0.0)

    best_idx = int(np.argmax(sharpe))
    worst_idx = int(np.argmin(sharpe))
    mean_sharpe = float(np.mean(sharpe))
    q1, q3 = np.percentile(sharpe, [25, 75])

    best = {"entry": float(entry_vec[best_idx]), "exit": float(exit_vec[best_idx]), "sharpe": float(sharpe[best_idx])}
    worst = {"entry": float(entry_vec[worst_idx]), "exit": float(exit_vec[worst_idx]), "sharpe": float(sharpe[worst_idx])}
    points = [
        {"entry": float(e), "exit": float(x), "sharpe": float(s)}
        for e, x, s in zip(entry_vec.tolist(), exit_vec.tolist(), sharpe.tolist())
    ]

    return {
        "grid": {
            "entry": [round(float(x), 4) for x in entry_grid.tolist()],
            "exit": [round(float(x), 4) for x in exit_grid.tolist()],
        },
        "best": best,
        "worst": worst,
        "mean_sharpe": mean_sharpe,
        "median_sharpe": float(np.median(sharpe)),
        "iqr_sharpe": float(q3 - q1),
        "count": len(entry_vec),
        "points": points,
        "heatmap_grid": {
            "entry": entry_vec.tolist(),
            "exit": exit_vec.tolist(),
            "sharpe": sharpe.tolist(),
        },
    }


def _scan_threshold_stability(
    probabilities: pd.Series,
    future_returns: pd.Series,
    cost_rate: float,
    base_entry: float,
    base_exit: float,
) -> dict[str, Any]:
    return scan_threshold_stability(probabilities, future_returns, cost_rate, base_entry, base_exit)


def _generate_validation_report(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    params: StrategyInput,
) -> dict[str, Any] | None:
    if dataset is None or dataset.empty or not feature_columns:
        return None
    if "target" not in dataset or "future_return" not in dataset:
        return None

    test_window = params.test_window if params.test_window is not None else 5
    splitter = PurgedWalkForwardSplit(
        train_window=params.train_window,
        test_window=max(int(test_window), 5),
        embargo=max(0, params.embargo_days),
    )
    cost_rate = (params.transaction_cost_bps + params.slippage_bps) / 10000.0
    long_threshold, short_threshold = _resolve_signal_thresholds(params)
    validation_slices: list[dict[str, Any]] = []
    model_params = params.ml_params or {}
    model_engine = (params.ml_model or "sk_gbdt").lower()
    if model_engine in {"lstm", "transformer", "seq_hybrid", "hybrid_seq", "fusion"}:
        model_engine = "sk_gbdt"
    labels_for_weight = dataset["target_multiclass"] if ("target_multiclass" in dataset and params.label_style == "triple_barrier") else dataset["target"]
    sample_weight = _maybe_get_sample_weight(labels_for_weight, params)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(len(dataset))):
        train_slice = dataset.iloc[train_idx]
        test_slice = dataset.iloc[test_idx]
        if train_slice["target"].nunique() < 2 or test_slice.empty:
            continue
        try:
            clf = _build_classifier(model_engine, model_params, feature_columns, params)
        except Exception:
            clf = _build_classifier("sk_gbdt", {}, feature_columns, params)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            if isinstance(clf, Pipeline):
                fit_kwargs["model__sample_weight"] = sample_weight[train_idx]
            else:
                fit_kwargs["sample_weight"] = sample_weight[train_idx]
        try:
            clf.fit(train_slice[feature_columns], train_slice["target"], **fit_kwargs)
            raw_proba = clf.predict_proba(test_slice[feature_columns])
            if isinstance(raw_proba, np.ndarray) and raw_proba.ndim > 1 and raw_proba.shape[1] > 1:
                proba = raw_proba[:, 1]
            else:
                proba = np.array(raw_proba).ravel()
        except Exception:
            continue
        long_threshold, short_threshold = _resolve_signal_thresholds(params)
        signal = np.where(
            proba >= long_threshold,
            1.0,
            np.where(proba <= short_threshold, -1.0, 0.0),
        )
        exposure = pd.Series(signal, index=test_slice.index, dtype=float).shift(fill_value=0.0)
        turnover = exposure.diff().abs().fillna(exposure.abs())
        pnl = exposure * test_slice["future_return"].fillna(0.0) - turnover * cost_rate
        fold_metrics = compute_validation_metrics(pnl)
        fold_metrics.update(
            {
                "fold": fold_idx + 1,
                "train_start": str(train_slice.index[0].date()) if hasattr(train_slice.index[0], "date") else str(train_slice.index[0]),
                "train_end": str(train_slice.index[-1].date()) if hasattr(train_slice.index[-1], "date") else str(train_slice.index[-1]),
                "test_start": str(test_slice.index[0].date()) if hasattr(test_slice.index[0], "date") else str(test_slice.index[0]),
                "test_end": str(test_slice.index[-1].date()) if hasattr(test_slice.index[-1], "date") else str(test_slice.index[-1]),
                "avg_position": float(exposure.abs().mean()),
            }
        )
        validation_slices.append(fold_metrics)

    if not validation_slices:
        return None

    summary = aggregate_oos_metrics(validation_slices)
    penalized = None
    sharpe_stats = summary.get("sharpe")
    if isinstance(sharpe_stats, dict):
        penalized = (sharpe_stats.get("mean") or 0.0) - (sharpe_stats.get("std") or 0.0)
    return {
        "slices": validation_slices,
        "summary": summary,
        "folds": len(validation_slices),
        "train_window": params.train_window,
        "test_window": params.test_window,
        "embargo": params.embargo_days,
        "penalized_sharpe": penalized,
    }


def build_threshold_heatmap(scan: dict[str, Any]) -> str | None:
    try:
        import matplotlib.pyplot as plt

        points = scan.get("heatmap_grid") or {}
        entry = np.array(points.get("entry") or [])
        exit = np.array(points.get("exit") or [])
        sharpe = np.array(points.get("sharpe") or [])
        if entry.size == 0 or exit.size == 0 or sharpe.size == 0:
            return None
        entry_levels = sorted(set(entry.tolist()))
        exit_levels = sorted(set(exit.tolist()))
        grid = np.full((len(exit_levels), len(entry_levels)), np.nan)
        for e, x, s in zip(entry, exit, sharpe):
            i = exit_levels.index(float(x))
            j = entry_levels.index(float(e))
            grid[i, j] = s
        fig, ax = plt.subplots(figsize=(5.5, 4))
        im = ax.imshow(grid, origin="lower", cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(entry_levels)))
        ax.set_xticklabels([f"{v:.2f}" for v in entry_levels], rotation=45, ha="right")
        ax.set_yticks(range(len(exit_levels)))
        ax.set_yticklabels([f"{v:.2f}" for v in exit_levels])
        ax.set_xlabel("Entry 阈值")
        ax.set_ylabel("Exit 阈值")
        ax.set_title("阈值稳定性热力图（Sharpe）")
        fig.colorbar(im, ax=ax, label="Sharpe")
        return fig_to_base64(fig)
    except Exception:
        return None


def _extract_class_prob(proba: np.ndarray, classes: np.ndarray | None, label: int, default: float = 0.5) -> np.ndarray:
    if classes is None or proba.ndim != 2:
        return np.full(len(proba), default)
    for idx, cls in enumerate(classes):
        if cls == label:
            return proba[:, idx]
    return np.full(len(proba), default)


def _resolve_signal_thresholds(params: StrategyInput) -> tuple[float, float]:
    long_threshold = float(getattr(params, "long_threshold", 0.5) or 0.5)
    short_threshold = float(getattr(params, "short_threshold", 0.3) or 0.3)
    long_threshold = max(0.0, min(long_threshold, 1.0))
    short_threshold = max(0.0, min(short_threshold, long_threshold))
    long_threshold = max(long_threshold, 0.5)
    short_threshold = min(short_threshold, 0.5)
    return long_threshold, short_threshold


def _pfws_predict(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    params: StrategyInput,
) -> tuple[pd.Series, pd.Series, list[float]]:
    splitter = PurgedWalkForwardSplit(
        train_window=params.train_window,
        test_window=max(params.test_window, 5),
        embargo=max(0, params.embargo_days),
    )
    probabilities = pd.Series(np.nan, index=dataset.index, dtype=float)
    raw_signal = pd.Series(np.nan, index=dataset.index, dtype=float)
    auc_scores: list[float] = []
    model_params = params.ml_params or {}
    labels_for_weight = dataset["target_multiclass"] if ("target_multiclass" in dataset and params.label_style == "triple_barrier") else dataset["target"]
    sample_weight = _maybe_get_sample_weight(labels_for_weight, params)
    long_threshold, short_threshold = _resolve_signal_thresholds(params)
    for train_idx, test_idx in splitter.split(len(dataset)):
        train_slice = dataset.iloc[train_idx]
        test_slice = dataset.iloc[test_idx]
        if train_slice["target"].nunique() < 2 or test_slice.empty:
            continue
        clf = _build_classifier(params.ml_model, model_params, feature_columns, params)
        fit_kwargs = {}
        if sample_weight is not None:
            if isinstance(clf, Pipeline):
                fit_kwargs["model__sample_weight"] = sample_weight[train_idx]
            else:
                fit_kwargs["sample_weight"] = sample_weight[train_idx]
        try:
            clf.fit(train_slice[feature_columns], train_slice["target"], **fit_kwargs)
            proba = clf.predict_proba(test_slice[feature_columns])[:, 1]
        except Exception:
            continue
        probabilities.iloc[test_idx] = proba
        raw_signal.iloc[test_idx] = np.where(
            proba >= long_threshold,
            1.0,
            np.where(proba <= short_threshold, -1.0, 0.0),
        )
        if roc_auc_score is not None and test_slice["target"].nunique() > 1:
            try:
                auc_scores.append(roc_auc_score(test_slice["target"], proba))
            except Exception:
                pass
    return probabilities, raw_signal, auc_scores


def run_ml_backtest(
    prices: pd.DataFrame,
    params: StrategyInput,
    context_features: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    if GradientBoostingClassifier is None or Pipeline is None or StandardScaler is None:
        raise QuantStrategyError("scikit-learn 未安装，无法启用机器学习策略。请运行 pip install scikit-learn。")

    params = replace(params)
    dataset, feature_columns = build_feature_matrix(prices, params)
    if context_features:
        for name, value in context_features.items():
            dataset[name] = float(value)
            if name not in feature_columns:
                feature_columns.append(name)
    if dataset.empty:
        raise QuantStrategyError("样本不足，无法构建机器学习特征矩阵。")

    required = max(params.train_window, params.long_window * 3, params.short_window * 6, 180)
    test_window = max(params.test_window, 5)
    if dataset.shape[0] <= required + test_window:
        raise QuantStrategyError("历史样本数量不足以完成走期训练，建议延长回测区间或减少窗口。")

    model_warnings: list[str] = []
    hyperopt_report: dict[str, Any] | None = None

    requested_model = (params.ml_model or "seq_hybrid").lower()
    if requested_model in {"auto", "fusion", "seq_hybrid", "hybrid_seq"}:
        if torch is None:
            params.ml_model = "sk_gbdt"
            model_warnings.append("未检测到深度学习后端，自动回退到 GBDT。")
        else:
            params.ml_model = "seq_hybrid"
    else:
        params.ml_model = requested_model

    if getattr(params, "enable_hyperopt", False):
        try:
            from .optimization import run_optuna_search  # type: ignore

            hyperopt_report = run_optuna_search(dataset, feature_columns, params)
            params.entry_threshold = float(hyperopt_report.get("entry_threshold", params.entry_threshold))
            params.exit_threshold = float(hyperopt_report.get("exit_threshold", params.exit_threshold))
            best_ml = hyperopt_report.get("ml_params") or {}
            if best_ml:
                params.ml_params = {**best_ml}
            model_warnings.append(f"已应用 Optuna 搜索结果（score={hyperopt_report.get('best_score', 0.0):.2f}, trials={hyperopt_report.get('trials_ran', 0)}）。")
        except Exception as exc:
            hyperopt_report = {"error": str(exc)}
            model_warnings.append(f"自动超参搜索失败：{exc}")

    validation_report = None
    probabilities = pd.Series(np.nan, index=dataset.index, dtype=float)
    raw_signal = pd.Series(0.0, index=dataset.index, dtype=float)
    predictions = pd.Series(np.nan, index=dataset.index, dtype=float)
    auc_scores: list[float] = []
    shap_img: str | None = None
    validation_slices: list[dict[str, Any]] = []
    tuned_entries: list[float] = []
    tuned_exits: list[float] = []
    cost_rate = (params.transaction_cost_bps + params.slippage_bps) / 10000.0

    splitter = PurgedWalkForwardSplit(train_window=params.train_window, test_window=max(params.test_window, 5), embargo=max(0, params.embargo_days))
    n_jobs = max(1, int(getattr(params, "walk_forward_jobs", 1) or 1))

    def _run_pfws_fold(fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, Any]:
        train_slice = dataset.iloc[train_idx]
        test_slice = dataset.iloc[test_idx]
        if train_slice["target"].nunique() < 2 or test_slice.empty:
            return {}

        val_len = max(1, int(len(train_slice) * max(0.05, min(0.4, params.val_ratio))))
        train_end = len(train_slice)
        val_start = max(0, train_end - val_len)
        core_train = train_slice.iloc[:val_start]
        val_slice = train_slice.iloc[val_start:train_end]

        model_name = (params.ml_model or "sk_gbdt").lower()
        if model_name == "lightgbm" and lgb is not None:
            lgb_kwargs = dict(learning_rate=0.05, n_estimators=400, max_depth=-1, subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=-1)
            if params.ml_params:
                lgb_kwargs.update({k: v for k, v in params.ml_params.items() if k in lgb_kwargs or True})
            estimator = lgb.LGBMClassifier(**lgb_kwargs)
            pipeline = Pipeline([("model", estimator)])
        elif model_name == "catboost" and CatBoostClassifier is not None:
            cb_kwargs = dict(learning_rate=0.05, depth=6, iterations=400, subsample=0.8, random_seed=42, verbose=False, allow_writing_files=False)
            if params.ml_params:
                cb_kwargs.update(params.ml_params)
            estimator = CatBoostClassifier(**cb_kwargs)
            pipeline = Pipeline([("model", estimator)])
        elif model_name in {"lstm", "transformer"}:
            try:
                estimator, _ = build_custom_sequence_model(model_name, feature_columns, params)
                pipeline = estimator
            except (RuntimeError, ValueError) as exc:
                raise QuantStrategyError(str(exc)) from exc
        else:
            sk_kwargs = dict(learning_rate=0.05, n_estimators=250, max_depth=3, subsample=0.8, random_state=42)
            if params.ml_params:
                sk_kwargs.update({k: v for k, v in params.ml_params.items() if k in sk_kwargs})
            estimator = GradientBoostingClassifier(**sk_kwargs)
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", estimator)])

        prob_model = None
        weight_labels = core_train["target_multiclass"] if ("target_multiclass" in core_train and params.label_style == "triple_barrier") else core_train["target"]
        sample_weight = _maybe_get_sample_weight(weight_labels, params)

        def _fit_model(pipe, X, y, *, sample_weight=None, extra_params=None):
            extra_params = extra_params or {}
            if sample_weight is not None:
                if isinstance(pipe, Pipeline):
                    pipe.fit(X, y, model__sample_weight=sample_weight, **extra_params)
                else:
                    pipe.fit(X, y, sample_weight=sample_weight, **extra_params)
            else:
                pipe.fit(X, y, **extra_params)

        if model_name == "lightgbm" and lgb is not None and not val_slice.empty and params.early_stopping_rounds:
            try:
                estimator.fit(
                    core_train[feature_columns],
                    core_train["target"],
                    eval_set=[(val_slice[feature_columns], val_slice["target"])],
                    eval_metric="binary_logloss",
                    early_stopping_rounds=params.early_stopping_rounds,
                    verbose=False,
                    sample_weight=sample_weight,
                )
                prob_model = estimator
            except Exception:
                _fit_model(pipeline, core_train[feature_columns], core_train["target"], sample_weight=sample_weight)
                prob_model = pipeline
        elif model_name == "catboost" and CatBoostClassifier is not None and not val_slice.empty and params.early_stopping_rounds:
            try:
                estimator.set_params(od_type="Iter", od_wait=params.early_stopping_rounds)
                estimator.fit(
                    core_train[feature_columns],
                    core_train["target"],
                    eval_set=(val_slice[feature_columns], val_slice["target"]),
                    sample_weight=sample_weight,
                )
                prob_model = estimator
            except Exception:
                _fit_model(pipeline, core_train[feature_columns], core_train["target"], sample_weight=sample_weight)
                prob_model = pipeline
        else:
            _fit_model(pipeline, core_train[feature_columns], core_train["target"], sample_weight=sample_weight)
            prob_model = pipeline

        calibrator = None
        if not val_slice.empty and params.calibrate_proba and IsotonicRegression is not None:
            try:
                _val_raw = prob_model.predict_proba(val_slice[feature_columns])[:, 1]
                calibrator = IsotonicRegression(out_of_bounds="clip").fit(_val_raw, val_slice["target"].values)
            except Exception:
                calibrator = None

        entry_thr = params.entry_threshold
        exit_thr = params.exit_threshold
        if params.optimize_thresholds and not val_slice.empty:
            val_proba = pd.Series(prob_model.predict_proba(val_slice[feature_columns])[:, 1], index=val_slice.index)
            if calibrator is not None:
                try:
                    val_proba = pd.Series(calibrator.transform(val_proba.values), index=val_proba.index)
                except Exception:
                    pass
            val_ret = val_slice["future_return"].fillna(0.0)
            e_opt, x_opt = tune_thresholds_on_validation(val_proba, val_ret, cost_rate, n_jobs=getattr(params, "threshold_jobs", 1))
            entry_thr, exit_thr = e_opt, x_opt

        raw_proba_arr = prob_model.predict_proba(test_slice[feature_columns])
        classes_ = getattr(prob_model, "classes_", None)
        if calibrator is not None:
            try:
                if raw_proba_arr.ndim == 1 or raw_proba_arr.shape[1] == 1:
                    raw_proba_arr = calibrator.transform(raw_proba_arr)
                else:
                    raw_proba_arr[:, :] = np.vstack([calibrator.transform(raw_proba_arr[:, i]) for i in range(raw_proba_arr.shape[1])]).T
            except Exception:
                pass

        if params.label_style == "triple_barrier" and raw_proba_arr.ndim == 2 and raw_proba_arr.shape[1] >= 2:
            proba_up = _extract_class_prob(raw_proba_arr, classes_, 1, default=0.5)
            proba_down = _extract_class_prob(raw_proba_arr, classes_, -1, default=0.5)
        else:
            proba_up = raw_proba_arr[:, 1] if raw_proba_arr.ndim == 2 and raw_proba_arr.shape[1] > 1 else np.array([])
            proba_down = np.zeros_like(proba_up)
        proba_vals = proba_up if proba_up.size else (raw_proba_arr[:, 1] if raw_proba_arr.ndim == 2 and raw_proba_arr.shape[1] > 1 else raw_proba_arr)

        pred_vals = np.argmax(raw_proba_arr, axis=1) if raw_proba_arr.ndim == 2 and raw_proba_arr.shape[1] > 1 else (proba_vals >= 0.5).astype(int)

        if params.ml_task == "hybrid":
            try:
                if model_name == "lightgbm" and lgb is not None:
                    reg = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=400, subsample=0.8, colsample_bytree=0.9, random_state=42)
                elif model_name == "catboost" and CatBoostClassifier is not None:
                    reg = CatBoostClassifier(learning_rate=0.05, iterations=400, depth=6, subsample=0.8, random_seed=42, verbose=False, allow_writing_files=False)
                else:
                    reg = GradientBoostingRegressor(random_state=42, n_estimators=160, learning_rate=0.05, max_depth=3)
                reg.fit(core_train[feature_columns], core_train["future_return"].fillna(0.0))
                magnitudes = reg.predict(test_slice[feature_columns])
                proba_vals = proba_vals * np.sign(magnitudes)
            except Exception:
                pass

        long_threshold, short_threshold = _resolve_signal_thresholds(params)
        signal_fold = np.where(
            proba_vals >= long_threshold,
            1.0,
            np.where(proba_vals <= short_threshold, -1.0, 0.0),
        )
        exposure_fold = pd.Series(signal_fold, index=test_slice.index, dtype=float).shift(fill_value=0.0)
        turnover = exposure_fold.diff().abs().fillna(exposure_fold.abs())
        pnl_fold = exposure_fold * test_slice["future_return"].fillna(0.0) - turnover * cost_rate
        fold_metrics = compute_validation_metrics(pnl_fold)
        fold_metrics.update(
            {
                "fold": fold_idx + 1,
                "train_start": str(train_slice.index[0].date()),
                "train_end": str(train_slice.index[-1].date()),
                "test_start": str(test_slice.index[0].date()),
                "test_end": str(test_slice.index[-1].date()),
                "avg_position": float(exposure_fold.abs().mean()),
            }
        )

        return {
            "prob": pd.Series(proba_vals, index=test_slice.index),
            "pred": pd.Series(pred_vals, index=test_slice.index),
            "signal": pd.Series(signal_fold, index=test_slice.index, dtype=float),
            "fold_metrics": fold_metrics,
            "entry": entry_thr,
            "exit": exit_thr,
            "auc": float(roc_auc_score(test_slice["target"], proba_vals)) if roc_auc_score is not None and test_slice["target"].nunique() > 1 else None,
            "pred_classes": pd.Series(pred_vals, index=test_slice.index),
        }

    fold_jobs = list(splitter.split(len(dataset)))
    if n_jobs > 1 and Parallel is not None:
        fold_results = Parallel(n_jobs=n_jobs)(delayed(_run_pfws_fold)(i, tr, te) for i, (tr, te) in enumerate(fold_jobs))
    else:
        fold_results = [_run_pfws_fold(i, tr, te) for i, (tr, te) in enumerate(fold_jobs)]

    for res in fold_results:
        if not res:
            continue
        probabilities.loc[res["prob"].index] = res["prob"]
        predictions.loc[res["pred"].index] = res["pred"]
        raw_signal.loc[res["signal"].index] = res["signal"]
        validation_slices.append(res["fold_metrics"])
        tuned_entries.append(res["entry"])
        tuned_exits.append(res["exit"])
        if res.get("auc") is not None:
            auc_scores.append(res["auc"])
        if res.get("pred_classes") is not None:
            predictions.loc[res["pred_classes"].index] = res["pred_classes"]

    if not validation_slices:
        raise QuantStrategyError("PFWS 样本外预测不足，无法完成回测，请调整窗口或延长历史区间。")

    summary_oos = aggregate_oos_metrics(validation_slices)
    penalized = None
    sharpe_stats = summary_oos.get("sharpe")
    if isinstance(sharpe_stats, dict):
        penalized = (sharpe_stats.get("mean") or 0.0) - (sharpe_stats.get("std") or 0.0)
    validation_report = {
        "slices": validation_slices,
        "summary": summary_oos,
        "folds": len(validation_slices),
        "train_window": params.train_window,
        "test_window": params.test_window,
        "embargo": params.embargo_days,
        "penalized_sharpe": penalized,
        "distributions": {
            k: [float(entry.get(k, 0.0)) for entry in validation_slices if entry.get(k) is not None]
            for k in ("sharpe", "cagr", "max_drawdown", "hit_ratio")
        },
    }
    oos_chart = build_oos_boxplot(validation_report["distributions"], "PFWS OOS 统计箱线图")
    if oos_chart:
        validation_report["chart"] = oos_chart

    try:
        params.entry_threshold = float(np.mean(tuned_entries)) if tuned_entries else params.entry_threshold
        params.exit_threshold = float(np.mean(tuned_exits)) if tuned_exits else params.exit_threshold
    except Exception:
        pass

    pfws_proba, pfws_signal, pfws_auc = _pfws_predict(dataset, feature_columns, params)
    if pfws_auc:
        auc_scores.extend(pfws_auc)

    dataset["probability"] = pfws_proba
    dataset["signal"] = pfws_signal
    probabilities = pfws_proba.combine_first(probabilities)
    raw_signal = pfws_signal.combine_first(raw_signal)

    # Liquidity guard: skip signals on ultra-low ADV or zero volume days
    liquidity_blocks = 0
    try:
        adv_series = dataset["adv"].fillna(0.0) if "adv" in dataset else pd.Series(0.0, index=dataset.index)
        vol_series = dataset["volume"].fillna(0.0) if "volume" in dataset else pd.Series(0.0, index=dataset.index)
        adv_median = float(adv_series.median())
        if adv_median > 0:
            floor = adv_median * 0.1
            illiquid_mask = (adv_series < floor) | (vol_series <= 0)
            liquidity_blocks = int(illiquid_mask.sum())
            if liquidity_blocks:
                dataset.loc[illiquid_mask, "signal"] = 0.0
                raw_signal.loc[illiquid_mask] = 0.0
                probabilities.loc[illiquid_mask] = probabilities.loc[illiquid_mask]
    except Exception:
        liquidity_blocks = 0

    filtered_signal = apply_signal_filters(dataset, raw_signal, probabilities, params)
    dataset["signal"] = filtered_signal
    raw_signal = filtered_signal

    threshold_scan = scan_threshold_stability(
        probabilities.dropna(),
        dataset["future_return"],
        cost_rate,
        params.entry_threshold,
        params.exit_threshold,
    )
    heatmap = build_threshold_heatmap(threshold_scan) if threshold_scan else None

    df_for_backtest = dataset.copy()
    df_for_backtest["position"] = raw_signal.reindex(dataset.index).fillna(0.0)

    # Construct PnL with event-driven execution + costs
    position = df_for_backtest["position"].astype(float).fillna(0.0)
    asset_returns = compute_realized_returns(df_for_backtest, params)
    df_for_backtest["asset_return"] = asset_returns
    volatility = asset_returns.rolling(20).std().fillna(0.0) * np.sqrt(252)
    leverage = calculate_target_leverage(position, volatility, params.volatility_target, params.max_leverage)
    exposure, overlay_events = enforce_risk_limits(position, leverage, asset_returns, params)
    df_for_backtest["volatility"] = volatility
    df_for_backtest["leverage"] = leverage.reindex(df_for_backtest.index).fillna(0.0)
    df_for_backtest, exec_stats, exec_events = run_event_backtest(
        df_for_backtest,
        exposure,
        params,
        leverage=leverage,
    )
    adv_hits = int(exec_stats.get("adv_hard_cap_hits") or 0)

    perf_total = float(df_for_backtest["cum_strategy"].iloc[-1] - 1) if not df_for_backtest.empty else 0.0
    ret_series = df_for_backtest["strategy_return"].fillna(0.0)
    vol_ann = float(ret_series.std() * math.sqrt(252)) if not ret_series.empty else 0.0
    sharpe = float((ret_series.mean() * 252) / (ret_series.std() * math.sqrt(252) + 1e-12)) if not ret_series.empty else 0.0
    mdd = calculate_max_drawdown(df_for_backtest["cum_strategy"]) if not df_for_backtest.empty else 0.0

    calibration = _calibration_summary(probabilities, dataset["target"])
    calibration_plot = build_calibration_plot(calibration) if calibration else None

    returns_series = df_for_backtest.get("strategy_return")
    cpcv = compute_cpcv_report(returns_series)
    stress_report = build_stress_report(returns_series)
    psi_ret = 0.0
    psi_proba = 0.0
    if returns_series is not None and not returns_series.dropna().empty:
        mid = len(returns_series) // 2
        psi_ret = compute_psi(returns_series.iloc[:mid], returns_series.iloc[mid:])
    if probabilities is not None and not probabilities.dropna().empty:
        mid_p = len(probabilities) // 2
        psi_proba = compute_psi(probabilities.iloc[:mid_p], probabilities.iloc[mid_p:])
    stats: dict[str, Any] = {
        "validation_report": validation_report,
        "validation_oos_summary": summary_oos,
        "validation_summary_compact": summary_oos,
        "auc": float(np.nanmean(auc_scores)) if auc_scores else None,
        "shap_img": shap_img,
        "calibration": calibration,
        "calibration_plot": calibration_plot,
        "threshold_scan": threshold_scan,
        "threshold_heatmap": heatmap,
        "runtime_warnings": model_warnings,
        "hyperopt": hyperopt_report,
        "total_return": perf_total,
        "volatility": vol_ann,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "execution": {
            **exec_stats,
            "exec_cost_total": float(df_for_backtest["execution_cost"].sum()) if "execution_cost" in df_for_backtest else 0.0,
            "txn_cost_total": float(df_for_backtest["transaction_cost"].sum()) if "transaction_cost" in df_for_backtest else 0.0,
        },
        "execution_stats": {
            "avg_coverage": exec_stats.get("avg_coverage"),
            "unfilled_ratio": exec_stats.get("unfilled_ratio"),
            "avg_spread_bps": exec_stats.get("avg_spread_bps"),
            "halt_days": exec_stats.get("halt_days"),
            "limit_days": exec_stats.get("limit_days"),
            "participation": exec_stats.get("participation"),
            "effective_participation": exec_stats.get("effective_participation"),
            "adv_hard_cap_hits": adv_hits,
        },
        "signal_thresholds": {"long": long_threshold, "short": short_threshold},
        "cpcv": cpcv or {},
        "stress_test": stress_report or {},
        "drift": {"psi_returns": psi_ret, "psi_probabilities": psi_proba},
    }
    if overlay_events:
        stats.setdefault("risk_events", []).extend(overlay_events)
    if exec_events:
        stats.setdefault("risk_events", []).extend(exec_events)
    # 阈值敏感度：记录分布，若极端差则提示
    if threshold_scan:
        stability = {
            "mean_sharpe": threshold_scan.get("mean_sharpe"),
            "median_sharpe": threshold_scan.get("median_sharpe"),
            "iqr_sharpe": threshold_scan.get("iqr_sharpe"),
            "best": threshold_scan.get("best"),
            "worst": threshold_scan.get("worst"),
        }
        stats["threshold_stability"] = stability
        try:
            best_s = float((threshold_scan.get("best") or {}).get("sharpe") or 0.0)
            worst_s = float((threshold_scan.get("worst") or {}).get("sharpe") or 0.0)
            if worst_s < 0 and best_s > 0 and (best_s - worst_s) > abs(best_s) * 0.8:
                stats.setdefault("risk_events", []).append("阈值敏感度高：最佳/最差 Sharpe 差异显著，建议收紧阈值或降低杠杆。")
        except Exception:
            pass
    if adv_hits:
        stats.setdefault("risk_events", [])
        stats["risk_events"].append(
            f"因 ADV 参与率上限({(params.max_adv_participation or 0.1):.0%}) 压缩 {adv_hits} 次 ML 仓位，避免不可成交。"
        )
        stats["adv_hits"] = adv_hits
    if liquidity_blocks:
        stats.setdefault("risk_events", [])
        stats["risk_events"].append(f"因成交额过低/停牌，跳过 {liquidity_blocks} 个交易日的 ML 信号。")
        stats["liquidity_blocks"] = liquidity_blocks
    if cpcv:
        worst = cpcv.get("worst_sharpe")
        p10 = cpcv.get("p10_sharpe")
        if worst is not None and worst < 0:
            stats.setdefault("risk_events", []).append(f"CPCV 最差 Sharpe={worst:.2f}，提示策略在部分分段失效。")
        if p10 is not None and p10 < 0.2:
            stats.setdefault("risk_events", []).append(f"CPCV 10% 分位 Sharpe={p10:.2f}，建议收紧阈值/降杠杆。")
    if exec_stats:
        coverage = exec_stats.get("avg_coverage")
        if coverage is not None and coverage < 0.7:
            stats.setdefault("risk_events", []).append(f"执行覆盖率偏低（{coverage:.0%}），建议降低参与率或放宽成交假设。")
        halt_days = exec_stats.get("halt_days")
        if halt_days is not None and halt_days > 0:
            stats.setdefault("risk_events", []).append(f"检测到停牌/无成交 {int(halt_days)} 天，已跳过成交。")
        limit_days = exec_stats.get("limit_days")
        if limit_days is not None and limit_days > 0:
            stats.setdefault("risk_events", []).append(f"检测到涨跌幅限制 {int(limit_days)} 天，已跳过成交。")
    if stress_report:
        worst_mdd = stress_report.get("worst_mdd")
        worst_sharpe = stress_report.get("worst_sharpe")
        if worst_mdd is not None and worst_mdd < -0.25:
            stats.setdefault("risk_events", []).append(f"压力测试最差回撤 {worst_mdd:.1%}，需关注极端行情下的仓位韧性。")
        if worst_sharpe is not None and worst_sharpe < 0:
            stats.setdefault("risk_events", []).append(f"压力测试最差 Sharpe={worst_sharpe:.2f}，建议降杠杆或收紧阈值。")
    if psi_ret > 0.25 or psi_proba > 0.25:
        stats.setdefault("risk_events", []).append(f"检测到分布漂移（PSI: returns={psi_ret:.2f}, prob={psi_proba:.2f}），建议降杠杆或重新训练。")

    # 根据稳健性/漂移/执行给出自动降风险系数
    safety = 1.0
    safety_reasons: list[str] = []
    coverage = exec_stats.get("avg_coverage") if exec_stats else None
    if coverage is not None and coverage < 0.7:
        safety *= 0.8
        safety_reasons.append(f"执行覆盖率 {coverage:.0%} 低")
    if psi_ret > 0.25 or psi_proba > 0.25:
        safety *= 0.85
        safety_reasons.append(f"PSI 漂移 returns={psi_ret:.2f}, prob={psi_proba:.2f}")
    if stress_report and stress_report.get("worst_mdd", 0) < -0.25:
        safety *= 0.75
        safety_reasons.append(f"压力回撤 {stress_report.get('worst_mdd'):.1%}")
    if cpcv and cpcv.get("worst_sharpe", 1) < 0:
        safety *= 0.8
        safety_reasons.append(f"CPCV 最差 Sharpe {cpcv.get('worst_sharpe'):.2f}")
    safety = max(0.3, min(1.0, safety))
    stats["safety_multiplier"] = safety
    stats["safety_reasons"] = safety_reasons

    record_metric("ml_backtest_completed", engine=params.ml_model, tuned=bool(params.optimize_thresholds))
    metrics: list[dict[str, Any]] = []
    return df_for_backtest, metrics, stats


__all__ = [
    "_calibration_summary",
    "_generate_validation_report",
    "_scan_threshold_stability",
    "build_calibration_plot",
    "build_feature_matrix",
    "build_labels",
    "build_threshold_heatmap",
    "compute_triple_barrier_labels",
    "load_best_ml_config",
    "run_ml_backtest",
    "scan_threshold_stability",
    "tune_thresholds_on_validation",
]
