from __future__ import annotations

from typing import Any, Iterable, Sequence, TYPE_CHECKING
import json
import time
import os
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # Optional backends
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None  # type: ignore

try:
    from catboost import CatBoostClassifier  # type: ignore
except Exception:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from .strategies import StrategyInput
from django.conf import settings

from .ml_models import build_custom_sequence_model

DEFAULT_STRATEGY_SEED = int(os.environ.get("STRATEGY_SEED", "42"))
DATA_CACHE_DIR = settings.DATA_CACHE_DIR
TRAINING_DIR = DATA_CACHE_DIR / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)


class PurgedWalkForwardSplit:
    """Time-series split that respects embargo to avoid leakage."""

    def __init__(self, train_window: int, test_window: int, embargo: int = 0):
        self.train_window = max(50, int(train_window))
        self.test_window = max(5, int(test_window))
        self.embargo = max(0, int(embargo))

    def split(self, n_samples: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        start = self.train_window
        while start + self.test_window <= n_samples:
            train_start = max(0, start - self.train_window)
            train_end = max(train_start + 10, start - self.embargo)
            if train_end - train_start < 50:
                break
            test_start = start
            test_end = min(test_start + self.test_window, n_samples)
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            if len(test_idx) < 5:
                break
            yield train_idx, test_idx
            start += self.test_window


def _build_classifier(
    engine: str,
    params_dict: dict[str, Any],
    feature_columns: Sequence[str],
    base: "StrategyInput",
):
    # Enforce deterministic seeds across all engines
    try:
        seed = int(getattr(base, "random_seed", DEFAULT_STRATEGY_SEED))
    except Exception:
        seed = DEFAULT_STRATEGY_SEED
    engine = (engine or "sk_gbdt").lower()
    if engine == "lightgbm":
        if lgb is None:
            raise RuntimeError("lightgbm 未安装，无法执行该模型的超参搜索。")
        model = lgb.LGBMClassifier(
            random_state=seed,
            n_jobs=-1,
            **params_dict,
        )
        return Pipeline([("model", model)])
    if engine == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost 未安装，无法执行该模型的超参搜索。")
        params = params_dict.copy()
        params.setdefault("random_seed", seed)
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        model = CatBoostClassifier(**params)
        return Pipeline([("model", model)])
    if engine in {"lstm", "transformer", "seq_hybrid", "hybrid_seq", "fusion"}:
        model, _ = build_custom_sequence_model(engine, feature_columns, base)
        return model

    model = GradientBoostingClassifier(random_state=seed, **params_dict)
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def _compute_slippage_cost(
    exposure_change: pd.Series,
    prices: pd.Series,
    volume: pd.Series,
    slippage: dict[str, Any] | None,
) -> pd.Series:
    """Compute slippage cost per period given exposure change."""

    if slippage is None:
        slippage = {"type": "linear", "bps": 0.0}
    model = (slippage.get("type") or "linear").lower()
    bps = float(slippage.get("bps", 0.0)) / 10000.0
    filled_prices = prices.reindex(exposure_change.index).ffill().bfill().fillna(0.0)
    notionals = exposure_change.abs() * filled_prices
    if model == "linear":
        return notionals * bps
    if model == "sqrt":
        eta = float(slippage.get("eta", 1.0))
        vol = (
            slippage.get("volatility")
            if isinstance(slippage.get("volatility"), pd.Series)
            else prices.pct_change().rolling(slippage.get("vol_window", 20)).std()
        )
        vol = vol.reindex(exposure_change.index).ffill().bfill().fillna(0.0)
        if vol.abs().sum() == 0:
            vol = pd.Series(1.0, index=exposure_change.index)
        participation = np.abs(exposure_change.fillna(0.0)) / volume.reindex(exposure_change.index).replace(0, np.nan)
        impact = eta * vol * np.sqrt(participation.replace([np.inf, -np.inf], 0.0).fillna(0.0))
        return impact * filled_prices
    if model == "spread":
        spread = float(slippage.get("spread", 0.0))
        depth = float(slippage.get("depth", 1.0))
        return (spread / max(depth, 1e-9)) * exposure_change.abs()
    if model == "impact":
        eta = float(slippage.get("eta", 1.0))
        vol_window = int(slippage.get("vol_window", 20))
        sigma = prices.pct_change().rolling(vol_window, min_periods=5).std().reindex(exposure_change.index)
        sigma = sigma.ffill().bfill().fillna(0.0)
        depth = volume.reindex(exposure_change.index).replace(0, np.nan)
        participation = np.abs(exposure_change.fillna(0.0)) / depth
        impact = eta * sigma * np.sqrt(participation.replace([np.inf, -np.inf], 0.0).fillna(0.0))
        return impact * filled_prices
    if model == "spread_depth":
        # 双因子：点差 + 深度，允许自定义冲击斜率
        spread = float(slippage.get("spread", 0.0))
        depth = volume.reindex(exposure_change.index).fillna(0.0)
        depth_factor = float(slippage.get("depth_factor", 1.0))
        # 基础点差成本
        base = (spread / (depth.replace(0, np.nan))) * exposure_change.abs()
        base = base.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        # 深度冲击项（可堆叠）
        impact = depth_factor * (exposure_change.abs() / depth.replace(0, np.nan))
        impact = impact.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return (base + impact) * filled_prices
    return notionals * bps


def _simulate_returns(
    probabilities: np.ndarray,
    index: pd.Index,
    future_returns: pd.Series,
    entry: float,
    exit: float,
    cost_rate: float,
    *,
    slippage: dict[str, Any] | None = None,
    prices: pd.Series | None = None,
    volume: pd.Series | None = None,
    borrow_cost_bps: float = 0.0,
    long_borrow_cost_bps: float | None = None,
    short_borrow_cost_bps: float | None = None,
    adv: pd.Series | None = None,
    max_adv_participation: float | None = None,
) -> tuple[pd.Series, dict[str, float]]:
    proba_series = pd.Series(probabilities, index=index)
    signals = pd.Series(
        np.where(proba_series >= entry, 1.0, np.where(proba_series <= exit, -1.0, 0.0)),
        index=index,
    )
    exposure = signals.fillna(0.0)
    adv_series = adv.reindex(index).fillna(0.0) if adv is not None else None
    adv_hits = 0
    if adv_series is not None and max_adv_participation:
        cap = adv_series * max_adv_participation
        capped = exposure.where(exposure.abs() <= cap, 0.0)
        adv_hits = int((capped != exposure).sum())
        exposure = capped
    pnl = exposure * future_returns.reindex(index).fillna(0.0)
    turnover = signals.diff().abs().fillna(signals.abs())
    linear_cost = turnover * cost_rate

    price_series = prices.reindex(index).ffill().bfill() if prices is not None else pd.Series(1.0, index=index)
    volume_series = volume.reindex(index).fillna(0.0) if volume is not None else pd.Series(0.0, index=index)
    slippage_cost = _compute_slippage_cost(turnover, price_series, volume_series, slippage)

    lbps = borrow_cost_bps if long_borrow_cost_bps is None else long_borrow_cost_bps
    sbps = borrow_cost_bps if short_borrow_cost_bps is None else short_borrow_cost_bps
    long_daily = float(lbps) / 10000.0 / 252.0
    short_daily = float(sbps) / 10000.0 / 252.0
    borrow_cost = (
        exposure.clip(lower=0.0) * long_daily
        + (-exposure.clip(upper=0.0)) * short_daily
    )

    total_cost = linear_cost + slippage_cost + borrow_cost
    return pnl - total_cost, {
        "linear_cost": float(linear_cost.sum()),
        "slippage_cost": float(slippage_cost.sum()),
        "borrow_cost": float(borrow_cost.sum()),
        "adv_rejections": adv_hits,
    }


def _score_series(series: pd.Series) -> dict[str, float]:
    cleaned = series.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return {
            "sharpe": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "hit_ratio": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "cvar_95": 0.0,
            "recovery_period": 0.0,
            "loss_streak": 0,
        }
    ann_factor = 252
    mean = float(cleaned.mean())
    std = float(cleaned.std())
    sharpe = np.sqrt(ann_factor) * mean / std if std > 0 else 0.0
    cumulative = (1 + cleaned).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = float(drawdown.min())
    # 恢复期：距离上次净值高点的最大间隔（按样本个数计）
    recovery_period = 0
    last_peak = 0
    for idx, val in enumerate(drawdown):
        if val == 0:
            recovery_period = max(recovery_period, idx - last_peak)
            last_peak = idx
    if drawdown.iloc[-1] < 0:
        recovery_period = max(recovery_period, len(drawdown) - 1 - last_peak)
    periods = len(cleaned)
    cagr = cumulative.iloc[-1] ** (ann_factor / periods) - 1 if periods > 0 else 0.0
    hits = (cleaned > 0).sum()
    total = (cleaned != 0).sum()
    hit_ratio = float(hits / total) if total else 0.0
    downside = cleaned.copy()
    downside[downside > 0] = 0
    downside_std = float(np.sqrt((downside**2).mean())) if not downside.empty else 0.0
    sortino = np.sqrt(ann_factor) * mean / downside_std if downside_std > 0 else 0.0
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0
    # CVaR95 （简单尾部均值）
    q95 = float(np.quantile(cleaned, 0.05))
    tail = cleaned[cleaned <= q95]
    cvar_95 = float(tail.mean()) if not tail.empty else q95
    # 最长连亏
    loss_streak = 0
    current = 0
    for r in cleaned:
        if r < 0:
            current += 1
            loss_streak = max(loss_streak, current)
        else:
            current = 0
    return {
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "max_drawdown": float(max_drawdown),
        "hit_ratio": float(hit_ratio),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "cvar_95": float(cvar_95),
        "recovery_period": float(recovery_period),
        "loss_streak": int(loss_streak),
    }


def _suggest_params(trial: Any, engine: str) -> dict[str, Any]:
    engine = (engine or "sk_gbdt").lower()
    if engine == "lightgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        }
    if engine == "catboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "iterations": trial.suggest_int("iterations", 200, 800, step=50),
            "depth": trial.suggest_int("depth", 4, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    if engine == "lstm":
        return {
            "sequence_length": trial.suggest_int("sequence_length", 16, 64, step=4),
            "hidden_dim": trial.suggest_int("hidden_dim", 32, 160, step=16),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "epochs": trial.suggest_int("epochs", 6, 20),
            "batch_size": trial.suggest_int("batch_size", 32, 96, step=16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-3, log=True),
        }
    if engine == "transformer":
        return {
            "sequence_length": trial.suggest_int("sequence_length", 16, 64, step=4),
            "model_dim": trial.suggest_int("model_dim", 64, 192, step=16),
            "num_heads": trial.suggest_int("num_heads", 2, 6),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "epochs": trial.suggest_int("epochs", 6, 18),
            "batch_size": trial.suggest_int("batch_size", 32, 96, step=16),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        }
    if engine in {"seq_hybrid", "hybrid_seq", "fusion"}:
        return {
            "sequence_length": trial.suggest_int("sequence_length", 16, 64, step=4),
            "hidden_dim": trial.suggest_int("hidden_dim", 32, 160, step=16),
            "lstm_num_layers": trial.suggest_int("lstm_num_layers", 1, 3),
            "lstm_dropout": trial.suggest_float("lstm_dropout", 0.0, 0.4),
            "lstm_epochs": trial.suggest_int("lstm_epochs", 6, 20),
            "model_dim": trial.suggest_int("model_dim", 64, 192, step=16),
            "num_heads": trial.suggest_int("num_heads", 2, 6),
            "transformer_num_layers": trial.suggest_int("transformer_num_layers", 1, 4),
            "transformer_dropout": trial.suggest_float("transformer_dropout", 0.0, 0.3),
            "transformer_epochs": trial.suggest_int("transformer_epochs", 6, 18),
            "validation_ratio": trial.suggest_float("validation_ratio", 0.1, 0.3),
            "score_tolerance": trial.suggest_float("score_tolerance", 0.005, 0.05),
        }
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=25),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


def _evaluate_candidate(
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    engine: str,
    entry_thr: float,
    exit_thr: float,
    model_params: dict[str, Any],
    splitter: PurgedWalkForwardSplit,
    cost_rate: float,
    base_params: "StrategyInput",
) -> dict[str, float]:
    fold_stats: list[dict[str, float]] = []
    auc_scores: list[float] = []

    def _aggregate(values: list[float]) -> dict[str, float]:
        arr = np.array(values, dtype=float)
        if arr.size == 0:
            return {}
        q1, q3 = np.percentile(arr, [25, 75])
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "iqr": float(q3 - q1),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }

    for train_idx, test_idx in splitter.split(len(dataset)):
        train_slice = dataset.iloc[train_idx]
        test_slice = dataset.iloc[test_idx]
        if train_slice["target"].nunique() < 2:
            continue
        clf = _build_classifier(engine, model_params, feature_columns, base_params)
        clf.fit(train_slice[feature_columns], train_slice["target"])
        proba = clf.predict_proba(test_slice[feature_columns])[:, 1]
        net_returns, _ = _simulate_returns(
            proba,
            test_slice.index,
            test_slice["future_return"],
            entry_thr,
            exit_thr,
            cost_rate,
            slippage=getattr(base_params, "slippage_model", None),
            prices=test_slice.get("adj close") if "adj close" in test_slice else None,
            volume=test_slice.get("volume") if "volume" in test_slice else None,
            borrow_cost_bps=getattr(base_params, "borrow_cost_bps", 0.0),
            long_borrow_cost_bps=getattr(base_params, "long_borrow_cost_bps", None),
            short_borrow_cost_bps=getattr(base_params, "short_borrow_cost_bps", None),
            adv=test_slice.get("adv") if "adv" in test_slice else None,
            max_adv_participation=getattr(base_params, "max_adv_participation", None),
        )
        stats = _score_series(net_returns)
        fold_stats.append(stats)
        if test_slice["target"].nunique() > 1:
            try:
                auc_scores.append(roc_auc_score(test_slice["target"], proba))
            except Exception:
                pass
    if not fold_stats:
        return {
            "score": float("-inf"),
            "sharpe": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "hit_ratio": 0.0,
            "auc": float("nan"),
            "folds": 0,
            "oos_summary": {},
        }
    summary = {
        "sharpe": _aggregate([s["sharpe"] for s in fold_stats]),
        "cagr": _aggregate([s["cagr"] for s in fold_stats]),
        "max_drawdown": _aggregate([s["max_drawdown"] for s in fold_stats]),
        "hit_ratio": _aggregate([s["hit_ratio"] for s in fold_stats]),
        "cvar_95": _aggregate([s.get("cvar_95", 0.0) for s in fold_stats]),
        "recovery_period": _aggregate([s.get("recovery_period", 0.0) for s in fold_stats]),
        "loss_streak": _aggregate([s.get("loss_streak", 0.0) for s in fold_stats]),
    }
    mean_sharpe = summary["sharpe"].get("mean", 0.0)
    std_sharpe = summary["sharpe"].get("std", 0.0)
    penalized = mean_sharpe - std_sharpe
    agg = {
        "sharpe": float(mean_sharpe),
        "cagr": float(summary.get("cagr", {}).get("mean", 0.0)),
        "max_drawdown": float(summary.get("max_drawdown", {}).get("mean", 0.0)),
        "hit_ratio": float(summary.get("hit_ratio", {}).get("mean", 0.0)),
        "cvar_95": float(summary.get("cvar_95", {}).get("mean", 0.0)),
        "recovery_period": float(summary.get("recovery_period", {}).get("mean", 0.0)),
        "loss_streak": float(summary.get("loss_streak", {}).get("mean", 0.0)),
        "auc": float(np.nanmean(auc_scores)) if auc_scores else float("nan"),
        "folds": len(fold_stats),
        "oos_summary": summary,
        "penalized_sharpe": float(penalized),
    }
    agg["score"] = penalized + 0.3 * agg["hit_ratio"] - 0.1 * abs(agg["max_drawdown"])
    return agg


def run_optuna_search(
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    params: Any,
) -> dict[str, Any]:
    """Run Optuna search for ML momentum strategy."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Optuna 未安装，无法执行自动超参数搜索。请先 pip install optuna。") from exc

    # Ensure deterministic seeds inside search
    random.seed(DEFAULT_STRATEGY_SEED)
    np.random.seed(DEFAULT_STRATEGY_SEED)

    if dataset.empty or not feature_columns:
        raise RuntimeError("样本不足，无法执行自动超参搜索。")

    trials = max(1, int(getattr(params, "hyperopt_trials", 20)))
    timeout = getattr(params, "hyperopt_timeout", None)
    timeout = int(timeout) if timeout else None
    engine = getattr(params, "ml_model", "sk_gbdt")
    train_window = getattr(params, "train_window", 252)
    test_window = getattr(params, "test_window", 21)
    embargo = getattr(params, "embargo_days", 5)
    splitter = PurgedWalkForwardSplit(train_window, test_window, embargo)
    cost_rate = (getattr(params, "transaction_cost_bps", 8.0) + getattr(params, "slippage_bps", 5.0)) / 10000.0
    study_name = f"{getattr(params, 'ticker', 'unknown')}_{engine}_hyperopt"
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)

    start_ts = time.time()

    def objective(trial: optuna.trial.Trial) -> float:
        entry_thr = trial.suggest_float("entry_threshold", 0.52, 0.7)
        exit_thr = trial.suggest_float("exit_threshold", 0.28, 0.48)
        if exit_thr >= 0.5 or entry_thr <= 0.5 or (entry_thr - exit_thr) < 0.05:
            raise optuna.TrialPruned()
        model_hyperparams = _suggest_params(trial, engine)
        metrics = _evaluate_candidate(
            dataset,
            feature_columns,
            engine,
            entry_thr,
            exit_thr,
            model_hyperparams,
            splitter,
            cost_rate,
            params,
        )
        if not np.isfinite(metrics["score"]):
            raise optuna.TrialPruned()
        trial.set_user_attr("metrics", metrics)
        return metrics["score"]

    study.optimize(objective, n_trials=trials, timeout=timeout, n_jobs=1, show_progress_bar=False)
    if not study.best_trials:
        raise RuntimeError("自动超参搜索未得到有效结果，请检查数据窗口或减少约束。")

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    ml_params = {k: v for k, v in best_params.items() if k not in {"entry_threshold", "exit_threshold"}}
    best_metrics = best_trial.user_attrs.get("metrics", {})
    result = {
        "ticker": getattr(params, "ticker", "").upper(),
        "engine": engine,
        "entry_threshold": float(best_params["entry_threshold"]),
        "exit_threshold": float(best_params["exit_threshold"]),
        "ml_params": ml_params,
        "best_score": float(best_trial.value),
        "fold_metrics": best_metrics,
        "oos_summary": best_metrics.get("oos_summary") if isinstance(best_metrics, dict) else {},
        "penalized_sharpe": best_metrics.get("penalized_sharpe") if isinstance(best_metrics, dict) else None,
        "trials_ran": len(study.trials),
        "duration_seconds": round(time.time() - start_ts, 2),
    }

    out_path = TRAINING_DIR / f"hyperopt_{result['ticker'] or 'unknown'}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    return result
