from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
try:  # optional dependency for p-value computation
    from scipy.stats import norm  # type: ignore
except Exception:  # pragma: no cover
    norm = None  # type: ignore
import random


def _newey_west_variance(series: pd.Series, lag: int | None = None) -> float:
    """Return the variance of the sample mean using Newey-West HAC estimator."""

    clean = series.dropna().astype(float)
    n = len(clean)
    if n < 5:
        return 0.0
    residuals = clean - clean.mean()
    if lag is None:
        lag = int(np.ceil(np.sqrt(n)))
    gamma0 = float(np.dot(residuals, residuals) / n)
    var = gamma0
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        cov = float(np.dot(residuals[k:], residuals[:-k]) / n)
        var += 2 * weight * cov
    return max(var / n, 0.0)


def _deflated_sharpe(sharpe: float, n_obs: int, trials: int) -> float | None:
    if n_obs <= 2:
        return None
    trials = max(1, trials)
    if trials == 1:
        return sharpe
    z = math.sqrt(2 * math.log(trials))
    if not math.isfinite(z) or z <= 0:
        return sharpe
    c = z + (math.log(math.pi) + math.log(math.log(trials))) / (2 * z)
    variance = max((1 - c**2) / (n_obs - 1), 1e-9)
    return (sharpe - c) / math.sqrt(variance)


def compute_robust_sharpe(
    returns: pd.Series,
    *,
    annual_factor: int = 252,
    trials: int = 1,
) -> dict[str, Any]:
    """
    Estimate Sharpe standard error using Newey-West variance and compute deflated Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Strategy net returns (periodic).
    annual_factor : int
        Trading periods per year.
    trials : int
        Number of strategy variations evaluated (used for DSR adjustment).
    """

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(clean)
    if n < 5:
        return {}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    if std == 0:
        return {}
    sharpe = math.sqrt(annual_factor) * mean / std
    var_mean = _newey_west_variance(clean)
    if var_mean <= 0:
        return {"sharpe": sharpe}
    std_error = math.sqrt(annual_factor) * math.sqrt(var_mean) / std
    z = 1.96
    ci = (sharpe - z * std_error, sharpe + z * std_error)
    dsr = _deflated_sharpe(sharpe, n, trials)
    payload: dict[str, Any] = {
        "sharpe": sharpe,
        "std_error": std_error,
        "ci": ci,
    }
    if dsr is not None:
        payload["deflated_sharpe"] = dsr
    return payload


def compute_white_reality_check(
    returns: pd.Series,
    *,
    trials: int = 1,
    annual_factor: int = 252,
) -> dict[str, Any]:
    """
    Approximate White's Reality Check by adjusting Sharpe significance for multiple tests.

    Uses Newey-West variance to estimate Sharpe standard error and applies a
    Bonferroni-style correction on the two-sided p-value. This is a lightweight
    alternative to full bootstrap RC that still highlights data-snooping risk.
    """

    robust = compute_robust_sharpe(returns, annual_factor=annual_factor, trials=trials)
    sharpe = robust.get("sharpe") if robust else None
    std_error = robust.get("std_error") if robust else None
    if sharpe is None or std_error in (None, 0):
        return {}
    z_score = sharpe / std_error
    if norm is None:
        return {"sharpe": sharpe, "z_score": z_score, "trials": trials}
    p_value = float(2 * (1 - norm.cdf(abs(z_score))))
    adjusted = min(1.0, p_value * max(1, trials))
    return {
        "sharpe": sharpe,
        "z_score": z_score,
        "p_value": p_value,
        "p_value_adjusted": adjusted,
        "trials": trials,
    }


def _choose_block_size(n: int, block_size: int | None) -> int:
    if block_size and block_size > 0:
        return int(block_size)
    # Rule-of-thumb: sqrt(n)/2, min 2
    return max(2, int(math.sqrt(max(n, 2)) / 2))


def _stationary_bootstrap(series: np.ndarray, block_size: int, n_samples: int, seed: int | None = None) -> np.ndarray:
    """Simple stationary bootstrap for dependent returns."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    n = len(series)
    out = np.empty((n_samples, n))
    for i in range(n_samples):
        idx = []
        t = random.randint(0, n - 1)
        for _ in range(n):
            idx.append(t)
            if random.random() < 1.0 / max(block_size, 1):
                t = random.randint(0, n - 1)
            else:
                t = (t + 1) % n
        out[i] = series[idx]
    return out


def compute_white_reality_check_bootstrap(
    returns: pd.Series,
    *,
    trials: int = 1,
    block_size: int | None = None,
    bootstrap_samples: int = 500,
    annual_factor: int = 252,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Bootstrap-based White's Reality Check / SPA-style p-value for Sharpe.

    - Uses stationary bootstrap to preserve dependence.
    - If `trials` > 1, computes p-value against the max Sharpe across trials
      using the bootstrap distribution (proxy for model selection bias).
    """

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(clean)
    if n < 20:
        return {}
    obs_sharpe = float(np.sqrt(annual_factor) * clean.mean() / clean.std(ddof=1)) if clean.std(ddof=1) > 0 else 0.0
    if obs_sharpe == 0:
        return {}

    blk = _choose_block_size(n, block_size)
    boot_matrix = _stationary_bootstrap(clean.to_numpy(), block_size=blk, n_samples=bootstrap_samples, seed=seed)
    sharpe_samples = []
    for sample in boot_matrix:
        std = sample.std(ddof=1)
        if std == 0:
            sharpe_samples.append(0.0)
        else:
            sharpe_samples.append(float(np.sqrt(annual_factor) * sample.mean() / std))
    sharpe_samples = np.array(sharpe_samples)
    if trials > 1:
        # approximate max over trials by sampling the max of (trials) bootstrap sharpe draws
        pooled = []
        for i in range(0, len(sharpe_samples), trials):
            chunk = sharpe_samples[i : i + trials]
            if chunk.size == 0:
                continue
            pooled.append(float(np.max(chunk)))
        if pooled:
            sharpe_samples = np.array(pooled)
    p_boot = float((sharpe_samples >= obs_sharpe).mean()) if sharpe_samples.size else 1.0
    return {
        "sharpe_observed": obs_sharpe,
        "p_value_bootstrap": p_boot,
        "bootstrap_samples": int(bootstrap_samples),
        "block_size": int(blk),
        "trials": int(trials),
        "sharpe_bootstrap_mean": float(np.mean(sharpe_samples)) if sharpe_samples.size else None,
        "sharpe_bootstrap_std": float(np.std(sharpe_samples)) if sharpe_samples.size else None,
    }


def compute_spa_pvalue(
    returns: pd.Series,
    *,
    block_size: int | None = None,
    bootstrap_samples: int = 500,
    annual_factor: int = 252,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Hansen SPA-style p-value for Sharpe: center returns and bootstrap max Sharpe.
    Simplified: centers series to remove sample mean (controls data snooping) and
    bootstraps Sharpe distribution to compute p-value of observed Sharpe.
    """

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(clean)
    if n < 20:
        return {}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    if std == 0:
        return {}
    obs_sharpe = math.sqrt(annual_factor) * mean / std

    centered = clean - mean
    blk = _choose_block_size(n, block_size)
    boot_matrix = _stationary_bootstrap(centered.to_numpy(), block_size=blk, n_samples=bootstrap_samples, seed=seed)
    sharpe_samples = []
    for sample in boot_matrix:
        s_std = sample.std(ddof=1)
        if s_std == 0:
            sharpe_samples.append(0.0)
        else:
            sharpe_samples.append(float(math.sqrt(annual_factor) * sample.mean() / s_std))
    sharpe_samples = np.array(sharpe_samples)
    p_spa = float((sharpe_samples >= obs_sharpe).mean()) if sharpe_samples.size else 1.0
    return {
        "sharpe_observed": obs_sharpe,
        "p_value_spa": p_spa,
        "bootstrap_samples": int(bootstrap_samples),
        "block_size": int(blk),
        "sharpe_bootstrap_mean": float(np.mean(sharpe_samples)) if sharpe_samples.size else None,
        "sharpe_bootstrap_std": float(np.std(sharpe_samples)) if sharpe_samples.size else None,
    }


def calculate_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """Compute Conditional VaR / Expected Shortfall at given confidence level."""

    clean = returns.dropna().astype(float)
    if clean.empty:
        return 0.0
    cutoff = np.quantile(clean, 1 - alpha)
    tail = clean[clean <= cutoff]
    if tail.empty:
        return float(cutoff)
    return float(tail.mean())


def recovery_period_days(cumulative: pd.Series) -> int:
    """Computes the longest time to recover from drawdowns (in days)."""

    if cumulative.empty:
        return 0
    peaks = cumulative.cummax()
    underwater = cumulative / peaks - 1
    max_recovery = 0
    last_peak_idx = cumulative.index[0]
    for idx, val in underwater.items():
        if val == 0:
            # recovered
            recovery = (idx - last_peak_idx).days if hasattr(idx, "__sub__") else 0
            max_recovery = max(max_recovery, recovery)
            last_peak_idx = idx
    # If never recovered at the end
    if underwater.iloc[-1] < 0:
        recovery = (underwater.index[-1] - last_peak_idx).days if hasattr(underwater.index[-1], "__sub__") else 0
        max_recovery = max(max_recovery, recovery)
    return int(max_recovery)
