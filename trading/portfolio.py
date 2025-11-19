from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def _equal_weights(keys: Iterable[str]) -> dict[str, float]:
    keys = list(keys)
    n = len(keys)
    if n == 0:
        return {}
    w = 1.0 / n
    return {k: w for k in keys}


def _erc_weights(pnl_dict: Dict[str, pd.Series]) -> dict[str, float]:
    keys = list(pnl_dict.keys())
    if not keys:
        return {}
    rets = pd.DataFrame({k: pnl_dict[k].reindex_like(next(iter(pnl_dict.values()))) for k in keys}).fillna(0.0)
    cov = rets.cov()
    try:
        diag = np.diag(cov)
        diag = np.where(diag == 0, 1e-8, diag)
        inv_vol = 1.0 / np.sqrt(diag)
        inv_vol = np.where(np.isfinite(inv_vol), inv_vol, 0.0)
        if inv_vol.sum() == 0:
            return _equal_weights(keys)
        weights = inv_vol / inv_vol.sum()
        return {k: float(weights[idx]) for idx, k in enumerate(keys)}
    except Exception:
        return _equal_weights(keys)


def _corr_weights(pnl_dict: Dict[str, pd.Series]) -> dict[str, float]:
    keys = list(pnl_dict.keys())
    if not keys:
        return {}
    rets = pd.DataFrame({k: pnl_dict[k] for k in keys}).fillna(0.0)
    corr = rets.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol = rets.std().replace(0, np.nan)
    mean_corr = corr.abs().mean().replace(0, np.nan)
    score = 1.0 / (vol * mean_corr)
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if score.sum() == 0:
        return _equal_weights(keys)
    weights = score / score.sum()
    return {k: float(weights.get(k, 0.0)) for k in keys}


def combine(pnl_dict: Dict[str, pd.Series], scheme: str = "equal") -> tuple[pd.Series, dict[str, float]]:
    """Combine multiple PnL series into a portfolio using given weighting scheme.

    Args:
        pnl_dict: mapping from symbol to PnL/returns series.
        scheme: 'equal' or 'erc' (risk parity/vol parity).
    Returns:
        combined PnL series, and the weight mapping used.
    """

    if not pnl_dict:
        return pd.Series(dtype=float), {}
    keys = list(pnl_dict.keys())
    if scheme == "erc":
        weights = _erc_weights(pnl_dict)
    elif scheme == "corr":
        weights = _corr_weights(pnl_dict)
    else:
        weights = _equal_weights(keys)
    aligned = pd.DataFrame({k: pnl_dict[k] for k in keys}).fillna(0.0)
    weight_vec = np.array([weights.get(k, 0.0) for k in keys])
    combined = aligned.dot(weight_vec)
    return combined, weights


def portfolio_stats(pnl: pd.Series) -> dict[str, float]:
    """Compute portfolio-level risk stats for combined PnL."""

    clean = pnl.dropna().astype(float)
    if clean.empty:
        return {}
    ann = 252.0
    cumulative = (1 + clean).cumprod()
    ret = float(cumulative.iloc[-1] - 1)
    vol = float(clean.std() * np.sqrt(ann))
    sharpe = float(np.sqrt(ann) * clean.mean() / clean.std()) if clean.std() > 0 else 0.0
    max_dd = float((cumulative / cumulative.cummax() - 1).min())
    downside = clean.copy()
    downside[downside > 0] = 0
    sortino = float(np.sqrt(ann) * clean.mean() / downside.std(ddof=0)) if downside.std(ddof=0) > 0 else 0.0
    # Simple tail risk proxies
    q95 = float(np.quantile(clean, 0.05))
    cvar95 = float(clean[clean <= q95].mean()) if (clean <= q95).any() else q95
    # Recovery metrics
    cumulative = (1 + clean).cumprod()
    peaks = cumulative.cummax()
    underwater = cumulative / peaks - 1
    recovery_days = 0
    last_peak = cumulative.index[0]
    for idx, val in underwater.items():
        if val == 0:
            recovery_days = max(recovery_days, (idx - last_peak).days if hasattr(idx, "__sub__") else recovery_days)
            last_peak = idx
    if underwater.iloc[-1] < 0:
        recovery_days = max(recovery_days, (underwater.index[-1] - last_peak).days if hasattr(underwater.index[-1], "__sub__") else recovery_days)
    loss_streak = int((clean < 0).astype(int).groupby((clean >= 0).astype(int).cumsum()).sum().max()) if not clean.empty else 0
    return {
        "total_return": ret,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "cvar_95": cvar95,
        "recovery_days": recovery_days,
        "loss_streak": loss_streak,
    }
