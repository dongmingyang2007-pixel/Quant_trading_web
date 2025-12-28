from __future__ import annotations

from typing import Dict, Iterable, Mapping

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


def build_target_weights(
    scores: Mapping[str, float],
    *,
    allow_short: bool = True,
    max_weight: float | None = None,
    min_weight: float | None = None,
    max_holdings: int | None = None,
) -> dict[str, float]:
    """Map raw scores into normalized portfolio weights."""
    if not scores:
        return {}
    items = [(k, float(v)) for k, v in scores.items() if v is not None]
    if max_holdings:
        items = sorted(items, key=lambda x: abs(x[1]), reverse=True)[: max(1, int(max_holdings))]
    longs = {k: max(0.0, v) for k, v in items}
    shorts = {k: min(0.0, v) for k, v in items} if allow_short else {}
    long_sum = sum(longs.values())
    short_sum = sum(abs(v) for v in shorts.values())
    weights: dict[str, float] = {}
    if long_sum > 0:
        for k, v in longs.items():
            weights[k] = v / long_sum
    if short_sum > 0:
        for k, v in shorts.items():
            weights[k] = weights.get(k, 0.0) - (abs(v) / short_sum)
    if not weights:
        return {}
    if max_weight is not None:
        max_weight = abs(float(max_weight))
        for k, v in list(weights.items()):
            weights[k] = float(np.clip(v, -max_weight, max_weight))
    if min_weight is not None and min_weight > 0:
        min_weight = float(min_weight)
        for k, v in list(weights.items()):
            if v > 0 and v < min_weight:
                weights[k] = 0.0
            if v < 0 and abs(v) < min_weight:
                weights[k] = 0.0
    # Re-normalize to keep total gross exposure at 1.0
    gross = sum(abs(v) for v in weights.values())
    if gross > 0:
        weights = {k: v / gross for k, v in weights.items()}
    return weights


def apply_sector_caps(
    weights: Mapping[str, float],
    sector_map: Mapping[str, str] | None,
    sector_caps: Mapping[str, float] | None,
) -> dict[str, float]:
    if not weights or not sector_map or not sector_caps:
        return dict(weights)
    capped = dict(weights)
    for sector, cap in sector_caps.items():
        members = [s for s, sec in sector_map.items() if sec == sector and s in capped]
        if not members:
            continue
        total = sum(capped[m] for m in members)
        if abs(total) <= cap:
            continue
        scale = cap / abs(total) if total else 1.0
        for sym in members:
            capped[sym] = capped[sym] * scale
    return capped


def cap_turnover(
    previous: Mapping[str, float],
    target: Mapping[str, float],
    turnover_cap: float | None,
) -> dict[str, float]:
    if turnover_cap is None or turnover_cap <= 0:
        return dict(target)
    keys = set(previous) | set(target)
    prev = {k: float(previous.get(k, 0.0)) for k in keys}
    tgt = {k: float(target.get(k, 0.0)) for k in keys}
    turnover = sum(abs(tgt[k] - prev[k]) for k in keys)
    if turnover <= turnover_cap or turnover == 0:
        return tgt
    scale = turnover_cap / turnover
    return {k: prev[k] + (tgt[k] - prev[k]) * scale for k in keys}


def build_trade_list(
    target_weights: Mapping[str, float],
    current_positions: Mapping[str, float] | None,
    prices: Mapping[str, float],
    *,
    capital: float,
    lot_size: int = 1,
    min_trade_value: float = 0.0,
) -> list[dict[str, float | str]]:
    """Translate target weights into executable trade list."""
    trades: list[dict[str, float | str]] = []
    positions = {k: float(v) for k, v in (current_positions or {}).items()}
    capital = max(float(capital), 0.0)
    lot_size = max(int(lot_size), 1)
    for symbol, weight in target_weights.items():
        price = float(prices.get(symbol, 0.0) or 0.0)
        if price <= 0:
            continue
        target_value = float(weight) * capital
        current_qty = float(positions.get(symbol, 0.0))
        current_value = current_qty * price
        delta_value = target_value - current_value
        if abs(delta_value) < min_trade_value:
            continue
        raw_qty = delta_value / price
        qty = int(abs(raw_qty) // lot_size) * lot_size
        if qty <= 0:
            continue
        side = "BUY" if raw_qty > 0 else "SELL"
        trades.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": float(qty),
                "price": round(price, 4),
                "notional": round(price * qty, 2),
            }
        )
    return trades
