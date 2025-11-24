from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import StrategyInput


def enforce_min_holding(position: pd.Series, min_days: int, probability: pd.Series | None = None) -> pd.Series:
    """维持最小持仓天数，避免频繁换仓抖动。"""
    if min_days <= 1 or position.empty:
        return position.fillna(0.0)
    adjusted = position.copy()
    last_pos = 0.0
    last_change_idx: int | None = None
    for idx, value in enumerate(position):
        if np.isnan(value):
            adjusted.iloc[idx] = last_pos
            continue
        if last_change_idx is None:
            last_pos = value
            last_change_idx = idx
            continue
        if value == last_pos:
            adjusted.iloc[idx] = value
            continue
        if idx - last_change_idx < min_days:
            # 如果概率可用，则在高置信度时允许提前切换
            if probability is not None:
                try:
                    prob = float(probability.iloc[idx])
                except Exception:
                    prob = None
                if prob is not None and prob >= 0.7:
                    adjusted.iloc[idx] = value
                    last_pos = value
                    last_change_idx = idx
                    continue
            adjusted.iloc[idx] = last_pos
        else:
            adjusted.iloc[idx] = value
            last_pos = value
            last_change_idx = idx
    return adjusted.fillna(0.0)


def calculate_target_leverage(
    position: pd.Series,
    realized_vol: pd.Series,
    target_vol: float = 0.15,
    max_leverage: float = 3.0,
) -> pd.Series:
    """按目标年化波动缩放杠杆，空仓时强制为 0。"""
    vol = realized_vol.replace(0, np.nan)
    leverage = target_vol / vol
    leverage = leverage.clip(lower=0, upper=max_leverage).fillna(0)
    leverage = leverage.where(position != 0, 0)
    return leverage


def apply_vol_targeting(
    exposure: pd.Series,
    asset_returns: pd.Series,
    params: StrategyInput,
) -> tuple[pd.Series, list[str]]:
    """根据 realized vol 对仓位进行动态缩放。"""
    events: list[str] = []
    target = params.target_vol
    if target is None or target <= 0:
        return exposure, events
    window = max(10, int(params.vol_target_window or 60))
    rolling_vol = asset_returns.rolling(window).std().fillna(0.0) * np.sqrt(252)
    current_vol = rolling_vol.reindex(exposure.index).ffill().bfill()
    scale = target / current_vol.replace(0, np.nan)
    scale = scale.clip(lower=0.0, upper=3.0).fillna(0.0)
    scaled = exposure * scale
    if not scaled.equals(exposure):
        events.append(f"波动率目标生效：目标 {target:.2f}，窗口 {window} 天，缩放上限 3x。")
    return scaled, events


def enforce_risk_limits(
    position: pd.Series,
    leverage: pd.Series,
    asset_returns: pd.Series,
    params: StrategyInput,
) -> tuple[pd.Series, list[str]]:
    """综合日内损失/最大回撤/波动率目标的仓位调节。"""
    exposure = (position * leverage).fillna(0.0)
    exposure = exposure.clip(lower=-params.max_leverage, upper=params.max_leverage)
    exposure, vol_events = apply_vol_targeting(exposure, asset_returns, params)

    intraday_limit = max(params.intraday_loss_limit or 0.0, 0.0)
    max_drawdown_stop = max(params.max_drawdown_stop or 0.0, 0.0)
    prev_exposure = 0.0
    cumulative = 1.0
    peak = 1.0
    day_start_val = 1.0
    current_day: datetime.date | None = None
    stopped = False
    stop_dates: list[str] = []
    resumed_dates: list[str] = []
    limit_hits = 0
    adjusted = []
    index = exposure.index
    for i, exp in enumerate(exposure):
        try:
            ts = index[i]
        except Exception:
            ts = None
        ret = asset_returns.iloc[i] if i < len(asset_returns) else 0.0
        if ts is not None and (current_day is None or ts.date() != current_day):
            day_start_val = cumulative
            current_day = ts.date()

        cumulative = max(1e-9, cumulative * (1 + prev_exposure * ret))
        peak = max(peak, cumulative)
        drawdown = cumulative / peak - 1

        intraday_drawdown = cumulative / day_start_val - 1
        if not stopped and intraday_limit > 0 and intraday_drawdown <= -intraday_limit:
            stopped = True
            stop_dates.append(f"{ts.date()}(日内止损)")
        elif stopped and intraday_drawdown >= -0.01 and drawdown >= -0.05:
            stopped = False
            resumed_dates.append(str(ts.date()))

        if not stopped and max_drawdown_stop > 0 and drawdown <= -max_drawdown_stop:
            stopped = True
            stop_dates.append(str(ts.date()))
        elif stopped and drawdown >= -0.02:
            stopped = False
            resumed_dates.append(str(ts.date()))

        prev_exposure = exp
        adjusted.append(0.0 if stopped else exp)

    exposure_series = pd.Series(adjusted, index=index)
    events: list[str] = []
    if limit_hits:
        events.append(f"日曝险限制被触发 {limit_hits} 次，系统自动压降仓位。")
    if stop_dates:
        latest = stop_dates[-1]
        if resumed_dates:
            events.append(f"最大回撤止损在 {latest} 生效，{resumed_dates[-1]} 恢复交易。")
        else:
            events.append(f"最大回撤止损在 {latest} 生效，尚未恢复交易。")
    events.extend(vol_events)
    return exposure_series, events


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """计算最大回撤。"""
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1
    return drawdown.min()


def calculate_drawdown_series(cumulative_returns: pd.Series) -> pd.Series:
    """返回逐日回撤序列。"""
    rolling_max = cumulative_returns.cummax()
    return cumulative_returns / rolling_max - 1


__all__ = [
    "apply_vol_targeting",
    "calculate_drawdown_series",
    "calculate_max_drawdown",
    "calculate_target_leverage",
    "enforce_min_holding",
    "enforce_risk_limits",
]
