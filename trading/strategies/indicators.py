from __future__ import annotations

from datetime import date
from typing import Any, Tuple
import math

import numpy as np
import pandas as pd

from .config import StrategyInput, QuantStrategyError


def compute_indicators(
    prices: pd.DataFrame, short_window: int, long_window: int, rsi_period: int
) -> pd.DataFrame:
    """计算趋势/波动/成交量等技术指标，用于信号和特征构建。"""
    if long_window <= short_window:
        raise QuantStrategyError("The long window must be greater than the short window.")

    prices = prices.sort_index().copy()
    prices["sma_short"] = prices["adj close"].rolling(window=short_window).mean()
    prices["sma_long"] = prices["adj close"].rolling(window=long_window).mean()

    # RSI (Wilder's smoothing)
    delta = prices["adj close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    alpha = 1 / max(rsi_period, 1)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    prices["rsi"] = (100 - (100 / (1 + rs))).fillna(0)

    prices["ema_short"] = prices["adj close"].ewm(span=max(short_window // 2, 2), adjust=False).mean()
    prices["ema_long"] = prices["adj close"].ewm(span=max(long_window // 2, 4), adjust=False).mean()
    prices["ema_trend"] = prices["ema_short"] / prices["ema_long"] - 1

    # Bollinger Bands
    rolling_std = prices["adj close"].rolling(window=long_window).std()
    prices["boll_up"] = prices["sma_long"] + 2 * rolling_std
    prices["boll_dn"] = prices["sma_long"] - 2 * rolling_std

    # MACD
    ema_fast = prices["adj close"].ewm(span=12, adjust=False).mean()
    ema_slow = prices["adj close"].ewm(span=26, adjust=False).mean()
    prices["macd"] = ema_fast - ema_slow
    prices["macd_signal"] = prices["macd"].ewm(span=9, adjust=False).mean()
    prices["macd_hist"] = prices["macd"] - prices["macd_signal"]

    returns = prices["adj close"].pct_change()
    prices["return_1d"] = returns
    prices["return_5d"] = prices["adj close"].pct_change(5)
    prices["return_21d"] = prices["adj close"].pct_change(21)
    direction = np.sign(returns).fillna(0)
    streak_group = (direction != direction.shift()).cumsum()
    prices["return_streak"] = direction.groupby(streak_group).cumsum()

    prices["vol_10d"] = returns.rolling(10).std()
    prices["vol_20d"] = returns.rolling(20).std()
    prices["vol_60d"] = returns.rolling(60).std()

    prices["momentum_short"] = prices["adj close"] / prices["sma_short"] - 1
    prices["momentum_long"] = prices["adj close"] / prices["sma_long"] - 1

    high = prices["high"] if "high" in prices.columns else prices["adj close"]
    low = prices["low"] if "low" in prices.columns else prices["adj close"]
    prev_close = prices["adj close"].shift()
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).fillna(0.0)
    true_range = tr_components.max(axis=1)
    atr_window = max(14, min(long_window, 60))
    prices["atr_14"] = true_range.rolling(atr_window).mean()
    prices["atr_pct"] = prices["atr_14"] / prices["adj close"]

    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_sum = true_range.rolling(atr_window).sum().replace(0, np.nan)
    plus_di = pd.Series(plus_dm, index=prices.index).rolling(atr_window).sum() * 100 / tr_sum
    minus_di = pd.Series(minus_dm, index=prices.index).rolling(atr_window).sum() * 100 / tr_sum
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    prices["adx_14"] = dx.rolling(atr_window).mean()

    volume = prices["volume"].fillna(0)
    vol_ma20 = volume.rolling(20).mean()
    prices["volume_z"] = (volume - vol_ma20) / (vol_ma20.replace(0, np.nan))
    prices["volume_trend"] = volume.pct_change(5).replace([np.inf, -np.inf], np.nan)
    direction = np.sign(prices["adj close"].diff().fillna(0))
    prices["obv"] = (direction * volume).cumsum()
    hl_range = (high - low).replace(0, np.nan)
    mf_multiplier = ((prices["adj close"] - low) - (high - prices["adj close"])) / hl_range
    mf_volume = mf_multiplier.fillna(0) * volume
    vol_sum = volume.rolling(20).sum().replace(0, np.nan)
    prices["cmf_20"] = mf_volume.rolling(20).sum() / vol_sum
    prices["skew_21"] = returns.rolling(21).skew()
    prices["kurt_21"] = returns.rolling(21).kurt()
    if "price_z" not in prices.columns:
        prices["price_z"] = (
            (prices["adj close"] - prices["adj close"].rolling(64, min_periods=20).mean())
            / (prices["adj close"].rolling(64, min_periods=20).std().replace(0, np.nan))
        ).fillna(0.0)

    _attach_forward_returns(prices)
    prices["label"] = (prices["forward_return"] > 0).astype(int)

    return prices


def _normalized_open_prices(prices: pd.DataFrame) -> pd.Series:
    if "open" not in prices:
        return prices["adj close"]
    open_series = prices["open"].copy()
    close = prices.get("close")
    if close is not None:
        ratio = prices["adj close"] / close.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        open_series = open_series * ratio
    return open_series.fillna(prices["adj close"])


def _attach_forward_returns(prices: pd.DataFrame) -> None:
    adj_close = prices["adj close"]
    next_close = adj_close.shift(-1)
    prices["forward_return_close"] = (next_close / adj_close) - 1
    adj_open = _normalized_open_prices(prices)
    next_open = adj_open.shift(-1)
    prices["forward_return_open"] = (next_open / adj_close) - 1
    # 盘中收益：当日开到收
    prices["forward_return_open_to_close"] = (adj_close / adj_open.replace(0, np.nan)) - 1
    # 隔夜收益：收盘到次日开盘
    prices["forward_return_overnight"] = prices["forward_return_open"]
    prices["forward_return"] = prices["forward_return_close"]


def _select_forward_return(
    frame: pd.DataFrame,
    params: StrategyInput,
) -> tuple[pd.Series, str]:
    """
    根据配置选择标签/未来收益路径。

    优先级：label_return_path > return_path > 默认 close_to_close。
    支持 close_to_close / close_to_open / open_to_close 三种。
    """
    path = (getattr(params, "label_return_path", None) or getattr(params, "return_path", "close_to_close")).lower()
    path = path if path in {"close_to_open", "open_to_close"} else "close_to_close"
    key_map = {
        "close_to_close": "forward_return_close",
        "close_to_open": "forward_return_open",
        "open_to_close": "forward_return_open_to_close",
    }
    series = frame.get(key_map[path])
    if series is None:
        # fallback to basic close-to-close
        series = frame["adj close"].pct_change().shift(-1)
    return series, path


def _compute_asset_returns(frame: pd.DataFrame, params: StrategyInput) -> pd.Series:
    """根据 return_path 选择资产收益序列。"""
    ret_path = getattr(params, "return_path", "close_to_close")
    adj_close = frame["adj close"]
    if ret_path == "close_to_open":
        try:
            adj_open = _normalized_open_prices(frame)
            next_open = adj_open.shift(-1)
            returns = (next_open / adj_close) - 1
        except Exception:
            returns = adj_close.shift(-1) / adj_close - 1
        return returns.fillna(0.0)
    if ret_path == "open_to_close":
        try:
            adj_open = _normalized_open_prices(frame)
            returns = (adj_close / adj_open.replace(0, np.nan)) - 1
        except Exception:
            returns = adj_close / adj_close.shift(1) - 1
        return returns.fillna(0.0)
    # default close-to-close
    return adj_close.pct_change().fillna(0.0)


__all__ = [
    "compute_indicators",
    "_normalized_open_prices",
    "_attach_forward_returns",
    "_select_forward_return",
    "_compute_asset_returns",
]
