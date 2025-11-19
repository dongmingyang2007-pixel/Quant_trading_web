from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any, Optional, Tuple
import os
import textwrap
import base64
import io
import math
import re
import hashlib
import random
import time
import json

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import dates as mdates

from django.utils.translation import gettext_lazy as _
from django.utils.safestring import mark_safe
from django.conf import settings

from .headlines import estimate_readers
from .data_sources import collect_auxiliary_data, AuxiliaryData
from . import screener
from .preprocessing import FeatureStore, sanitize_price_history
from .ml_models import build_custom_sequence_model
from .reinforcement import build_reinforcement_playbook, train_value_agent
from .rl_agents import build_rl_agent
from .backtest_logger import BacktestLogEntry, append_log, top_runs
from .observability import record_metric, track_latency
from .http_client import http_client, HttpClientError
from .security import sanitize_html_fragment
from .optimization import PurgedWalkForwardSplit, _simulate_returns, _compute_slippage_cost
from .risk_stats import (
    compute_robust_sharpe,
    calculate_cvar,
    recovery_period_days,
    compute_white_reality_check,
    compute_white_reality_check_bootstrap,
    compute_spa_pvalue,
)
from .validation import (
    build_walk_forward_report,
    build_purged_kfold_schedule,
    compute_tail_risk_summary,
    collect_repro_metadata,
)

DEFAULT_STRATEGY_SEED = int(os.environ.get("STRATEGY_SEED", "42"))
DEFAULT_SEED_META = {"strategy_seed": DEFAULT_STRATEGY_SEED}
try:  # optional RL utils
    from stable_baselines3.common.utils import set_random_seed as sb3_set_seed  # type: ignore
except Exception:  # pragma: no cover
    sb3_set_seed = None  # type: ignore
try:  # Optional heavy dependency, validated in requirements
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.isotonic import IsotonicRegression
    from sklearn.base import clone
    try:
        from sklearn.utils.class_weight import compute_sample_weight
    except Exception:  # pragma: no cover - optional utility
        compute_sample_weight = None  # type: ignore[assignment]
except ImportError:  # pragma: no cover - fallback handled at runtime
    GradientBoostingClassifier = None  # type: ignore[assignment]
    GradientBoostingRegressor = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
    IsotonicRegression = None  # type: ignore[assignment]
    def clone(estimator):  # type: ignore[no-redef]
        return estimator

# Optional external engines
try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None  # type: ignore

try:
    from catboost import CatBoostClassifier  # type: ignore
except Exception:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore

try:  # optional parallel backend
    from joblib import Parallel, delayed  # type: ignore
except Exception:  # pragma: no cover
    Parallel = None  # type: ignore
    def delayed(func):  # type: ignore
        return func

try:  # Optional deep learning backend
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

DEFAULT_STRATEGY_SEED = int(os.environ.get("STRATEGY_SEED", "42"))


def _ensure_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    if torch is not None:
        try:
            torch.manual_seed(seed)
        except Exception:  # pragma: no cover
            pass
    if sb3_set_seed is not None:
        try:
            sb3_set_seed(seed)
        except Exception:
            pass


def _collect_env_info() -> dict[str, Any]:
    """收集轻量环境版本信息，便于复现实验。"""
    info: dict[str, Any] = {}
    try:
        import sys

        info["python"] = sys.version.split()[0]
    except Exception:
        pass
    try:
        info["numpy"] = np.__version__
    except Exception:
        pass
    try:
        import sklearn  # type: ignore

        info["sklearn"] = getattr(sklearn, "__version__", None)
    except Exception:
        pass
    try:
        info["torch"] = getattr(torch, "__version__", None) if torch is not None else None
    except Exception:
        pass
    try:
        import stable_baselines3 as sb3  # type: ignore

        info["stable_baselines3"] = getattr(sb3, "__version__", None)
    except Exception:
        pass
    try:
        info["pandas"] = pd.__version__
    except Exception:
        pass
    return info

try:  # Optional graph analytics
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

try:  # Optional statistical baselines
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
except Exception:  # pragma: no cover
    ARIMA = None  # type: ignore

try:
    from statsmodels.tsa.vector_ar.var_model import VAR  # type: ignore
except Exception:  # pragma: no cover
    VAR = None  # type: ignore

try:  # Optional sentiment analyzer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
except Exception:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore

try:
    from sklearn.neural_network import MLPClassifier
except Exception:  # pragma: no cover
    MLPClassifier = None  # type: ignore

plt.switch_backend("Agg")
try:
    import matplotlib as mpl
    mpl.rcParams["axes.unicode_minus"] = False
    # Prefer common CJK fonts on macOS/Windows/Linux with sensible fallbacks
    mpl.rcParams["font.sans-serif"] = [
        "PingFang SC",  # macOS
        "Hiragino Sans GB",
        "Microsoft YaHei",  # Windows
        "Noto Sans CJK SC",  # Linux common
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
except Exception:
    pass


class QuantStrategyError(Exception):
    """Custom exception for strategy failures."""


@dataclass(slots=True)
class StrategyInput:
    ticker: str
    benchmark_ticker: Optional[str]
    start_date: date
    end_date: date
    short_window: int
    long_window: int
    rsi_period: int
    include_plots: bool
    show_ai_thoughts: bool
    risk_profile: str
    capital: float
    strategy_engine: str = "ml_momentum"
    volatility_target: float = 0.15
    transaction_cost_bps: float = 8.0
    slippage_bps: float = 5.0
    min_holding_days: int = 3
    train_window: int = 252
    test_window: int = 21
    entry_threshold: float = 0.55
    exit_threshold: float = 0.45
    max_leverage: float = 2.5
    # ML pipeline controls
    ml_task: str = "direction"  # 'direction' | 'hybrid'
    val_ratio: float = 0.15
    embargo_days: int = 5
    optimize_thresholds: bool = True
    ml_model: str = "seq_hybrid"  # 默认自动融合 LSTM+Transformer；如后端缺失将自动回退
    ml_params: dict[str, Any] | None = None
    auto_apply_best_config: bool = True
    calibrate_proba: bool = True
    early_stopping_rounds: int = 50
    use_feature_cache: bool = True
    enable_hyperopt: bool = False
    hyperopt_trials: int = 20
    hyperopt_timeout: int = 120
    max_drawdown_stop: float = 0.25
    daily_exposure_limit: float = 1.5
    dl_sequence_length: int = 32
    dl_hidden_dim: int = 64
    dl_dropout: float = 0.2
    dl_epochs: int = 12
    dl_batch_size: int = 64
    dl_num_layers: int = 2
    rl_engine: str = "value_iter"
    rl_params: dict[str, Any] | None = None
    validation_slices: int = 3
    out_of_sample_ratio: float = 0.2
    execution_liquidity_buffer: float = 0.05
    execution_penalty_bps: float = 6.0
    execution_mode: str = "adv"  # 'adv' or 'limit'
    class_weight_mode: str = "balanced"  # imbalance handling
    focal_gamma: float = 2.0
    dynamic_exposure_multiplier: float = 1.0
    label_return_path: str | None = None  # override label/future_return path; defaults to return_path
    intraday_loss_limit: float = 0.08
    slippage_model: dict[str, Any] | None = None
    borrow_cost_bps: float = 0.0
    long_borrow_cost_bps: float = 0.0
    short_borrow_cost_bps: float = 0.0
    max_adv_participation: float = 0.1
    target_vol: float | None = None
    vol_target_window: int = 60
    # triple barrier labeling (optional)
    label_style: str = "direction"  # 'direction' | 'triple_barrier'
    tb_up: float = 0.03
    tb_down: float = 0.03
    tb_max_holding: int = 10
    tb_dynamic: bool = False
    tb_vol_multiplier: float = 1.2
    tb_vol_window: int = 20
    interest_keywords: list[str] | None = None
    investment_horizon: str = "medium"
    experience_level: str = "novice"
    primary_goal: str = "growth"
    return_path: str = "close_to_close"  # 'close_to_close' | 'close_to_open'
    # 支持标签/回测收益口径：close_to_close | close_to_open | open_to_close
    # 若前端需要拆分盘中/隔夜，可用 label_return_path 切换
    request_id: str | None = None
    user_id: str | None = None
    model_version: str | None = None
    data_version: str | None = None
    exec_latency_ms: float | None = None
    include_walk_forward_report: bool = False
    walk_forward_horizon_days: int | None = None
    walk_forward_step_days: int | None = None
    walk_forward_jobs: int = 1
    force_pfws: bool = True
    # stats / reporting controls
    stats_enable_bootstrap: bool = True
    stats_bootstrap_samples: int = 600
    stats_bootstrap_block: int | None = None
    enforce_pfws_only: bool = False  # 若为 True，禁止非 PFWS 切分的训练/验证
    random_seed: int = DEFAULT_STRATEGY_SEED
    threshold_jobs: int = 1


@dataclass(slots=True)
class StrategyOutcome:
    engine: str
    backtest: pd.DataFrame
    metrics: list[dict[str, Any]]
    stats: dict[str, Any]
    weight: float = 1.0


DATA_CACHE_DIR = settings.DATA_CACHE_DIR
DATA_CACHE_DIR.mkdir(exist_ok=True)
FEATURE_STORE = FeatureStore()


def extract_context_features(auxiliary: AuxiliaryData) -> dict[str, float]:
    """Flatten auxiliary data (macro/flows/sentiment/options) into numeric features for ML."""
    features: dict[str, float] = {}

    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    macro = auxiliary.macro or {}
    for key, entry in macro.items():
        if not isinstance(entry, dict) or not entry.get("available"):
            continue
        short = entry.get("short") or key
        features[f"macro_{short}_chg5"] = _as_float(entry.get("change_5d"))
        features[f"macro_{short}_chg21"] = _as_float(entry.get("change_21d"))
        features[f"macro_{short}_trend"] = {"上升": 1.0, "下降": -1.0}.get(entry.get("trend"), 0.0)

    flows = auxiliary.capital_flows or {}
    for key, entry in flows.items():
        if key == "_summary":
            continue
        if not isinstance(entry, dict) or not entry.get("available"):
            continue
        features[f"flow_{key}_momentum"] = _as_float(entry.get("momentum_21d"))
        features[f"flow_{key}_volatility"] = _as_float(entry.get("volatility"))
        features[f"flow_{key}_adv"] = _as_float(entry.get("avg_dollar_volume"))
        features[f"flow_{key}_signal"] = _as_float(entry.get("flow_signal"))
    summary = flows.get("_summary") or {}
    if summary:
        features["flow_risk_appetite"] = {"Risk-On": 1.0, "Neutral": 0.0, "Risk-Off": -1.0}.get(
            summary.get("risk_appetite"), 0.0
        )
        features["flow_growth_vs_value"] = _as_float(summary.get("growth_vs_value"))
        features["flow_duration_vs_equity"] = _as_float(summary.get("duration_vs_equity"))

    sentiment = auxiliary.news_sentiment or {}
    if sentiment.get("available"):
        features["sentiment_score"] = _as_float(sentiment.get("avg_score"))
        features["sentiment_samples"] = _as_float(sentiment.get("sample_size"))

    options = auxiliary.options_metrics or {}
    if options.get("available"):
        features["options_call_iv"] = _as_float(options.get("call_iv"))
        features["options_put_iv"] = _as_float(options.get("put_iv"))
        features["options_put_call"] = _as_float(options.get("put_call_ratio"))

    global_macro = auxiliary.global_macro or {}
    summary_macro = global_macro.get("summary") or {}
    for key, value in summary_macro.items():
        features[f"global_{key}"] = {"放缓": -1.0, "温和": 0.3, "观望": 0.0, "中性": 0.0}.get(value, _as_float(value))

    fundamentals = auxiliary.fundamentals or {}
    for key, value in fundamentals.items():
        if isinstance(value, (int, float)):
            features[f"fund_{key}"] = float(value)

    return features

METRIC_DESCRIPTIONS = {
    "策略累计收益率": "策略净值相对起点的总涨幅，衡量绝对收益表现。",
    "买入持有收益率": "直接持有标的（不做调仓）的回报，可用于衡量策略超额收益。",
    "夏普比率": "单位波动获得的超额收益，>1 表示风险调整后表现良好。",
    "最大回撤": "净值从高点到低点的最大跌幅，越小越稳健。",
    "索提诺比率": "仅惩罚下跌波动的收益风险比，更关注下行风险。",
    "年化波动率": "收益波动程度，反映风险水平。",
    "年化复合收益率": "复利视角的年化收益，更贴近日常收益口径。",
    "卡玛比率": "年化收益与最大回撤之比，衡量单位回撤创造收益能力。",
    "胜率": "盈利交易日占比，可反映策略稳定性。",
    "单日平均盈亏": "平均上涨日与下跌日的收益率，衡量盈亏对称度。",
    "平均持仓比例": "在市场中的平均资金暴露程度。",
    "平均杠杆（波动率目标）": "为达到目标波动率所需的平均杠杆倍数。",
    "日度95%VaR": "在 95% 情况下，单日最大可能亏损的保守估计。",
    "日度95%CVaR": "落在 VaR 置信区间外的平均亏损，衡量尾部风险。",
    "交易日数量": "回测样本的交易日数量。",
    "基准累计收益率": "选择的对比指数或资产在回测期内的收益。",
    "基准年化波动率": "基准资产的波动水平。",
    "基准夏普比率": "基准资产的风险调整后收益表现。",
    "策略相对基准α": "策略相对基准的年化超额收益（CAPM α）。",
    "β系数": "策略对基准波动的敏感度，>1 表示更敏感。",
    "与基准相关系数": "策略与基准收益的同步度，接近 0 更有分散度。",
    "信息比率": "单位跟踪误差的超额收益，越高代表超额收益稳定。",
    "跟踪误差": "策略相对基准收益差的波动度。",
    "预测胜率": "模型预测方向与实际收益方向一致的比例。",
    "ROC-AUC": "预测概率区分上涨与下跌的能力，0.5 为随机水准。",
    "年化换手率": "一年内的仓位变动次数衡量交易频率。",
    "平均持仓天数": "单次开仓持有的平均持续时间。",
    "成本占收益比": "交易成本与策略总收益的比例，用于衡量成本侵蚀。",
}

RISK_PROFILE_LABELS = {
    "conservative": "保守型",
    "balanced": "均衡型",
    "aggressive": "进取型",
}


def build_metric(label: str, value: str) -> dict[str, str]:
    return {
        "label": label,
        "value": value,
        "explain": METRIC_DESCRIPTIONS.get(label, ""),
    }


def format_currency(value: float) -> str:
    try:
        return f"{value:,.0f}"
    except ValueError:
        return str(value)


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fetch_remote_strategy_overrides(params: StrategyInput) -> dict[str, Any]:
    """Optionally pull remote overrides (weights/notes) for strategy coordination."""
    endpoint = os.getenv("STRATEGY_UPDATE_ENDPOINT")
    if not endpoint:
        return {}
    try:
        headers = {}
        auth_token = os.getenv("STRATEGY_UPDATE_AUTH_TOKEN")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        response = http_client.get(
            endpoint,
            params={
                "ticker": params.ticker.upper(),
                "engine": params.strategy_engine,
                "benchmark": (params.benchmark_ticker or "").upper(),
            },
            headers=headers,
            timeout=3,
        )
        data = response.json()
        return data if isinstance(data, dict) else {}
    except (HttpClientError, ValueError, json.JSONDecodeError):
        return {}


def fetch_price_data(ticker: str, start: date, end: date) -> Tuple[pd.DataFrame, list[str]]:
    """Fetch historical price data using yfinance with local cache fallback."""
    warnings: list[str] = []
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency load
        raise QuantStrategyError(
            "yfinance is required to download market data. Install it first."
        ) from exc

    download_error: Exception | None = None
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            actions=False,
            repair=True,
            threads=False,
        )
    except Exception as exc:  # pragma: no cover - network failure
        download_error = exc
        data = pd.DataFrame()

    if data.empty:
        cache_file = DATA_CACHE_DIR / f"{ticker.upper()}.csv"
        if cache_file.exists():
            cached = pd.read_csv(cache_file, parse_dates=["date"])
            cached = cached.set_index("date").sort_index()
            cached = cached.rename(columns=str.lower)
            required_cols = {"close", "adj close", "volume"}
            missing = required_cols - set(cached.columns)
            if missing:
                raise QuantStrategyError(
                    f"本地缓存文件缺少列 {missing}. 请确保包含 Close, Adj Close, Volume."
                )
            warnings.append(
                f"已从本地缓存 {cache_file} 读取数据。若需最新行情，请联网后刷新缓存。"
            )
            data = cached.loc[(cached.index.date >= start) & (cached.index.date <= end)]
        else:
            if download_error:
                raise QuantStrategyError(
                    "无法从 Yahoo Finance 下载行情，请确认网络可用或在 "
                    f"{DATA_CACHE_DIR} 放置名为 {ticker.upper()}.csv 的历史数据文件。"
                ) from download_error
            raise QuantStrategyError("No market data was returned. Check the ticker and dates.")

    if isinstance(data.columns, pd.MultiIndex):
        try:
            # Try to select the ticker level explicitly (handles single ticker requests)
            data = data.xs(ticker, axis=1, level=-1)
        except (KeyError, ValueError):
            # Fallback: drop the last level if ticker not found as expected
            data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str.lower)
    data.index = pd.to_datetime(data.index)
    if "adj close" not in data.columns and "close" in data.columns:
        data["adj close"] = data["close"]
    if "close" not in data.columns and "adj close" in data.columns:
        data["close"] = data["adj close"]
    missing_price_cols = {"close", "adj close"} - set(data.columns)
    if missing_price_cols:
        raise QuantStrategyError(
            f"关键价格字段缺失：{', '.join(sorted(missing_price_cols))}。请尝试其他数据源或标的。"
        )
    # forward fill if needed and drop missing adj close rows
    data[["adj close", "close"]] = data[["adj close", "close"]].ffill()
    if data["adj close"].isna().all():
        raise QuantStrategyError("Adjusted close prices are unavailable for this ticker.")
    if "volume" not in data.columns:
        data["volume"] = np.nan

    # Persist a normalized CSV cache on successful fetch for offline use
    try:
        cache_file = DATA_CACHE_DIR / f"{ticker.upper()}.csv"
        cache_cols = ["open", "high", "low", "close", "adj close", "volume"]
        missing_cols = [col for col in cache_cols if col not in data.columns]
        for col in missing_cols:
            if col in {"open", "high", "low", "close"} and "close" in data.columns:
                data[col] = data["close"]
            elif col == "adj close" and "close" in data.columns:
                data[col] = data["close"]
            elif col == "volume":
                data[col] = 0.0
        to_cache = data[cache_cols].copy()
        to_cache = to_cache.reset_index().rename(columns={"index": "date"})
        to_cache["date"] = pd.to_datetime(to_cache["date"]).dt.strftime("%Y-%m-%d")
        to_cache.to_csv(cache_file, index=False)
    except Exception:
        # Non-fatal: caching should not break pipeline
        pass

    base_cols = ["open", "high", "low", "close", "adj close", "volume"]
    for col in ("open", "high", "low", "close"):
        if col not in data.columns and "close" in data.columns:
            data[col] = data["close"]
    if "volume" not in data.columns:
        data["volume"] = 0.0
    return data[base_cols].dropna(subset=["adj close"]), warnings


def compute_indicators(
    prices: pd.DataFrame, short_window: int, long_window: int, rsi_period: int
) -> pd.DataFrame:
    """Compute comprehensive technical indicators used for trading decisions."""
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
    Choose forward return series based on label_return_path/return_path.

    Priority: label_return_path > return_path > default close_to_close.
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


def _load_latest_walk_forward_report(params: StrategyInput) -> dict[str, Any] | None:
    """Load the latest walk-forward report if available on disk for display."""
    training_dir = DATA_CACHE_DIR / "training"
    if not training_dir.exists():
        return None
    files = sorted(training_dir.glob("walk_forward_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        payload["path"] = os.fspath(files[0])
        payload["generated_at"] = payload.get("generated_at") or files[0].stem.replace("walk_forward_report_", "")
        payload["reports_count"] = len(payload.get("reports", []))
        return payload
    except Exception:
        return None


def _compute_asset_returns(frame: pd.DataFrame, params: StrategyInput) -> pd.Series:
    """Select asset return series based on configured return_path."""

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


def _calibration_summary(probs: pd.Series, labels: pd.Series, bins: int = 10) -> dict[str, Any]:
    """Simple reliability stats for probability calibration."""
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


def _build_calibration_plot(calib: dict[str, Any]) -> str | None:
    """Render a reliability curve from calibration buckets."""
    try:
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
        # 次轴显示样本量
        ax2 = ax.twinx()
        ax2.bar(preds, counts, alpha=0.15, color="#10b981", width=0.05, label="分桶样本数")
        ax2.set_ylabel("样本数")
        ax.legend(loc="upper left")
        return fig_to_base64(fig)
    except Exception:
        return None


def _pfws_predict(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    params: StrategyInput,
) -> tuple[pd.Series, pd.Series, list[float]]:
    """Use PurgedWalkForwardSplit to generate out-of-sample probabilities/predictions."""

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
            proba >= params.entry_threshold,
            1.0,
            np.where(proba <= params.exit_threshold, -1.0, 0.0),
        )
        if roc_auc_score is not None and test_slice["target"].nunique() > 1:
            try:
                auc_scores.append(roc_auc_score(test_slice["target"], proba))
            except Exception:
                pass
    return probabilities, raw_signal, auc_scores


def build_feature_matrix(prices: pd.DataFrame, params: StrategyInput) -> tuple[pd.DataFrame, list[str]]:
    """
    Construct an ML-ready feature matrix with aligned targets.

    Returns a tuple of (dataset, feature_columns).
    """
    cache_key: str | None = None
    if getattr(params, "use_feature_cache", True):
        try:
            cache_key = FEATURE_STORE.fingerprint(prices, params)
            cached = FEATURE_STORE.load(cache_key)
            if cached:
                dataset, feature_columns, _ = cached
                if "target" in dataset:
                    return dataset, feature_columns
                # 缓存缺失标签时强制重建，避免旧版本污染
        except Exception:
            cache_key = None

    dataset = prices.copy()
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

    feature_columns = [
        "return_1d",
        "return_5d",
        "return_21d",
        "vol_10d",
        "vol_20d",
        "vol_60d",
        "momentum_short",
        "momentum_long",
        "ema_trend",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi",
        "volume_z",
        "volume_trend",
        "atr_14",
        "atr_pct",
        "adx_14",
        "obv",
        "cmf_20",
        "skew_21",
        "kurt_21",
        "return_streak",
        "price_z",
    ]

    # Derivative features for crossover strength
    dataset["sma_diff"] = dataset["sma_short"] - dataset["sma_long"]
    dataset["sma_ratio"] = dataset["sma_short"] / dataset["sma_long"] - 1
    dataset["boll_bandwidth"] = (dataset["boll_up"] - dataset["boll_dn"]) / dataset["sma_long"]
    feature_columns.extend(["sma_diff", "sma_ratio", "boll_bandwidth"])

    # Apply labeling via helper for复用/缓存
    labeled = build_labels(dataset, params)
    if params.label_style == "triple_barrier":
        if "target_multiclass" not in labeled:
            raise QuantStrategyError("三重闸标签生成失败，缺少 target_multiclass。")
        dataset["target"] = labeled["target"]
        dataset["target_multiclass"] = labeled["target_multiclass"]
        # 记录动态阈值轨迹用于稳定性分析
        dynamic_up = labeled.get("tb_up_active")
        dynamic_down = labeled.get("tb_down_active")
        dataset["tb_up_active"] = dynamic_up
        dataset["tb_down_active"] = dynamic_down
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
    # heuristic: symbols with dots (e.g., .SS) often equities; default mega
    return "mega"


def _load_best_ml_config(ticker: str) -> tuple[str | None, dict[str, Any] | None]:
    """Return (engine, params) from training cache if available for ticker category."""
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


def _tune_thresholds_on_validation(
    proba: pd.Series,
    returns: pd.Series,
    cost_rate: float,
    grid_long: list[float] | None = None,
    grid_short: list[float] | None = None,
    n_jobs: int = 1,
) -> tuple[float, float]:
    # Default grids
    n = len(proba.dropna())
    step_long = 0.02 if n < 150 else 0.015
    step_short = 0.02 if n < 150 else 0.015
    grid_long = grid_long or [round(x, 3) for x in np.arange(0.54, 0.71, step_long)]
    grid_short = grid_short or [round(x, 3) for x in np.arange(0.29, 0.48, step_short)]
    candidates = [(e, x) for e in grid_long for x in grid_short if 0.0 < x < 0.5 < e < 1.0]
    if not candidates:
        return 0.55, 0.45
    if n_jobs and n_jobs > 1:
        try:
            from joblib import Parallel, delayed  # type: ignore

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


def _scan_threshold_stability(
    probabilities: pd.Series,
    future_returns: pd.Series,
    cost_rate: float,
    base_entry: float,
    base_exit: float,
) -> dict[str, Any]:
    """Evaluate a grid of thresholds in a vectorized fashion and summarize Sharpe stability."""

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

    p_arr = proba.values[:, None]  # shape (n, 1)
    ret_arr = aligned_returns.values[:, None]
    entry_mat = entry_vec[None, :]  # shape (1, m)
    exit_mat = exit_vec[None, :]

    signals = np.where(
        p_arr >= entry_mat,
        1.0,
        np.where(p_arr <= exit_mat, -1.0, 0.0),
    )
    exposure = np.vstack([np.zeros((1, signals.shape[1])), signals[:-1, :]])
    turnover = np.vstack([np.abs(exposure[0, :]), np.abs(np.diff(exposure, axis=0))])
    pnl = exposure * ret_arr
    linear_cost = turnover * cost_rate
    pnl_net = pnl - linear_cost

    # portfolio stats vectorized over columns
    mean = pnl_net.mean(axis=0)
    std = pnl_net.std(axis=0)
    ann_factor = 252.0
    sharpe = np.where(std > 0, np.sqrt(ann_factor) * mean / std, 0.0)

    best_idx = int(np.argmax(sharpe))
    worst_idx = int(np.argmin(sharpe))
    sharpe_arr = sharpe
    mean_sharpe = float(np.mean(sharpe_arr))
    q1, q3 = np.percentile(sharpe_arr, [25, 75])

    best = {"entry": float(entry_vec[best_idx]), "exit": float(exit_vec[best_idx]), "sharpe": float(sharpe_arr[best_idx])}
    worst = {"entry": float(entry_vec[worst_idx]), "exit": float(exit_vec[worst_idx]), "sharpe": float(sharpe_arr[worst_idx])}
    points = [
        {"entry": float(e), "exit": float(x), "sharpe": float(s)}
        for e, x, s in zip(entry_vec.tolist(), exit_vec.tolist(), sharpe_arr.tolist())
    ]

    return {
        "grid": {
            "entry": [round(float(x), 4) for x in entry_grid.tolist()],
            "exit": [round(float(x), 4) for x in exit_grid.tolist()],
        },
        "best": best,
        "worst": worst,
        "mean_sharpe": mean_sharpe,
        "median_sharpe": float(np.median(sharpe_arr)),
        "iqr_sharpe": float(q3 - q1),
        "count": len(entry_vec),
        "points": points,
        "heatmap_grid": {
            "entry": entry_vec.tolist(),
            "exit": exit_vec.tolist(),
            "sharpe": sharpe_arr.tolist(),
        },
    }


def _build_threshold_heatmap(scan: dict[str, Any]) -> str | None:
    """Render a heatmap of Sharpe over entry/exit thresholds."""

    try:
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


def compute_triple_barrier_labels(
    price: pd.Series,
    up: float | pd.Series,
    down: float | pd.Series,
    max_holding: int,
) -> tuple[pd.Series, pd.Series]:
    idx = price.index
    arr = price.values
    n = len(arr)
    binary = np.full(n, np.nan)
    multi = np.zeros(n)
    up_arr = np.asarray(up.reindex(idx).ffill().bfill().fillna(up.mean() if isinstance(up, pd.Series) else up)) if isinstance(up, pd.Series) else np.full(n, up)
    down_arr = np.asarray(down.reindex(idx).ffill().bfill().fillna(down.mean() if isinstance(down, pd.Series) else down)) if isinstance(down, pd.Series) else np.full(n, down)
    for i in range(n - 1):
        p0 = arr[i]
        horizon = min(n - i - 1, max_holding)
        if horizon <= 0:
            break
        future = arr[i + 1 : i + 1 + horizon]
        rets = future / p0 - 1
        up_thr = float(up_arr[i]) if i < len(up_arr) else float(up_arr[-1])
        dn_thr = float(down_arr[i]) if i < len(down_arr) else float(down_arr[-1])
        hit_up = np.where(rets >= up_thr)[0]
        hit_dn = np.where(rets <= -dn_thr)[0]
        t_up = hit_up[0] if hit_up.size else np.inf
        t_dn = hit_dn[0] if hit_dn.size else np.inf
        if t_up < t_dn:
            binary[i] = 1
            multi[i] = 1
        elif t_dn < t_up:
            binary[i] = 0
            multi[i] = -1
        else:
            terminal = 1 if rets[-1] > 0 else 0
            binary[i] = terminal
            multi[i] = 0
    return pd.Series(binary, index=idx), pd.Series(multi, index=idx)


def build_labels(frame: pd.DataFrame, params: StrategyInput) -> pd.DataFrame:
    """
    Build labels (direction or triple-barrier) and attach to frame for downstream caching/复用.
    """
    data = frame.copy()
    selected_return, label_path = _select_forward_return(data, params)
    data["forward_return"] = selected_return
    data["label_return_path"] = label_path
    # Ensure required columns exist
    if "forward_return" not in data and "adj close" in data:
        data["forward_return"] = data["adj close"].pct_change().shift(-1)
    if params.label_style == "triple_barrier":
        dynamic_up = params.tb_up
        dynamic_down = params.tb_down
        if params.tb_dynamic:
            try:
                vol_proxy = data.get("atr_pct")
                if vol_proxy is None or vol_proxy.isna().all():
                    daily_ret = data["adj close"].pct_change().abs()
                    vol_proxy = daily_ret.rolling(window=max(5, params.tb_vol_window)).std()
                vol_proxy = vol_proxy.ffill().bfill()
                dynamic_up = np.maximum(params.tb_up, vol_proxy * params.tb_vol_multiplier)
                dynamic_down = np.maximum(params.tb_down, vol_proxy * params.tb_vol_multiplier)
            except Exception:
                dynamic_up = params.tb_up
                dynamic_down = params.tb_down
        binary, multiclass = compute_triple_barrier_labels(
            data["adj close"],
            up=dynamic_up,
            down=dynamic_down,
            max_holding=params.tb_max_holding,
        )
        data["target"] = binary
        data["target_multiclass"] = multiclass
        data["tb_up_active"] = dynamic_up if isinstance(dynamic_up, pd.Series) else pd.Series(dynamic_up, index=data.index)
        data["tb_down_active"] = dynamic_down if isinstance(dynamic_down, pd.Series) else pd.Series(dynamic_down, index=data.index)
    else:
        fr = selected_return.fillna(0.0)
        data["target"] = (fr > 0).astype(int)
    return data


def _summarize_series_stats(series: pd.Series) -> dict[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {}
    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=0)),
    }


def _tb_summary_from_dataset(dataset: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if "tb_up_active" in dataset:
        stats = _summarize_series_stats(dataset["tb_up_active"])
        if stats:
            summary["up"] = stats
    if "tb_down_active" in dataset:
        stats = _summarize_series_stats(dataset["tb_down_active"])
        if stats:
            summary["down"] = stats
    if "label_return_path" in dataset:
        summary["return_path"] = str(dataset["label_return_path"].iloc[0])
    return summary


def enforce_min_holding(
    signals: pd.Series,
    min_days: int,
    probabilities: Optional[pd.Series] = None,
) -> pd.Series:
    """Ensure each position is held for at least `min_days` unless confidence flips strongly."""
    if min_days <= 1:
        return signals

    result = []
    current = 0
    days_in_trade = 0
    prob_series = probabilities if probabilities is not None else pd.Series(0.5, index=signals.index)
    prob_series = prob_series.reindex(signals.index).fillna(0.5)

    for idx, sig in signals.items():
        desired = int(sig)
        prob = float(prob_series.loc[idx])

        if current == 0:
            if desired != 0:
                current = desired
                days_in_trade = 1
            result.append(current)
            continue

        if desired == current or desired == 0:
            if desired == 0 and days_in_trade < min_days and 0.35 < prob < 0.65:
                # Maintain position until minimum holding reached unless conviction is high
                result.append(current)
                days_in_trade += 1
            elif desired == 0 and days_in_trade < min_days and (prob <= 0.35 or prob >= 0.65):
                # Allow exit when model confidence strongly signals flat positioning
                current = 0
                days_in_trade = 0
                result.append(current)
            elif desired == 0:
                current = 0
                days_in_trade = 0
                result.append(current)
            else:
                result.append(current)
                days_in_trade += 1
        else:  # Signal flipped
            strong_flip = (prob >= 0.65 and desired == 1) or (prob <= 0.35 and desired == -1)
            if days_in_trade < min_days and not strong_flip:
                result.append(current)
                days_in_trade += 1
            else:
                current = desired
                days_in_trade = 1
                result.append(current)

    return pd.Series(result, index=signals.index, dtype=float)


def apply_signal_filters(
    dataset: pd.DataFrame,
    raw_signal: pd.Series,
    probabilities: pd.Series,
    params: StrategyInput,
) -> pd.Series:
    """Apply trend and risk filters before running backtest."""
    filtered = raw_signal.copy()
    filtered = filtered.reindex(dataset.index).fillna(0.0)
    probs = probabilities.reindex(dataset.index).fillna(0.5)

    trend_long = dataset["sma_short"] > dataset["sma_long"]
    trend_short = dataset["sma_short"] < dataset["sma_long"]

    filtered = filtered.where(~((filtered > 0) & (~trend_long)), 0.0)
    filtered = filtered.where(~((filtered < 0) & (~trend_short)), 0.0)

    # Prevent entering overbought/oversold extremes unless conviction is high
    overbought = dataset["rsi"] > 75
    oversold = dataset["rsi"] < 25
    filtered = filtered.mask((filtered > 0) & overbought & (probs < 0.7), other=0.0)
    filtered = filtered.mask((filtered < 0) & oversold & (probs > 0.3), other=0.0)

    constrained = enforce_min_holding(filtered, params.min_holding_days, probs)
    return constrained


def enforce_risk_limits(
    position: pd.Series,
    leverage: pd.Series,
    asset_returns: pd.Series,
    params: StrategyInput,
) -> tuple[pd.Series, list[str]]:
    exposure_raw = (position * leverage).fillna(0.0)
    exposure_limit = max(float(params.daily_exposure_limit or 0.0), 0.0) * max(params.dynamic_exposure_multiplier, 0.5)
    max_drawdown_stop = max(float(params.max_drawdown_stop or 0.0), 0.0)
    index = exposure_raw.index
    adjusted = []
    limit_hits = 0
    stop_dates: list[str] = []
    resumed_dates: list[str] = []
    stopped = False
    prev_exposure = 0.0
    cumulative = 1.0
    peak = 1.0
    asset_ret = asset_returns.fillna(0.0).reindex(index).fillna(0.0)

    intraday_limit = max(params.intraday_loss_limit, 0.0)
    day_start_val = cumulative
    current_day = index[0].date() if len(index) else None

    for i, ts in enumerate(index):
        exp = float(exposure_raw.iloc[i])
        if exposure_limit > 0 and abs(exp) > exposure_limit + 1e-8:
            exp = float(np.clip(exp, -exposure_limit, exposure_limit))
            limit_hits += 1
        if stopped:
            exp = 0.0
        adjusted.append(exp)

        ret = float(asset_ret.iloc[i])
        if current_day and ts.date() != current_day:
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
    return exposure_series, events


def apply_vol_targeting(
    exposure: pd.Series,
    asset_returns: pd.Series,
    params: StrategyInput,
) -> tuple[pd.Series, list[str]]:
    """Scale exposure to align realized volatility with a target when configured."""

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


def apply_execution_model(
    backtest: pd.DataFrame,
    price_source: pd.DataFrame,
    params: StrategyInput,
) -> tuple[pd.DataFrame, list[str]]:
    if "volume" not in price_source.columns:
        return backtest, []
    price = backtest.get("adj close")
    if price is None or price.empty:
        return backtest, []
    raw_volume = price_source["volume"].reindex(backtest.index).ffill().bfill()
    dollar_volume = (raw_volume * price).rolling(20).mean()
    exposure_change = backtest["exposure"].diff().abs().fillna(backtest["exposure"].abs())
    turnover_value = exposure_change * params.capital
    liquidity_buffer = max(params.execution_liquidity_buffer, 0.01)
    liquidity_capacity = dollar_volume * liquidity_buffer
    # 若缺失成交量数据，直接返回并记录提示，避免虚高的惩罚
    if liquidity_capacity.isna().all() or liquidity_capacity.fillna(0.0).sum() == 0:
        return backtest, ["执行模型：缺少成交量数据，已跳过撮合成本估计。"]
    impact = turnover_value / liquidity_capacity.replace(0, np.nan)
    impact = impact.clip(lower=0.0, upper=5.0).fillna(0.0)
    if params.execution_mode == "limit":
        fill_prob = np.exp(-impact.clip(0, 5))
        penalty = (1 - fill_prob) * np.abs(backtest["strategy_return_gross"]) + impact * (params.execution_penalty_bps / 10000.0)
    else:
        penalty = impact * (params.execution_penalty_bps / 10000.0)
    backtest["execution_cost"] = penalty
    backtest["strategy_return"] = backtest["strategy_return"] - penalty
    events = []
    if penalty.sum() > 0:
        avg_impact = float(impact.replace([np.inf, -np.inf], np.nan).mean())
        events.append(f"执行撮合模型：平均冲击 {avg_impact:.2f}×ADV，额外成本 {penalty.sum():.4f}。")
    return backtest, events


def backtest_sma_strategy(prices: pd.DataFrame, params: StrategyInput) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """
    Run a simple moving average crossover backtest.

    Assumptions: fully invested when short SMA crosses above long SMA,
    in cash otherwise. Strategy return is calculated on adjusted close prices.
    """
    backtest = prices.dropna(subset=["sma_short", "sma_long"]).copy()
    backtest["signal"] = np.where(backtest["sma_short"] > backtest["sma_long"], 1.0, 0.0)
    backtest["position"] = enforce_min_holding(
        pd.Series(backtest["signal"], index=backtest.index), params.min_holding_days
    )

    asset_returns = _compute_asset_returns(backtest, params)
    backtest["asset_return"] = asset_returns
    backtest["volatility"] = asset_returns.rolling(window=20).std().fillna(0.0) * np.sqrt(252)
    backtest["leverage"] = calculate_target_leverage(
        backtest["position"], backtest["volatility"], params.volatility_target, params.max_leverage
    )

    exposure_series, overlay_events = enforce_risk_limits(
        backtest["position"],
        backtest["leverage"],
        asset_returns,
        params,
    )
    backtest["exposure"] = exposure_series
    with np.errstate(divide="ignore", invalid="ignore"):
        adj_position = backtest["exposure"] / backtest["leverage"].replace(0, np.nan)
    backtest["position"] = adj_position.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    exposure_change = backtest["exposure"].diff().abs().fillna(backtest["exposure"].abs())
    cost_rate = (params.transaction_cost_bps + params.slippage_bps) / 10000.0
    # ADV 参与率约束
    adv_hits = 0
    if "adv" in backtest and backtest["adv"].notna().any():
        max_part = max(0.0, min(1.0, params.max_adv_participation or 0.1))
        adv_limit = backtest["adv"].fillna(0.0) * max_part
        mask = backtest["exposure"].abs() > adv_limit
        adv_hits = int(mask.sum())
        capped = backtest["exposure"].where(~mask, 0.0)
        if adv_hits > 0:
            backtest["exposure"] = capped
            exposure_change = backtest["exposure"].diff().abs().fillna(backtest["exposure"].abs())
    backtest["transaction_cost"] = exposure_change * cost_rate

    shifted_exposure = (
        backtest["exposure"]
        if params.return_path in {"close_to_open", "open_to_close"}
        else backtest["exposure"].shift(fill_value=0)
    )
    backtest["strategy_return_gross"] = asset_returns * shifted_exposure
    long_daily = float(params.long_borrow_cost_bps or params.borrow_cost_bps) / 10000.0 / 252.0
    short_daily = float(params.short_borrow_cost_bps or params.borrow_cost_bps) / 10000.0 / 252.0
    borrow_cost = (
        backtest["exposure"].clip(lower=0.0) * long_daily
        + (-backtest["exposure"].clip(upper=0.0)) * short_daily
    )
    backtest["borrow_cost"] = borrow_cost
    backtest["strategy_return"] = backtest["strategy_return_gross"] - backtest["transaction_cost"] - borrow_cost
    backtest, execution_events = apply_execution_model(backtest, prices, params)
    backtest["cum_strategy"] = (1 + backtest["strategy_return"]).cumprod()
    backtest["cum_buy_hold"] = (1 + asset_returns).cumprod()

    metrics, stats = summarize_backtest(
        backtest,
        params,
        include_prediction=False,
        include_auc=False,
        feature_columns=[
            "sma_short",
            "sma_long",
            "rsi",
            "volatility",
            "position",
        ],
    )
    stats["cost_assumptions"] = {
        "slippage_model": params.slippage_model,
        "cost_rate": cost_rate,
        "long_borrow_bps": params.long_borrow_cost_bps or params.borrow_cost_bps,
        "short_borrow_bps": params.short_borrow_cost_bps or params.borrow_cost_bps,
        "adv_participation": params.max_adv_participation,
        "execution_mode": params.execution_mode,
    }
    aggregate_events = []
    if overlay_events:
        aggregate_events.extend(overlay_events)
    if adv_hits > 0:
        aggregate_events.append(f"因 ADV 参与率上限({params.max_adv_participation:.0%}) 清零 {adv_hits} 次仓位，避免不可成交。")
    aggregate_events.extend(execution_events)
    oos_report = _compute_oos_from_backtest(backtest, params)
    if oos_report:
        stats["validation_report_detected"] = "sma_pfws"
        stats["validation_oos_summary"] = oos_report.get("summary")
        stats["validation_oos_folds"] = oos_report.get("folds")
        stats["validation_penalized_sharpe"] = oos_report.get("penalized_sharpe")
        stats["validation_train_window"] = oos_report.get("train_window")
        stats["validation_test_window"] = oos_report.get("test_window")
        stats["validation_embargo"] = oos_report.get("embargo")
    if not oos_report:
        aggregate_events.append("提示：此策略样本外 PFWS 指标生成失败，当前仅展示全量回测结果。")
    if aggregate_events:
        stats["risk_events"] = aggregate_events
    return backtest, metrics, stats


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate maximum drawdown for a cumulative return series."""
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1
    return drawdown.min()


def calculate_drawdown_series(cumulative_returns: pd.Series) -> pd.Series:
    rolling_max = cumulative_returns.cummax()
    return cumulative_returns / rolling_max - 1


def format_table(backtest: pd.DataFrame) -> list[dict[str, Any]]:
    """Prepare a concise table for template rendering."""
    columns = [
        "signal",
        "position",
        "adj close",
        "sma_short",
        "sma_long",
        "rsi",
        "strategy_return",
        "leverage",
        "cum_strategy",
        "cum_buy_hold",
    ]
    if "probability" in backtest.columns:
        columns.insert(3, "probability")
    if "transaction_cost" in backtest.columns:
        columns.append("transaction_cost")

    subset = backtest[columns].tail(30)
    subset.index = subset.index.date
    return [
        {
            "date": idx.strftime("%Y-%m-%d") if isinstance(idx, datetime) else str(idx),
            "position": int(row["position"]),
            "signal": int(row["signal"]),
            "adj_close": round(float(row["adj close"]), 2),
            "sma_short": round(float(row["sma_short"]), 2),
            "sma_long": round(float(row["sma_long"]), 2),
            "rsi": round(float(row["rsi"]), 2),
            "daily_return": round(float(row["strategy_return"]), 4),
            "leverage": round(float(row["leverage"]), 2),
            "cum_strategy": round(float(row["cum_strategy"]), 4),
            "cum_buy_hold": round(float(row["cum_buy_hold"]), 4),
            "probability": round(float(row["probability"]), 3) if "probability" in subset.columns else None,
            "transaction_cost": round(float(row["transaction_cost"]), 5)
            if "transaction_cost" in subset.columns
            else None,
        }
        for idx, row in subset.iterrows()
    ]


def generate_charts(
    prices: pd.DataFrame,
    backtest: pd.DataFrame,
    benchmark_series: Optional[pd.DataFrame],
    params: StrategyInput,
) -> list[dict[str, str]]:
    charts: list[dict[str, str]] = []
    if prices.empty or backtest.empty:
        return charts

    date_formatter = mdates.DateFormatter("%Y-%m")
    benchmark_label = params.benchmark_ticker.upper() if params.benchmark_ticker else "基准"

    # 价格 + 均线 + RSI
    fig, (ax_price, ax_rsi) = plt.subplots(
        2,
        1,
        figsize=(11, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_price.plot(prices.index, prices["adj close"], color="#2563eb", label=f"{params.ticker.upper()} 收盘价")
    ax_price.plot(prices.index, prices["sma_short"], color="#10b981", label=f"短期均线({params.short_window})")
    ax_price.plot(prices.index, prices["sma_long"], color="#f59e0b", label=f"长期均线({params.long_window})")
    ax_price.set_ylabel("价格")
    ax_price.set_title("价格与均线信号")
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.2)

    ax_rsi.plot(prices.index, prices["rsi"], color="#7c3aed")
    ax_rsi.axhline(70, color="#ef4444", linestyle="--", linewidth=1, label="超买阈值 70")
    ax_rsi.axhline(30, color="#0ea5e9", linestyle="--", linewidth=1, label="超卖阈值 30")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.xaxis.set_major_formatter(date_formatter)
    ax_rsi.legend(loc="upper left")
    ax_rsi.grid(alpha=0.2)
    charts.append(
        {
            "title": "价格与技术指标",
            "img": fig_to_base64(fig),
        }
    )

    # 策略净值 vs 买入持有 vs 基准
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(backtest.index, backtest["cum_strategy"], label="策略净值", color="#2563eb")
    ax2.plot(backtest.index, backtest["cum_buy_hold"], label="买入持有", color="#9ca3af")
    if benchmark_series is not None and "benchmark_cum" in benchmark_series:
        ax2.plot(
            benchmark_series.index,
            benchmark_series["benchmark_cum"],
            label=f"基准净值（{benchmark_label}）",
            color="#f97316",
        )
    ax2.set_title("净值曲线对比")
    ax2.set_ylabel("累计收益倍数")
    ax2.grid(alpha=0.2)
    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_formatter(date_formatter)
    charts.append({"title": "净值曲线对比", "img": fig_to_base64(fig2)})

    # 回撤曲线
    fig3, ax3 = plt.subplots(figsize=(11, 3.5))
    strategy_drawdown = calculate_drawdown_series(backtest["cum_strategy"])
    ax3.fill_between(strategy_drawdown.index, strategy_drawdown, color="#ef4444", alpha=0.35, step="pre", label="策略回撤")
    ax3.plot(strategy_drawdown.index, strategy_drawdown, color="#b91c1c", linewidth=1)
    if benchmark_series is not None and "benchmark_cum" in benchmark_series:
        benchmark_drawdown = calculate_drawdown_series(benchmark_series["benchmark_cum"])
        ax3.plot(
            benchmark_drawdown.index,
            benchmark_drawdown,
            color="#f59e0b",
            linewidth=1,
            linestyle="--",
            label="基准回撤",
        )
    ax3.set_title("最大回撤跟踪")
    ax3.set_ylabel("回撤比例")
    ax3.grid(alpha=0.2)
    ax3.legend(loc="lower left")
    ax3.xaxis.set_major_formatter(date_formatter)
    charts.append({"title": "回撤分析", "img": fig_to_base64(fig3)})

    # 日度收益分布
    fig4, ax4 = plt.subplots(figsize=(11, 4))
    ax4.hist(
        backtest["strategy_return"].dropna(),
        bins=50,
        alpha=0.6,
        label="策略日收益",
        color="#2563eb",
    )
    ax4.hist(
        backtest["adj close"].pct_change().dropna(),
        bins=50,
        alpha=0.4,
        label="标的日收益（买入持有）",
        color="#9ca3af",
    )
    if benchmark_series is not None:
        ax4.hist(
            benchmark_series["benchmark_return"].dropna(),
            bins=50,
            alpha=0.4,
            label="基准日收益",
            color="#f97316",
        )
    ax4.set_title("日度收益分布（风险对比）")
    ax4.set_xlabel("日收益率")
    ax4.set_ylabel("频次")
    ax4.legend(loc="upper right")
    ax4.grid(alpha=0.2)
    charts.append({"title": "收益分布", "img": fig_to_base64(fig4)})

    if "probability" in backtest.columns:
        fig5, ax5 = plt.subplots(figsize=(11, 3.5))
        ax5.plot(
            backtest.index,
            backtest["probability"].clip(0, 1),
            color="#0ea5e9",
            label="模型多头概率",
        )
        ax5.fill_between(
            backtest.index,
            backtest["probability"].clip(0, 1),
            color="#0ea5e9",
            alpha=0.15,
        )
        ax5.axhline(0.5, color="#9ca3af", linewidth=1, linestyle="--")
        ax5.set_ylim(0, 1)
        ax5.set_title("模型信号强度（多头概率）")
        ax5.set_ylabel("P(上涨)")
        ax5.grid(alpha=0.2)
        ax5.xaxis.set_major_formatter(date_formatter)
        ax5.legend(loc="upper left")
        charts.append({"title": "信号强度", "img": fig_to_base64(fig5)})

    # 60日滚动夏普与波动
    try:
        rwin = 60
        ret = backtest["strategy_return"].fillna(0)
        roll_mean = ret.rolling(rwin).mean()
        roll_std = ret.rolling(rwin).std()
        roll_sharpe = (roll_mean / roll_std).replace([np.inf, -np.inf], np.nan)
        roll_vol = roll_std * np.sqrt(252)
        fig6, ax6 = plt.subplots(figsize=(11, 3.5))
        ax6.plot(ret.index, roll_sharpe, color="#0f766e", label=f"滚动夏普({rwin}日)")
        ax6.set_ylabel("Sharpe")
        ax6_t = ax6.twinx()
        ax6_t.plot(ret.index, roll_vol, color="#a855f7", alpha=0.6, label=f"滚动波动({rwin}日)")
        ax6_t.set_ylabel("年化波动")
        ax6.set_title("滚动夏普与波动")
        ax6.grid(alpha=0.2)
        ax6.xaxis.set_major_formatter(date_formatter)
        charts.append({"title": "滚动夏普与波动", "img": fig_to_base64(fig6)})
    except Exception:
        pass

    # 60日滚动β（若有基准）
    if benchmark_series is not None and "benchmark_return" in benchmark_series:
        try:
            win = 60
            tmp = backtest[["strategy_return"]].join(benchmark_series[["benchmark_return"]], how="inner").dropna()
            cov = tmp["strategy_return"].rolling(win).cov(tmp["benchmark_return"])
            var = tmp["benchmark_return"].rolling(win).var()
            roll_beta = (cov / var).replace([np.inf, -np.inf], np.nan)
            fig7, ax7 = plt.subplots(figsize=(11, 3.5))
            ax7.plot(roll_beta.index, roll_beta, color="#f59e0b")
            ax7.axhline(1.0, color="#9ca3af", linestyle="--", linewidth=1)
            ax7.set_title(f"滚动β（{win}日）")
            ax7.grid(alpha=0.2)
            ax7.xaxis.set_major_formatter(date_formatter)
            charts.append({"title": "滚动β", "img": fig_to_base64(fig7)})
        except Exception:
            pass

    # 概率校准/提升图（仅 ML 策略可用）
    if "probability" in backtest.columns:
        try:
            outcome = (backtest["asset_return"].shift(-1) > 0).astype(int)
            dfc = pd.DataFrame({"p": backtest["probability"].clip(0, 1), "y": outcome}).dropna()
            bins = np.linspace(0, 1, 11)
            dfc["bin"] = np.digitize(dfc["p"], bins) - 1
            calib = dfc.groupby("bin").agg(p_mean=("p", "mean"), y_rate=("y", "mean"))
            lift = (
                dfc.assign(q=pd.qcut(dfc["p"], 10, duplicates="drop"))
                .groupby("q", observed=False)
                .agg(r=("y", "mean"))
            )
            fig8, ax8 = plt.subplots(1, 2, figsize=(12, 4))
            ax8[0].plot([0, 1], [0, 1], "--", color="#9ca3af")
            ax8[0].plot(calib["p_mean"], calib["y_rate"], marker="o", color="#2563eb")
            ax8[0].set_title("概率校准（可靠性图）")
            ax8[0].set_xlabel("预测概率")
            ax8[0].set_ylabel("实际上涨率")
            ax8[1].bar(range(len(lift)), lift["r"], color="#10b981")
            ax8[1].set_title("分位提升（高概率组应更高胜率）")
            ax8[1].set_xlabel("概率分位(低→高)")
            ax8[1].set_ylabel("上涨率")
            fig8.tight_layout()
            charts.append({"title": "校准与提升", "img": fig_to_base64(fig8)})
        except Exception:
            pass

    # 曝露/杠杆与换手
    try:
        expo = (backtest["position"] * backtest["leverage"]).fillna(0.0)
        turnover = expo.diff().abs().fillna(expo.abs())
        fig9, ax9 = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax9[0].plot(backtest.index, expo, color="#0ea5e9", label="曝露=仓位×杠杆")
        ax9[0].axhline(0, color="#9ca3af", linewidth=1)
        ax9[0].set_title("曝露与换手")
        ax9[0].legend(loc="upper left")
        ax9[0].grid(alpha=0.2)
        ax9[1].bar(backtest.index, turnover, color="#94a3b8")
        ax9[1].set_ylabel("|Δ曝露|")
        ax9[1].grid(alpha=0.2)
        ax9[1].xaxis.set_major_formatter(date_formatter)
        charts.append({"title": "曝露与换手", "img": fig_to_base64(fig9)})
    except Exception:
        pass

    # 未来情景预测（乐观/中性/悲观）——基于最近一年收益与当前信号构造区间
    try:
        hist = backtest["strategy_return"].dropna().tail(252)
        if not hist.empty:
            mu = float(hist.mean())
            sigma = float(hist.std())
            z = 0.84  # 约 80% 分位
            # 若有模型概率，用其对期望收益做自适应调制
            if "probability" in backtest.columns:
                p0 = float(backtest["probability"].dropna().iloc[-1])
                adj = (p0 - 0.5) * sigma * 2  # 信号越强，越偏离均值
            else:
                adj = 0.0
            r_mid = mu + adj
            r_opt = r_mid + z * sigma
            r_pes = r_mid - z * sigma
            horizon = 60  # 未来 60 个交易日
            last_date = backtest.index[-1]
            future_idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="B")
            base = float(backtest["cum_strategy"].iloc[-1])
            mid_path = base * np.cumprod(np.ones(horizon) * (1 + r_mid))
            opt_path = base * np.cumprod(np.ones(horizon) * (1 + r_opt))
            pes_path = base * np.cumprod(np.ones(horizon) * (1 + r_pes))

            # 画图：最近 60 日历史 + 未来乐观/悲观
            fig10, ax10 = plt.subplots(figsize=(11, 4))
            hist_idx = backtest.index[-60:]
            ax10.plot(hist_idx, backtest.loc[hist_idx, "cum_strategy"], color="#2563eb", label="历史净值")
            ax10.plot(future_idx, mid_path, color="#f59e0b", label="中性情形")
            ax10.plot(future_idx, opt_path, color="#10b981", label="乐观情形")
            ax10.plot(future_idx, pes_path, color="#ef4444", label="悲观情形")
            ax10.fill_between(future_idx, pes_path, opt_path, color="#e5e7eb", alpha=0.6, label="区间")
            ax10.set_title("未来 60 日情景预测（乐观/悲观）")
            ax10.set_ylabel("净值倍数")
            ax10.grid(alpha=0.2)
            ax10.legend(loc="upper left")
            ax10.xaxis.set_major_formatter(date_formatter)
            charts.append({"title": "情景预测（60日）", "img": fig_to_base64(fig10)})
    except Exception:
        pass

    return charts


def _run_quant_pipeline_inner(params: StrategyInput) -> dict[str, Any]:
    """Execute the end-to-end workflow and return context for rendering."""
    if params.start_date >= params.end_date:
        raise QuantStrategyError("Start date must be earlier than end date.")

    # 使用请求级随机种子，确保可复现
    try:
        _ensure_global_seed(int(getattr(params, "random_seed", DEFAULT_STRATEGY_SEED)))
    except Exception:
        _ensure_global_seed(DEFAULT_STRATEGY_SEED)

    warnings: list[str] = []
    # Auto-apply best ML config when available
    if params.strategy_engine in {"ml_momentum", "multi_combo", "rl_policy"} and params.auto_apply_best_config:
        engine, mlp = _load_best_ml_config(params.ticker)
        if engine:
            try:
                params = replace(params, ml_model=engine, ml_params=mlp or params.ml_params)
                warnings.append(f"已根据训练缓存自动应用最优引擎：{engine}。可在表单取消自动应用或手动覆盖参数。")
            except Exception:
                pass
    prices, fetch_warnings = fetch_price_data(params.ticker, params.start_date, params.end_date)
    warnings.extend(fetch_warnings)
    prices, quality_report = sanitize_price_history(prices)
    warnings.extend(quality_report.notes)

    min_required = max(
        params.long_window + params.rsi_period,
        params.long_window * 3,
        params.train_window + params.test_window,
        200,
    )
    if prices.shape[0] < min_required:
        buffer_days = max(params.long_window * 3, 365)
        extended_start = params.start_date - timedelta(days=buffer_days)
        warnings.append(
            f"原始区间内数据不足，已自动向前扩展至 {extended_start.isoformat()} "
            "以满足指标计算所需的历史长度。"
        )
        prices, extended_warnings = fetch_price_data(params.ticker, extended_start, params.end_date)
        warnings.extend(extended_warnings)
        prices, extended_report = sanitize_price_history(prices)
        warnings.extend(extended_report.notes)

    prices = compute_indicators(prices, params.short_window, params.long_window, params.rsi_period)
    if prices.empty:
        raise QuantStrategyError(
            "可用数据不足以计算指标，请尝试延长回测窗口或缩短均线周期。"
        )

    market_context = fetch_market_context(params)
    auxiliary = collect_auxiliary_data(params, market_context or {})
    context_features = extract_context_features(auxiliary)

    remote_overrides = fetch_remote_strategy_overrides(params)
    if remote_overrides.get("note"):
        warnings.append(str(remote_overrides["note"]))

    combo_details: list[dict[str, Any]] = []
    component_outcomes: list[StrategyOutcome] = []
    ensemble_weights: dict[str, float] = {}

    ml_context: tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]] | None = None

    if params.strategy_engine in {"ml_momentum", "rl_policy", "multi_combo"}:
        ml_context = run_ml_backtest(prices, params, context_features)
        ml_backtest, ml_metrics, ml_stats = ml_context
        warnings.extend(ml_stats.pop("runtime_warnings", []))
        ml_outcome = StrategyOutcome("机器学习动量", ml_backtest, ml_metrics, ml_stats)

        if params.strategy_engine == "ml_momentum":
            component_outcomes.append(ml_outcome)
            backtest, metrics, stats = ml_backtest, ml_metrics, ml_stats
        elif params.strategy_engine == "rl_policy":
            rl_backtest, rl_metrics, rl_stats = run_rl_policy_backtest(prices, params, ml_context, context_features)
            component_outcomes.append(StrategyOutcome("强化学习策略", rl_backtest, rl_metrics, rl_stats))
            backtest, metrics, stats = rl_backtest, rl_metrics, rl_stats
        else:
            component_outcomes.append(ml_outcome)
            try:
                sma_backtest, sma_metrics, sma_stats = backtest_sma_strategy(prices, params)
                warnings.extend(sma_stats.pop("runtime_warnings", []))
                component_outcomes.append(StrategyOutcome("传统双均线", sma_backtest, sma_metrics, sma_stats))
            except QuantStrategyError as exc:
                warnings.append(f"组合策略中的双均线部分计算失败：{exc}")
            try:
                rl_backtest, rl_metrics, rl_stats = run_rl_policy_backtest(prices, params, ml_context, context_features)
                component_outcomes.append(StrategyOutcome("强化学习策略", rl_backtest, rl_metrics, rl_stats))
            except QuantStrategyError as exc:
                warnings.append(f"强化学习策略生成失败：{exc}")
            combined_outcome, ensemble_weights = combine_strategy_outcomes(component_outcomes, params, overrides=remote_overrides)
            backtest, metrics, stats = combined_outcome.backtest, combined_outcome.metrics, combined_outcome.stats
            combo_details = [
                {
                    "engine": outcome.engine,
                    "backtest": outcome.backtest,
                    "metrics": outcome.metrics,
                    "stats": outcome.stats,
                    "weight": ensemble_weights.get(outcome.engine, outcome.weight),
                }
                for outcome in component_outcomes
            ]
    else:
        backtest, metrics, stats = backtest_sma_strategy(prices, params)
        component_outcomes.append(StrategyOutcome("传统双均线", backtest, metrics, stats))

    warnings.extend(stats.pop("runtime_warnings", []))

    if not ensemble_weights and component_outcomes:
        ensemble_weights = {component_outcomes[0].engine: 1.0}

    benchmark_metrics: list[dict[str, str]] = []
    benchmark_stats: dict[str, float] | None = None
    benchmark_series: pd.DataFrame | None = None
    benchmark_label = ""

    if params.benchmark_ticker:
        benchmark_label = params.benchmark_ticker.upper()
        benchmark_prices, bench_warnings = fetch_price_data(
            params.benchmark_ticker,
            params.start_date,
            params.end_date,
        )
        warnings.extend(bench_warnings)
        benchmark_prices, bench_quality = sanitize_price_history(benchmark_prices)
        warnings.extend(bench_quality.notes)
        if benchmark_prices.empty:
            warnings.append(f"未能获取基准 {benchmark_label} 的行情数据，已跳过对比分析。")
        else:
            benchmark_returns = benchmark_prices["adj close"].pct_change().fillna(0)
            combined = backtest[["strategy_return"]].join(
                benchmark_returns.rename("benchmark_return"),
                how="inner",
            ).dropna()
            if combined.empty:
                warnings.append("基准与策略的交易日无交集，基准对比已跳过。")
            else:
                combined["benchmark_cum"] = (1 + combined["benchmark_return"]).cumprod()
                annual_factor = stats["annual_factor"]
                benchmark_total_return = combined["benchmark_cum"].iloc[-1] - 1
                benchmark_vol = combined["benchmark_return"].std() * np.sqrt(annual_factor)
                rf = _get_risk_free_rate_annual()
                benchmark_sharpe = calculate_sharpe(
                    combined["benchmark_return"], trading_days=annual_factor, risk_free_rate=rf
                )
                correlation = combined["strategy_return"].corr(combined["benchmark_return"])
                beta = calculate_beta(combined["strategy_return"], combined["benchmark_return"])
                # CAPM-style annualized alpha using risk-free rate
                strat_excess_daily = combined["strategy_return"].mean() - rf / annual_factor
                bench_excess_daily = combined["benchmark_return"].mean() - rf / annual_factor
                alpha = annual_factor * (strat_excess_daily - beta * bench_excess_daily)
                relative = combined["strategy_return"] - combined["benchmark_return"]
                tracking_error = relative.std() * np.sqrt(annual_factor)
                info_ratio = (
                    relative.mean() * annual_factor / tracking_error if tracking_error != 0 else 0.0
                )

                benchmark_metrics = [
                    build_metric("基准累计收益率", format_percentage(benchmark_total_return)),
                    build_metric("基准年化波动率", format_percentage(benchmark_vol)),
                    build_metric("基准夏普比率", f"{benchmark_sharpe:.2f}"),
                    build_metric("策略相对基准α", format_percentage(alpha)),
                    build_metric("β系数", f"{beta:.2f}"),
                    build_metric("与基准相关系数", f"{correlation:.2f}"),
                    build_metric("信息比率", f"{info_ratio:.2f}"),
                    build_metric("跟踪误差", format_percentage(tracking_error)),
                ]
                benchmark_stats = {
                    "total_return": benchmark_total_return,
                    "volatility": benchmark_vol,
                    "sharpe": benchmark_sharpe,
                    "alpha": alpha,
                    "beta": beta,
                    "correlation": correlation,
                    "info_ratio": info_ratio,
                    "tracking_error": tracking_error,
                }
                benchmark_series = combined

    charts = generate_charts(prices, backtest, benchmark_series, params) if params.include_plots else []
    if params.include_plots and stats.get("shap_img"):
        charts.append({"title": "特征重要性（SHAP）", "img": stats.get("shap_img")})
    recommendations = generate_recommendations(stats, benchmark_stats, params, market_context)
    related_portfolios = build_related_portfolios(params, market_context, params.capital)
    key_takeaways = build_key_takeaways(stats, benchmark_stats, params)
    user_guidance = build_user_guidance(stats, benchmark_stats, params)
    advanced_research = build_flagship_research_bundle(
        params=params,
        prices=prices,
        backtest=backtest,
        stats=stats,
        benchmark_stats=benchmark_stats,
        market_context=market_context or {},
        combo_details=combo_details,
    )
    try:
        feature_dataset_for_analysis, feature_columns_for_analysis = build_feature_matrix(prices, params)
    except Exception:
        feature_dataset_for_analysis, feature_columns_for_analysis = None, []
    statistical_bundle = build_statistical_baselines(prices, params)
    deep_signal_bundle = (
        run_deep_signal_model(feature_dataset_for_analysis, feature_columns_for_analysis)
        if feature_dataset_for_analysis is not None and feature_columns_for_analysis
        else None
    )
    multimodal_bundle = build_multimodal_bundle(
        params,
        feature_dataset_for_analysis,
        market_context,
        fundamentals_override=auxiliary.fundamentals,
        macro_bundle=auxiliary.macro,
    )
    knowledge_bundle = build_knowledge_graph_bundle(params, market_context, feature_dataset_for_analysis)
    factor_scorecard = build_factor_scorecard(
        prices,
        feature_dataset_for_analysis,
        auxiliary.fundamentals,
    )
    factor_effectiveness = analyze_factor_effectiveness(feature_dataset_for_analysis, feature_columns_for_analysis)
    risk_dashboard = build_risk_dashboard(stats, benchmark_stats)
    mlops_report = build_mlops_report(params, stats)
    macro_highlight = summarize_macro_highlight(auxiliary.macro)
    scenario_simulation = build_scenario_simulation(backtest, stats)
    opportunity_radar = build_opportunity_radar(params, factor_effectiveness, knowledge_bundle)
    ensemble_bundle = build_model_ensemble_view(
        statistical_bundle,
        stats,
        deep_signal_bundle,
        graph_bundle=knowledge_bundle,
        factor_bundle=factor_effectiveness,
    )
    model_weights = compute_model_weights(
        statistical_bundle,
        stats,
        deep_signal_bundle,
        knowledge_bundle,
        factor_effectiveness,
    )
    executive_briefing = build_executive_briefing(
        params,
        ensemble_bundle,
        model_weights,
        risk_dashboard,
        knowledge_bundle,
        factor_effectiveness,
        multimodal_bundle,
        deep_signal_bundle,
        scenario_simulation,
        opportunity_radar,
    )
    rl_playbook = build_reinforcement_playbook(
        backtest,
        (params.transaction_cost_bps + params.slippage_bps) / 10000.0,
    )
    user_questions = build_user_questions(
        stats,
        recommendations,
        risk_dashboard,
        model_weights,
        ensemble_bundle,
        scenario_simulation,
        opportunity_radar,
    )
    advisor_playbook = build_advisor_playbook(
        stats,
        user_guidance,
        recommendations,
        scenario_simulation,
        risk_dashboard,
        opportunity_radar,
        macro_highlight,
    )

    combo_results: list[dict[str, Any]] = []
    if params.strategy_engine == "multi_combo" and combo_details:
        for idx, entry in enumerate(combo_details):
            stats_item = entry["stats"]
            metrics_item = entry["metrics"]
            if idx == 0:
                guidance_item = user_guidance
            else:
                guidance_item = build_user_guidance(stats_item, None, params)
            engine_name = entry["engine"] + ("（主）" if idx == 0 else "")
            combo_results.append(
                {
                    "engine": engine_name,
                    "metrics": metrics_item,
                    "stats": {
                        "total_return": stats_item.get("total_return"),
                        "cagr": stats_item.get("cagr"),
                        "sharpe": stats_item.get("sharpe"),
                        "max_drawdown": stats_item.get("max_drawdown"),
                        "volatility": stats_item.get("volatility"),
                        "annual_turnover": stats_item.get("annual_turnover"),
                        "average_holding_days": stats_item.get("average_holding_days"),
                    },
                    "weight": entry.get("weight"),
                    "quick_summary": guidance_item.get("quick_summary", []),
                    "confidence_label": guidance_item.get("confidence_label"),
                    "confidence_score": guidance_item.get("confidence_score"),
                    "action_plan": guidance_item.get("action_plan", []),
                    "risk_alerts": guidance_item.get("risk_alerts", []),
                }
            )

    remote_meta = {k: remote_overrides.get(k) for k in ("source", "version", "timestamp") if remote_overrides.get(k) is not None}

    engine_label = (
        _("机器学习动量 + 风险控制")
        if params.strategy_engine == "ml_momentum"
        else _("组合策略（主策略：机器学习动量）")
        if params.strategy_engine == "multi_combo"
        else _("双均线动量框架")
    )

    walk_forward_report = build_walk_forward_report(backtest.get("strategy_return"))
    purged_schedule = build_purged_kfold_schedule(
        feature_dataset_for_analysis.index if isinstance(feature_dataset_for_analysis, pd.DataFrame) else None,
        n_splits=max(2, params.validation_slices or 3),
        embargo=getattr(params, "embargo_days", 5),
    )
    stats["validation_report"] = {
        "walk_forward": walk_forward_report,
        "purged_kfold": purged_schedule,
    }
    stats["tail_risk_summary"] = compute_tail_risk_summary(backtest.get("strategy_return"))
    metadata = collect_repro_metadata(params)
    risk_controls = {
        "volatility_target": {
            "target": params.volatility_target,
            "realized": stats.get("volatility"),
        },
        "tail_risk": stats.get("tail_risk_summary"),
        "max_drawdown_stop": params.max_drawdown_stop,
        "daily_exposure_limit": params.daily_exposure_limit,
    }
    label_meta = {
        "label_style": params.label_style,
        "tb_up": params.tb_up,
        "tb_down": params.tb_down,
        "tb_dynamic": params.tb_dynamic,
        "tb_vol_multiplier": params.tb_vol_multiplier,
        "tb_vol_window": params.tb_vol_window,
        "tb_max_holding": params.tb_max_holding,
        "return_path": params.return_path,
    }
    if stats.get("tb_dynamic_summary"):
        label_meta["tb_dynamic_summary"] = stats.get("tb_dynamic_summary")
    if hasattr(params, "tb_up_applied") or hasattr(params, "tb_down_applied"):
        label_meta["tb_applied"] = {
            "up": getattr(params, "tb_up_applied", params.tb_up),
            "down": getattr(params, "tb_down_applied", params.tb_down),
        }

    walk_forward_training: dict[str, Any] | None = None
    if getattr(params, "include_walk_forward_report", False):
        walk_forward_training = _load_latest_walk_forward_report(params)
        if walk_forward_training:
            stats["walk_forward_training"] = walk_forward_training
        else:
            warnings.append(
                "未找到 walk-forward 报告，请运行 trading.mlops.walk_forward_train 生成最新报告。"
            )

    result_payload = {
        "ticker": params.ticker.upper(),
        "start_date": params.start_date.strftime("%Y-%m-%d"),
        "end_date": params.end_date.strftime("%Y-%m-%d"),
        "metrics": metrics,
        "benchmark_ticker": benchmark_label if benchmark_metrics else "",
        "benchmark_metrics": benchmark_metrics,
        "recent_rows": format_table(backtest),
        "warnings": warnings,
        "stats": stats,
        "benchmark_stats": benchmark_stats,
        "charts": charts,
        "recommendations": recommendations,
        "related_portfolios": related_portfolios,
        "key_takeaways": key_takeaways,
        "market_context": market_context,
        "risk_profile": RISK_PROFILE_LABELS.get(params.risk_profile, params.risk_profile),
        "capital": params.capital,
        "engine": params.strategy_engine,
        "engine_label": engine_label,
        "params": {
            "ticker": params.ticker.upper(),
            "benchmark": params.benchmark_ticker,
            "start_date": params.start_date.isoformat(),
            "end_date": params.end_date.isoformat(),
            "short_window": params.short_window,
            "long_window": params.long_window,
            "rsi_period": params.rsi_period,
            "volatility_target": params.volatility_target,
            "transaction_cost_bps": params.transaction_cost_bps,
            "slippage_bps": params.slippage_bps,
            "min_holding_days": params.min_holding_days,
            "strategy_engine": params.strategy_engine,
            "risk_profile": params.risk_profile,
            "capital": params.capital,
            "investment_horizon": params.investment_horizon,
            "experience_level": params.experience_level,
            "primary_goal": params.primary_goal,
            "interest_keywords": params.interest_keywords,
            "max_drawdown_stop": params.max_drawdown_stop,
            "daily_exposure_limit": params.daily_exposure_limit,
            "rl_engine": params.rl_engine,
            "validation_slices": params.validation_slices,
            "out_of_sample_ratio": params.out_of_sample_ratio,
            "execution_liquidity_buffer": params.execution_liquidity_buffer,
            "execution_penalty_bps": params.execution_penalty_bps,
            "return_path": params.return_path,
            "slippage_model": params.slippage_model,
            "borrow_cost_bps": params.borrow_cost_bps,
            "long_borrow_cost_bps": params.long_borrow_cost_bps,
            "short_borrow_cost_bps": params.short_borrow_cost_bps,
            "max_adv_participation": params.max_adv_participation,
            "class_weight_mode": params.class_weight_mode,
            "focal_gamma": params.focal_gamma,
            "tb_dynamic": params.tb_dynamic,
            "tb_vol_multiplier": params.tb_vol_multiplier,
            "tb_vol_window": params.tb_vol_window,
            "include_walk_forward_report": params.include_walk_forward_report,
            "walk_forward_horizon_days": params.walk_forward_horizon_days,
            "walk_forward_step_days": params.walk_forward_step_days,
            "walk_forward_jobs": params.walk_forward_jobs,
            "validation_summary": stats.get("validation_summary_compact"),
            "threshold_scan_summary": stats.get("threshold_scan_summary"),
        },
        "guidance": user_guidance,
        "combo_results": combo_results,
        "include_plots": params.include_plots,
        "show_ai_thoughts": params.show_ai_thoughts,
        "advanced_research": advanced_research,
        "statistical_bundle": statistical_bundle,
        "deep_signal_bundle": deep_signal_bundle,
        "multimodal_bundle": multimodal_bundle,
        "ensemble_bundle": ensemble_bundle,
        "knowledge_bundle": knowledge_bundle,
        "factor_scorecard": factor_scorecard,
        "macro_bundle": auxiliary.macro,
        "event_bundle": auxiliary.events,
        "financial_snapshot": auxiliary.financials,
        "capital_flows": auxiliary.capital_flows,
        "news_sentiment": auxiliary.news_sentiment,
        "options_metrics": auxiliary.options_metrics,
        "global_macro_context": auxiliary.global_macro,
        "factor_effectiveness": factor_effectiveness,
        "model_weights": model_weights,
        "risk_dashboard": risk_dashboard,
        "mlops_report": mlops_report,
        "executive_briefing": executive_briefing,
        "user_questions": user_questions,
        "macro_highlight": macro_highlight,
        "scenario_simulation": scenario_simulation,
        "opportunity_radar": opportunity_radar,
        "advisor_playbook": advisor_playbook,
        "rl_playbook": rl_playbook,
        "validation_report": stats.get("validation_report"),
        "calibration": stats.get("calibration"),
        "ensemble_breakdown": {
            "weights": ensemble_weights,
            "available": params.strategy_engine == "multi_combo",
        },
        "remote_config": remote_meta,
        "metadata": metadata,
        "risk_controls": risk_controls,
        "return_path": params.return_path,
        "label_meta": label_meta,
        "repro": metadata,
        "walk_forward_training": walk_forward_training,
    }
    if stats.get("threshold_scan"):
        result_payload["params"]["threshold_scan"] = stats.get("threshold_scan")
    extra_risk_alerts = list(user_guidance.get("risk_alerts", []))
    if stats.get("risk_events"):
        extra_risk_alerts.extend(stats.get("risk_events", []))
        warnings.extend(stats.get("risk_events", []))
    result_payload.update(
        {
            "quick_summary": user_guidance.get("quick_summary", []),
            "action_plan": user_guidance.get("action_plan", []),
            "risk_alerts": extra_risk_alerts,
            "education_tips": user_guidance.get("education_tips", []),
            "confidence_label": user_guidance.get("confidence_label"),
            "confidence_score": user_guidance.get("confidence_score"),
            "experience_label": user_guidance.get("experience_label"),
            "investment_horizon_label": user_guidance.get("investment_horizon_label"),
            "primary_goal_label": user_guidance.get("primary_goal_label"),
            "disclaimer": user_guidance.get("disclaimer"),
        }
    )
    hyperopt_report = stats.get("hyperopt_report")
    if hyperopt_report:
        result_payload["hyperopt_report"] = hyperopt_report
    progress_plan = _build_progress_plan(params, result_payload)
    result_payload["progress_plan"] = progress_plan
    _sanitize_analysis_sections(result_payload)

    result_payload["task_feedback"] = {
        "remaining_steps": progress_plan["remaining"],
        "eta_seconds": progress_plan["eta_seconds"],
    }
    return result_payload


def _sanitize_analysis_sections(payload: dict[str, Any]) -> None:
    """Sanitize LLM / HTML fragments before渲染到模板，避免 XSS。"""
    global_macro = payload.get("global_macro_context")
    if isinstance(global_macro, dict):
        if global_macro.get("data"):
            global_macro["data"] = mark_safe(sanitize_html_fragment(global_macro["data"]))
        if global_macro.get("message"):
            global_macro["message"] = mark_safe(sanitize_html_fragment(global_macro["message"]))
        summary = global_macro.get("summary")
        if isinstance(summary, dict):
            for key, value in list(summary.items()):
                summary[key] = mark_safe(sanitize_html_fragment(value))

    events = payload.get("event_bundle")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("title"):
                event["title"] = sanitize_html_fragment(event["title"])
            if event.get("summary"):
                event["summary"] = sanitize_html_fragment(event["summary"])


def _build_progress_plan(params: StrategyInput, payload: dict[str, Any]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [
        {"key": "stats", "label": _("核心统计"), "eta_seconds": 2, "ready": True},
        {
            "key": "visuals",
            "label": _("图表/风控面板"),
            "eta_seconds": 3,
            "ready": bool(payload.get("charts")) and bool(params.include_plots),
        },
        {
            "key": "ai_insights",
            "label": _("AI 结论"),
            "eta_seconds": 4,
            "ready": not params.show_ai_thoughts,
        },
    ]
    remaining = sum(1 for step in steps if not step["ready"])
    eta_seconds = sum(step["eta_seconds"] for step in steps if not step["ready"])
    message = (
        _("核心统计已就绪，后续将依次补充图表与 AI 解读。")
        if remaining
        else _("所有步骤已完成。")
    )
    return {
        "steps": steps,
        "remaining": remaining,
        "eta_seconds": eta_seconds,
        "message": message,
    }


def run_quant_pipeline(params: StrategyInput) -> dict[str, Any]:
    """Wrapper with统一的耗时 metrics + request metadata."""
    pipeline_started = time.perf_counter()
    success = False
    error_message: str | None = None
    result_payload: dict[str, Any] | None = None
    try:
        if isinstance(params.start_date, str):
            params.start_date = datetime.fromisoformat(params.start_date).date()
        if isinstance(params.end_date, str):
            params.end_date = datetime.fromisoformat(params.end_date).date()
    except Exception:
        pass
    _ensure_global_seed(getattr(params, "random_seed", DEFAULT_STRATEGY_SEED))
    try:
        result = _run_quant_pipeline_inner(params)
        if isinstance(result, dict):
            result_payload = result
        success = True
        return result
    except Exception as exc:
        error_message = str(exc)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - pipeline_started) * 1000.0
        try:
            params.exec_latency_ms = elapsed_ms
        except Exception:
            pass
        if result_payload is not None:
            metadata = result_payload.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["exec_latency_ms"] = round(elapsed_ms, 2)
        record_metric(
            "backtest.pipeline",
            ticker=params.ticker.upper(),
            engine=params.strategy_engine,
            user_id=params.user_id,
            request_id=params.request_id,
            duration_ms=round(elapsed_ms, 2),
            success=success,
            error=error_message,
        )
    # function end


def fig_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.tight_layout()
    # Increase DPI for clarity so charts are readable in smaller cards
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def format_percentage(value: float) -> str:
    """Format a decimal return as percentage string."""
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value:.2%}"


def calculate_sharpe(returns: pd.Series, trading_days: int, risk_free_rate: float = 0.0) -> float:
    excess_returns = returns - risk_free_rate / trading_days
    std = returns.std()
    if std == 0:
        return 0.0
    return np.sqrt(trading_days) * excess_returns.mean() / std


def calculate_sortino(returns: pd.Series, trading_days: int, risk_free_rate: float = 0.0) -> float:
    downside = returns.copy()
    downside[downside > 0] = 0
    downside_std = np.sqrt((downside**2).mean())
    if downside_std == 0:
        return 0.0
    avg_excess = returns.mean() - risk_free_rate / trading_days
    return np.sqrt(trading_days) * avg_excess / downside_std


def _get_risk_free_rate_annual() -> float:
    try:
        return float(os.environ.get("RISK_FREE_RATE_ANNUAL", "0.0"))
    except ValueError:
        return 0.0


def calculate_cagr(cumulative_returns: pd.Series, trading_days: int) -> float:
    if cumulative_returns.empty:
        return 0.0
    total_return = cumulative_returns.iloc[-1]
    periods = cumulative_returns.shape[0]
    if periods <= 1 or total_return <= 0:
        return 0.0
    return total_return ** (trading_days / periods) - 1


def calculate_calmar(cagr: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return 0.0
    return cagr / abs(max_drawdown)


def calculate_hit_ratio(returns: pd.Series) -> float:
    positive = (returns > 0).sum()
    total = (returns != 0).sum()
    if total == 0:
        return 0.0
    return positive / total


def calculate_avg_gain_loss(returns: pd.Series) -> tuple[float, float]:
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    avg_gain = positive.mean() if not positive.empty else 0.0
    avg_loss = negative.mean() if not negative.empty else 0.0
    return avg_gain, avg_loss


def calculate_holding_periods(position: pd.Series) -> float:
    """Compute the average holding period (in trading days) for non-zero positions."""
    if position.empty:
        return 0.0
    clean = position.fillna(0).round().astype(int)
    durations: list[int] = []
    current = clean.iloc[0]
    length = 1 if current != 0 else 0

    for value in clean.iloc[1:]:
        if value == current and value != 0:
            length += 1
        else:
            if current != 0 and length > 0:
                durations.append(length)
            current = value
            length = 1 if current != 0 else 0
    if current != 0 and length > 0:
        durations.append(length)
    return float(np.mean(durations)) if durations else 0.0


def calculate_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    variance = benchmark_returns.var()
    if variance == 0:
        return 0.0
    covariance = strategy_returns.cov(benchmark_returns)
    return covariance / variance


def calculate_target_leverage(
    position: pd.Series,
    realized_vol: pd.Series,
    target_vol: float = 0.15,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Scale exposure to hit target annualized volatility."""
    vol = realized_vol.replace(0, np.nan)
    leverage = target_vol / vol
    leverage = leverage.clip(lower=0, upper=max_leverage).fillna(0)
    leverage = leverage.where(position != 0, 0)
    return leverage


def calculate_var_cvar(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    """Compute daily Value at Risk and Conditional VaR at given confidence."""
    if returns.empty:
        return 0.0, 0.0
    quantile = returns.quantile(1 - alpha)
    var = min(quantile, 0.0)
    tail_losses = returns[returns <= quantile]
    cvar = tail_losses.mean() if not tail_losses.empty else 0.0
    return var, cvar


def build_core_metrics(
    stats: dict[str, Any],
    *,
    include_prediction: bool = False,
    include_auc: bool = False,
) -> list[dict[str, str]]:
    """Render the standard metric cards from a stats dictionary."""
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    metrics = [
        build_metric("策略累计收益率", format_percentage(_float(stats.get("total_return")))),
        build_metric("买入持有收益率", format_percentage(_float(stats.get("buy_hold_return")))),
        build_metric("夏普比率", f"{_float(stats.get('sharpe')):.2f}"),
        build_metric(
            "夏普置信区间",
            f"[{_float(stats.get('sharpe_ci_lower')):.2f}, {_float(stats.get('sharpe_ci_upper')):.2f}]"
            if stats.get("sharpe_ci_lower") is not None and stats.get("sharpe_ci_upper") is not None
            else "N/A",
        ),
        build_metric(
            "Deflated Sharpe",
            f"{_float(stats.get('deflated_sharpe')):.2f}" if stats.get("deflated_sharpe") is not None else "N/A",
        ),
        build_metric("最大回撤", format_percentage(_float(stats.get("max_drawdown")))),
        build_metric("索提诺比率", f"{_float(stats.get('sortino')):.2f}"),
        build_metric("年化波动率", format_percentage(_float(stats.get("volatility")))),
        build_metric("年化复合收益率", format_percentage(_float(stats.get("cagr")))),
        build_metric("卡玛比率", f"{_float(stats.get('calmar')):.2f}"),
        build_metric("胜率", format_percentage(_float(stats.get("hit_ratio")))),
        build_metric(
            "单日平均盈亏",
            f"{format_percentage(_float(stats.get('avg_gain')))} / {format_percentage(_float(stats.get('avg_loss')))}",
        ),
        build_metric("平均持仓比例", format_percentage(_float(stats.get("avg_exposure")))),
        build_metric("平均杠杆（波动率目标）", f"{_float(stats.get('avg_leverage')):.2f}x"),
        build_metric("日度95%VaR", format_percentage(-_float(stats.get("var_95")))),
        build_metric("日度95%CVaR", format_percentage(-_float(stats.get("cvar_95")))),
        build_metric("最长回撤恢复期(TWR)", f"{_float(stats.get('twr_days')):.0f} 天" if stats.get("twr_days") is not None else "N/A"),
        build_metric("持续亏损天数", str(int(_float(stats.get("loss_streak"))))),
        build_metric("最长恢复期", f"{_float(stats.get('recovery_days')):.0f} 天" if stats.get("recovery_days") is not None else "N/A"),
        build_metric(
            "White RC 调整p值",
            f"{_float(stats.get('sharpe_pvalue_adjusted')):.3f}" if stats.get("sharpe_pvalue_adjusted") is not None else "N/A",
        ),
    ]

    # OOS（PFWS）指标：显示均值/方差/IQR，帮助识别样本外稳定性
    validation_summary = (
        stats.get("validation_summary_compact")
        or stats.get("validation_oos_summary")
        or {}
    )
    sharpe_oos = validation_summary.get("sharpe") if isinstance(validation_summary, dict) else None
    if isinstance(sharpe_oos, dict) and sharpe_oos:
        metrics.append(
            build_metric(
                "OOS夏普(均值±std)",
                f"{_float(sharpe_oos.get('mean')):.2f} ± {_float(sharpe_oos.get('std')):.2f}",
            )
        )
        metrics.append(
            build_metric(
                "OOS夏普IQR",
                f"{_float(sharpe_oos.get('iqr')):.2f}",
            )
        )

    if include_prediction and stats.get("prediction_accuracy") is not None:
        metrics.append(build_metric("预测胜率", format_percentage(_float(stats.get("prediction_accuracy")))))
    if include_auc and stats.get("auc") is not None:
        auc = _float(stats.get("auc"), float("nan"))
        value = "N/A" if math.isnan(auc) else f"{auc:.2f}"
        metrics.append(build_metric("ROC-AUC", value))
    if stats.get("calibration"):
        calib = stats["calibration"]
        brier = _float(calib.get("brier"), float("nan"))
        brier_val = "N/A" if math.isnan(brier) else f"{brier:.4f}"
        metrics.append(build_metric("Brier Score", brier_val))

    metrics.extend(
        [
            build_metric("年化换手率", format_percentage(_float(stats.get("annual_turnover")))),
            build_metric("平均持仓天数", f"{_float(stats.get('average_holding_days')):.1f}"),
            build_metric("成本占收益比", format_percentage(_float(stats.get("cost_ratio")))),
            build_metric("交易日数量", str(int(_float(stats.get("trading_days"), 0.0)))),
        ]
    )
    return metrics


def summarize_backtest(
    backtest: pd.DataFrame,
    params: StrategyInput,
    *,
    include_prediction: bool = False,
    include_auc: bool = False,
    feature_columns: Optional[list[str]] = None,
    shap_img: Optional[str] = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if backtest.empty:
        raise QuantStrategyError("回测结果为空，无法生成统计指标。")

    annual_factor = 252
    rf = _get_risk_free_rate_annual()

    net_returns = backtest.get("strategy_return")
    if net_returns is None:
        raise QuantStrategyError("回测结果缺少 strategy_return 列。")
    net_returns = net_returns.astype(float).fillna(0.0)

    asset_returns = backtest.get("asset_return")
    if asset_returns is None:
        price_series = backtest.get("adj close")
        if price_series is None:
            raise QuantStrategyError("回测结果缺少 asset_return 与 adj close 列。")
        asset_returns = price_series.astype(float).pct_change().fillna(0.0)
    else:
        asset_returns = asset_returns.astype(float).fillna(0.0)

    cum_strategy = (1 + net_returns).cumprod()
    cum_buy_hold = (1 + asset_returns).cumprod()

    total_return = float(cum_strategy.iloc[-1] - 1) if not cum_strategy.empty else 0.0
    buy_hold_return = float(cum_buy_hold.iloc[-1] - 1) if not cum_buy_hold.empty else 0.0
    max_drawdown = calculate_max_drawdown(cum_strategy)
    sharpe = calculate_sharpe(net_returns, trading_days=annual_factor, risk_free_rate=rf)
    sortino = calculate_sortino(net_returns, trading_days=annual_factor, risk_free_rate=rf)
    volatility = net_returns.std() * np.sqrt(annual_factor)
    cagr = calculate_cagr(cum_strategy, trading_days=annual_factor)
    calmar = calculate_calmar(cagr, max_drawdown)
    hit_ratio = calculate_hit_ratio(net_returns)
    avg_gain, avg_loss = calculate_avg_gain_loss(net_returns)

    position_series = backtest.get("position")
    position_series = position_series.astype(float) if position_series is not None else pd.Series(0.0, index=backtest.index)
    avg_exposure = float(position_series.abs().mean()) if not position_series.empty else 0.0

    leverage_series = backtest.get("leverage")
    leverage_series = leverage_series.astype(float) if leverage_series is not None else pd.Series(0.0, index=backtest.index)
    avg_leverage = float(leverage_series.mean()) if not leverage_series.empty else 0.0

    var_95, cvar_95 = calculate_var_cvar(net_returns, alpha=0.95)
    # Recovery metrics
    cumulative_curve = (1 + net_returns).cumprod()
    recovery_days = recovery_period_days(cumulative_curve)
    loss_streak = int((net_returns < 0).astype(int).groupby((net_returns >= 0).astype(int).cumsum()).sum().max()) if not net_returns.empty else 0
    es_95 = calculate_cvar(net_returns, alpha=0.95)

    exposure_series = backtest.get("exposure")
    if exposure_series is not None:
        exposure_series = exposure_series.astype(float)
        exposure_change = exposure_series.diff().abs().fillna(exposure_series.abs())
    else:
        exposure_change = position_series.diff().abs().fillna(position_series.abs())
    daily_turnover = float(exposure_change.mean()) if not exposure_change.empty else 0.0
    annual_turnover = daily_turnover * annual_factor

    transaction_cost = backtest.get("transaction_cost")
    if transaction_cost is not None:
        transaction_cost = transaction_cost.astype(float).fillna(0.0)
    else:
        transaction_cost = pd.Series(0.0, index=backtest.index)
    execution_cost = backtest.get("execution_cost")
    if execution_cost is not None:
        execution_cost = execution_cost.astype(float).fillna(0.0)
    else:
        execution_cost = pd.Series(0.0, index=backtest.index)
    borrow_cost_series = backtest.get("borrow_cost")
    if borrow_cost_series is not None:
        borrow_cost_series = borrow_cost_series.astype(float).fillna(0.0)
    else:
        borrow_cost_series = pd.Series(0.0, index=backtest.index)
    total_cost = float(transaction_cost.sum() + execution_cost.sum() + borrow_cost_series.sum())

    strategy_return_gross = backtest.get("strategy_return_gross")
    if strategy_return_gross is not None:
        strategy_return_gross = strategy_return_gross.astype(float).fillna(0.0)
        gross_exposure = float(np.abs(strategy_return_gross).sum())
    else:
        # 还原未扣成本的收益，用于成本占比等统计
        strategy_return_gross = net_returns + transaction_cost + execution_cost + borrow_cost_series
        gross_exposure = float(np.abs(strategy_return_gross).sum())
    cost_base = float(exposure_change.abs().sum()) if not exposure_change.empty else 0.0
    cost_ratio = total_cost / max(cost_base, 1e-9)

    avg_holding = calculate_holding_periods(position_series) if not position_series.empty else 0.0

    prediction_accuracy = None
    if include_prediction:
        signal_series = backtest.get("signal")
        if signal_series is not None:
            direction_prediction = np.sign(signal_series.astype(float).fillna(0.0))
            actual_direction = np.sign(asset_returns.shift(-1).fillna(0.0))
            align_mask = direction_prediction != 0
            if align_mask.any():
                prediction_accuracy = float((direction_prediction[align_mask] == actual_direction[align_mask]).sum()) / int(align_mask.sum())
            else:
                prediction_accuracy = 0.0

    auc = float("nan")
    if include_auc and roc_auc_score is not None and "probability" in backtest:
        proba = backtest["probability"].astype(float)
        actual = (asset_returns.shift(-1) > 0).astype(int)
        mask = proba.notna() & actual.notna()
        if mask.sum() > 1 and actual[mask].nunique() > 1:
            try:
                auc = float(roc_auc_score(actual[mask], proba[mask].clip(0, 1)))
            except ValueError:
                auc = float("nan")

    recent_window = net_returns.tail(60)
    if recent_window.empty or recent_window.std() == 0:
        recent_sharpe_60d = 0.0
    else:
        recent_sharpe_60d = float(np.sqrt(252) * recent_window.mean() / (recent_window.std() + 1e-12))

    stats = {
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "volatility": volatility,
        "cagr": cagr,
        "calmar": calmar,
        "hit_ratio": hit_ratio,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "avg_exposure": avg_exposure,
        "avg_leverage": avg_leverage,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "es_95": es_95,
        "annual_turnover": annual_turnover,
        "average_holding_days": avg_holding,
        "cost_ratio": cost_ratio,
        "total_cost": total_cost,
        "transaction_cost_total": float(transaction_cost.sum()),
        "execution_cost_total": float(execution_cost.sum()),
        "borrow_cost_total": float(borrow_cost_series.sum()),
        "trading_days": int(backtest.shape[0]),
        "annual_factor": annual_factor,
        "recent_sharpe_60d": recent_sharpe_60d,
        "recovery_days": recovery_days,
        "loss_streak": loss_streak,
        "feature_columns": feature_columns or [],
        "feature_count": len(feature_columns or []),
        "shap_img": shap_img,
        "twr_days": recovery_days,  # alias for time-to-recovery
        "return_path": getattr(params, "return_path", "close_to_close"),
        "label_return_path": getattr(params, "label_return_path", None) or getattr(params, "return_path", "close_to_close"),
        "pfws_train_window": getattr(params, "train_window", None),
        "pfws_test_window": getattr(params, "test_window", None),
        "pfws_embargo": getattr(params, "embargo_days", None),
        "pfws_enforced": bool(getattr(params, "enforce_pfws_only", False)),
    }
    stats.setdefault(
        "cost_assumptions",
        {
            "slippage_model": getattr(params, "slippage_model", None),
            "cost_rate": (getattr(params, "transaction_cost_bps", 0.0) + getattr(params, "slippage_bps", 0.0)) / 10000.0,
            "long_borrow_bps": getattr(params, "long_borrow_cost_bps", None) or getattr(params, "borrow_cost_bps", None),
            "short_borrow_bps": getattr(params, "short_borrow_cost_bps", None) or getattr(params, "borrow_cost_bps", None),
            "adv_participation": getattr(params, "max_adv_participation", None),
            "execution_mode": getattr(params, "execution_mode", None),
        },
    )

    # 统一记录执行假设与防泄漏策略，便于前端/导出提示
    exec_mode = stats["return_path"]
    label_mode = stats["label_return_path"]
    pfws_policy = "enforced" if stats["pfws_enforced"] else "not_enforced"
    stats["execution_assumptions"] = {
        "return_path": exec_mode,
        "label_return_path": label_mode,
        "description": f"执行口径={exec_mode}，标签口径={label_mode}",
    }
    stats["pfws_policy"] = {
        "status": pfws_policy,
        "train_window": stats["pfws_train_window"],
        "test_window": stats["pfws_test_window"],
        "embargo": stats["pfws_embargo"],
        "note": "PFWS 强制仅对 ML 训练/验证生效；传统/确定性策略无样本外切分。" if not stats["pfws_enforced"] else "全局启用 PFWS 强制，禁止非 PFWS 切分。",
    }

    if include_prediction:
        stats["prediction_accuracy"] = prediction_accuracy if prediction_accuracy is not None else float("nan")
    if include_auc:
        stats["auc"] = auc

    robust = compute_robust_sharpe(net_returns, annual_factor=annual_factor, trials=max(1, int(getattr(params, "hyperopt_trials", 1) or 1)))
    if robust:
        stats["sharpe_std_error"] = robust.get("std_error")
        stats["sharpe_ci"] = robust.get("ci")
        if robust.get("ci") and isinstance(robust.get("ci"), tuple):
            stats["sharpe_ci_lower"] = robust["ci"][0]
            stats["sharpe_ci_upper"] = robust["ci"][1]
        if robust.get("deflated_sharpe") is not None:
            stats["deflated_sharpe"] = robust.get("deflated_sharpe")
    white_rc = compute_white_reality_check(
        net_returns,
        trials=max(1, int(getattr(params, "hyperopt_trials", 1) or 1)),
        annual_factor=annual_factor,
    )
    if white_rc:
        stats["sharpe_pvalue"] = white_rc.get("p_value")
        stats["sharpe_pvalue_adjusted"] = white_rc.get("p_value_adjusted")
        stats["sharpe_zscore"] = white_rc.get("z_score")
    # Bootstrap-based White's RC / SPA
    enable_bootstrap = bool(getattr(params, "stats_enable_bootstrap", True))
    bootstrap_samples = max(0, int(getattr(params, "stats_bootstrap_samples", 600) or 0))
    bootstrap_block = getattr(params, "stats_bootstrap_block", None)
    if enable_bootstrap and bootstrap_samples > 0:
        white_boot = compute_white_reality_check_bootstrap(
            net_returns,
            trials=max(1, int(getattr(params, "hyperopt_trials", 1) or 1)),
            block_size=bootstrap_block,
            bootstrap_samples=bootstrap_samples,
            annual_factor=annual_factor,
            seed=getattr(params, "random_seed", DEFAULT_STRATEGY_SEED),
        )
        if white_boot:
            stats["sharpe_pvalue_bootstrap"] = white_boot.get("p_value_bootstrap")
            stats["sharpe_bootstrap_mean"] = white_boot.get("sharpe_bootstrap_mean")
            stats["sharpe_bootstrap_std"] = white_boot.get("sharpe_bootstrap_std")
            stats["sharpe_bootstrap_block"] = white_boot.get("block_size")
        spa = compute_spa_pvalue(
            net_returns,
            block_size=bootstrap_block,
            bootstrap_samples=bootstrap_samples,
            annual_factor=annual_factor,
            seed=getattr(params, "random_seed", DEFAULT_STRATEGY_SEED),
        )
        if spa:
            stats["sharpe_pvalue_spa"] = spa.get("p_value_spa")
            stats["sharpe_spa_block"] = spa.get("block_size")
            stats["sharpe_spa_bootstrap_mean"] = spa.get("sharpe_bootstrap_mean")
            stats["sharpe_spa_bootstrap_std"] = spa.get("sharpe_bootstrap_std")

    metrics = build_core_metrics(stats, include_prediction=include_prediction, include_auc=include_auc)
    return metrics, stats


def combine_strategy_outcomes(
    outcomes: list[StrategyOutcome],
    params: StrategyInput,
    overrides: Optional[dict[str, Any]] = None,
) -> tuple[StrategyOutcome, dict[str, float]]:
    """Blend multiple strategy outcomes into a single ensemble result."""
    if not outcomes:
        raise QuantStrategyError("没有可合并的策略结果。")
    if len(outcomes) == 1:
        single = outcomes[0]
        return single, {single.engine: 1.0}

    overrides = overrides or {}
    override_weights = overrides.get("weights") if isinstance(overrides.get("weights"), dict) else {}

    raw_weights: list[float] = []
    for outcome in outcomes:
        key = outcome.engine
        weight = override_weights.get(key)
        if weight is None and "（" in key:
            weight = override_weights.get(key.split("（", 1)[0])
        if weight is None:
            weight = outcome.weight
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = 1.0
        raw_weights.append(max(weight, 0.0))

    total_weight = sum(raw_weights)
    if total_weight <= 0:
        raw_weights = [1.0 for _ in outcomes]
        total_weight = float(len(outcomes))

    normalized_weights = [w / total_weight for w in raw_weights]
    weights_map = {out.engine: w for out, w in zip(outcomes, normalized_weights)}

    base = outcomes[0].backtest.copy()
    index = base.index

    numeric_cols = [
        "strategy_return_gross",
        "transaction_cost",
        "signal",
        "position",
        "leverage",
        "exposure",
    ]
    combined_series: dict[str, pd.Series] = {
        col: pd.Series(0.0, index=index, dtype=float) for col in numeric_cols
    }
    probability_values = pd.Series(0.0, index=index, dtype=float)
    probability_weights = pd.Series(0.0, index=index, dtype=float)

    for outcome, weight in zip(outcomes, normalized_weights):
        df = outcome.backtest.reindex(index)
        for col in numeric_cols:
            if col in df:
                combined_series[col] = combined_series[col] + df[col].astype(float).fillna(0.0) * weight
        if "probability" in df:
            proba = df["probability"].astype(float)
            mask = proba.notna()
            probability_values.loc[mask] += proba[mask] * weight
            probability_weights.loc[mask] += weight

    ensemble = base.copy()
    for col, series in combined_series.items():
        ensemble[col] = series

    if probability_weights.gt(0).any():
        prob_series = probability_values.copy()
        mask = probability_weights > 0
        prob_series.loc[mask] = probability_values.loc[mask] / probability_weights.loc[mask]
        prob_series.loc[~mask] = np.nan
        ensemble["probability"] = prob_series

    ensemble["signal"] = np.clip(ensemble.get("signal", pd.Series(0.0, index=index)), -1.0, 1.0)
    ensemble["position"] = np.clip(ensemble.get("position", pd.Series(0.0, index=index)), -1.0, 1.0)
    ensemble["leverage"] = ensemble.get("leverage", pd.Series(0.0, index=index)).clip(lower=0.0)
    ensemble["exposure"] = ensemble["position"] * ensemble["leverage"]
    ensemble["strategy_return_gross"] = ensemble.get("strategy_return_gross", pd.Series(0.0, index=index))
    ensemble["transaction_cost"] = ensemble.get("transaction_cost", pd.Series(0.0, index=index))
    ensemble["strategy_return"] = ensemble["strategy_return_gross"] - ensemble["transaction_cost"]
    ensemble["cum_strategy"] = (1 + ensemble["strategy_return"]).cumprod()
    if "asset_return" in ensemble:
        ensemble["cum_buy_hold"] = (1 + ensemble["asset_return"].fillna(0.0)).cumprod()

    has_probability = any("probability" in out.backtest.columns for out in outcomes)
    feature_union = sorted({col for out in outcomes for col in out.stats.get("feature_columns", [])})
    shap_img = next((out.stats.get("shap_img") for out in outcomes if out.stats.get("shap_img")), None)

    metrics, stats = summarize_backtest(
        ensemble,
        params,
        include_prediction=True,
        include_auc=has_probability,
        feature_columns=feature_union,
        shap_img=shap_img,
    )

    component_breakdown = []
    for outcome, weight in zip(outcomes, normalized_weights):
        component_breakdown.append(
            {
                "engine": outcome.engine,
                "weight": weight,
                "sharpe": outcome.stats.get("sharpe"),
                "total_return": outcome.stats.get("total_return"),
                "max_drawdown": outcome.stats.get("max_drawdown"),
            }
        )

    stats.update(
        {
            "weights": weights_map,
            "component_breakdown": component_breakdown,
        }
    )

    combined_outcome = StrategyOutcome(
        engine="组合策略",
        backtest=ensemble,
        metrics=metrics,
        stats=stats,
        weight=1.0,
    )
    return combined_outcome, weights_map


def _compute_validation_metrics(pnl: pd.Series) -> dict[str, float]:
    pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()
    if pnl.empty:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "hit_ratio": 0.0}
    cumulative = (1 + pnl).cumprod()
    return {
        "sharpe": float(calculate_sharpe(pnl, 252)),
        "cagr": float(calculate_cagr(cumulative, 252)),
        "max_drawdown": float(calculate_max_drawdown(cumulative)),
        "hit_ratio": float(calculate_hit_ratio(pnl)),
    }


def _compute_oos_from_backtest(backtest: pd.DataFrame, params: StrategyInput) -> dict[str, Any] | None:
    """Compute simple PFWS OOS metrics（可用于传统/确定性/RL 回测）."""
    if backtest.empty or "strategy_return" not in backtest:
        return None
    total = len(backtest)
    if total < params.train_window + params.test_window:
        return None
    splitter = PurgedWalkForwardSplit(
        train_window=params.train_window,
        test_window=max(params.test_window, 5),
        embargo=max(0, params.embargo_days),
    )
    slices: list[dict[str, Any]] = []
    for fold_idx, (_, test_idx) in enumerate(splitter.split(total)):
        test_returns = backtest["strategy_return"].iloc[test_idx].fillna(0.0)
        if test_returns.empty:
            continue
        metrics = _compute_validation_metrics(test_returns)
        metrics.update(
            {
                "fold": fold_idx + 1,
                "test_start": str(test_returns.index[0].date()) if hasattr(test_returns.index[0], "date") else str(test_returns.index[0]),
                "test_end": str(test_returns.index[-1].date()) if hasattr(test_returns.index[-1], "date") else str(test_returns.index[-1]),
            }
        )
        slices.append(metrics)
    if not slices:
        return None
    summary = _aggregate_oos_metrics(slices)
    sharpe_stats = summary.get("sharpe") or {}
    penalized = (sharpe_stats.get("mean") or 0.0) - (sharpe_stats.get("std") or 0.0)
    distributions = {
        k: [float(entry.get(k, 0.0)) for entry in slices if entry.get(k) is not None] for k in ("sharpe", "cagr", "max_drawdown", "hit_ratio")
    }
    return {
        "slices": slices,
        "summary": summary,
        "folds": len(slices),
        "penalized_sharpe": penalized,
        "train_window": params.train_window,
        "test_window": params.test_window,
        "embargo": params.embargo_days,
        "distributions": distributions,
    }


def _balanced_sample_weight(labels: pd.Series) -> np.ndarray | None:
    """Compute balanced sample weights for binary labels if sklearn helper is available."""
    if compute_sample_weight is None:
        return None
    try:
        return compute_sample_weight("balanced", labels)
    except Exception:
        return None


def _maybe_get_sample_weight(labels: pd.Series, params: StrategyInput) -> np.ndarray | None:
    """Return sample weight for imbalance handling (balanced or focal)."""
    # Triple-barrier 默认启用 balanced，除非显式禁用
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


def _summarize_multiclass_accuracy(target: pd.Series, pred: pd.Series) -> dict[str, Any]:
    """Per-class accuracy for triple-barrier multiclass targets."""
    target = target.dropna()
    pred = pred.reindex(target.index)
    mask = pred.notna()
    target = target[mask]
    pred = pred[mask]
    if target.empty:
        return {}
    classes = sorted(target.unique())
    per_class: dict[str, float] = {}
    correct = (target == pred)
    for cls in classes:
        cls_mask = target == cls
        per_class[str(int(cls))] = float(correct[cls_mask].mean()) if cls_mask.any() else 0.0
    macro_acc = float(correct.mean())
    return {"per_class_acc": per_class, "macro_acc": macro_acc}


def _confusion_summary(target: pd.Series, pred: pd.Series) -> dict[str, Any]:
    """Simple confusion counts for -1/0/1 labels."""
    target = target.dropna()
    pred = pred.reindex(target.index)
    mask = pred.notna()
    target = target[mask]
    pred = pred[mask]
    if target.empty:
        return {}
    labels = sorted(set(target.unique()) | set(pred.unique()))
    counts: dict[str, dict[str, int]] = {}
    for t in labels:
        row: dict[str, int] = {}
        for p in labels:
            row[str(p)] = int(((target == t) & (pred == p)).sum())
        counts[str(t)] = row
    return {"labels": labels, "matrix": counts}


def _extract_class_prob(proba: np.ndarray, classes: np.ndarray | None, label: int, default: float = 0.5) -> np.ndarray:
    if classes is None or proba.ndim == 1 or proba.shape[1] == 1:
        return np.full(proba.shape[0], default, dtype=float)
    try:
        idx = list(classes).index(label)
        return proba[:, idx]
    except Exception:
        return np.full(proba.shape[0], default, dtype=float)


def _generate_validation_report(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    params: StrategyInput,
) -> dict[str, Any] | None:
    if getattr(params, "enforce_pfws_only", False) and getattr(params, "validation_slices", 0) <= 1:
        # 强制 PFWS 时，不运行任何非PFWS验证
        return None
    if params.validation_slices <= 1:
        return None
    total = len(dataset)
    if total < params.train_window + params.test_window:
        return None
    cost_rate = (params.transaction_cost_bps + params.slippage_bps) / 10000.0
    splitter = PurgedWalkForwardSplit(
        train_window=params.train_window,
        test_window=max(params.test_window, 5),
        embargo=max(0, params.embargo_days),
    )
    slices: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(total)):
        train_slice = dataset.iloc[train_idx]
        test_slice = dataset.iloc[test_idx]
        if train_slice["target"].nunique() < 2 or test_slice.empty:
            continue
        scaler = StandardScaler()
        model_name = (params.ml_model or "sk_gbdt").lower()
        try:
            X_train = scaler.fit_transform(train_slice[feature_columns])
            X_test = scaler.transform(test_slice[feature_columns])
            sample_weight = _balanced_sample_weight(train_slice["target"])

            estimator = None
            if model_name == "lightgbm" and lgb is not None:
                estimator = lgb.LGBMClassifier(
                    learning_rate=0.05,
                    n_estimators=120,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.9,
                    random_state=42,
                )
            elif model_name == "catboost" and CatBoostClassifier is not None:
                estimator = CatBoostClassifier(
                    learning_rate=0.05,
                    iterations=150,
                    depth=6,
                    subsample=0.8,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                )
            elif GradientBoostingClassifier is not None:
                estimator = GradientBoostingClassifier(
                    learning_rate=0.05,
                    n_estimators=180,
                    max_depth=3,
                    subsample=0.8,
                    random_state=42,
                )
            # fallback to logistic regression for calibrated probability
            if estimator is None and LogisticRegression is not None:
                estimator = LogisticRegression(max_iter=400, class_weight="balanced")
            if estimator is None:
                continue
            if not isinstance(estimator, Pipeline):
                pipe = Pipeline([("model", estimator)])
            else:
                pipe = estimator
            fit_params = {}
            if sample_weight is not None:
                if isinstance(pipe, Pipeline):
                    fit_params["model__sample_weight"] = sample_weight
                else:
                    fit_params["sample_weight"] = sample_weight
            pipe.fit(X_train, train_slice["target"], **fit_params)
            proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X_test)
        except Exception:
            continue
        signal = np.where(
            proba >= params.entry_threshold,
            1.0,
            np.where(proba <= params.exit_threshold, -1.0, 0.0),
        )
        exposure = pd.Series(signal, index=test_slice.index, dtype=float).shift(fill_value=0.0)
        turnover = exposure.diff().abs().fillna(exposure.abs())
        pnl = exposure * test_slice["future_return"].fillna(0.0) - turnover * cost_rate
        metrics = _compute_validation_metrics(pnl)
        metrics.update(
            {
                "fold": fold_idx + 1,
                "train_start": str(train_slice.index[0].date()),
                "train_end": str(train_slice.index[-1].date()),
                "test_start": str(test_slice.index[0].date()),
                "test_end": str(test_slice.index[-1].date()),
                "avg_position": float(exposure.abs().mean()),
            }
        )
        slices.append(metrics)
        if len(slices) >= params.validation_slices:
            break
    if not slices:
        return None
    summary = _aggregate_oos_metrics(slices)
    penalized = None
    sharpe_stats = summary.get("sharpe")
    if isinstance(sharpe_stats, dict):
        penalized = (sharpe_stats.get("mean") or 0.0) - (sharpe_stats.get("std") or 0.0)
    return {
        "slices": slices,
        "summary": summary,
        "folds": len(slices),
        "train_window": params.train_window,
        "test_window": params.test_window,
        "embargo": params.embargo_days,
        "penalized_sharpe": penalized,
    }


def _aggregate_oos_metrics(slices: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not slices:
        return summary
    for key in ("sharpe", "cagr", "max_drawdown", "hit_ratio"):
        values = [float(entry.get(key, 0.0)) for entry in slices if entry.get(key) is not None]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "iqr": float(q3 - q1),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }
    return summary


def _build_oos_boxplot(distributions: dict[str, list[float]], title: str) -> str | None:
    """Create a simple boxplot (base64) for OOS metrics."""
    try:
        metrics = []
        labels = []
        for key, values in distributions.items():
            if not values:
                continue
            metrics.append(values)
            labels.append(key.upper())
        if not metrics:
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(metrics, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#93c5fd", alpha=0.7))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        return fig_to_base64(fig)
    except Exception:
        return None


def run_ml_backtest(
    prices: pd.DataFrame,
    params: StrategyInput,
    context_features: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    if GradientBoostingClassifier is None or Pipeline is None or StandardScaler is None:
        raise QuantStrategyError(
            "scikit-learn 未安装，无法启用机器学习策略。请运行 pip install scikit-learn。"
        )

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
        raise QuantStrategyError(
            "历史样本数量不足以完成走期训练，建议延长回测区间或减少窗口。"
        )

    model_warnings: list[str] = []
    risk_events: list[str] = []
    hyperopt_report: dict[str, Any] | None = None

    # 统一默认：自动融合 LSTM+Transformer；若缺少深度学习后端则回退到 GBDT
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
            from .optimization import run_optuna_search

            hyperopt_report = run_optuna_search(dataset, feature_columns, params)
            params.entry_threshold = float(hyperopt_report.get("entry_threshold", params.entry_threshold))
            params.exit_threshold = float(hyperopt_report.get("exit_threshold", params.exit_threshold))
            best_ml = hyperopt_report.get("ml_params") or {}
            if best_ml:
                params.ml_params = {**best_ml}
            model_warnings.append(
                f"已应用 Optuna 搜索结果（score={hyperopt_report.get('best_score', 0.0):.2f}, trials={hyperopt_report.get('trials_ran', 0)}）。"
            )
        except Exception as exc:
            hyperopt_report = {"error": str(exc)}
            model_warnings.append(f"自动超参搜索失败：{exc}")

    # 统一使用 PFWS 进行训练/预测与 OOS 汇总
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

    splitter = PurgedWalkForwardSplit(
        train_window=params.train_window,
        test_window=max(params.test_window, 5),
        embargo=max(0, params.embargo_days),
    )
    if getattr(params, "enforce_pfws_only", False) and splitter is None:
        raise QuantStrategyError("已启用强制 PFWS，无法使用其他切分方案。")
    n_jobs = max(1, int(getattr(params, "walk_forward_jobs", 1) or 1))

    def _run_pfws_fold(fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, Any]:
        train_slice = dataset.iloc[train_idx]
        test_slice = dataset.iloc[test_idx]
        if train_slice["target"].nunique() < 2 or test_slice.empty:
            return {}

        embargo = max(0, int(params.embargo_days))
        val_len = max(1, int(len(train_slice) * max(0.05, min(0.4, params.val_ratio))))
        train_end = max(1, len(train_slice) - embargo)
        val_start = max(0, train_end - val_len)
        core_train = train_slice.iloc[:val_start]
        val_slice = train_slice.iloc[val_start:train_end]

        model_name = (params.ml_model or "sk_gbdt").lower()
        if model_name == "lightgbm" and lgb is not None:
            lgb_kwargs = dict(
                learning_rate=0.05,
                n_estimators=400,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
            )
            if params.ml_params:
                lgb_kwargs.update({k: v for k, v in params.ml_params.items() if k in lgb_kwargs or True})
            estimator = lgb.LGBMClassifier(**lgb_kwargs)
            pipeline = Pipeline([("model", estimator)])
        elif model_name == "catboost" and CatBoostClassifier is not None:
            cb_kwargs = dict(
                learning_rate=0.05,
                depth=6,
                iterations=400,
                subsample=0.8,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            )
            if params.ml_params:
                cb_kwargs.update(params.ml_params)
            estimator = CatBoostClassifier(**cb_kwargs)
            pipeline = Pipeline([("model", estimator)])
        elif model_name in {"lstm", "transformer"}:
            try:
                estimator, descriptor = build_custom_sequence_model(model_name, feature_columns, params)
                pipeline = estimator
            except (RuntimeError, ValueError) as exc:
                raise QuantStrategyError(str(exc)) from exc
        else:
            sk_kwargs = dict(
                learning_rate=0.05,
                n_estimators=250,
                max_depth=3,
                subsample=0.8,
                random_state=42,
            )
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
            e_opt, x_opt = _tune_thresholds_on_validation(
                val_proba,
                val_ret,
                cost_rate,
                n_jobs=getattr(params, "threshold_jobs", 1),
            )
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

        signal_fold = np.where(
            proba_vals >= entry_thr,
            1.0,
            np.where((proba_down if proba_down.size else proba_vals) >= (1 - exit_thr), -1.0, 0.0),
        )
        exposure_fold = pd.Series(signal_fold, index=test_slice.index, dtype=float).shift(fill_value=0.0)
        turnover = exposure_fold.diff().abs().fillna(exposure_fold.abs())
        pnl_fold = exposure_fold * test_slice["future_return"].fillna(0.0) - turnover * cost_rate
        fold_metrics = _compute_validation_metrics(pnl_fold)
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

    summary_oos = _aggregate_oos_metrics(validation_slices)
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
    # 可视化：OOS 箱线/均值±std 图（base64）
    oos_chart = _build_oos_boxplot(validation_report["distributions"], "PFWS OOS 统计箱线图")
    if oos_chart:
        validation_report["chart"] = oos_chart

        try:
            params.entry_threshold = float(np.mean(tuned_entries)) if tuned_entries else params.entry_threshold
            params.exit_threshold = float(np.mean(tuned_exits)) if tuned_exits else params.exit_threshold
        except Exception:
            pass
    if params.label_style == "triple_barrier" and "target_multiclass" in dataset:
        try:
            mc_target = dataset["target_multiclass"].reindex(backtest.index)
            # 取概率最大类作为预测，若缺失则用信号方向
            if "prediction" in backtest:
                mc_pred = backtest["prediction"].reindex(backtest.index)
            else:
                mc_pred = np.sign(backtest["signal"]).reindex(backtest.index)
            mc_metrics = _summarize_multiclass_accuracy(mc_target, mc_pred)
            confusion = _confusion_summary(mc_target, mc_pred)
            if mc_metrics:
                stats["tb_multiclass_metrics"] = mc_metrics
            if confusion:
                stats["tb_confusion"] = confusion
            # F1/Precision/Recall（macro + per-class）
            labels = sorted(set(mc_target.dropna().unique()) | set(mc_pred.dropna().unique()))
            per_class = {}
            for c in labels:
                t_mask = mc_target == c
                p_mask = mc_pred == c
                tp = int((t_mask & p_mask).sum())
                fp = int((~t_mask & p_mask).sum())
                fn = int((t_mask & ~p_mask).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                per_class[str(int(c)) if isinstance(c, (int, float)) else str(c)] = {
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                }
            if per_class:
                macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))
                macro_prec = float(np.mean([v["precision"] for v in per_class.values()]))
                macro_rec = float(np.mean([v["recall"] for v in per_class.values()]))
                stats["tb_prf"] = {
                    "per_class": per_class,
                    "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
                }
            calib_chart = None
            if stats.get("calibration"):
                calib_chart = _build_calibration_plot(stats["calibration"])
            if calib_chart:
                stats["calibration_chart"] = calib_chart
        except Exception:
            pass
    fallback_signal = np.where(dataset["sma_short"] > dataset["sma_long"], 1.0, -1.0)
    probabilities = probabilities.fillna(0.5)
    raw_signal = raw_signal.fillna(pd.Series(fallback_signal, index=dataset.index))

    constrained_position = apply_signal_filters(dataset, raw_signal, probabilities, params)
    constrained_position = constrained_position.clip(lower=-1.0, upper=1.0)

    backtest = pd.DataFrame(index=dataset.index)
    backtest["adj close"] = dataset["adj close"]
    backtest["signal"] = raw_signal
    backtest["probability"] = probabilities
    backtest["position"] = constrained_position
    backtest = backtest.join(dataset[["sma_short", "sma_long", "rsi", "adv"]])

    asset_returns = _compute_asset_returns(backtest, params)
    backtest["asset_return"] = asset_returns
    backtest["volatility"] = asset_returns.rolling(20).std().fillna(0.0) * np.sqrt(252)

    backtest["leverage"] = calculate_target_leverage(
        backtest["position"], backtest["volatility"], params.volatility_target, params.max_leverage
    )
    exposure_series, overlay_events = enforce_risk_limits(
        backtest["position"],
        backtest["leverage"],
        asset_returns,
        params,
    )
    exposure_series, vol_events = apply_vol_targeting(exposure_series, asset_returns, params)
    overlay_events.extend(vol_events)
    if overlay_events:
        risk_events.extend(overlay_events)
    backtest["exposure"] = exposure_series
    with np.errstate(divide="ignore", invalid="ignore"):
        adj_position = backtest["exposure"] / backtest["leverage"].replace(0, np.nan)
    backtest["position"] = adj_position.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # 确保 volume 可用（用于滑点/ADV 估计）
    try:
        backtest["volume"] = prices["volume"].reindex(backtest.index).ffill().bfill()
    except Exception:
        backtest["volume"] = np.nan

    exposure_change = backtest["exposure"].diff().abs().fillna(backtest["exposure"].abs())
    cost_rate = (params.transaction_cost_bps + params.slippage_bps) / 10000.0

    # ADV 参与率约束：超过最大参与率则归零并计数
    adv_hits = 0
    adv_max_part = max(0.0, min(1.0, params.max_adv_participation or 0.1))
    if "adv" in backtest and backtest["adv"].notna().any():
        adv_limit = backtest["adv"].fillna(0.0) * adv_max_part
        mask = backtest["exposure"].abs() > adv_limit
        adv_hits = int(mask.sum())
        capped = backtest["exposure"].where(~mask, 0.0)
        if adv_hits > 0:
            backtest["exposure"] = capped
            exposure_change = backtest["exposure"].diff().abs().fillna(backtest["exposure"].abs())

    # 基础成交成本（线性 bps）
    linear_cost = exposure_change * cost_rate

    # 滑点模型
    slippage_model = params.slippage_model or {"type": "linear", "bps": params.transaction_cost_bps}
    slippage_cost = _compute_slippage_cost(
        exposure_change,
        backtest["adj close"],
        backtest.get("volume", pd.Series(0.0, index=backtest.index)),
        slippage_model,
    )

    # 融资/借券成本（按日）
    long_daily = float(params.long_borrow_cost_bps or params.borrow_cost_bps) / 10000.0 / 252.0
    short_daily = float(params.short_borrow_cost_bps or params.borrow_cost_bps) / 10000.0 / 252.0
    borrow_cost = (
        backtest["exposure"].clip(lower=0.0) * long_daily
        + (-backtest["exposure"].clip(upper=0.0)) * short_daily
    )

    backtest["transaction_cost"] = linear_cost + slippage_cost + borrow_cost
    cost_detail = {
        "linear_cost": float(linear_cost.sum()),
        "slippage_cost": float(slippage_cost.sum()),
        "borrow_cost": float(borrow_cost.sum()),
        "slippage_model": slippage_model,
        "max_adv_participation": params.max_adv_participation,
        "long_borrow_cost_bps": params.long_borrow_cost_bps or params.borrow_cost_bps,
        "short_borrow_cost_bps": params.short_borrow_cost_bps or params.borrow_cost_bps,
        "adv_rejections": adv_hits,
    }
    if adv_hits > 0:
        risk_events.append(f"ADV 参与率上限({adv_max_part:.0%}) 触发 {adv_hits} 次，超额仓位已清零。")

    backtest["strategy_return_gross"] = (
        backtest["exposure"] * asset_returns
        if params.return_path == "close_to_open"
        else backtest["exposure"].shift(fill_value=0.0) * asset_returns
    )
    backtest["strategy_return"] = backtest["strategy_return_gross"] - backtest["transaction_cost"]
    try:
        backtest["volume"] = prices["volume"].reindex(backtest.index).ffill().bfill()
    except Exception:
        backtest["volume"] = np.nan
    backtest, execution_events = apply_execution_model(backtest, prices, params)
    risk_events.extend(execution_events)

    backtest["cum_strategy"] = (1 + backtest["strategy_return"]).cumprod()
    backtest["cum_buy_hold"] = (1 + asset_returns).cumprod()
    metrics, stats = summarize_backtest(
        backtest,
        params,
        include_prediction=True,
        include_auc=True,
        feature_columns=list(feature_columns),
        shap_img=shap_img,
    )
    threshold_scan = _scan_threshold_stability(
        probabilities,
        dataset["future_return"],
        cost_rate=(params.transaction_cost_bps + params.slippage_bps) / 10000.0,
        base_entry=params.entry_threshold,
        base_exit=params.exit_threshold,
    )
    if threshold_scan:
        stats["threshold_scan"] = threshold_scan
        heatmap = _build_threshold_heatmap(threshold_scan)
        if heatmap:
            stats["threshold_scan_chart"] = heatmap
    stats["cost_assumptions"] = {
        "slippage_model": slippage_model,
        "cost_rate": cost_rate,
        "long_borrow_bps": params.long_borrow_cost_bps or params.borrow_cost_bps,
        "short_borrow_bps": params.short_borrow_cost_bps or params.borrow_cost_bps,
        "adv_participation": params.max_adv_participation,
        "execution_mode": params.execution_mode,
    }
    # 概率校准可视化
    if stats.get("calibration") and "calibration_chart" not in stats:
        try:
            chart = _build_calibration_plot(stats["calibration"])
            if chart:
                stats["calibration_chart"] = chart
        except Exception:
            pass

    direction_prediction = np.sign(backtest["signal"])
    actual_direction = dataset["return_sign"].reindex(backtest.index).fillna(0.0)
    align_mask = direction_prediction != 0
    prediction_accuracy = (
        float((direction_prediction[align_mask] == actual_direction[align_mask]).sum()) / max(1, int(align_mask.sum()))
        if align_mask.any()
        else stats.get("prediction_accuracy", float("nan"))
    )

    auc_mean = float(np.nanmean(auc_scores)) if auc_scores else stats.get("auc", float("nan"))

    stats.update(
        {
            "prediction_accuracy": prediction_accuracy,
            "auc": auc_mean,
            "auc_window_scores": auc_scores,
            "shap_img": shap_img,
            "feature_columns": list(feature_columns),
            "feature_count": len(feature_columns),
            "risk_events": risk_events,
            "execution_cost_detail": cost_detail,
            "slippage_assumptions": slippage_model,
            "seeds": collect_repro_metadata(params).get("seeds"),
            "validation_summary": stats.get("validation_summary_compact"),
            "threshold_scan_summary": stats.get("threshold_scan_summary"),
            "risk_compact": {
                "cvar_95": stats.get("cvar_95"),
                "recovery_days": stats.get("recovery_days"),
                "loss_streak": stats.get("loss_streak"),
                "white_rc_padj": stats.get("sharpe_pvalue_adjusted"),
            },
        }
    )
    if params.label_style == "triple_barrier" and "target_multiclass" in dataset:
        try:
            mc_target = dataset["target_multiclass"].reindex(backtest.index)
            mc_pred = np.sign(backtest["signal"]).reindex(backtest.index)
            mc_metrics = _summarize_multiclass_accuracy(mc_target, mc_pred)
            if mc_metrics:
                stats["tb_multiclass_metrics"] = mc_metrics
        except Exception:
            pass
    try:
        calib = _calibration_summary(probabilities, dataset["target"])
        if calib:
            stats["calibration"] = calib
    except Exception:
        pass
    if hyperopt_report:
        stats["hyperopt_report"] = hyperopt_report
    if model_warnings:
        stats["runtime_warnings"] = model_warnings
    if validation_report:
        stats["validation_report"] = validation_report
        summary = validation_report.get("summary") or {}
        # propagate full OOS summary for downstream consumers (charts/export/API)
        stats["validation_oos_summary"] = summary
        stats["validation_oos_folds"] = validation_report.get("folds")
        for key, payload in summary.items():
            if isinstance(payload, dict) and "mean" in payload:
                stats[f"validation_{key}_mean"] = payload.get("mean")
                stats[f"validation_{key}_std"] = payload.get("std")
        # propagate a compact summary for result.params/metrics consumers
        stats["validation_summary_compact"] = {
            k: {"mean": v.get("mean"), "std": v.get("std"), "iqr": v.get("iqr")}
            for k, v in summary.items()
            if isinstance(v, dict)
        }
        if validation_report.get("penalized_sharpe") is not None:
            stats["validation_penalized_sharpe"] = validation_report.get("penalized_sharpe")
        if validation_report.get("chart"):
            stats["validation_chart"] = validation_report.get("chart")
        stats["validation_train_window"] = validation_report.get("train_window")
        stats["validation_test_window"] = validation_report.get("test_window")
        stats["validation_embargo"] = validation_report.get("embargo")
    if params.label_style == "triple_barrier":
        try:
            label_counts = dataset.get("target_multiclass", dataset.get("target")).value_counts(dropna=True)
            stats["label_distribution"] = {str(int(k)) if isinstance(k, (int, float)) else str(k): int(v) for k, v in label_counts.items()}
        except Exception:
            pass
    # Attach triple-barrier dynamic thresholds summary for snapshots/exports
    if params.label_style == "triple_barrier":
        tb_summary = _tb_summary_from_dataset(dataset)
        if tb_summary:
            stats["tb_dynamic_summary"] = tb_summary
            try:
                params.tb_up_applied = tb_summary.get("up", {}).get("mean", params.tb_up)
                params.tb_down_applied = tb_summary.get("down", {}).get("mean", params.tb_down)
            except Exception:
                pass
    if threshold_scan:
        stats["threshold_scan_summary"] = {
            "best": threshold_scan.get("best"),
            "worst": threshold_scan.get("worst"),
            "mean_sharpe": threshold_scan.get("mean_sharpe"),
            "median_sharpe": threshold_scan.get("median_sharpe"),
            "iqr_sharpe": threshold_scan.get("iqr_sharpe"),
            "grid": threshold_scan.get("grid"),
        }

    metrics = build_core_metrics(stats, include_prediction=True, include_auc=True)
    try:
        append_log(
            BacktestLogEntry(
                record_id=stats.get("record_id", params.ticker + str(datetime.now())),
                timestamp=datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                ticker=params.ticker.upper(),
                engine=params.strategy_engine,
                sharpe=float(stats.get("sharpe") or 0.0),
                total_return=float(stats.get("total_return") or 0.0),
                max_drawdown=float(stats.get("max_drawdown") or 0.0),
                validation_sharpe=(stats.get("validation_report") or {}).get("mean_sharpe"),
                execution_cost=float(backtest.get("execution_cost", pd.Series(dtype=float)).sum()) if "execution_cost" in backtest else 0.0,
                notes=list(stats.get("risk_events", []))[:4],
                request_id=params.request_id,
                user_id=params.user_id,
                model_version=params.model_version or params.ml_model,
                data_version=params.data_version or os.environ.get("MARKET_DATA_VERSION"),
                latency_ms=params.exec_latency_ms,
                seeds=stats.get("seeds") if isinstance(stats.get("seeds"), dict) else None,
                versions=(collect_repro_metadata(params) or {}).get("versions"),
            )
        )
    except Exception:
        pass
    return backtest, metrics, stats


def run_rl_policy_backtest(
    prices: pd.DataFrame,
    params: StrategyInput,
    ml_context: tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]] | None = None,
    context_features: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    if ml_context is None:
        base_backtest, base_metrics, base_stats = run_ml_backtest(prices, params, context_features)
    else:
        base_backtest, base_metrics, base_stats = ml_context
    cost_rate = (params.transaction_cost_bps + params.slippage_bps) / 10000.0
    try:
        agent = build_rl_agent(getattr(params, "rl_engine", "value_iter"), cost_rate, params.rl_params or {})
    except RuntimeError as exc:
        raise QuantStrategyError(str(exc)) from exc
    if not agent.fit_from_backtest(base_backtest):
        raise QuantStrategyError("无法训练强化学习代理，可能缺少概率信号或样本不足。")
    raw_signal = agent.signal_series(base_backtest)
    probs = base_backtest.get("probability", pd.Series(0.5, index=base_backtest.index)).fillna(0.5)
    position = enforce_min_holding(raw_signal, params.min_holding_days, probs)

    rl_backtest = base_backtest.copy()
    rl_backtest["signal"] = raw_signal
    rl_backtest["position"] = position
    asset_returns = rl_backtest["asset_return"].fillna(0.0)
    rl_backtest["volatility"] = asset_returns.rolling(20).std().fillna(0.0) * np.sqrt(252)
    rl_backtest["leverage"] = calculate_target_leverage(
        rl_backtest["position"], rl_backtest["volatility"], params.volatility_target, params.max_leverage
    )
    exposure_series, overlay_events = enforce_risk_limits(
        rl_backtest["position"],
        rl_backtest["leverage"],
        asset_returns,
        params,
    )
    rl_backtest["exposure"] = exposure_series
    exposure_change = exposure_series.diff().abs().fillna(exposure_series.abs())

    # ADV 参与率约束（与 ML/传统回测保持一致）
    adv_hits = 0
    adv_max_part = max(0.0, min(1.0, params.max_adv_participation or 0.1))
    if "adv" in rl_backtest and rl_backtest["adv"].notna().any():
        adv_limit = rl_backtest["adv"].fillna(0.0) * adv_max_part
        mask = rl_backtest["exposure"].abs() > adv_limit
        adv_hits = int(mask.sum())
        capped = rl_backtest["exposure"].where(~mask, 0.0)
        if adv_hits > 0:
            rl_backtest["exposure"] = capped
            exposure_change = rl_backtest["exposure"].diff().abs().fillna(rl_backtest["exposure"].abs())

    rl_backtest["transaction_cost"] = exposure_change * cost_rate
    rl_backtest["strategy_return_gross"] = (
        rl_backtest["exposure"] * asset_returns
        if params.return_path in {"close_to_open", "open_to_close"}
        else rl_backtest["exposure"].shift(fill_value=0.0) * asset_returns
    )
    long_daily = float(params.long_borrow_cost_bps or params.borrow_cost_bps) / 10000.0 / 252.0
    short_daily = float(params.short_borrow_cost_bps or params.borrow_cost_bps) / 10000.0 / 252.0
    borrow_cost = (
        rl_backtest["exposure"].clip(lower=0.0) * long_daily
        + (-rl_backtest["exposure"].clip(upper=0.0)) * short_daily
    )
    rl_backtest["strategy_return"] = rl_backtest["strategy_return_gross"] - rl_backtest["transaction_cost"] - borrow_cost
    try:
        rl_backtest["volume"] = prices["volume"].reindex(rl_backtest.index).ffill().bfill()
    except Exception:
        rl_backtest["volume"] = np.nan
    rl_backtest, rl_execution_events = apply_execution_model(rl_backtest, prices, params)
    rl_backtest["cum_strategy"] = (1 + rl_backtest["strategy_return"]).cumprod()
    rl_backtest["cum_buy_hold"] = (1 + rl_backtest["asset_return"]).cumprod()

    metrics, stats = summarize_backtest(
        rl_backtest,
        params,
        include_prediction=True,
        include_auc=True,
        feature_columns=base_stats.get("feature_columns", []),
        shap_img=base_stats.get("shap_img"),
    )
    stats["rl_playbook"] = agent.playbook
    events = list(stats.get("risk_events", []))
    events.extend(overlay_events)
    if adv_hits > 0:
        events.append(f"RL 回测：ADV 参与率上限({adv_max_part:.0%}) 触发 {adv_hits} 次，超额仓位已清零。")
    events.extend(rl_execution_events)
    stats["risk_events"] = events
    runtime_notes = list(stats.get("runtime_warnings", []))
    runtime_notes.append(f"强化学习策略采用 {params.rl_engine or 'value_iter'} 代理。")
    stats["runtime_warnings"] = runtime_notes
    stats["base_engine"] = base_stats.get("source_engine", "ml_momentum")
    stats["cost_assumptions"] = {
        "slippage_model": params.slippage_model,
        "cost_rate": cost_rate,
        "long_borrow_bps": params.long_borrow_cost_bps or params.borrow_cost_bps,
        "short_borrow_bps": params.short_borrow_cost_bps or params.borrow_cost_bps,
        "adv_participation": params.max_adv_participation,
        "execution_mode": params.execution_mode,
    }
    oos_report = _compute_oos_from_backtest(rl_backtest, params)
    if not oos_report:
        # 回退：使用整段序列作为简易 OOS 统计，确保字段存在
        vm = _compute_validation_metrics(rl_backtest["strategy_return"].fillna(0.0))
        oos_report = {
            "slices": [
                {
                    **vm,
                    "fold": 1,
                    "test_start": str(rl_backtest.index[0].date()) if len(rl_backtest.index) else "",
                    "test_end": str(rl_backtest.index[-1].date()) if len(rl_backtest.index) else "",
                }
            ],
            "summary": _aggregate_oos_metrics([vm]),
            "folds": 1,
            "penalized_sharpe": vm.get("sharpe", 0.0) - 0.0,
            "train_window": params.train_window,
            "test_window": params.test_window,
            "embargo": params.embargo_days,
            "distributions": {
                k: [float(vm.get(k, 0.0))] for k in ("sharpe", "cagr", "max_drawdown", "hit_ratio")
            },
        }
    if oos_report:
        stats["validation_report_detected"] = "rl_pfws"
        stats["validation_oos_summary"] = oos_report.get("summary")
        stats["validation_oos_folds"] = oos_report.get("folds")
        stats["validation_penalized_sharpe"] = oos_report.get("penalized_sharpe")
        stats["validation_train_window"] = oos_report.get("train_window")
        stats["validation_test_window"] = oos_report.get("test_window")
        stats["validation_embargo"] = oos_report.get("embargo")
    else:
        events.append("提示：RL 策略未能生成 PFWS 样本外指标，当前为全量回测。")
        stats["risk_events"] = events
    metrics = build_core_metrics(stats, include_prediction=True, include_auc=True)
    return rl_backtest, metrics, stats


def generate_recommendations(
    stats: dict[str, Any],
    benchmark_stats: Optional[dict[str, Any]],
    params: StrategyInput,
    market_context: dict[str, Any],
) -> list[dict[str, Any]]:
    risk = (params.risk_profile or "balanced").lower()
    risk_label = RISK_PROFILE_LABELS.get(risk, risk)
    base_allocations = {
        "conservative": 0.4,
        "balanced": 0.6,
        "aggressive": 0.75,
    }
    core_weight = base_allocations.get(risk, 0.6)
    hedge_weight = max(0.0, 1 - core_weight)
    overlay_weight = 0.0

    sharpe = stats.get("sharpe", 0.0) or 0.0
    cagr = stats.get("cagr", 0.0) or 0.0
    max_drawdown = stats.get("max_drawdown", 0.0) or 0.0
    volatility = stats.get("volatility", 0.0) or 0.0
    annual_factor = stats.get("annual_factor", 252)
    trading_days = max(1, stats.get("trading_days", 1))
    capital = params.capital or 0.0

    benchmark_total_return = 0.0
    benchmark_vol = 0.0
    benchmark_correlation = 0.0
    benchmark_mu = 0.0

    if benchmark_stats:
        benchmark_total_return = benchmark_stats.get("total_return", 0.0) or 0.0
        benchmark_vol = benchmark_stats.get("volatility", 0.0) or 0.0
        benchmark_correlation = benchmark_stats.get("correlation", 0.0) or 0.0
        benchmark_mu = (1 + benchmark_total_return) ** (annual_factor / trading_days) - 1 if trading_days else 0.0

    def fmt_pct(value: float) -> str:
        return format_percentage(value)

    def breakdown(weights: list[tuple[str, float]]) -> list[dict[str, str]]:
        breakdown_rows: list[dict[str, str]] = []
        for label, pct in weights:
            amount = capital * pct if capital else 0.0
            breakdown_rows.append(
                {
                    "label": label,
                    "percent": fmt_pct(pct),
                    "amount": format_currency(amount) if capital else "—",
                }
            )
        return breakdown_rows

    def project_return(weight_strategy: float, weight_benchmark: float) -> tuple[str, float]:
        strategy_mu = cagr
        strategy_vol = volatility
        combined_mu = weight_strategy * strategy_mu + weight_benchmark * benchmark_mu
        combined_var = (weight_strategy * strategy_vol) ** 2 + (weight_benchmark * benchmark_vol) ** 2
        combined_var += 2 * weight_strategy * weight_benchmark * strategy_vol * benchmark_vol * benchmark_correlation
        combined_var = max(combined_var, 0.0)
        combined_vol = math.sqrt(combined_var)
        if combined_vol == 0:
            success_prob = 1.0 if combined_mu > 0 else 0.0
        else:
            success_prob = normal_cdf(combined_mu / combined_vol)
        expected_final = capital * (1 + combined_mu)
        projection = (
            f"投入 {format_currency(capital)} 约一年后期望价值 {format_currency(expected_final)}"
            if capital
            else "未提供资金规模，建议输入可支配资金以估算收益。"
        )
        if capital:
            projection += f"，获得正收益的概率约 {success_prob * 100:.1f}%。"
        return projection, success_prob

    def build_timeline(plan_name: str, success_prob: Optional[float]) -> list[dict[str, str]]:
        best_prob = f"{(success_prob or 0) * 100:.1f}%" if success_prob is not None else "—"
        base_prob = f"{format_percentage(cagr)} / {format_percentage(volatility)}"
        return [
            {
                "phase": "起步（0-1个月）",
                "base": f"确认信号并分批建仓，执行预设风控：滚动回撤 {format_percentage(stats.get('avg_exposure', 0.0) or 0.0)} 以内保持仓位。",
                "best": f"若信号强劲、成交量放大，可加速完成建仓；预计正收益概率 {best_prob}。",
                "worst": "若建仓阶段市场高波动，建议降低仓位 10%-20%，并辅以指数或期权对冲。",
            },
            {
                "phase": "持有管理（1-6个月）",
                "base": f"按月再平衡，保持策略/防御仓位结构；关注夏普 {stats.get('sharpe', 0.0):.2f} 与回撤走势。",
                "best": "若策略净值新高且回撤受控，可将盈利部分滚入同主题增强资产或分红。",
                "worst": f"若出现连续三周胜率 <40% 或回撤触发 {format_percentage(stats.get('var_95', 0.0) or 0.0)} VaR，执行止损并回归现金。",
            },
            {
                "phase": "评估与再配置（6-12个月）",
                "base": f"滚动检视收益/波动比（当前 {base_prob}），根据宏观与行业信号决定是否延续策略。",
                "best": "若策略跑赢基准 α 保持正数，可逐步提高目标仓位或扩充至相关行业篮子。",
                "worst": "若趋势逆转或基本面恶化，建议转换到低波策略/债券资产，保留核心盈利。",
            },
        ]

    recommendations: list[dict[str, Any]] = []

    projection, base_success_prob = project_return(core_weight, 0.0 if not benchmark_stats else hedge_weight * 0.0)
    recommendations.append(
        {
            "title": "核心动量组合",
            "subtitle": f"以策略信号为主资产，适用于{risk_label}投资者维护收益与波动的平衡。",
            "allocation": [
                f"{fmt_pct(core_weight)} 配置本策略（可通过 ETF/股票篮子复制）",
                f"{fmt_pct(hedge_weight)} 留作现金或货币基金以缓冲波动" if hedge_weight > 0 else "如风险承受度允许，可短期保持满仓策略",
            ],
            "breakdown": breakdown(
                [
                    (f"{params.ticker.upper()} 策略组合", core_weight),
                    ("现金 / 货币基金", hedge_weight),
                ]
            ),
            "projection": projection,
            "actions": textwrap.dedent(
                f"""
                当前夏普比率 {sharpe:.2f}，单位风险补偿依旧健康；最大回撤 {fmt_pct(max_drawdown)}，建议设置
                20% 的动态仓位缓冲（回撤触发时逐步减仓）。可采用“月度再平衡 + 10% 回撤止损”框架来维持纪律。
                """
            ).strip(),
            "success_probability": base_success_prob * 100 if base_success_prob is not None else None,
            "timeline": build_timeline("核心动量组合", base_success_prob),
        }
    )

    if benchmark_stats:
        hedge_allocation = fmt_pct(hedge_weight) if hedge_weight > 0 else "适量"
        projection, success_prob = project_return(core_weight, hedge_weight)
        recommendations.append(
            {
                "title": "防御性对冲组合",
                "subtitle": f"将策略与基准 {params.benchmark_ticker.upper()} 组合，削弱单一资产风险。",
                "allocation": [
                    f"{fmt_pct(core_weight)} 策略多元资产/动量篮子",
                    f"{hedge_allocation} {params.benchmark_ticker.upper()} 指数基金或防御资产",
                ],
                "breakdown": breakdown(
                    [
                        (f"{params.ticker.upper()} 策略组合", core_weight),
                        (f"{params.benchmark_ticker.upper()} 指数/ETF", hedge_weight),
                    ]
                ),
                "projection": projection,
                "actions": textwrap.dedent(
                    f"""
                    基准波动率 {fmt_pct(benchmark_vol)}，与策略的相关系数 {benchmark_correlation:.2f}。
                    建议使用 60-90 天滚动 β 做风险预算，β>1.2 时提高指数/国债权重；β<0.8 时可增配策略。
                    当前组合预估正收益概率约 {success_prob * 100:.1f}% ，相较单一资产更平滑。
                    """
                ).strip(),
                "success_probability": success_prob * 100 if success_prob is not None else None,
                "timeline": build_timeline("防御性对冲组合", success_prob),
            }
        )
    else:
        projection, success_prob = project_return(core_weight, 0.0)
        recommendations.append(
            {
                "title": "防御性现金垫层",
                "subtitle": "无可用基准时，以现金/短久期债券为缓冲区。",
                "allocation": [
                    f"{fmt_pct(core_weight)} 策略组合",
                    f"{fmt_pct(hedge_weight)} 现金、短久期债券或货币基金",
                ],
                "breakdown": breakdown(
                    [
                        (f"{params.ticker.upper()} 策略组合", core_weight),
                        ("现金 / 短久期债券", hedge_weight),
                    ]
                ),
                "projection": projection,
                "actions": textwrap.dedent(
                    f"""
                    建议保留至少 {fmt_pct(hedge_weight)} 的随时可用资金。当策略回撤超过 {fmt_pct(volatility)} 时，
                    采用分批补仓方式（例如每下降 5% 加仓 10%）提升均值回归收益，同时保留止损纪律。
                    """
                ).strip(),
                "success_probability": success_prob * 100 if success_prob is not None else None,
                "timeline": build_timeline("防御性现金垫层", success_prob),
            }
        )

    if risk == "aggressive" and sharpe > 1:
        overlay_weight = min(0.3, max(0.0, cagr * 1.5))
        effective_weight = min(0.9, core_weight + overlay_weight)
        projection, success_prob = project_return(effective_weight, hedge_weight)
        recommendations.append(
            {
                "title": "进取型增强策略",
                "subtitle": "适合具备衍生品经验的投资者，在波动可控时提升收益目标。",
                "allocation": [
                    f"{fmt_pct(effective_weight)} 策略组合或杠杆 ETF",
                    f"{fmt_pct(max(0.0, 1 - effective_weight))} 现金/保证金仓位",
                    "信号强劲时可使用股指期货、牛市价差或看涨期权做 1.2x~1.4x 杠杆",
                ],
                "breakdown": breakdown(
                    [
                        (f"{params.ticker.upper()} 策略组合/杠杆 ETF", effective_weight),
                        ("现金 / 保证金仓位", max(0.0, 1 - effective_weight)),
                    ]
                ),
                "projection": projection,
                "actions": textwrap.dedent(
                    f"""
                    建议设定两条风控线：一是策略滚动年化波动率超过 25% 或 VaR 超过 {fmt_pct(-stats.get('var_95', 0.0) or 0.0)} 时降杠杆；
                    二是 RSI 连续高于 70 且价量背离时锁定利润，将仓位回落至基础权重。
                    预估正收益概率约 {success_prob * 100:.1f}%。
                    """
                ).strip(),
                "success_probability": success_prob * 100 if success_prob is not None else None,
                "timeline": build_timeline("进取型增强策略", success_prob),
            }
        )
    else:
        projection, success_prob = project_return(core_weight * 0.8, hedge_weight)
        recommendations.append(
            {
                "title": "再平衡与风险预算",
                "subtitle": "通过趋势过滤与资产再分配稳健提升风险收益比。",
                "allocation": [
                    "核心策略：信号向上时保持 80% 仓位，信号疲弱或 RSI>70 时减至 40%-50%",
                    "防御资产：股指回撤或宏观数据恶化时，增配国债/黄金 ETF 各 10%",
                ],
                "breakdown": breakdown(
                    [
                        (f"{params.ticker.upper()} 策略组合（动态）", core_weight * 0.8),
                        ("现金 / 防御资产", max(0.0, 1 - core_weight * 0.8)),
                    ]
                ),
                "projection": projection,
                "actions": textwrap.dedent(
                    f"""
                    建议“日内不操作、周度检查、月度再平衡”，并设定胜率阈值（连续三周 <40% 暂停交易）。
                    当前方案预估正收益概率约 {success_prob * 100:.1f}% ，适合希望稳健增值的{risk_label}投资者。
                    """
                ).strip(),
                "success_probability": success_prob * 100 if success_prob is not None else None,
                "timeline": build_timeline("再平衡与风险预算", success_prob),
            }
        )

    return recommendations


def build_related_portfolios(
    params: StrategyInput, market_context: dict[str, Any], capital: float
) -> list[dict[str, Any]]:
    primary = params.ticker.upper()
    tickers = market_context.get("tickers", []) if market_context else []
    top_related = tickers[:4]
    related: list[dict[str, Any]] = []

    if top_related:
        allocation = 1 / len(top_related) if top_related else 0
        related.append(
            {
                "title": "行业主题增强组合",
                "description": (
                    f"根据最新资讯筛选出与 {primary} 同行业或供应链关系紧密的标的，"
                    "采用等权配置获取主题溢价，同时在组合层面分散单一公司的特有风险。"
                ),
                "tickers": top_related,
                "breakdown": [
                    {
                        "label": symbol,
                        "percent": format_percentage(allocation),
                        "amount": format_currency(capital * allocation) if capital else "—",
                    }
                    for symbol in top_related
                ],
                "rationale": "适合卫星仓位，观察行业共振趋势时逐步放大配置。",
            }
        )

    defensive_bundle = ["GLD", "TLT", "BIL"]
    related.append(
        {
            "title": "稳健对冲组合",
            "description": (
                "通过贵金属（GLD）、长期国债（TLT）与短期现金替代（BIL）构建的低相关性篮子，"
                "在系统性风险上升时可缓冲权益投资的波动，并提供流动性来源。"
            ),
            "tickers": defensive_bundle,
            "breakdown": [
                {
                    "label": "GLD 黄金 ETF",
                    "percent": "30.00%",
                    "amount": format_currency(capital * 0.3) if capital else "—",
                },
                {
                    "label": "TLT 20年期美债 ETF",
                    "percent": "40.00%",
                    "amount": format_currency(capital * 0.4) if capital else "—",
                },
                {
                    "label": "BIL 超短期国库券 ETF",
                    "percent": "30.00%",
                    "amount": format_currency(capital * 0.3) if capital else "—",
                },
            ],
            "rationale": "用于在市场突发回撤时快速对冲或作为现金管理池。",
        }
    )

    global_bundle = ["ACWI", "QQQ", "SPY"]
    related.append(
        {
            "title": "全球资产配置组合",
            "description": (
                "聚焦全球多元资产：ACWI 提供全球宽基敞口，QQQ 捕捉科技成长，SPY 代表美股核心仓位，"
                "适合作为策略之外的长期资产基石。"
            ),
            "tickers": global_bundle,
            "breakdown": [
                {
                    "label": "ACWI 全球宽基",
                    "percent": "40.00%",
                    "amount": format_currency(capital * 0.4) if capital else "—",
                },
                {
                    "label": "QQQ 纳斯达克100",
                    "percent": "30.00%",
                    "amount": format_currency(capital * 0.3) if capital else "—",
                },
                {
                    "label": "SPY 标普500",
                    "percent": "30.00%",
                    "amount": format_currency(capital * 0.3) if capital else "—",
                },
            ],
            "rationale": "用于建立核心长期仓位，与策略组合形成‘核心+卫星’配置框架。",
        }
    )

    return related


def build_statistical_baselines(prices: pd.DataFrame, params: StrategyInput) -> dict[str, Any]:
    """Generate ARIMA/VAR baselines to benchmark ML策略."""
    closes = prices.get("adj close", pd.Series(dtype=float)).dropna()
    volumes = prices.get("volume", pd.Series(dtype=float)).dropna()
    baseline: dict[str, Any] = {"arima": None, "var": None, "diagnostics": []}

    if ARIMA is not None and closes.shape[0] >= 60:
        try:
            ideal_order = (1, 1, 1)
            model = ARIMA(closes, order=ideal_order)
            res = model.fit()
            forecast = res.forecast(steps=5)
            conf_int = res.get_forecast(steps=5).conf_int(alpha=0.1)
            baseline["arima"] = {
                "order": ideal_order,
                "aic": float(res.aic),
                "bic": float(res.bic),
                "forecast": [float(v) for v in forecast],
                "conf_int": conf_int.round(4).values.tolist(),
                "summary": f"ARIMA{ideal_order} AIC={res.aic:.1f}",
            }
        except Exception as exc:  # pragma: no cover - statsmodels optional
            baseline["diagnostics"].append(f"ARIMA 预测失败：{exc}")
    elif ARIMA is None:
        baseline["diagnostics"].append("缺少 statsmodels，无法生成 ARIMA 基线。")
    else:
        baseline["diagnostics"].append("数据量不足，ARIMA 需要至少 60 条有效价格。")

    if VAR is not None and closes.shape[0] >= 80 and volumes.shape[0] >= 80:
        try:
            aligned = pd.concat(
                [closes.pct_change().dropna().rename("return"), volumes.pct_change().dropna().rename("volume")],
                axis=1,
            ).dropna()
            model = VAR(aligned)
            res = model.fit(maxlags=4, ic="aic")
            forecast = res.forecast(aligned.values[-res.k_ar :], steps=5)
            baseline["var"] = {
                "lags": res.k_ar,
                "aic": float(res.aic),
                "forecast": forecast.tolist(),
                "summary": f"VAR(lag={res.k_ar}) 捕捉价格-量的互动",
            }
        except Exception as exc:  # pragma: no cover
            baseline["diagnostics"].append(f"VAR 拟合失败：{exc}")
    elif VAR is None:
        baseline["diagnostics"].append("缺少 statsmodels，无法生成 VAR 基线。")
    else:
        baseline["diagnostics"].append("数据量不足，VAR 需要至少 80 条有效样本。")

    return baseline


def _safe_get(info: dict[str, Any], *keys: str) -> Any:
    value: Any = info
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def build_factor_snapshot(params: StrategyInput) -> dict[str, Any]:
    fundamentals: dict[str, Any] = {}
    try:
        ticker = yf.Ticker(params.ticker)
        fundamentals = ticker.info or {}
    except Exception:
        fundamentals = {}

    factor_rows: list[dict[str, Any]] = []
    mappings = [
        ("市盈率 (PE)", ("trailingPE",)),
        ("预期PE", ("forwardPE",)),
        ("市净率 (PB)", ("priceToBook",)),
        ("净利率", ("profitMargins",)),
        ("ROE", ("returnOnEquity",)),
        ("收入同比增长", ("revenueGrowth",)),
        ("自由现金流/收入", ("freeCashflow",)),
        ("Beta", ("beta",)),
    ]
    for label, path in mappings:
        value = _safe_get(fundamentals, *path)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            if abs(value) < 1:
                formatted = f"{value*100:.2f}%"
            else:
                formatted = f"{value:.2f}"
        else:
            formatted = str(value)
        factor_rows.append({"label": label, "value": formatted})

    sector = fundamentals.get("sector") or ""
    industry = fundamentals.get("industry") or ""
    cap = fundamentals.get("marketCap")
    cap_display = f"{cap/1e9:.2f}B" if isinstance(cap, (int, float)) else "—"

    return {
        "meta": {
            "sector": sector,
            "industry": industry,
            "market_cap": cap_display,
            "summary": fundamentals.get("longBusinessSummary") or "",
        },
        "factors": factor_rows,
    }


def build_sentiment_snapshot(market_context: dict[str, Any]) -> dict[str, Any]:
    analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
    sentiments: list[dict[str, Any]] = []
    aggregate = 0.0
    if analyzer and market_context and market_context.get("news"):
        for item in market_context["news"]:
            title = item.get("title") or ""
            snippet = item.get("snippet") or ""
            joined = f"{title} {snippet}".strip()
            if not joined:
                continue
            score = analyzer.polarity_scores(joined)["compound"]
            aggregate += score
            sentiments.append(
                {
                    "title": title,
                    "score": score,
                    "label": "利好" if score > 0.1 else "利空" if score < -0.1 else "中性",
                    "url": item.get("url"),
                }
            )
    reason = ""
    avg_score = 0.0
    if sentiments:
        avg_score = aggregate / len(sentiments)
        if avg_score > 0.15:
            reason = "整体情绪偏正面，短期动能有望延续。"
        elif avg_score < -0.15:
            reason = "新闻舆情偏负面，需关注回撤风险。"
        else:
            reason = "情绪中性，可重点观察基本面与技术信号。"
    elif analyzer is None:
        reason = "缺少 vaderSentiment，推荐 pip install vaderSentiment 以启用情绪分析。"
    else:
        reason = "近期无足够新闻供情绪分析参考。"

    return {
        "available": bool(sentiments),
        "average": avg_score,
        "insight": reason,
        "items": sentiments[:6],
    }


def build_multimodal_bundle(
    params: StrategyInput,
    feature_dataset: pd.DataFrame | None,
    market_context: dict[str, Any],
    fundamentals_override: dict[str, Any] | None = None,
    macro_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = None
    if fundamentals_override:
        market_cap = fundamentals_override.get("marketCap")
        if isinstance(market_cap, (int, float)):
            cap_display = f"{market_cap/1e9:.2f}B"
        else:
            cap_display = market_cap
        meta = {
            "sector": fundamentals_override.get("sector"),
            "industry": fundamentals_override.get("industry"),
            "market_cap": cap_display or "—",
            "summary": fundamentals_override.get("summary", ""),
        }
        factor_map = {
            "利润率": fundamentals_override.get("profitMargins"),
            "营业利润率": fundamentals_override.get("operatingMargins"),
            "净资产收益率": fundamentals_override.get("returnOnEquity"),
            "资产回报率": fundamentals_override.get("returnOnAssets"),
            "收入增长": fundamentals_override.get("revenueGrowth"),
            "季度盈利增长": fundamentals_override.get("earningsQuarterlyGrowth"),
        }
        factors = [
            {"label": label, "value": f"{value*100:.2f}%"} for label, value in factor_map.items() if isinstance(value, (int, float))
        ]
        snapshot = {"meta": meta, "factors": factors}

    if not snapshot:
        snapshot = build_factor_snapshot(params)
    sentiment = build_sentiment_snapshot(market_context)

    technical_momentums: dict[str, Any] | None = None
    if feature_dataset is not None and not feature_dataset.empty:
        latest = feature_dataset.iloc[-1]
        technical_momentums = {
            "short_return": format_percentage(float(latest.get("return_5d", np.nan))),
            "mid_return": format_percentage(float(latest.get("return_21d", np.nan))),
            "momentum_short": float(latest.get("momentum_short", 0.0)),
            "momentum_long": float(latest.get("momentum_long", 0.0)),
            "rsi": float(latest.get("rsi", np.nan)),
        }

    return {
        "fundamentals": snapshot,
        "sentiment": sentiment,
        "technicals": technical_momentums,
        "macro": macro_bundle,
    }


def run_deep_signal_model(dataset: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any] | None:
    if dataset is None or dataset.empty or len(feature_columns) < 3:
        return None

    df = dataset.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns + ["target", "future_return"])
    if df.shape[0] < 160:
        return None

    values = df[feature_columns].astype(float).values
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0) + 1e-12
    normalized = (values - mean) / std

    y = df["target"].astype(float).values
    window = min(48, max(12, len(feature_columns)))
    sequences: list[np.ndarray] = []
    labels: list[float] = []
    for idx in range(window, len(normalized)):
        seq = normalized[idx - window : idx]
        if np.isnan(seq).any():
            continue
        sequences.append(seq)
        labels.append(y[idx])

    if len(sequences) < 150:
        return None

    X_np = np.stack(sequences)
    y_np = np.array(labels)
    split = int(len(X_np) * 0.8)
    if split <= 0 or split >= len(X_np) - 10:
        return None

    X_train = X_np[:split]
    y_train = y_np[:split]
    X_test = X_np[split:]
    y_test = y_np[split:]

    reports: list[dict[str, Any]] = []

    if torch is not None and nn is not None:
        device = torch.device("cpu")

        def train_sequence_model(name: str, model: nn.Module, epochs: int = 20, lr: float = 0.003) -> dict[str, Any] | None:
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
            y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
            X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
            try:
                for epoch in range(epochs):
                    perm = torch.randperm(X_train_t.size(0))
                    X_train_epoch = X_train_t[perm]
                    y_train_epoch = y_train_t[perm]
                    for start in range(0, X_train_epoch.size(0), 32):
                        end = start + 32
                        xb = X_train_epoch[start:end]
                        yb = y_train_epoch[start:end]
                        optimizer.zero_grad()
                        pred = model(xb)
                        loss = criterion(pred, yb)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
                        optimizer.step()
                model.eval()
                with torch.no_grad():
                    probs = model(X_test_t).cpu().numpy().flatten()
                    preds = (probs >= 0.5).astype(int)
                    accuracy = float((preds == y_test.astype(int)).mean())
                    confidence = float(probs[-1]) if len(probs) else None
                return {
                    "name": name,
                    "accuracy": accuracy,
                    "confidence": confidence,
                    "sample": int(len(X_test)),
                }
            except Exception:
                return None

        class LSTMHead(nn.Module):
            def __init__(self, input_size: int):
                super().__init__()
                hidden = min(96, max(24, input_size * 2))
                self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden // 2, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                logits = self.head(last)
                return torch.sigmoid(logits)

        class TransformerHead(nn.Module):
            def __init__(self, input_size: int, seq_len: int):
                super().__init__()
                d_model = min(96, max(32, input_size * 2))
                self.seq_len = seq_len
                self.input_proj = nn.Linear(input_size, d_model)
                self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
                nhead = max(1, d_model // 16)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=0.2,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model // 2, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                seq_len = x.size(1)
                if seq_len < self.seq_len:
                    pad = self.seq_len - seq_len
                    x = torch.nn.functional.pad(x, (0, 0, pad, 0))
                    seq_len = self.seq_len
                elif seq_len > self.seq_len:
                    x = x[:, -self.seq_len :, :]
                    seq_len = self.seq_len
                h = self.input_proj(x) + self.pos_embed[:, :seq_len, :]
                enc = self.encoder(h)
                pooled = enc.mean(dim=1)
                logits = self.head(pooled)
                return torch.sigmoid(logits)

        lstm_model = LSTMHead(len(feature_columns)).to(device)
        lstm_report = train_sequence_model(
            name=f"LSTM (window={window})",
            model=lstm_model,
            epochs=22,
            lr=0.003,
        )
        if lstm_report:
            reports.append(lstm_report)

        transformer_model = TransformerHead(len(feature_columns), seq_len=window).to(device)
        transformer_report = train_sequence_model(
            name=f"Transformer (window={window})",
            model=transformer_model,
            epochs=18,
            lr=0.0025,
        )
        if transformer_report:
            reports.append(transformer_report)

    if MLPClassifier is not None:
        split_idx = int(len(df) * 0.8)
        if 0 < split_idx < len(df) - 10:
            X_train_df = df[feature_columns].iloc[:split_idx]
            X_test_df = df[feature_columns].iloc[split_idx:]
            y_train_df = df["target"].iloc[:split_idx]
            y_test_df = df["target"].iloc[split_idx:]

            scaler = StandardScaler() if StandardScaler else None
            if scaler is not None:
                X_train_df = pd.DataFrame(
                    scaler.fit_transform(X_train_df),
                    index=X_train_df.index,
                    columns=X_train_df.columns,
                )
                X_test_df = pd.DataFrame(
                    scaler.transform(X_test_df),
                    index=X_test_df.index,
                    columns=X_test_df.columns,
                )

            try:
                clf = MLPClassifier(
                    hidden_layer_sizes=(128, 64, 16),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.001,
                    max_iter=400,
                    random_state=42,
                )
                clf.fit(X_train_df, y_train_df)
                score = float(clf.score(X_test_df, y_test_df))
                proba = clf.predict_proba(X_test_df)[:, 1] if hasattr(clf, "predict_proba") else None
                confidence = float(proba[-1]) if proba is not None and len(proba) else None
                reports.append(
                    {
                        "name": "MLP Baseline",
                        "accuracy": score,
                        "confidence": confidence,
                        "sample": len(X_test_df),
                    }
                )
            except Exception:
                pass

    if not reports:
        return None

    reports_sorted = sorted(reports, key=lambda item: item.get("accuracy", 0.0), reverse=True)
    best = reports_sorted[0]
    return {
        "accuracy": best.get("accuracy"),
        "confidence": best.get("confidence"),
        "sample": best.get("sample"),
        "best_model": best.get("name"),
        "models": reports_sorted,
    }


def build_model_ensemble_view(
    statistical_bundle: dict[str, Any] | None,
    ml_stats: dict[str, Any],
    deep_bundle: dict[str, Any] | None,
    graph_bundle: dict[str, Any] | None = None,
    factor_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: list[str] = []
    confidence_score = 0.0
    anchors = []

    if statistical_bundle and statistical_bundle.get("arima"):
        stats = statistical_bundle["arima"]
        forecast_value = stats.get("forecast", [0])[0]
        summary.append(f"ARIMA 基线预测 5 日变动 {forecast_value:.2f}，用于短线对照。")
        confidence_score += 0.18
        anchors.append("统计基线")
    if deep_bundle:
        acc = deep_bundle.get("accuracy")
        note = deep_bundle.get("best_model", "深度信号")
        if acc is not None:
            summary.append(f"{note} 准确率 {acc*100:.1f}% ，样本 {deep_bundle.get('sample')}。")
            confidence_score += min(0.3, max(0.12, acc))
        else:
            summary.append(note)
            confidence_score += 0.12
        anchors.append("深度信号")
        model_details = deep_bundle.get("models")
        if model_details and len(model_details) > 1:
            best_second = model_details[1]
            summary.append(
                f"备选模型 {best_second['name']} 准确率 {best_second['accuracy']*100:.1f}%。"
            )
    if ml_stats:
        sharpe = ml_stats.get("sharpe", 0.0) or 0.0
        win_rate = format_percentage(ml_stats.get("hit_ratio", 0.0) or 0.0)
        summary.append(f"机器学习引擎夏普 {sharpe:.2f}，胜率 {win_rate}。")
        confidence_score += min(0.32, max(0.1, sharpe / 3))
        anchors.append("机器学习")

    if graph_bundle and graph_bundle.get("available"):
        density = graph_bundle["stats"].get("density", 0.0)
        risk_score = graph_bundle.get("risk_score")
        summary.append(f"图谱密度 {density:.3f}，网络风险评分 {risk_score:.2f}。")
        confidence_score += min(0.12, max(-0.1, 0.15 - density))
        anchors.append("图谱")

    if factor_bundle and factor_bundle.get("available"):
        composite = factor_bundle.get("composite", 0.0)
        summary.append(f"多因子综合 Z 分 {composite:.2f}，Top 信号：{', '.join(item['name'] for item in factor_bundle.get('top_factors', [])[:3])}.")
        confidence_score += min(0.18, max(0.0, abs(composite)))
        anchors.append("因子")

    blended_comment = "、".join(anchors) + " 多模型组合" if anchors else "组合策略"
    recommendation = (
        "多模型观点一致，可维持当前仓位并关注风险阈值。"
        if confidence_score >= 0.65
        else "模型观点分歧，建议降低仓位或等待信号共识。"
    )
    return {
        "summary": summary,
        "confidence": round(confidence_score, 2),
        "recommendation": recommendation,
        "title": blended_comment,
    }


def analyze_factor_effectiveness(
    dataset: pd.DataFrame | None,
    feature_columns: list[str] | None,
) -> dict[str, Any]:
    if dataset is None or dataset.empty or not feature_columns:
        return {"available": False, "message": "缺少特征矩阵，无法评估因子表现。"}

    df = dataset.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns + ["future_return"])
    if df.shape[0] < 120:
        return {"available": False, "message": "样本不足，无法计算因子统计。"}

    future_return = df["future_return"].astype(float)
    if future_return.std() == 0:
        return {"available": False, "message": "收益序列缺乏波动，无法计算因子表现。"}

    factors: list[dict[str, Any]] = []
    for col in feature_columns:
        series = df[col].astype(float)
        if series.nunique() < 10 or series.std() == 0:
            continue
        z = (series - series.mean()) / (series.std() + 1e-12)
        ic = float(np.corrcoef(z, future_return)[0, 1])
        ranks = z.rank(pct=True)
        high = future_return[ranks >= 0.8]
        low = future_return[ranks <= 0.2]
        long_short = float((high.mean() - low.mean()) * 100) if not high.empty and not low.empty else 0.0
        factors.append(
            {
                "name": col,
                "ic": round(ic, 3),
                "long_short": round(long_short, 3),
                "strength": round(abs(ic) + abs(long_short) / 100, 3),
            }
        )

    if not factors:
        return {"available": False, "message": "无有效因子信号。"}

    factors_sorted = sorted(factors, key=lambda item: item["strength"], reverse=True)
    composite = round(float(np.mean([item["ic"] for item in factors_sorted[:5]])), 3)
    return {
        "available": True,
        "factors": factors_sorted,
        "top_factors": factors_sorted[:5],
        "composite": composite,
        "message": "IC 与长短组合收益评估因子质量，可结合实盘进一步验证。",
    }


def build_knowledge_graph_bundle(
    params: StrategyInput,
    market_context: dict[str, Any],
    feature_dataset: pd.DataFrame | None,
) -> dict[str, Any]:
    ticker = params.ticker.upper()
    if not market_context or not market_context.get("news"):
        return {"available": False, "message": "近期缺少可用于构建图谱的资讯。"}
    if nx is None:
        return {"available": False, "message": "未安装 networkx，无法生成知识图谱。"}

    G = nx.Graph()
    G.add_node(ticker, type="ticker")

    sector = None
    industry = None
    try:
        info = yf.Ticker(params.ticker).info or {}
        sector = info.get("sector")
        industry = info.get("industry")
    except Exception:
        pass
    if sector:
        G.add_node(f"行业:{sector}", type="sector")
        G.add_edge(ticker, f"行业:{sector}", weight=1.0)
    if industry:
        G.add_node(f"子行业:{industry}", type="industry")
        G.add_edge(ticker, f"子行业:{industry}", weight=1.0)

    token_pattern = re.compile(r"\b[A-Z]{2,6}\b")
    for item in market_context.get("news", []):
        title = item.get("title") or ""
        snippet = item.get("snippet") or ""
        raw = f"{title} {snippet}"
        tokens = {tok for tok in token_pattern.findall(raw) if not tok.isdigit()}
        if not tokens:
            continue
        weight = 1.0 + float(item.get("score", 0) or 0)
        for tok in tokens:
            if tok not in G:
                G.add_node(tok, type="ticker")
            G.add_edge(ticker, tok, weight=G.get_edge_data(ticker, tok, {"weight": 0}).get("weight", 0) + weight)
        if len(tokens) > 1:
            listed = sorted(tokens)
            for i in range(len(listed) - 1):
                a, b = listed[i], listed[i + 1]
                if a == b:
                    continue
                G.add_edge(a, b, weight=G.get_edge_data(a, b, {"weight": 0}).get("weight", 0) + 0.5)

    if feature_dataset is not None and "momentum_long" in feature_dataset.columns:
        latest = feature_dataset.tail(1).iloc[0]
        momentum = float(latest.get("momentum_long", 0.0))
        G.nodes[ticker]["momentum"] = momentum

    centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(
        ((node, data, centrality.get(node, 0.0)) for node, data in G.nodes(data=True) if node != ticker),
        key=lambda item: item[2],
        reverse=True,
    )[:6]
    highlights = [
        {
            "node": node,
            "centrality": round(score, 3),
            "type": data.get("type", "ticker"),
        }
        for node, data, score in sorted_nodes
    ]
    stats = {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "density": round(nx.density(G), 4) if G.number_of_nodes() > 1 else 0.0,
    }
    centrality_values = [score for _, _, score in sorted_nodes] or [0.0]
    avg_centrality = float(np.mean(centrality_values))
    risk_score = round(float(avg_centrality + stats["density"]), 3)
    insight = (
        "网络集中度较高，需关注行业联动风险。"
        if risk_score > 1.2
        else "网络分散度适中，事件传导风险可控。"
    )
    return {
        "available": True,
        "stats": stats,
        "highlights": highlights,
        "risk_score": risk_score,
        "message": insight,
    }


def build_factor_scorecard(
    prices: pd.DataFrame,
    feature_dataset: pd.DataFrame | None,
    fundamentals: dict[str, Any] | None,
) -> dict[str, Any]:
    if prices is None or prices.empty:
        return {"available": False, "message": "缺少价格数据，无法生成因子得分。"}

    closes = prices.get("adj close", pd.Series(dtype=float)).dropna()
    if closes.shape[0] < 120:
        return {"available": False, "message": "样本不足，无法生成稳定因子。"}

    returns = closes.pct_change().dropna()
    momentum_21 = (1 + returns).rolling(21).apply(lambda x: np.prod(1 + x) - 1).dropna()
    momentum_score = float(momentum_21.tail(1).iloc[0]) if not momentum_21.empty else 0.0
    volatility = float(returns.tail(60).std() * np.sqrt(252)) if returns.shape[0] >= 60 else float(returns.std())
    drawdown_series = (1 + returns).cumprod()
    drawdown = float(calculate_max_drawdown(drawdown_series))

    fundamentals = fundamentals or {}
    factors_raw = [
        ("动量", momentum_score),
        ("波动率", -volatility),
        ("最大回撤", -abs(drawdown)),
    ]
    for label, key in [
        ("盈利能力 (ROE)", "returnOnEquity"),
        ("利润率", "profitMargins"),
        ("估值 (1/PE)", "trailingPE"),
    ]:
        value = fundamentals.get(key)
        if value is None:
            continue
        val = float(value)
        if label.startswith("估值"):
            val = 1 / val if val not in (0, np.nan) else 0.0
        factors_raw.append((label, val))

    if feature_dataset is not None and "volume_z" in feature_dataset.columns:
        liquidity = float(feature_dataset["volume_z"].tail(1).iloc[0])
        factors_raw.append(("成交活跃度", liquidity))

    scores = []
    values = np.array([item[1] for item in factors_raw], dtype=float)
    if values.size:
        mean = np.mean(values)
        std = np.std(values) + 1e-9
        zscores = (values - mean) / std
        for (label, raw), z in zip(factors_raw, zscores):
            scores.append({"factor": label, "raw": round(raw, 4), "score": round(float(z), 3)})

    composite = round(float(np.mean([entry["score"] for entry in scores])) if scores else 0.0, 3)
    return {
        "available": True,
        "scores": scores,
        "composite": composite,
        "message": "Z-score 标准化后的多因子得分，可用于参考仓位调整。",
    }


def compute_model_weights(
    statistical_bundle: dict[str, Any] | None,
    ml_stats: dict[str, Any] | None,
    deep_bundle: dict[str, Any] | None,
    knowledge_bundle: dict[str, Any] | None,
    factor_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    def _add(name: str, score: float, rationale: str) -> None:
        if score > 0:
            candidates.append({"name": name, "score": score, "rationale": rationale})

    if ml_stats:
        sharpe = ml_stats.get("sharpe", 0.0) or 0.0
        max_dd = abs(ml_stats.get("max_drawdown", 0.0) or 0.0)
        score = max(0.25, min(0.7, 0.45 + sharpe / 5 - max_dd))
        rationale = f"夏普 {sharpe:.2f}、最大回撤 {format_percentage(max_dd)}"
        _add("机器学习主策略", score, rationale)

    if deep_bundle and deep_bundle.get("accuracy") is not None:
        acc = float(deep_bundle["accuracy"])
        confidence = float(deep_bundle.get("confidence") or 0.5)
        score = max(0.1, min(0.35, acc * 0.8 + confidence * 0.2))
        best = deep_bundle.get("best_model", "深度模型")
        rationale = f"{best} 准确率 {acc*100:.1f}%、置信度 {confidence:.2f}"
        _add("深度信号", score, rationale)

    if statistical_bundle and statistical_bundle.get("arima"):
        arima = statistical_bundle["arima"]
        aic = float(arima.get("aic", 0.0))
        score = max(0.05, min(0.2, 0.2 - aic / 6000))
        rationale = f"ARIMA AIC {aic:.1f}，提供短线基准"
        _add("统计基线", score, rationale)

    if knowledge_bundle and knowledge_bundle.get("available"):
        risk_score = float(knowledge_bundle.get("risk_score", 0.0))
        adjustment = max(-0.15, min(0.1, 0.12 - risk_score / 4))
        rationale = f"网络风险评分 {risk_score:.2f}"
        if adjustment > 0:
            _add("图谱风控调节", adjustment, rationale)

    if factor_bundle and factor_bundle.get("available"):
        composite = float(factor_bundle.get("composite", 0.0))
        score = max(0.05, min(0.25, abs(composite)))
        top_names = ", ".join(item["name"] for item in factor_bundle.get("top_factors", [])[:3])
        direction = "正向" if composite >= 0 else "反向"
        rationale = f"{direction}信号 · Top 因子：{top_names}"
        _add("因子信号", score, rationale)

    if not candidates:
        return {"available": False, "allocations": [], "message": "缺少可用于分配权重的模型评分。"}

    total = sum(item["score"] for item in candidates)
    if total <= 0:
        return {"available": False, "allocations": [], "message": "模型评分为零，无法生成权重。"}

    allocations = [
        {
            "name": item["name"],
            "weight": round(item["score"] / total, 3),
            "rationale": item["rationale"],
        }
        for item in candidates
    ]
    return {"available": True, "allocations": allocations}


def build_risk_dashboard(
    stats: dict[str, Any],
    benchmark_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = []
    var_value = stats.get("var_95")
    cvar_value = stats.get("cvar_95")
    metrics.append(
        {
            "label": "最大回撤",
            "value": format_percentage(stats.get("max_drawdown", 0.0) or 0.0),
            "comment": "历史净值从高到低的最大跌幅，衡量情绪压力。",
        }
    )
    metrics.append(
        {
            "label": "波动率",
            "value": format_percentage(stats.get("volatility", 0.0) or 0.0),
            "comment": "年化标准差，反映收益的波动程度。",
        }
    )
    metrics.append(
        {
            "label": "夏普比",
            "value": f"{stats.get('sharpe', 0.0):.2f}",
            "comment": "单位风险获取的超额收益，>1 代表风险调整后表现良好。",
        }
    )
    if var_value is not None:
        metrics.append(
            {
                "label": "日度 VaR 95%",
                "value": format_percentage(-var_value),
                "comment": "在正常市场条件下，日内损失超过该值的概率约为 5%。",
            }
        )
    if cvar_value is not None:
        metrics.append(
            {
                "label": "日度 CVaR 95%",
                "value": format_percentage(-cvar_value),
                "comment": "在极端情况下的平均损失（最差 5% 状况的均值）。",
            }
        )
    if stats.get("annual_turnover") is not None:
        metrics.append(
            {
                "label": "年化换手率",
                "value": format_percentage(stats.get("annual_turnover", 0.0) or 0.0),
                "comment": "仓位调换频率，可用于评估交易成本敏感性。",
            }
        )
    benchmark_delta = None
    if benchmark_stats:
        benchmark_delta = float(stats.get("cagr", 0.0) or 0.0) - float(benchmark_stats.get("total_return", 0.0) or 0.0)
    risk_level = "高" if stats.get("max_drawdown", 0.0) <= -0.2 or stats.get("volatility", 0.0) >= 0.35 else "中"
    if stats.get("volatility", 0.0) <= 0.15 and (stats.get("max_drawdown", 0.0) or 0.0) > -0.1:
        risk_level = "低"
    insight = (
        "策略风险偏高，需严格控制仓位并结合对冲。"
        if risk_level == "高"
        else "策略风险处于可接受水平，建议保持纪律性调仓。"
    )
    return {
        "metrics": metrics,
        "risk_level": risk_level,
        "benchmark_delta": benchmark_delta,
        "insight": insight,
    }


def build_mlops_report(params: StrategyInput, stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "auto_retrain": params.auto_apply_best_config,
        "train_window": params.train_window,
        "test_window": params.test_window,
        "last_trading_days": stats.get("trading_days"),
        "recent_sharpe": stats.get("recent_sharpe_60d"),
        "notes": "建议每个训练窗口或当滚动夏普 < 0 时重新训练模型，并记录指标漂移情况。",
    }


def build_scenario_simulation(
    backtest: pd.DataFrame,
    stats: dict[str, Any],
    horizon_days: int = 21,
    simulations: int = 400,
) -> dict[str, Any]:
    returns = backtest.get("strategy_return")
    if returns is None or returns.dropna().shape[0] < max(60, horizon_days):
        return {
            "available": False,
            "message": "历史样本不足，暂无法生成情景模拟。",
        }
    series = returns.dropna().astype(float)
    series = series[np.isfinite(series)]
    if series.empty:
        return {
            "available": False,
            "message": "收益序列无有效值。",
        }
    rng = np.random.default_rng(seed=42)
    sample = rng.choice(series.values, size=(simulations, horizon_days), replace=True)
    cumulative = np.prod(1 + sample, axis=1) - 1
    optimistic, base, defensive = np.quantile(cumulative, [0.85, 0.55, 0.15])
    expected = float(np.mean(cumulative))

    def scenario(label: str, value: float, description: str, allocation_hint: str) -> dict[str, Any]:
        return {
            "label": label,
            "return": format_percentage(float(value)),
            "description": description,
            "allocation_hint": allocation_hint,
        }

    volatility = stats.get("volatility", 0.0) or 0.0
    max_drawdown = stats.get("max_drawdown", 0.0) or 0.0
    scenarios = [
        scenario(
            "乐观冲刺",
            optimistic,
            "市场情绪向好且信号同步，适合分批加仓并预留止盈。",
            "考虑将盈利滚入进取型资产，同时保留 10% 现金缓冲。",
        ),
        scenario(
            "稳健基线",
            base,
            "沿用当前策略与仓位，按周再平衡即可。",
            "保持模型权重不变，关注交易成本与胜率。",
        ),
        scenario(
            "防守下行",
            defensive,
            "若市场震荡或突发利空，需提前设置回撤阈值。",
            "跌幅扩大时将仓位降至 40%-60%，并结合防御性组合。",
        ),
    ]
    return {
        "available": True,
        "horizon_days": horizon_days,
        "scenarios": scenarios,
        "volatility": format_percentage(volatility),
        "max_drawdown": format_percentage(max_drawdown),
        "expected_return": format_percentage(expected),
        "insight": (
            f"若维持历史节奏，未来 {horizon_days} 日平均回报约 {format_percentage(expected)}；"
            f"基线情形 {format_percentage(base)}，需重点防守回撤 {format_percentage(defensive)}。"
        ),
        "notes": "基于历史收益的自助抽样模拟，实际表现将受到宏观与流动性影响。",
    }


def build_opportunity_radar(
    params: StrategyInput,
    factor_effectiveness: dict[str, Any] | None,
    knowledge_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        page = screener.fetch_page(size=30)
    except Exception as exc:
        return {
            "available": False,
            "message": f"获取行情失败：{exc}",
        }
    rows = page.get("rows", []) if isinstance(page, dict) else []
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        price = row.get("price")
        change_pct = row.get("change_pct")
        if price is None or change_pct is None:
            continue
        name = row.get("name") or row.get("ticker") or "未命名"
        clean_rows.append(
            {
                "ticker": row.get("ticker"),
                "name": name,
                "price": float(price),
                "change_pct": float(change_pct),
            }
        )
    if not clean_rows:
        return {
            "available": False,
            "message": "暂无实时行情数据。",
        }
    leaders = sorted(clean_rows, key=lambda x: x["change_pct"], reverse=True)[:4]
    laggards = sorted(clean_rows, key=lambda x: x["change_pct"])[:4]
    steady = sorted(clean_rows, key=lambda x: abs(x["change_pct"]))[:4]
    factor_hint = ""
    if factor_effectiveness and factor_effectiveness.get("available"):
        top_factors = factor_effectiveness.get("top_factors", [])
        if top_factors:
            factor_hint = top_factors[0]["name"]
    knowledge_hint = ""
    if knowledge_bundle and knowledge_bundle.get("available"):
        highlights = knowledge_bundle.get("highlights") or []
        if highlights:
            knowledge_hint = highlights[0]["node"]
    summary = "结合胜率和因子信号，关注涨幅领先的高景气板块，同时在回调名单寻找低吸机会。"
    if factor_hint:
        summary += f" 当前主导因子：{factor_hint}。"
    if knowledge_hint:
        summary += f" 图谱热点：{knowledge_hint}。"
    return {
        "available": True,
        "leaders": leaders,
        "laggards": laggards,
        "steady": steady,
        "summary": summary,
        "insight": (
            f"涨幅榜关注：{', '.join(item['ticker'] for item in leaders[:3]) or '暂无'}；"
            f"回调区：{', '.join(item['ticker'] for item in laggards[:3]) or '暂无'}；"
            f"防守仓观察：{', '.join(item['ticker'] for item in steady[:2]) or '暂无'}。"
        ),
    }


def summarize_macro_highlight(macro_bundle: dict[str, Any]) -> str:
    if not macro_bundle:
        return "暂无宏观提示"
    insights: list[str] = []
    for entry in macro_bundle.values():
        if not isinstance(entry, dict):
            continue
        if not entry.get("available"):
            continue
        label = entry.get("short") or entry.get("label")
        trend = entry.get("trend")
        change_21 = entry.get("change_21d")
        if isinstance(change_21, (int, float)):
            change_text = f"{change_21:+.2f}%"
        else:
            change_text = "—"
        insights.append(f"{label}: {trend}（21日 {change_text}）")
    return "；".join(insights) if insights else "暂无宏观提示"


def build_executive_briefing(
    params: StrategyInput,
    ensemble_bundle: dict[str, Any] | None,
    model_weights: dict[str, Any] | None,
    risk_dashboard: dict[str, Any] | None,
    knowledge_bundle: dict[str, Any] | None,
    factor_effectiveness: dict[str, Any] | None,
    multimodal_bundle: dict[str, Any] | None,
    deep_bundle: dict[str, Any] | None,
    scenario_bundle: dict[str, Any] | None = None,
    opportunity_bundle: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []

    if ensemble_bundle:
        confidence = ensemble_bundle.get("confidence")
        status = f"{confidence:.2f}" if confidence is not None else "—"
        cards.append(
            {
                "title": "模型共识",
                "status": status,
                "body": (
                    "; ".join(ensemble_bundle.get("summary", [])[:2])
                    or "多模型评估完成，可查看详细信心来源。"
                ),
                "cta_label": "查看详情",
                "cta_href": "#advisor-pane",
            }
        )

    if model_weights and model_weights.get("available"):
        allocations = model_weights.get("allocations", [])
        top = allocations[0] if allocations else {}
        formatted = f"{top.get('name')} {top.get('weight', 0):.2f}" if top else "—"
        cards.append(
            {
                "title": "策略权重",
                "status": formatted,
                "body": allocations[0].get("rationale", "") if allocations else "暂无可用权重信息。",
                "cta_label": "组合建议",
                "cta_href": "#model-weights-title",
            }
        )

    if scenario_bundle and scenario_bundle.get("available"):
        scenarios = scenario_bundle.get("scenarios", [])
        base_return = scenarios[1]["return"] if len(scenarios) >= 2 else "—"
        optimistic = scenarios[0]["return"] if scenarios else "—"
        defensive = scenarios[-1]["return"] if scenarios else "—"
        cards.append(
            {
                "title": "情景模拟",
                "status": f"基线 {base_return}",
                "body": f"乐观 {optimistic} / 防守 {defensive}，配合仓位节奏执行。",
                "cta_label": "查看情景",
                "cta_href": "#scenario-board",
            }
        )

    if risk_dashboard:
        cards.append(
            {
                "title": "风险雷达",
                "status": risk_dashboard.get("risk_level", "—"),
                "body": (risk_dashboard.get("insight") or "")[:160],
                "cta_label": "风险指标",
                "cta_href": "#risk-dashboard-title",
            }
        )

    if knowledge_bundle and knowledge_bundle.get("available"):
        insight = knowledge_bundle.get("message", "")
        cards.append(
            {
                "title": "知识图谱",
                "status": f"风险 {knowledge_bundle.get('risk_score', 0):.2f}",
                "body": insight[:160],
                "cta_label": "图谱洞察",
                "cta_href": "#knowledge-graph-title",
            }
        )

    if factor_effectiveness and factor_effectiveness.get("available"):
        composite = factor_effectiveness.get("composite")
        cards.append(
            {
                "title": "因子信号",
                "status": f"IC {composite:.2f}" if composite is not None else "—",
                "body": factor_effectiveness.get("message", "")[:160],
                "cta_label": "查看因子",
                "cta_href": "#factor-effectiveness-title",
            }
        )

    if multimodal_bundle:
        sentiment = multimodal_bundle.get("sentiment", {})
        sentiment_avg = sentiment.get("average")
        macro_text = ""
        macro_bundle = multimodal_bundle.get("macro") or {}
        if isinstance(macro_bundle, dict):
            available = [entry for entry in macro_bundle.values() if isinstance(entry, dict) and entry.get("available")]
            if available:
                first = available[0]
                macro_text = f"；宏观：{first.get('label')} {first.get('trend')}"
        body_text = (sentiment.get("insight", "") or "情绪中性")[:160]
        if macro_text:
            body_text = f"{body_text}{macro_text}"
        cards.append(
            {
                "title": "多模态情绪",
                "status": f"{sentiment_avg:.2f}" if sentiment_avg is not None else "—",
                "body": body_text,
                "cta_label": "市场信息",
                "cta_href": "#macro-dashboard-title",
            }
        )

    if deep_bundle:
        status = deep_bundle.get("best_model", "深度信号")
        cards.append(
            {
                "title": "深度评估",
                "status": status,
                "body": f"准确率 {deep_bundle.get('accuracy', 0.0)*100:.1f}% ，样本 {deep_bundle.get('sample', '-')}",
                "cta_label": "AI 研判",
                "cta_href": "#ai-panel",
            }
        )

    if opportunity_bundle and opportunity_bundle.get("available"):
        leaders = opportunity_bundle.get("leaders", [])
        highlight = ", ".join(item["ticker"] for item in leaders[:2]) if leaders else "暂无"
        cards.append(
            {
                "title": "机会雷达",
                "status": highlight,
                "body": opportunity_bundle.get("summary", "")[:160],
                "cta_label": "机会详情",
                "cta_href": "#opportunity-radar",
            }
        )

    # Remove potential duplicates if same href
    unique_cards: dict[str, dict[str, Any]] = {}
    for card in cards:
        key = card.get("title")
        if key and key not in unique_cards:
            unique_cards[key] = card
    return list(unique_cards.values())


def build_user_questions(
    stats: dict[str, Any],
    recommendations: list[dict[str, Any]],
    risk_dashboard: dict[str, Any] | None,
    model_weights: dict[str, Any] | None,
    ensemble_bundle: dict[str, Any] | None,
    scenario_bundle: dict[str, Any] | None,
    opportunity_bundle: dict[str, Any] | None,
) -> list[dict[str, str]]:
    answers: list[dict[str, str]] = []

    cagr = stats.get("cagr", 0.0) or 0.0
    sharpe = stats.get("sharpe", 0.0) or 0.0
    answers.append(
        {
            "question": "能赚多少？",
            "answer": (
                f"历史回测显示年化复合收益率 {format_percentage(cagr)}，夏普比 {sharpe:.2f}。"
                " 以过去表现估算，10 万本金若维持同样节奏，一年期望收益约 "
                f"{format_currency(100000 * (1 + cagr))}（不含交易费用与滑点）。"
            ),
        }
    )

    if recommendations:
        first_plan = recommendations[0]
        answers.append(
            {
                "question": "怎么投才能赚？",
                "answer": (
                    f"金融顾问建议：{first_plan['title']}。"
                    f" 核心执行：{first_plan['actions']}"
                ),
            }
        )

    risk_text = "风险等级：中，维持纪律性调仓。"
    if risk_dashboard:
        risk_text = f"风险等级：{risk_dashboard.get('risk_level', '—')}。{risk_dashboard.get('insight', '')}"
    answers.append(
        {
            "question": "风险如何控制？",
            "answer": risk_text,
        }
    )

    if model_weights and model_weights.get("available"):
        weight_desc = "; ".join(
            f"{item['name']} {item['weight']:.2f}"
            for item in model_weights.get("allocations", [])[:3]
        )
        answers.append(
            {
                "question": "有没有组合投资建议？",
                "answer": (
                    f"根据模型评分，推荐的信号组合权重：{weight_desc}。"
                    " 可在组合策略页查看详细分配。"
                ),
            }
        )

    if ensemble_bundle:
        answers.append(
            {
                "question": "模型观点一致吗？",
                "answer": (
                    f"综合置信度 {ensemble_bundle.get('confidence', 0.0):.2f}，"
                    "整体观点：" + "；".join(ensemble_bundle.get("summary", [])[:2])
                ),
            }
        )

    if scenario_bundle and scenario_bundle.get("available"):
        scenarios = scenario_bundle.get("scenarios", [])
        base = scenarios[1]["return"] if len(scenarios) >= 2 else "—"
        defensive = scenarios[-1]["return"] if scenarios else "—"
        answers.append(
            {
                "question": "最坏情况会怎样？",
                "answer": (
                    f"情景模拟显示：基线 {base}，若出现突发回撤约为 {defensive}。"
                    " 建议提前设定回撤阈值并结合防御资产调仓。"
                ),
            }
        )

    if opportunity_bundle and opportunity_bundle.get("available"):
        leaders = opportunity_bundle.get("leaders", [])
        highlight = ", ".join(item["ticker"] for item in leaders[:3]) if leaders else "暂无"
        answers.append(
            {
                "question": "市场上有什么机会？",
                "answer": (
                    f"机会雷达聚焦：{highlight}，可结合策略权重逐步建仓。"
                    " 同时关注回调名单寻找低风险切入点。"
                ),
            }
        )

    return answers


def build_advisor_playbook(
    stats: dict[str, Any],
    user_guidance: dict[str, Any],
    recommendations: list[dict[str, Any]],
    scenario_bundle: dict[str, Any] | None,
    risk_dashboard: dict[str, Any] | None,
    opportunity_bundle: dict[str, Any] | None,
    macro_highlight: str | None,
) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []

    def shorten(text: str, width: int = 92) -> str:
        return textwrap.shorten(text, width=width, placeholder="…")

    def add_section(title: str, points: list[str], tag: str | None = None) -> None:
        clean = [point for point in points if point]
        if not clean:
            return
        entry: dict[str, Any] = {"title": title, "points": clean[:4]}
        if tag:
            entry["tag"] = tag
        sections.append(entry)

    cagr = stats.get("cagr", 0.0) or 0.0
    total_return = stats.get("total_return", 0.0) or 0.0
    sharpe = stats.get("sharpe", 0.0) or 0.0
    profit_points = [
        f"历史年化 {format_percentage(cagr)} · 累计收益 {format_percentage(total_return)}",
        f"风险调整后夏普比 {sharpe:.2f}",
    ]
    if scenario_bundle and scenario_bundle.get("available"):
        scenarios = scenario_bundle.get("scenarios", [])
        horizon = scenario_bundle.get("horizon_days")
        if scenarios:
            base_entry = scenarios[1] if len(scenarios) >= 2 else scenarios[0]
            defensive_entry = scenarios[-1]
            base = base_entry.get("return", "—")
            defensive = defensive_entry.get("return", "—")
        else:
            base = "—"
            defensive = "—"
        profit_points.append(
            f"未来 {horizon} 日基线 {base}，防守情形约 {defensive}"
        )
        insight = scenario_bundle.get("insight")
        if insight:
            profit_points.append(shorten(insight))
    add_section("盈利空间", profit_points, tag="收益概览")

    action_points: list[str] = []
    if recommendations:
        first_plan = recommendations[0]
        action_points.append(shorten(f"{first_plan['title']}：{first_plan['actions']}"))
    for step in (user_guidance.get("action_plan") or [])[:2]:
        action_points.append(shorten(f"{step['title']}：{step['detail']}"))
    add_section("操作路线", action_points, tag="执行计划")

    risk_points: list[str] = []
    risk_alerts = user_guidance.get("risk_alerts") or []
    risk_points.extend(risk_alerts[:2])
    if risk_dashboard:
        risk_points.append(
            f"风险等级：{risk_dashboard.get('risk_level', '—')} · {shorten(risk_dashboard.get('insight', ''))}"
        )
        metrics = risk_dashboard.get("metrics") or []
        if metrics:
            first_metric = metrics[0]
            risk_points.append(
                shorten(f"{first_metric['label']}：{first_metric['value']}（{first_metric['comment']}）")
            )
    add_section("风险防守", risk_points, tag="风控")

    opportunity_points: list[str] = []
    if opportunity_bundle and opportunity_bundle.get("available"):
        opportunity_points.append(shorten(opportunity_bundle.get("summary", "")))
        opportunity_points.append(shorten(opportunity_bundle.get("insight", "")))
    if macro_highlight:
        opportunity_points.append(shorten(f"宏观提示：{macro_highlight}"))
    interest_keywords = user_guidance.get("primary_goal_label")
    if interest_keywords:
        opportunity_points.append(shorten(f"目标定位：{interest_keywords}"))
    add_section("机会与提醒", opportunity_points, tag="市场观察")

    return {
        "available": bool(sections),
        "sections": sections,
    }


def _compute_horizon_statistics(backtest: pd.DataFrame, window: int) -> dict[str, Any] | None:
    if backtest.empty or backtest.shape[0] < max(5, window):
        return None
    window_data = backtest.tail(window)
    if window_data.empty:
        return None
    returns = window_data.get("strategy_return")
    if returns is None or returns.empty:
        return None
    cumulative = (1 + returns).cumprod()
    horizon_return = float(cumulative.iloc[-1] - 1)
    horizon_vol = float(returns.std() * np.sqrt(252)) if returns.std() is not None else 0.0
    horizon_sharpe = float(np.sqrt(252) * returns.mean() / (returns.std() + 1e-12))
    hit_ratio = float((returns > 0).mean()) if not returns.empty else 0.0
    latest_prob = float(window_data.get("probability", pd.Series(dtype=float)).dropna().iloc[-1]) if "probability" in window_data.columns and not window_data["probability"].dropna().empty else None
    confidence = normal_cdf(horizon_return / (horizon_vol + 1e-9)) if horizon_vol else 0.5
    drawdown = float(calculate_max_drawdown(cumulative)) if not cumulative.empty else 0.0
    return {
        "window": window,
        "return": horizon_return,
        "volatility": horizon_vol,
        "sharpe": horizon_sharpe,
        "drawdown": drawdown,
        "hit_ratio": hit_ratio,
        "latest_probability": latest_prob,
        "confidence": confidence,
    }


def build_flagship_research_bundle(
    params: StrategyInput,
    prices: pd.DataFrame,
    backtest: pd.DataFrame,
    stats: dict[str, Any],
    benchmark_stats: dict[str, Any] | None,
    market_context: dict[str, Any],
    combo_details: list[dict[str, Any]],
) -> dict[str, Any]:
    horizons_spec = [
        ("短线（日内~5日）", 5),
        ("中线（20~60日）", 60),
        ("长线（120~252日）", 252),
    ]
    horizon_rows: list[dict[str, Any]] = []
    for label, window in horizons_spec:
        metrics = _compute_horizon_statistics(backtest, window)
        if not metrics:
            continue
        horizon_rows.append(
            {
                "label": label,
                "window": window,
                "return": format_percentage(metrics["return"]),
                "volatility": format_percentage(metrics["volatility"]),
                "sharpe": f"{metrics['sharpe']:.2f}",
                "drawdown": format_percentage(metrics["drawdown"]),
                "hit_ratio": format_percentage(metrics["hit_ratio"]),
                "confidence": f"{metrics['confidence'] * 100:.1f}%",
                "latest_probability": metrics["latest_probability"],
            }
        )

    data_sources = []
    if not prices.empty:
        coverage = f"{prices.index.min().date()} → {prices.index.max().date()}"
        data_sources.append(
            {
                "name": "行情数据 (OHLCV)",
                "type": "结构化",
                "status": "已接入",
                "coverage": coverage,
                "notes": "由 yfinance 提供的日线级行情，可扩展至分钟级。",
            }
        )
    data_sources.extend(
        [
            {
                "name": "技术指标/特征库",
                "type": "特征工程",
                "status": "已构建",
                "coverage": f"当前使用 {stats.get('feature_count', 0)} 个核心特征",
                "notes": "支持指标扩展与复用型 Feature Store。",
            },
            {
                "name": "宏观与财务因子",
                "type": "结构化",
                "status": "集成中",
                "notes": "预留接入 FRED/财报 API，需进行频率对齐与因子暴露计算。",
            },
            {
                "name": "新闻/公告/社交媒体",
                "type": "非结构化",
                "status": "试运行" if market_context.get("news") else "待接入",
                "notes": "利用 DuckDuckGo/自建爬虫抽取情绪指标，当前结果用于资讯摘要。",
            },
            {
                "name": "金融知识图谱",
                "type": "关联数据",
                "status": "规划中",
                "notes": "准备将公司-行业-事件关系写入图数据库，结合 GNN 捕捉联动。",
            },
        ]
    )

    feature_columns = list(stats.get("feature_columns") or [])
    feature_library: dict[str, list[str]] = {
        "技术指标": [],
        "量价结构": [],
        "基本面/风格": [],
        "NLP/事件": [],
    }
    for col in feature_columns:
        name = col.lower()
        if any(tag in name for tag in ["sma", "ema", "rsi", "macd", "boll", "momentum"]):
            feature_library["技术指标"].append(col)
        elif any(tag in name for tag in ["volume", "turnover", "volatility", "vwap"]):
            feature_library["量价结构"].append(col)
        elif any(tag in name for tag in ["pe", "roe", "growth", "value", "quality"]):
            feature_library["基本面/风格"].append(col)
        elif any(tag in name for tag in ["sentiment", "news", "event", "text"]):
            feature_library["NLP/事件"].append(col)
        else:
            feature_library.setdefault("其他", []).append(col)

    model_stack: list[dict[str, Any]] = []
    primary_engine = params.strategy_engine
    if primary_engine == "ml_momentum":
        model_stack.append(
            {
                "layer": "短线动量",
                "models": ["Gradient Boosting / LightGBM", "概率校准 (Isotonic)", "阈值优化"],
                "objective": "预测未来 1-5 日方向与幅度，输出概率作为风控阈值。",
                "status": "在线",
            }
        )
        model_stack.append(
            {
                "layer": "中线 / 因子",
                "models": ["传统因子打分", "rolling Sharpe"],
                "objective": "提供组合锚点与风格暴露，辅助多策略集成。",
                "status": "在研",
            }
        )
    elif primary_engine == "multi_combo":
        for entry in combo_details:
            model_stack.append(
                {
                    "layer": entry["engine"],
                    "models": ["核心回测引擎"],
                    "objective": "与主策略并行计算，Stacking 组合信号。",
                    "status": "在线",
                }
            )
    else:
        model_stack.append(
            {
                "layer": "规则基线",
                "models": ["均线交叉", "波动率目标"],
                "objective": "提供稳健基线，便于评估机器学习增益。",
                "status": "在线",
            }
        )

    uncertainty = {
        "volatility": format_percentage(stats.get("volatility", 0.0)),
        "value_at_risk": format_percentage(-(stats.get("var_95", 0.0) or 0.0)),
        "conditional_var": format_percentage(-(stats.get("cvar_95", 0.0) or 0.0)),
        "recent_sharpe": f"{stats.get('recent_sharpe_60d', 0.0):.2f}"
        if "recent_sharpe_60d" in stats
        else "—",
        "notes": "VaR/CVaR 基于历史收益估计，后续将联动贝叶斯置信区间与 MC Dropout。",
    }

    training_days = stats.get("trading_days", len(backtest))
    mlops = {
        "sample_size": training_days,
        "train_window": params.train_window,
        "test_window": params.test_window,
        "embargo_days": params.embargo_days,
        "auto_retrain": params.auto_apply_best_config,
        "comment": "数据/模型版本已经持久化，支持 walk-forward 再训练与灰度验证。",
    }

    risk_controls = {
        "entry_threshold": params.entry_threshold,
        "exit_threshold": params.exit_threshold,
        "vol_target": params.volatility_target,
        "max_leverage": params.max_leverage,
        "stop_suggestion": "建议设定动态止损（最大回撤 × 0.6）与分级减仓阈值。",
    }

    knowledge_graph = {
        "status": "规划中",
        "next": "接入供应链/事件知识图谱，使用 GNN 建模联动风险与主题轮动。",
        "market_links": market_context.get("tickers", [])[:6],
    }

    benchmark = {
        "vol": format_percentage(benchmark_stats.get("volatility", 0.0)) if benchmark_stats else None,
        "sharpe": f"{benchmark_stats.get('sharpe', 0.0):.2f}" if benchmark_stats else None,
        "alpha": format_percentage(benchmark_stats.get("alpha", 0.0)) if benchmark_stats else None,
    }

    return {
        "multi_horizon": horizon_rows,
        "data_sources": data_sources,
        "feature_library": feature_library,
        "model_stack": model_stack,
        "uncertainty": uncertainty,
        "mlops": mlops,
        "risk_controls": risk_controls,
        "knowledge_graph": knowledge_graph,
        "benchmark_snapshot": benchmark,
    }


def build_key_takeaways(
    stats: dict[str, Any],
    benchmark_stats: Optional[dict[str, Any]],
    params: StrategyInput,
) -> list[str]:
    takeaways: list[str] = []
    risk_label = RISK_PROFILE_LABELS.get(params.risk_profile, params.risk_profile)
    takeaways.append(
        f"策略年化复合收益率为 {format_percentage(stats.get('cagr', 0.0) or 0.0)}，"
        f"夏普比率 {stats.get('sharpe', 0.0):.2f}。"
    )
    takeaways.append(
        f"最大回撤 {format_percentage(stats.get('max_drawdown', 0.0) or 0.0)}，平均杠杆 {stats.get('avg_leverage', 0.0):.2f}x，"
        "建议配合风控阈值执行仓位管理。"
    )
    if stats.get("prediction_accuracy") is not None:
        auc = stats.get("auc")
        auc_text = f"，ROC-AUC {auc:.2f}" if isinstance(auc, (int, float)) and not math.isnan(auc) else ""
        takeaways.append(
            f"模型方向预测胜率 {format_percentage(stats.get('prediction_accuracy', 0.0) or 0.0)}{auc_text}。"
        )
    turnover = stats.get("annual_turnover")
    if turnover is not None:
        takeaways.append(
            f"年化换手率约 {format_percentage(turnover or 0.0)}，平均持仓 {stats.get('average_holding_days', 0.0):.1f} 天，"
            f"交易成本占收益比 {format_percentage(stats.get('cost_ratio', 0.0) or 0.0)}。"
        )
    if benchmark_stats:
        takeaways.append(
            f"相对基准 {params.benchmark_ticker.upper()} 的 α 为 {format_percentage(benchmark_stats.get('alpha', 0.0) or 0.0)}，"
            f"β 为 {benchmark_stats.get('beta', 0.0):.2f}，信息比率 {benchmark_stats.get('info_ratio', 0.0):.2f}。"
        )
    else:
        takeaways.append("无可用基准数据，可考虑添加指数或行业 ETF 作为对照与对冲。")
    if params.capital:
        takeaways.append(
            f"以可支配资金 {format_currency(params.capital)} 计，若按核心配置执行，"
            f"一年期的期望收益约为 {format_currency(params.capital * (1 + stats.get('cagr', 0.0) or 0.0))}。"
        )
    takeaways.append(
        f"基于风险偏好（{risk_label}）建议组合中保留 {format_percentage(1 - stats.get('avg_exposure', 0.0) or 0.0)} 的缓冲资产以应对极端情况。"
    )
    return takeaways


def estimate_confidence(stats: dict[str, Any]) -> tuple[str, float]:
    sharpe = stats.get("sharpe", 0.0) or 0.0
    max_drawdown = abs(stats.get("max_drawdown", 0.0) or 0.0)
    cagr = stats.get("cagr", 0.0) or 0.0
    score = 0.0
    if sharpe >= 1.2:
        score += 0.4
    elif sharpe >= 0.8:
        score += 0.25
    elif sharpe >= 0.5:
        score += 0.15
    if cagr >= 0.15:
        score += 0.3
    elif cagr >= 0.08:
        score += 0.2
    elif cagr >= 0.04:
        score += 0.1
    if max_drawdown <= 0.15:
        score += 0.3
    elif max_drawdown <= 0.25:
        score += 0.2
    elif max_drawdown <= 0.35:
        score += 0.1
    score = max(0.0, min(score, 0.95))
    if score >= 0.7:
        label = "高"
    elif score >= 0.45:
        label = "中"
    else:
        label = "低"
    return label, score


def build_user_guidance(
    stats: dict[str, Any],
    benchmark_stats: Optional[dict[str, Any]],
    params: StrategyInput,
) -> dict[str, Any]:
    horizon_labels = {
        "short": "短期（0-6 个月）",
        "medium": "中期（6-24 个月）",
        "long": "长期（24 个月以上）",
    }
    experience_labels = {
        "novice": "新手",
        "intermediate": "进阶投资者",
        "advanced": "专业投资者",
    }
    goal_messages = {
        "growth": "以净值增长为主，侧重追踪趋势机会并控制回撤。",
        "income": "关注稳定现金流，可搭配分红/债券资产平滑收益。",
        "preserve": "以资金安全为先，策略信号用于增强收益但优先守住本金。",
    }

    cagr = stats.get("cagr", 0.0) or 0.0
    total_return = stats.get("total_return", 0.0) or 0.0
    max_drawdown = stats.get("max_drawdown", 0.0) or 0.0
    sharpe = stats.get("sharpe", 0.0) or 0.0
    volatility = stats.get("volatility", 0.0) or 0.0
    capital = params.capital or 0.0

    horizon_label = horizon_labels.get(params.investment_horizon, "中期（6-24 个月）")
    experience_label = experience_labels.get(params.experience_level, "新手")
    goal_text = goal_messages.get(params.primary_goal, goal_messages["growth"])

    benchmark_alpha = None
    if benchmark_stats:
        benchmark_alpha = benchmark_stats.get("alpha", 0.0)

    quick_summary: list[str] = []
    quick_summary.append(
        f"策略在回测期的累计收益约 {format_percentage(total_return)}，年化复合收益率 {format_percentage(cagr)}。"
    )
    quick_summary.append(
        f"最大回撤 {format_percentage(max_drawdown)}，夏普比 {sharpe:.2f}，波动率 {format_percentage(volatility)}。"
    )
    if benchmark_alpha is not None:
        quick_summary.append(
            f"相对基准的年化超额收益（α）约 {format_percentage(benchmark_alpha)}。"
        )
    quick_summary.append(f"目标定位：{goal_text} 建议持有周期：{horizon_label}。")

    action_plan: list[dict[str, str]] = []
    capital_text = (
        f"当前可支配资金 {format_currency(capital)}，建议保留 10%-20% 的现金备用。"
        if capital
        else "未填写资金规模，可结合自身预算分配仓位。"
    )
    action_plan.append(
        {
            "title": "准备阶段（本周内）",
            "detail": (
                f"{capital_text} 完成账户/券商准备，确认手续费与交易规则。"
                " 对照策略参数生成观察清单，并将基准指数纳入监控。"
            ),
            "priority": "高",
        }
    )
    action_plan.append(
        {
            "title": "执行阶段（持仓运行）",
            "detail": (
                f"依据模型信号和再平衡节奏：当短均线高于长均线且信号概率 >55% 时建仓，"
                f"回撤达到 {format_percentage(max_drawdown or 0.0)} 或 RSI>70 时分批降仓。"
                " 每周查看胜率与净值曲线，必要时将盈利锁定到现金或防御资产。"
            ),
            "priority": "高",
        }
    )
    action_plan.append(
        {
            "title": "复盘与升级（每月/季度）",
            "detail": (
                "记录实际收益与计划差异，关注夏普比和换手率是否符合预期；"
                "结合宏观/行业变化，适时调整关注关键词与对冲资产。"
            ),
            "priority": "中",
        }
    )

    risk_alerts: list[str] = []
    if max_drawdown <= -0.2:
        risk_alerts.append("历史最大回撤超过 20%，建仓时务必控制仓位并设置止损。")
    if volatility >= 0.3:
        risk_alerts.append("策略波动率偏高，适合少量试仓或搭配债券/货币基金分散风险。")
    if stats.get("cost_ratio", 0.0) and stats.get("cost_ratio", 0.0) > 0.25:
        risk_alerts.append("交易成本占收益比例较高，需关注滑点与换手率，避免频繁调仓。")
    recent_sharpe = stats.get("recent_sharpe_60d", 0.0) or 0.0
    if recent_sharpe < 0:
        risk_alerts.append("近 60 日滚动夏普为负，建议降仓或启用对冲，并在下个训练窗口重训模型。")
    if not risk_alerts:
        risk_alerts.append("请持续关注市场突发事件，必要时降低杠杆或暂停交易。")

    education_tips: list[str] = []
    if params.experience_level == "novice":
        education_tips.append("夏普比衡量单位风险获取的收益，>1 代表风险调整后表现健康。")
        education_tips.append("最大回撤表示净值从高点跌到低点的幅度，是评估情绪压力的重要指标。")
        education_tips.append("建议先使用模拟账户熟悉信号，再逐步投入真实资金。")
    elif params.experience_level == "intermediate":
        education_tips.append("定期跟踪策略在不同市场环境下的收益分布，评估是否需要多策略组合。")
        education_tips.append("通过滚动窗口重新训练模型，可避免过拟合特定时间段。")
    else:
        education_tips.append("可考虑将策略纳入多因子框架，与价值/质量因子组合，提升稳健性。")
        education_tips.append("建议监控信息比率与跟踪误差，衡量相对基准的超额收益质量。")

    confidence_label, confidence_score = estimate_confidence(stats)

    disclaimer = (
        "所有建议基于历史回测结果，不构成投资承诺。实际交易需结合个人风险承受能力、"
        "资金规划与市场环境，设置止损并谨慎使用杠杆。"
    )

    return {
        "quick_summary": quick_summary,
        "action_plan": action_plan,
        "risk_alerts": risk_alerts,
        "education_tips": education_tips,
        "confidence_label": confidence_label,
        "confidence_score": round(confidence_score, 3),
        "experience_label": experience_label,
        "investment_horizon_label": horizon_label,
        "primary_goal_label": goal_text,
        "disclaimer": disclaimer,
    }


def fetch_market_context(params: StrategyInput) -> dict[str, Any]:
    context: dict[str, Any] = {
        "message": "",
        "news": [],
        "tickers": [],
        "analysis": "",
    }

    if os.environ.get("ENABLE_WEB_SEARCH", "1") == "0":
        context["message"] = (
            "当前运行环境未启用外部网络搜索。可在服务器设置 ENABLE_WEB_SEARCH=1 并配置代理后获取最新资讯。"
        )
        return context

    sector = ""
    industry = ""
    info: dict[str, Any] = {}
    try:
        ticker_info = yf.Ticker(params.ticker)
        info = ticker_info.info or {}
        sector = info.get("sector", "") or ""
        industry = info.get("industry", "") or ""
    except Exception:
        sector = ""
        industry = ""

    interest_terms: list[str] = []
    if params.interest_keywords:
        interest_terms.extend(params.interest_keywords)
    if sector:
        interest_terms.append(sector)
    if industry:
        interest_terms.append(industry)
    if params.benchmark_ticker:
        interest_terms.append(params.benchmark_ticker.upper())

    def normalize_term(term: str) -> str:
        if not term:
            return ""
        term = term.strip()
        return term

    interest_terms = list(dict.fromkeys(filter(None, (normalize_term(term) for term in interest_terms))))

    year = params.end_date.year
    ticker_upper = params.ticker.upper()
    base_queries = [
        f"{ticker_upper} 投资 前景 {year}",
        f"{ticker_upper} 新闻 {year}",
    ]
    english_queries = [
        f"{ticker_upper} stock news {year}",
        f"{ticker_upper} outlook {year}",
        f"{ticker_upper} earnings updates",
    ]
    base_queries.extend(english_queries)
    def quote_if_needed(term: str) -> str:
        if not term:
            return term
        if any(ch.isspace() for ch in term) or re.search(r"[\u4e00-\u9fff]", term):
            return f'"{term}"'
        return term

    if interest_terms:
        combined_terms = " ".join(quote_if_needed(t) for t in interest_terms[:2])
        base_queries.insert(0, f"{ticker_upper} {combined_terms} {year}")
        for term in interest_terms[:4]:
            base_queries.append(f"{quote_if_needed(term)} {ticker_upper} 最新 动态")
    if industry and industry not in interest_terms:
        base_queries.append(f"{industry} 行业 趋势 {year}")

    region = os.environ.get("DDG_REGION", "wt-wt")
    safesearch = os.environ.get("DDG_SAFESEARCH", "off")
    proxy = os.environ.get("DDG_PROXY")

    rate_limit_exceptions: tuple[type[Exception], ...] = ()
    ddg_client_cls: Any | None = None
    try:
        from ddgs import DDGS as _DDGS  # type: ignore

        ddg_client_cls = _DDGS
        try:
            from ddgs.exceptions import RatelimitException as _DDGRateLimit  # type: ignore

            rate_limit_exceptions = (_DDGRateLimit,)
        except Exception:
            rate_limit_exceptions = ()
    except ImportError:
        try:
            from duckduckgo_search import DDGS as _DDGS  # type: ignore

            ddg_client_cls = _DDGS
            try:
                from duckduckgo_search.exceptions import (  # type: ignore
                    RatelimitException as _DDGRateLimit,
                )

                rate_limit_exceptions = (_DDGRateLimit,)
            except Exception:
                rate_limit_exceptions = ()
        except ImportError:
            context["message"] = "缺少 duckduckgo-search / ddgs 依赖，无法执行在线搜索。"
            return context

    if ddg_client_cls is None:
        context["message"] = "未找到可用的 DuckDuckGo 搜索客户端。"
        return context

    def run_duck_query(ddgs_client: Any, query: str) -> list[dict[str, str]]:
        """Retry DuckDuckGo新闻/文本查询，自动处理限流。"""
        wait_seconds = 1.0
        for _ in range(3):
            try:
                news_items = list(
                    ddgs_client.news(
                        query,
                        region=region,
                        safesearch=safesearch,
                        max_results=6,
                    )
                )
                if news_items:
                    return news_items
            except Exception as exc:
                if rate_limit_exceptions and isinstance(exc, rate_limit_exceptions):
                    time.sleep(min(6.0, wait_seconds + random.random()))
                    wait_seconds *= 1.8
                    continue
                break
            time.sleep(0.5 + random.random() * 0.5)

        wait_seconds = 1.0
        for _ in range(2):
            try:
                text_items = list(
                    ddgs_client.text(
                        query,
                        region=region,
                        safesearch=safesearch,
                        max_results=6,
                    )
                )
                if text_items:
                    return text_items
            except Exception as exc:
                if rate_limit_exceptions and isinstance(exc, rate_limit_exceptions):
                    time.sleep(min(5.0, wait_seconds + random.random()))
                    wait_seconds *= 1.7
                    continue
                break
            time.sleep(0.4 + random.random() * 0.4)
        return []

    try:
        proxies = {"http": proxy, "https": proxy} if proxy else None
        aggregated: list[dict[str, str]] = []
        seen_queries: set[str] = set()
        with ddg_client_cls(proxies=proxies) as ddgs:  # type: ignore
            for q in base_queries:
                query_key = q.lower()
                if query_key in seen_queries:
                    continue
                seen_queries.add(query_key)
                search_results = run_duck_query(ddgs, q)
                aggregated.extend(search_results)
                if len(aggregated) >= 18:
                    break

        if not aggregated:
            context["message"] = "已尝试 DuckDuckGo 搜索，但未检索到相关新闻条目。"
            return context

        ticker_candidates: set[str] = set()
        unique_items: list[dict[str, str]] = []
        interest_lower = [term.lower() for term in interest_terms]
        interest_lower.append(params.ticker.lower())
        interest_lower = list(dict.fromkeys(filter(None, interest_lower)))
        ticker_pattern = re.compile(rf"\\b{re.escape(ticker_upper)}\\b", re.IGNORECASE)

        def _alias_variants(raw_alias: str) -> list[str]:
            alias = raw_alias.lower().strip()
            if not alias:
                return []
            variants = [alias]
            normalized = re.sub(r"[^\w\s]", " ", alias)
            normalized = re.sub(r"\s+", " ", normalized).strip()
            if normalized and normalized not in variants:
                variants.append(normalized)
            compact = re.sub(r"[^a-z0-9]", "", alias)
            if compact and compact not in variants:
                variants.append(compact)
            if normalized:
                first_token = normalized.split(" ")[0]
                if first_token and first_token not in variants:
                    variants.append(first_token)
            return [variant for variant in variants if variant]

        company_aliases: list[str] = []
        for alias_key in ("shortName", "longName"):
            alias_value = info.get(alias_key)
            if alias_value:
                company_aliases.extend(_alias_variants(str(alias_value)))
        if not company_aliases:
            company_aliases.append(params.ticker.lower())

        def normalize_story(item: dict[str, Any]) -> dict[str, Any] | None:
            title = item.get("title") or item.get("heading") or ""
            url = item.get("url") or item.get("href") or ""
            snippet = item.get("body") or item.get("excerpt") or item.get("snippet") or ""
            if not url:
                return None
            image_url = (
                item.get("image")
                or item.get("image_url")
                or item.get("img")
                or item.get("thumbnail")
                or ""
            )
            if image_url.startswith("//"):
                image_url = "https:" + image_url
            source = item.get("source") or item.get("publisher") or item.get("site") or ""
            published = item.get("published") or item.get("date") or item.get("time") or ""
            return {
                "title": title or "相关新闻",
                "url": url,
                "snippet": snippet,
                "image": image_url,
                "source": source,
                "published": published,
                "raw_score": item.get("score", 0) or 0,
            }

        normalized_items: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in aggregated:
            story = normalize_story(item)
            if not story:
                continue
            if story["url"] in seen_urls:
                continue
            seen_urls.add(story["url"])
            normalized_items.append(story)

        def _story_match_flags(story: dict[str, Any]) -> tuple[bool, bool, bool, str]:
            combined_text = f"{story['title']} {story['snippet']}"
            text_mix = combined_text.lower()
            matches_ticker = bool(ticker_pattern.search(combined_text))
            matches_alias = any(alias in text_mix for alias in company_aliases if alias)
            matches_interest = any(term and term in text_mix for term in interest_lower)
            return matches_ticker, matches_alias, matches_interest, text_mix

        for story in normalized_items:
            matches_ticker, matches_alias, matches_interest, text_mix = _story_match_flags(story)
            if not (matches_ticker or matches_alias or matches_interest):
                continue
            score = 0
            if matches_ticker:
                score += 3
            if matches_alias:
                score += 2
            if matches_interest:
                score += 1
            if industry and industry.lower() in text_mix:
                score += 1
            if sector and sector.lower() in text_mix:
                score += 1
            raw_score = story.get("raw_score", 0) or 0
            score += min(3, int(raw_score) // 10 if isinstance(raw_score, (int, float)) else 0)
            unique_items.append(
                {
                    "title": story["title"],
                    "url": story["url"],
                    "snippet": story["snippet"],
                    "score": score,
                    "image": story["image"],
                    "source": story["source"],
                    "published": story["published"],
                }
            )

            upper_words = {
                token
                for token in re.findall(r"[A-Z]{2,6}", f"{story['title']} {story['snippet']}")
                if not token.isdigit()
            }
            ticker_candidates.update(upper_words)

        if not unique_items and normalized_items:
            context[
                "message"
            ] = "DuckDuckGo 未返回足够精确的条目，以下结果为放宽条件后的最新资讯。"
            for story in normalized_items[:6]:
                unique_items.append(
                    {
                        "title": story["title"],
                        "url": story["url"],
                        "snippet": story["snippet"],
                        "score": story.get("raw_score", 0) or 0,
                        "image": story["image"],
                        "source": story["source"],
                        "published": story["published"],
                    }
                )

        unique_items.sort(key=lambda item: item.get("score", 0), reverse=True)
        filtered_items = [item for item in unique_items if item.get("score", 0) > 0]
        top_items = (filtered_items or unique_items)[:8]
        context["news"] = [
            {
                "title": item["title"],
                "url": item["url"],
                "snippet": item["snippet"],
                "image": item.get("image") or "",
                "source": item.get("source") or "",
                "published": item.get("published") or "",
            }
            for item in top_items
        ]
        focus_cards: list[dict[str, str]] = []
        for item in top_items[:6]:
            story_hash = hashlib.sha256(item["url"].encode("utf-8")).hexdigest()
            focus_cards.append(
                {
                    "id": story_hash[:12],
                    "title": item["title"],
                    "url": item["url"],
                    "snippet": item["snippet"],
                    "image": item.get("image") or "",
                    "source": item.get("source") or "",
                    "published": item.get("published") or "",
                    "readers": estimate_readers(story_hash[:16], item.get("score", 0)),
                }
            )
        context["focus_news"] = focus_cards

        context["tickers"] = sorted(ticker_candidates - {params.ticker.upper()})
        snippets = [item.get("snippet", "") for item in context["news"] if item.get("snippet")]
        analysis_parts: list[str] = []
        if sector:
            analysis_parts.append(f"所属行业：{sector}")
        if industry:
            analysis_parts.append(f"细分领域：{industry}")
        if snippets:
            analysis_parts.append("新闻摘要：" + " ".join(snippets[:2]))
        if interest_terms:
            analysis_parts.append("关注主题：" + "、".join(interest_terms[:4]))
        context["analysis"] = "；".join(analysis_parts)
        context["message"] = "以下为根据 DuckDuckGo 检索到的重点新闻摘要，可用于拓展与公司相关的标的。"
        context["interest_terms"] = interest_terms
    except Exception as exc:
        context["message"] = (
            "尝试调用外部搜索接口失败，请确认网络可达且 DuckDuckGo API 可用。"
            f" 错误详情：{exc}"
        )

    return context
