from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any, Optional, Tuple
import os
import random
import json

import numpy as np
import pandas as pd

from django.conf import settings


from ..data_sources import AuxiliaryData
from ..rl_agents import build_rl_agent
from ..http_client import http_client, HttpClientError
from ..network import get_requests_session, resolve_retry_config, retry_call_result
from ..alpaca_data import fetch_stock_bars_frame, resolve_alpaca_credentials
from .config import (
    QuantStrategyError,
    StrategyInput,
)
from .ml_engine import (
    run_ml_backtest,
)
from .metrics import build_core_metrics, compute_validation_metrics, aggregate_oos_metrics
from .event_engine import compute_realized_returns, run_event_backtest
from .risk import (
    calculate_target_leverage,
    enforce_min_holding,
    enforce_risk_limits,
)
from .store import DATA_CACHE_DIR

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

RISK_PROFILE_LABELS = {
    "conservative": "保守型",
    "balanced": "均衡型",
    "aggressive": "进取型",
}


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


def fetch_price_data(
    ticker: str,
    start: date,
    end: date,
    *,
    user_id: str | None = None,
) -> Tuple[pd.DataFrame, list[str]]:
    """Fetch historical price data using yfinance with local cache fallback."""
    warnings: list[str] = []
    data_source = "yfinance"
    cache_path: str | None = None

    def _shorten_error(exc: Exception | None) -> str:
        if not exc:
            return ""
        message = str(exc).strip().replace("\n", " ")
        if not message:
            return exc.__class__.__name__
        if len(message) > 140:
            return f"{message[:137]}..."
        return message
    download_error: Exception | None = None
    data = pd.DataFrame()
    key_id, secret = resolve_alpaca_credentials(user_id=user_id)
    if key_id and secret:
        try:
            alpaca_end = end + timedelta(days=1)
            frame = fetch_stock_bars_frame(
                [ticker],
                start=start,
                end=alpaca_end,
                timeframe="1Day",
                feed="sip",
                adjustment="split",
                user_id=user_id,
            )
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                if isinstance(frame.columns, pd.MultiIndex):
                    try:
                        data = frame.xs(ticker.upper(), axis=1, level=1)
                    except (KeyError, ValueError):
                        data = pd.DataFrame()
                else:
                    data = frame
                if isinstance(data, pd.DataFrame) and not data.empty:
                    data_source = "alpaca"
        except Exception as exc:
            download_error = exc

    if data.empty:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - dependency load
            raise QuantStrategyError(
                "缺少可用行情源。请配置 Alpaca API Key，或安装 yfinance 后再试。"
            ) from exc

        retry_config = resolve_retry_config(
            retries=os.environ.get("MARKET_FETCH_MAX_RETRIES"),
            backoff=os.environ.get("MARKET_FETCH_RETRY_BACKOFF"),
            default_timeout=getattr(settings, "MARKET_DATA_TIMEOUT_SECONDS", None),
        )
        session = get_requests_session(retry_config.timeout)

        def _empty_frame(value: object) -> bool:
            return not isinstance(value, pd.DataFrame) or value.empty

        try:
            def _download() -> pd.DataFrame:
                return yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    actions=False,
                    repair=True,
                    threads=False,
                    timeout=retry_config.timeout,
                    session=session,
                )

            data = retry_call_result(_download, config=retry_config, should_retry=_empty_frame)
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
            reason = _shorten_error(download_error)
            fallback_note = (
                f"线上行情下载失败，已回退到本地缓存。原因：{reason}"
                if reason
                else "线上行情返回为空，已回退到本地缓存。"
            )
            warnings.append(fallback_note)
            warnings.append(
                f"已从本地缓存 {cache_file} 读取数据。若需最新行情，请联网后刷新缓存。"
            )
            data = cached.loc[(cached.index.date >= start) & (cached.index.date <= end)]
            data_source = "csv_cache"
            cache_path = os.fspath(cache_file)
            data.attrs["data_fetch_note"] = fallback_note
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
    data, start, end = _apply_listing_window(data, ticker, start, end, warnings)
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
        to_cache.index = pd.to_datetime(to_cache.index)
        to_cache.index.name = "date"
        to_cache = to_cache.reset_index()
        if "Date" in to_cache.columns and "date" not in to_cache.columns:
            to_cache = to_cache.rename(columns={"Date": "date"})
        if "date" not in to_cache.columns:
            to_cache = to_cache.rename(columns={to_cache.columns[0]: "date"})
        to_cache["date"] = pd.to_datetime(to_cache["date"]).dt.strftime("%Y-%m-%d")
        to_cache.to_csv(cache_file, index=False)
        cache_path = os.fspath(cache_file)
    except Exception:
        # Non-fatal: caching should not break pipeline
        pass

    base_cols = ["open", "high", "low", "close", "adj close", "volume"]
    for col in ("open", "high", "low", "close"):
        if col not in data.columns and "close" in data.columns:
            data[col] = data["close"]
    if "volume" not in data.columns:
        data["volume"] = 0.0
    data.attrs["data_source"] = data_source
    if cache_path:
        data.attrs["cache_path"] = cache_path
    return data[base_cols].dropna(subset=["adj close"]), warnings


@lru_cache(maxsize=1)
def _load_listing_status() -> pd.DataFrame:
    path = DATA_CACHE_DIR / "alpha_listing_status.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(col).strip().lower() for col in df.columns]
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def _parse_listing_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            parsed = pd.to_datetime(text, errors="coerce")  # type: ignore[call-arg]
        except Exception:
            return None
        if pd.isna(parsed):
            return None
        return parsed.date()


def _apply_listing_window(
    data: pd.DataFrame,
    ticker: str,
    start: date,
    end: date,
    warnings: list[str],
) -> tuple[pd.DataFrame, date, date]:
    listing = _load_listing_status()
    if listing.empty or "symbol" not in listing.columns:
        warnings.append("缺少上市/退市信息缓存，无法校验幸存者偏差。")
        return data, start, end
    rows = listing[listing["symbol"] == ticker.upper()]
    if rows.empty:
        warnings.append(f"未找到 {ticker.upper()} 的上市/退市记录，无法校验幸存者偏差。")
        return data, start, end
    row = rows.iloc[0]
    ipo_date = _parse_listing_date(row.get("ipodate"))
    delist_date = _parse_listing_date(row.get("delistingdate"))
    status = str(row.get("status") or "").strip().lower()
    effective_start = start
    effective_end = end
    if ipo_date and start < ipo_date:
        effective_start = ipo_date
        warnings.append(f"{ticker.upper()} IPO 日期为 {ipo_date.isoformat()}，已将回测起点调整到上市日。")
    if delist_date and end > delist_date:
        effective_end = delist_date
        warnings.append(f"{ticker.upper()} 已于 {delist_date.isoformat()} 退市，回测截止日已相应截断。")
    if status and status != "active":
        warnings.append(f"{ticker.upper()} 当前状态为 {row.get('status') or status}，需注意退市/合并风险。")
    if effective_start > effective_end:
        raise QuantStrategyError("上市/退市日期导致回测区间无有效交易日，请调整开始/结束日期。")
    trimmed = data.loc[(data.index.date >= effective_start) & (data.index.date <= effective_end)]
    trimmed.attrs = dict(getattr(data, "attrs", {}))
    return trimmed, effective_start, effective_end


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
    asset_returns = compute_realized_returns(prices, params).reindex(rl_backtest.index).fillna(0.0)
    rl_backtest["asset_return"] = asset_returns
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
    if "volume" not in rl_backtest and "volume" in prices:
        rl_backtest["volume"] = prices["volume"].reindex(rl_backtest.index).ffill().bfill()
    if "adv" not in rl_backtest and "adv" in prices:
        rl_backtest["adv"] = prices["adv"].reindex(rl_backtest.index).ffill().bfill()
    rl_backtest, exec_stats, rl_execution_events = run_event_backtest(
        rl_backtest,
        exposure_series,
        params,
        leverage=rl_backtest["leverage"],
    )
    adv_hits = int(exec_stats.get("adv_hard_cap_hits") or 0)

    metrics, stats = summarize_backtest(
        rl_backtest,
        params,
        include_prediction=True,
        include_auc=True,
        feature_columns=base_stats.get("feature_columns", []),
        shap_img=base_stats.get("shap_img"),
    )
    stats["rl_playbook"] = agent.playbook
    stats["execution_stats"] = {
        "avg_coverage": exec_stats.get("avg_coverage"),
        "unfilled_ratio": exec_stats.get("unfilled_ratio"),
        "avg_spread_bps": exec_stats.get("avg_spread_bps"),
        "halt_days": exec_stats.get("halt_days"),
        "limit_days": exec_stats.get("limit_days"),
        "participation": exec_stats.get("participation"),
        "effective_participation": exec_stats.get("effective_participation"),
        "adv_hard_cap_hits": adv_hits,
    }
    events = list(stats.get("risk_events", []))
    events.extend(overlay_events)
    if adv_hits > 0:
        events.append(f"RL 回测：ADV 参与率上限({(params.max_adv_participation or 0.1):.0%}) 压缩 {adv_hits} 次仓位。")
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
        vm = compute_validation_metrics(rl_backtest["strategy_return"].fillna(0.0))
        oos_report = {
            "slices": [
                {
                    **vm,
                    "fold": 1,
                    "test_start": str(rl_backtest.index[0].date()) if len(rl_backtest.index) else "",
                    "test_end": str(rl_backtest.index[-1].date()) if len(rl_backtest.index) else "",
                }
            ],
            "summary": aggregate_oos_metrics([vm]),
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


def summarize_backtest(
    backtest: pd.DataFrame,
    params: StrategyInput,
    *,
    include_prediction: bool = False,
    include_auc: bool = False,
    feature_columns: Optional[list[str]] = None,
    shap_img: Optional[str] = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Proxy to pipeline.summarize_backtest for legacy imports without a hard module dependency."""
    from . import pipeline as _pipeline

    return _pipeline.summarize_backtest(
        backtest,
        params,
        include_prediction=include_prediction,
        include_auc=include_auc,
        feature_columns=feature_columns,
        shap_img=shap_img,
    )


def _compute_oos_from_backtest(backtest: pd.DataFrame, params: StrategyInput) -> dict[str, Any] | None:
    """Proxy to pipeline._compute_oos_from_backtest for legacy imports."""
    from . import pipeline as _pipeline

    return _pipeline._compute_oos_from_backtest(backtest, params)
