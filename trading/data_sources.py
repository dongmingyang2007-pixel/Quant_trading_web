from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from django.conf import settings

from .cache_utils import build_cache_key, cache_memoize
from .network import get_requests_session, resolve_retry_config, retry_call, retry_call_result
from .security import sanitize_html_fragment
from . import market_data
MACRO_TICKERS = {
    "^VIX": {"label": "芝加哥VIX波动率", "short": "VIX"},
    "^TNX": {"label": "美国10年期国债收益率", "short": "10Y"},
    "DX-Y.NYB": {"label": "美元指数 (DXY)", "short": "DXY"},
    "GC=F": {"label": "纽约金价", "short": "Gold"},
    "CL=F": {"label": "WTI 原油", "short": "WTI"},
    "HYG": {"label": "高收益债ETF (HYG)", "short": "HY Credit"},
    "BTC-USD": {"label": "比特币", "short": "BTC"},
}

GLOBAL_MACRO_FILE = settings.LEARNING_CONTENT_DIR / "global_macro.json"

SENTIMENT_LEXICON = {
    "bullish": ["beat", "增长", "扩张", "上涨", "好于预期", "record profit", "surge", "acceleration", "upgrade"],
    "bearish": ["亏损", "回撤", "降指引", "减产", "裁员", "裁撤", "downgrade", "warning", "probe", "miss"],
}


@dataclass(slots=True)
class AuxiliaryData:
    macro: Dict[str, Any]
    fundamentals: Dict[str, Any]
    financials: Dict[str, Any]
    events: List[Dict[str, Any]]
    capital_flows: Dict[str, Any]
    news_sentiment: Dict[str, Any]
    options_metrics: Dict[str, Any]
    global_macro: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {
            "macro": self.macro,
            "fundamentals": self.fundamentals,
            "financials": self.financials,
            "events": self.events,
            "capital_flows": self.capital_flows,
            "news_sentiment": self.news_sentiment,
            "options_metrics": self.options_metrics,
            "global_macro": self.global_macro,
        }


def _select_panel(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    """Normalize yfinance output (multi-index vs single) to a flat panel."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    columns = frame.columns
    if isinstance(columns, pd.MultiIndex):
        level0 = columns.get_level_values(0)
        target = field if field in level0 else None
        if target is None and field == "Adj Close" and "Close" in level0:
            target = "Close"
        if target is None and field == "Volume" and "Volume" in level0:
            target = "Volume"
        if target is None:
            for candidate in ("Close", "Adj Close", "Open"):
                if candidate in level0:
                    target = candidate
                    break
        if target is None:
            return pd.DataFrame()
        return frame.xs(target, level=0, axis=1)
    if field in frame.columns:
        return frame[field]
    if field == "Adj Close" and "Close" in frame.columns:
        return frame["Close"]
    return frame


def _yf_retry_config(timeout: float | None = None):
    return resolve_retry_config(
        timeout=timeout,
        retries=os.environ.get("MARKET_FETCH_MAX_RETRIES"),
        backoff=os.environ.get("MARKET_FETCH_RETRY_BACKOFF"),
        default_timeout=getattr(settings, "MARKET_DATA_TIMEOUT_SECONDS", None),
    )


def _empty_frame(value: object) -> bool:
    return not isinstance(value, pd.DataFrame) or value.empty


def _download_macro_series(
    end_date: date,
    lookback_days: int = 730,
    *,
    user_id: str | None = None,
) -> Dict[str, Any]:
    """拉取核心宏观指标（VIX、10Y、DXY），并计算最新值及动量。"""
    start = end_date - timedelta(days=lookback_days)
    records: Dict[str, Any] = {}
    fetch_failed = False
    try:
        raw = market_data.fetch(
            list(MACRO_TICKERS.keys()),
            start=start,
            end=end_date + timedelta(days=1),
            interval="1d",
            cache=True,
            ttl=getattr(settings, "MACRO_DATA_CACHE_TTL", 600),
            user_id=user_id,
        )
        if _empty_frame(raw):
            fetch_failed = True
            data = pd.DataFrame()
        else:
            data = _select_panel(raw, "Adj Close")
            if data.empty:
                fetch_failed = True
    except Exception:
        data = pd.DataFrame()
        fetch_failed = True

    if isinstance(data, pd.Series):
        data = data.to_frame()

    for symbol, meta in MACRO_TICKERS.items():
        series = data.get(symbol)
        if series is None or series.dropna().empty:
            message = "缺少历史数据"
            if fetch_failed or data.empty:
                message = "暂未获取到最新数据（可能离线或接口限制）"
            records[symbol] = {
                "label": meta["label"],
                "short": meta["short"],
                "available": False,
                "message": message,
            }
            continue
        series = series.dropna()
        latest = float(series.iloc[-1])
        change_5d = float((series.iloc[-1] / series.iloc[-5] - 1) * 100) if len(series) >= 5 else None
        change_21d = float((series.iloc[-1] / series.iloc[-21] - 1) * 100) if len(series) >= 21 else None
        trend = "上升" if change_21d and change_21d > 0 else "下降" if change_21d and change_21d < 0 else "平稳"
        records[symbol] = {
            "label": meta["label"],
            "short": meta["short"],
            "available": True,
            "latest": round(latest, 4),
            "change_5d": None if change_5d is None else round(change_5d, 2),
            "change_21d": None if change_21d is None else round(change_21d, 2),
            "trend": trend,
        }
    return records


def _fetch_macro_series(
    end_date: date,
    lookback_days: int = 730,
    *,
    user_id: str | None = None,
) -> Dict[str, Any]:
    cache_key = build_cache_key("macro-series", end_date.isoformat(), lookback_days)
    return cache_memoize(
        cache_key,
        lambda: _download_macro_series(end_date, lookback_days, user_id=user_id),
        getattr(settings, "MACRO_DATA_CACHE_TTL", 600),
    ) or {}


def _fetch_fundamental_snapshot(ticker: str) -> Dict[str, Any]:
    """使用 yfinance 抽取公司的基本面与财务摘要。"""
    fundamentals: Dict[str, Any] = {}
    financials: Dict[str, Any] = {}
    config = _yf_retry_config()
    session = get_requests_session(config.timeout)
    try:
        def _download() -> tuple[Dict[str, Any], Dict[str, Any]]:
            fundamentals_local: Dict[str, Any] = {}
            financials_local: Dict[str, Any] = {}
            yft = yf.Ticker(ticker, session=session)
            info = yft.info or {}
            for field in [
                "sector",
                "industry",
                "marketCap",
                "enterpriseValue",
                "profitMargins",
                "grossMargins",
                "operatingMargins",
                "returnOnEquity",
                "returnOnAssets",
                "revenueGrowth",
                "earningsQuarterlyGrowth",
            ]:
                value = info.get(field)
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    fundamentals_local[field] = float(value)
                else:
                    fundamentals_local[field] = value

            fundamentals_local["summary"] = info.get("longBusinessSummary") or ""

            def parse_financial_table(frame: pd.DataFrame | None, label: str) -> Dict[str, Any]:
                if frame is None or frame.empty:
                    return {}
                # yfinance financial tables columns are periods.
                latest_col = frame.columns[0]
                return {f"{label}:{idx}": float(val) for idx, val in frame[latest_col].dropna().items()}

            financials_local["income_statement"] = parse_financial_table(getattr(yft, "financials", None), "IS")
            financials_local["balance_sheet"] = parse_financial_table(getattr(yft, "balance_sheet", None), "BS")
            financials_local["cashflow"] = parse_financial_table(getattr(yft, "cashflow", None), "CF")
            return fundamentals_local, financials_local

        def _should_retry(result: object) -> bool:
            if not isinstance(result, tuple) or len(result) != 2:
                return True
            fundamentals_local, financials_local = result
            return not fundamentals_local and not financials_local

        fundamentals, financials = retry_call_result(
            _download,
            config=config,
            should_retry=_should_retry,
        )
    except Exception:
        fundamentals = {}
        financials = {}
    return fundamentals, financials


def _download_capital_flows_snapshot(
    end_date: date,
    lookback_days: int = 90,
    *,
    user_id: str | None = None,
) -> Dict[str, Any]:
    """
    通过代表性 ETF / 指数估算资金流向与风险偏好。
    """
    proxies = {
        "SPY": {"label": "美股大盘", "group": "equity"},
        "QQQ": {"label": "纳指科技", "group": "growth"},
        "IWM": {"label": "罗素小盘", "group": "smallcap"},
        "TLT": {"label": "20年期国债", "group": "duration"},
        "HYG": {"label": "高收益信用", "group": "credit"},
        "GLD": {"label": "黄金ETF", "group": "commodity"},
    }
    start = end_date - timedelta(days=lookback_days)
    flows: Dict[str, Any] = {}
    try:
        raw = market_data.fetch(
            list(proxies.keys()),
            start=start,
            end=end_date + timedelta(days=1),
            interval="1d",
            cache=True,
            ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
            user_id=user_id,
        )
    except Exception:
        raw = pd.DataFrame()
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    price_panel = _select_panel(raw, "Adj Close")
    volume_panel = _select_panel(raw, "Volume")
    for symbol, meta in proxies.items():
        price = price_panel.get(symbol)
        volume = volume_panel.get(symbol)
        if price is None or volume is None or price.dropna().empty:
            flows[symbol] = {
                "label": meta["label"],
                "group": meta["group"],
                "available": False,
                "message": "缺少最新数据",
            }
            continue
        price = price.dropna()
        volume = volume.reindex(price.index).ffill()
        returns = price.pct_change().fillna(0.0)
        adv = (volume * price).rolling(10).mean()
        latest_adv = float(adv.iloc[-1]) if not adv.dropna().empty else None
        flow_est = float((adv.diff().iloc[-1]) / adv.iloc[-2]) if adv.shape[0] > 2 and adv.iloc[-2] else None
        momentum_21d = float((price.iloc[-1] / price.iloc[-21] - 1) * 100) if len(price) > 21 else None
        flows[symbol] = {
            "label": meta["label"],
            "group": meta["group"],
            "available": True,
            "momentum_21d": None if momentum_21d is None else round(momentum_21d, 2),
            "volatility": round(float(returns.rolling(21).std().iloc[-1] * np.sqrt(252)), 3)
            if returns.shape[0] > 21
            else None,
            "avg_dollar_volume": None if latest_adv is None else round(latest_adv / 1e6, 2),
            "flow_signal": None if flow_est is None else round(flow_est * 100, 2),
        }
    # 计算风险偏好指标：高β vs 低β、成长 vs 价值
    try:
        equity = flows.get("SPY", {}).get("momentum_21d") or 0.0
        tech = flows.get("QQQ", {}).get("momentum_21d") or 0.0
        bonds = flows.get("TLT", {}).get("momentum_21d") or 0.0
        credit = flows.get("HYG", {}).get("momentum_21d") or 0.0
        flows["_summary"] = {
            "risk_appetite": "Risk-On" if tech > equity and credit > bonds else "Neutral" if tech >= equity else "Risk-Off",
            "growth_vs_value": tech - equity,
            "duration_vs_equity": bonds - equity,
        }
    except Exception:
        flows["_summary"] = {}
    return flows


def _fetch_capital_flows_snapshot(
    end_date: date,
    lookback_days: int = 90,
    *,
    user_id: str | None = None,
) -> Dict[str, Any]:
    cache_key = build_cache_key("capital-flows", end_date.isoformat(), lookback_days)
    return cache_memoize(
        cache_key,
        lambda: _download_capital_flows_snapshot(end_date, lookback_days, user_id=user_id),
        getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
    ) or {}


def _compute_news_sentiment(news_items: List[Dict[str, Any]] | None) -> Dict[str, Any]:
    if not news_items:
        return {"available": False, "message": "暂无新闻样本"}
    total = 0
    score = 0
    labeled: List[Dict[str, Any]] = []
    for item in news_items[:30]:
        title = (item.get("title") or "").lower()
        snippet = (item.get("snippet") or "").lower()
        text = f"{title} {snippet}"
        sentiment = 0
        for cue in SENTIMENT_LEXICON["bullish"]:
            if cue.lower() in text:
                sentiment += 1
        for cue in SENTIMENT_LEXICON["bearish"]:
            if cue.lower() in text:
                sentiment -= 1
        score += sentiment
        total += 1
        if sentiment != 0:
            labeled.append(
                {
                    "title": item.get("title"),
                    "score": sentiment,
                    "url": item.get("url"),
                }
            )
    avg = score / max(total, 1)
    label = "积极" if avg > 0.2 else "消极" if avg < -0.2 else "中性"
    return {
        "available": True,
        "avg_score": round(avg, 2),
        "label": label,
        "sample_size": total,
        "highlights": labeled[:6],
    }


def _fetch_options_metrics(ticker: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "available": False,
        "message": "暂无期权数据",
    }
    config = _yf_retry_config()
    session = get_requests_session(config.timeout)
    try:
        def _download() -> Dict[str, Any]:
            tk = yf.Ticker(ticker, session=session)
            expiries = tk.options
            if not expiries:
                return result
            expiry = expiries[0]
            chain = tk.option_chain(expiry)
            calls = chain.calls
            puts = chain.puts
            if calls.empty or puts.empty:
                return result
            call_iv = float(calls["impliedVolatility"].mean())
            put_iv = float(puts["impliedVolatility"].mean())
            put_call = float(puts["volume"].sum() / calls["volume"].sum()) if calls["volume"].sum() else None
            atm_call = calls.iloc[(calls["strike"] - tk.info.get("currentPrice", calls["strike"].median())).abs().idxmin()]
            atm_put = puts.iloc[(puts["strike"] - tk.info.get("currentPrice", puts["strike"].median())).abs().idxmin()]
            return {
                "available": True,
                "expiry": expiry,
                "call_iv": round(call_iv * 100, 2),
                "put_iv": round(put_iv * 100, 2),
                "put_call_ratio": None if put_call is None else round(put_call, 2),
                "atm_call_bid": float(atm_call.get("bid", 0.0)),
                "atm_put_bid": float(atm_put.get("bid", 0.0)),
            }

        result = retry_call(_download, config=config)
    except Exception:
        pass
    return result


def _load_global_macro_snapshot(end_date: date) -> Dict[str, Any]:
    if GLOBAL_MACRO_FILE.exists():
        try:
            with GLOBAL_MACRO_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, (dict, list)):
                    raw = json.dumps(data, ensure_ascii=False, indent=2)
                else:
                    raw = str(data)
                return {
                    "available": True,
                    "source": "custom",
                    "data": sanitize_html_fragment(raw),
                }
        except Exception:
            pass
    # fallback: derive from MACRO_TICKERS momentum
    derived = {key: value for key, value in MACRO_TICKERS.items()}
    summary = {
        "growth_nowcast": "放缓" if "^VIX" in derived else "中性",
        "inflation_watch": "温和",
        "policy_bias": "观望",
    }
    return {
        "available": False,
        "source": "derived",
        "summary": summary,
        "message": "未找到 global_macro.json ，已使用派生指标。",
    }


def _derive_event_signals(ticker: str, news_items: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    if not news_items:
        return []

    keywords = {
        "并购": ["acquire", "merger", "收购", "并购", "takeover"],
        "盈利预警": ["profit warning", "guidance cut", "亏损", "降指引"],
        "扩产": ["capacity", "expansion", "扩产", "产能"],
        "监管风险": ["lawsuit", "regulator", "监管", "调查"],
    }

    events: List[Dict[str, Any]] = []
    for item in news_items:
        title = item.get("title") or ""
        snippet = item.get("snippet") or ""
        url = item.get("url")
        lower = f"{title} {snippet}".lower()
        matched_types = []
        for tag, cues in keywords.items():
            if any(cue in lower for cue in cues):
                matched_types.append(tag)
        if matched_types:
            events.append(
                {
                    "title": title,
                    "types": matched_types,
                    "url": url,
                    "published": item.get("published"),
                }
            )
    return events[:10]


def collect_auxiliary_data(
    params,
    market_context: Dict[str, Any],
    *,
    user_id: str | None = None,
) -> AuxiliaryData:
    """统一收集宏观、基本面、财务与事件信息，供下游模块使用。"""
    end_date = params.end_date
    macro = _fetch_macro_series(end_date, user_id=user_id)
    fundamentals, financials = _fetch_fundamental_snapshot(params.ticker)
    events = _derive_event_signals(params.ticker, market_context.get("news") if market_context else None)
    sentiment = _compute_news_sentiment(market_context.get("news") if market_context else None)
    options = _fetch_options_metrics(params.ticker)
    global_macro = _load_global_macro_snapshot(end_date)
    return AuxiliaryData(
        macro=macro,
        fundamentals=fundamentals,
        financials=financials,
        events=events,
        capital_flows=_fetch_capital_flows_snapshot(end_date, user_id=user_id),
        news_sentiment=sentiment,
        options_metrics=options,
        global_macro=global_macro,
    )
