from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from django.conf import settings

from .alpaca_data import (
    fetch_stock_bars_frame,
    fetch_stock_snapshots,
)
from .cache_utils import build_cache_key, cache_memoize
from .security import sanitize_html_fragment

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

# 关键修复：所有 Proxy 必须是美股代码 (Stock Ticker)，不能包含 Crypto (如 BTC/USD)
# 否则调用 /v2/stocks/bars 接口会报错。
MACRO_TICKERS = {
    "^VIX": {"label": "芝加哥VIX波动率", "short": "VIX", "proxy": "VIXY"}, # ProShares VIX Short-Term Futures ETF
    "^TNX": {"label": "美国10年期国债", "short": "10Y", "proxy": "IEF"},  # iShares 7-10 Year Treasury Bond ETF
    "DX-Y.NYB": {"label": "美元指数 (DXY)", "short": "DXY", "proxy": "UUP"},  # Invesco DB US Dollar Index Bullish Fund
    "GC=F": {"label": "纽约金价", "short": "Gold", "proxy": "GLD"},     # SPDR Gold Shares
    "CL=F": {"label": "WTI 原油", "short": "WTI", "proxy": "USO"},     # United States Oil Fund
    "HYG": {"label": "高收益债ETF", "short": "HY Credit", "proxy": "HYG"}, # iShares iBoxx $ High Yield Corp Bond ETF
    "BTC-USD": {"label": "比特币", "short": "BTC", "proxy": "BITO"},    # 修复：使用 BITO (比特币策略ETF) 代替 Crypto 代码
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

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _calculate_trend(current: float, prev_5d: float | None, prev_21d: float | None) -> str:
    """Helper to determine trend string."""
    if prev_21d is None or prev_21d == 0:
        return "平稳"
    change = (current / prev_21d) - 1
    if change > 0.02:
        return "上升"
    if change < -0.02:
        return "下降"
    return "平稳"

def _safe_pct_change(current: float, previous: float | None) -> float | None:
    if previous is None or previous == 0:
        return None
    return round(((current / previous) - 1) * 100, 2)

# =============================================================================
# DATA FETCHING LOGIC (PURE ALPACA)
# =============================================================================

def _download_macro_series(
    end_date: date,
    lookback_days: int = 730,
    *,
    user_id: str | None = None,
) -> Dict[str, Any]:
    """
    使用 Alpaca 获取宏观代理 ETF 数据。
    """
    start = end_date - timedelta(days=lookback_days)
    records: Dict[str, Any] = {}
    
    # 1. 收集所有需要获取的 Proxy Symbol
    proxies_map = {k: v["proxy"] for k, v in MACRO_TICKERS.items() if v.get("proxy")}
    symbols_to_fetch = list(set(proxies_map.values())) # 去重
    
    # 2. 批量获取数据
    # 注意：如果某个 symbol 无效，Alpaca 可能会返回部分数据，我们需要健壮地处理
    try:
        df = fetch_stock_bars_frame(
            symbols_to_fetch,
            start=start,
            end=end_date,
            timeframe="1Day",
            user_id=user_id,
            feed="sip",
            adjustment="split",
        )
    except Exception:
        df = pd.DataFrame()

    # 3. 处理每个指标
    for original_ticker, meta in MACRO_TICKERS.items():
        proxy_symbol = meta.get("proxy")
        
        # 默认空状态
        record = {
            "label": meta["label"],
            "short": meta["short"],
            "available": False,
            "latest": 0.0,
            "change_5d": 0.0,
            "change_21d": 0.0,
            "trend": "无数据",
            "message": "数据暂不可用"
        }

        # 检查 DataFrame 是否包含该 symbol 的数据
        # fetch_stock_bars_frame 返回的 df 如果是多列，通常是 MultiIndex (Attribute, Symbol) 或 (Symbol, Attribute)
        # 这里的辅助函数通常返回 (Timestamp Index, Columns=Symbol MultiIndex)
        if proxy_symbol and not df.empty:
            # 兼容单层或多层索引
            series = None
            if isinstance(df.columns, pd.MultiIndex):
                # 尝试从 MultiIndex 中获取 Close
                if proxy_symbol in df.columns.get_level_values(0):
                     # 假设结构是 [Symbol][Attribute] 或 [Attribute][Symbol]
                     # 标准 alpaca_data 实现通常返回 Symbol 作为列名或一级索引
                     try:
                         series = df[proxy_symbol]["Close"]
                     except KeyError:
                         pass
            elif proxy_symbol in df.columns:
                 # 单一 Symbol 请求时可能直接返回 Close 列，但在多 Symbol 请求下不太可能
                 series = df[proxy_symbol]

            if series is not None:
                series = series.dropna()
                if not series.empty:
                    latest = float(series.iloc[-1])
                    val_5d = float(series.iloc[-5]) if len(series) >= 5 else latest
                    val_21d = float(series.iloc[-21]) if len(series) >= 21 else latest
                    
                    record.update({
                        "available": True,
                        "latest": round(latest, 4),
                        "change_5d": _safe_pct_change(latest, val_5d),
                        "change_21d": _safe_pct_change(latest, val_21d),
                        "trend": _calculate_trend(latest, val_5d, val_21d),
                        "message": None
                    })
        
        records[original_ticker] = record
        
    return records


def _fetch_macro_series(
    end_date: date,
    lookback_days: int = 730,
    *,
    user_id: str | None = None,
) -> Dict[str, Any]:
    cache_key = build_cache_key("macro-series-alpaca-v2", end_date.isoformat(), lookback_days)
    return cache_memoize(
        cache_key,
        lambda: _download_macro_series(end_date, lookback_days, user_id=user_id),
        getattr(settings, "MACRO_DATA_CACHE_TTL", 3600), 
    ) or {}


def _fetch_fundamental_snapshot(ticker: str, user_id: str | None = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Attempts to fetch basic info from Alpaca.
    """
    fundamentals: Dict[str, Any] = {
        "available": False, 
        "summary": "N/A",
        "marketCap": None,
        "peRatio": None,
        "eps": None
    }
    financials: Dict[str, Any] = {}
    
    try:
        snapshots = fetch_stock_snapshots([ticker], user_id=user_id)
        snapshot = snapshots.get(ticker) or snapshots.get(ticker.upper())

        if snapshot:
            daily_bar = snapshot.get("dailyBar", {})
            latest_trade = snapshot.get("latestTrade", {})
            latest_price = latest_trade.get("p") or daily_bar.get("c")
            
            fundamentals.update({
                "available": True,
                "sector": "Equity",
                "industry": "N/A",
                "currentPrice": latest_price,
                "volume": daily_bar.get("v"),
                # Alpaca Snapshots 不包含市值信息，这里保持 None，前端需兼容
            })
    except Exception:
        pass

    return fundamentals, financials


def _download_capital_flows_snapshot(
    end_date: date,
    lookback_days: int = 90,
    *,
    user_id: str | None = None,
) -> Dict[str, Any]:
    """
    通过代表性 ETF (SPY, QQQ, etc.) 估算资金流向。
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
        df = fetch_stock_bars_frame(
            list(proxies.keys()),
            start=start,
            end=end_date,
            timeframe="1Day",
            user_id=user_id,
            feed="sip",
            adjustment="split",
        )
    except Exception:
        df = pd.DataFrame()

    for symbol, meta in proxies.items():
        # 安全获取 Series
        price_series = None
        vol_series = None
        
        if not df.empty and isinstance(df.columns, pd.MultiIndex):
             if symbol in df.columns.get_level_values(0):
                 try:
                    price_series = df[symbol]["Close"]
                    vol_series = df[symbol]["Volume"]
                 except KeyError:
                    pass
        
        if price_series is None or price_series.dropna().empty:
            flows[symbol] = {
                "label": meta["label"],
                "group": meta["group"],
                "available": False,
                "message": "无数据",
            }
            continue
            
        try:
            price = price_series.dropna()
            volume = vol_series.fillna(0) if vol_series is not None else pd.Series(0, index=price.index)
            
            latest_price = float(price.iloc[-1])
            prev_21d = float(price.iloc[-21]) if len(price) >= 21 else latest_price
            
            momentum_21d = ((latest_price / prev_21d) - 1) * 100 if prev_21d else 0.0
            returns = price.pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0
            
            latest_volume = float(volume.iloc[-1]) if not volume.empty else 0.0
            avg_dollar_vol = (latest_price * latest_volume) / 1e6

            flows[symbol] = {
                "label": meta["label"],
                "group": meta["group"],
                "available": True,
                "momentum_21d": round(momentum_21d, 2),
                "volatility": round(volatility, 3),
                "avg_dollar_volume": round(avg_dollar_vol, 2),
                "flow_signal": 0.0,
            }
        except Exception:
             flows[symbol] = {"label": meta["label"], "available": False, "message": "计算错误"}

    # 汇总风险偏好
    try:
        tech = flows.get("QQQ", {}).get("momentum_21d", 0)
        equity = flows.get("SPY", {}).get("momentum_21d", 0)
        bonds = flows.get("TLT", {}).get("momentum_21d", 0)
        
        flows["_summary"] = {
            "risk_appetite": "Risk-On" if (tech > equity) else "Risk-Off",
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
    cache_key = build_cache_key("capital-flows-alpaca", end_date.isoformat(), lookback_days)
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
        title = (item.get("headline") or item.get("title") or "").lower()
        snippet = (item.get("summary") or item.get("snippet") or "").lower()
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
            labeled.append({
                "title": item.get("headline") or item.get("title"),
                "score": sentiment,
                "url": item.get("url"),
            })
            
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
    return {
        "available": False,
        "message": "期权数据暂不可用",
        "call_iv": None,
        "put_iv": None
    }


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
            
    summary = {
        "growth_nowcast": "Unknown",
        "inflation_watch": "Unknown",
        "policy_bias": "Unknown",
    }
    return {
        "available": False,
        "source": "derived",
        "summary": summary,
        "message": "未找到 global_macro.json",
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
        title = item.get("headline") or item.get("title") or ""
        snippet = (item.get("summary") or item.get("snippet") or "")
        url = item.get("url")
        published = item.get("created_at") or item.get("published_at") or item.get("datetime")
        
        lower = f"{title} {snippet}".lower()
        matched_types = []
        for tag, cues in keywords.items():
            if any(cue in lower for cue in cues):
                matched_types.append(tag)
        if matched_types:
            events.append({
                "title": title,
                "types": matched_types,
                "url": url,
                "published": published,
            })
    return events[:10]


def collect_auxiliary_data(
    params,
    market_context: Dict[str, Any],
    *,
    user_id: str | None = None,
) -> AuxiliaryData:
    end_date = params.end_date
    macro = _fetch_macro_series(end_date, user_id=user_id)
    fundamentals, financials = _fetch_fundamental_snapshot(params.ticker, user_id=user_id)
    
    news = market_context.get("news") if market_context else None
    events = _derive_event_signals(params.ticker, news)
    sentiment = _compute_news_sentiment(news)
    
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
