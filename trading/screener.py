
"""
Minimal screener module – only for today's top gainers/losers sidebar.

We keep:
- CORE_TICKERS_US: the universe the sidebar will look at
- _fetch_quote_snapshot(symbols): try Yahoo batch first, then per-symbol yfinance
- fetch_page(...): a simple fallback the view can call when primary fetch fails

All other original screener features (AlphaVantage listing, sectors, fundamentals,
formatting, etc.) are intentionally deleted because the app no longer needs them.
"""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

from django.utils.translation import gettext_lazy as _
from django.conf import settings

from .http_client import http_client, HttpClientError
from .observability import record_metric
from .cache_utils import cache_get_object, cache_set_object, build_cache_key

# yfinance is optional – we'll use it only as a fallback
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore


# Cache TTLs (Redis 优先，其次内存)
SCREENER_CACHE_TTL = getattr(settings, "SCREENER_CACHE_TTL", 180)


# ---------------------------------------------------------------------------
# universe to look at – enough for a daily "best/worst" panel
# ---------------------------------------------------------------------------
CORE_TICKERS_US: list[str] = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "TSLA",
    "META",
    "GOOGL",
    "GOOG",
    "AVGO",
    "AMD",
    "NFLX",
    "ADBE",
    "CRM",
    "INTC",
    "CSCO",
    "LIN",
    "NKE",
    "BA",
    "KO",
    "PEP",
    "JPM",
    "WMT",
    "V",
    "MA",
    "SPY",
    "QQQ",
    "XOM",
    "CVX",
    "UNH",
    "GS",
]

DEFAULT_MARKET = "us"
ALLOWED_MARKETS: dict[str, dict[str, Any]] = {
    "us": {"label": _("美国市场"), "aliases": {"usa", "nyse", "nasdaq"}},
    "hk": {"label": _("香港市场"), "aliases": {"hongkong", "hkex"}},
    "cn": {"label": _("中国A股"), "aliases": {"cn", "sh", "sz", "sse"}},
}

SECTOR_OPTIONS: dict[str, dict[str, Any]] = {
    "tech": {"label": _("科技板块"), "aliases": {"technology"}},
    "finance": {"label": _("金融板块"), "aliases": {"financial", "financials", "bank"}},
    "consumer": {"label": _("消费板块"), "aliases": {"consumer_discretionary", "consumer-staples"}},
    "energy": {"label": _("能源板块"), "aliases": {"oil", "gas", "energy"}},
    "healthcare": {"label": _("医疗健康"), "aliases": {"health", "medical"}},
    "industrials": {"label": _("工业板块"), "aliases": {"industrial"}},
}
DEFAULT_SECTOR_LABEL = _("全部板块")


def sanitize_market(raw: str | None) -> str:
    slug = (raw or DEFAULT_MARKET).strip().lower()
    if slug in ALLOWED_MARKETS:
        return slug
    for key, meta in ALLOWED_MARKETS.items():
        aliases = meta.get("aliases") or set()
        if slug in aliases:
            return key
    return DEFAULT_MARKET


def resolve_sector(value: str | None) -> tuple[str, str]:
    if not value:
        return DEFAULT_SECTOR_LABEL, ""
    slug = value.strip().lower()
    if slug in SECTOR_OPTIONS:
        meta = SECTOR_OPTIONS[slug]
        return meta["label"], slug
    for key, meta in SECTOR_OPTIONS.items():
        aliases = meta.get("aliases") or set()
        if slug in aliases:
            return meta["label"], key
    return DEFAULT_SECTOR_LABEL, ""


def _quote_cache_key(symbol: str) -> str:
    return build_cache_key("screener-quote", symbol.upper())


# ---------------------------------------------------------------------------
# main: fetch a quote snapshot
# ---------------------------------------------------------------------------
def _fetch_quote_snapshot(symbols: Sequence[str]) -> dict[str, dict[str, Any]]:
    """
    Try to get quotes for the given symbols.

    Strategy:
    1. use local cache if younger than QUOTE_CACHE_TTL_SECONDS
    2. try Yahoo batch endpoint
    3. if batch fails, fall back to per-symbol yfinance (if available)
    4. write everything we get back to the cache
    """
    results: dict[str, dict[str, Any]] = {}
    if not symbols:
        return results

    to_fetch: list[str] = []
    for s in symbols:
        sym = s.strip().upper()
        if not sym:
            continue
        cached = cache_get_object(_quote_cache_key(sym))
        if cached:
            results[sym] = cached
        else:
            to_fetch.append(sym)

    if not to_fetch:
        return results

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; QuantTradingSidebar/1.0)",
        "Accept": "application/json",
    }

    chunk_size = 40

    for idx in range(0, len(to_fetch), chunk_size):
        chunk = [s for s in to_fetch[idx : idx + chunk_size] if s]
        if not chunk:
            continue

        params = {
            "symbols": ",".join(chunk),
            "lang": "en-US",
            "region": "US",
        }

        try:
            # ① try Yahoo batch api
            resp = http_client.get(
                "https://query1.finance.yahoo.com/v7/finance/quote",
                params=params,
                headers=headers,
                timeout=10,
            )
            payload = resp.json()
            for item in payload.get("quoteResponse", {}).get("result", []):
                symbol = item.get("symbol")
                if not symbol:
                    continue
                item["_ts"] = time.time()
                results[symbol] = item
                cache_set_object(_quote_cache_key(symbol), item, SCREENER_CACHE_TTL)
        except (HttpClientError, ValueError, json.JSONDecodeError):
            # ② batch failed – use yfinance for each symbol
            if yf is not None:
                for symbol in chunk:
                    try:
                        tk = yf.Ticker(symbol)
                        fast = getattr(tk, "fast_info", None) or {}
                        info = getattr(tk, "info", None) or {}
                        price = (
                            fast.get("last_price")
                            or fast.get("lastPrice")
                            or fast.get("regularMarketPrice")
                            or info.get("regularMarketPrice")
                            or info.get("previousClose")
                        )
                        if price is None:
                            continue
                        change_pct = (
                            fast.get("regularMarketChangePercent")
                            or info.get("regularMarketChangePercent")
                        )
                        entry = {
                            "symbol": symbol,
                            "shortName": fast.get("shortName") or info.get("shortName") or symbol,
                            "regularMarketPrice": float(price),
                            "regularMarketChangePercent": float(change_pct) if change_pct is not None else 0.0,
                            "_ts": time.time(),
                        }
                        results[symbol] = entry
                        cache_set_object(_quote_cache_key(symbol), entry, SCREENER_CACHE_TTL)
                    except Exception:
                        continue
            # if yfinance is None, we just leave these symbols missing

    total_symbols = len([s for s in symbols if s and s.strip()])
    if total_symbols:
        record_metric(
            "screener.quote_snapshot",
            total=total_symbols,
            cache_hits=total_symbols - len(to_fetch),
            fetched=len(to_fetch),
        )
    return results


# ---------------------------------------------------------------------------
# very simple fallback for views._get_market_sidebar()
# ---------------------------------------------------------------------------
def fetch_page(
    offset: int = 0,
    size: int = 30,
    sector: str | None = None,
    industry: str | None = None,
    market: str | None = None,
    cache: Any | None = None,
) -> dict[str, Any]:
    """
    Minimal replacement for the old screener.fetch_page(...).

    We ignore sector/industry and simply:
    - take CORE_TICKERS_US
    - fetch quotes
    - build rows with ticker / price / change_pct
    """
    snapshot = _fetch_quote_snapshot(CORE_TICKERS_US)
    rows: list[dict[str, Any]] = []
    for sym in CORE_TICKERS_US:
        quote = snapshot.get(sym)
        if not quote:
            continue
        price = (
            quote.get("regularMarketPrice")
            or quote.get("postMarketPrice")
            or quote.get("preMarketPrice")
            or quote.get("previousClose")
        )
        change_pct = (
            quote.get("regularMarketChangePercent")
            or quote.get("postMarketChangePercent")
        )
        # change_pct in yahoo is already in percent, e.g. 1.23 means 1.23%
        rows.append(
            {
                "ticker": sym,
                "name": quote.get("shortName") or quote.get("longName") or sym,
                "price": float(price) if price is not None else None,
                "change_pct": float(change_pct) if change_pct is not None else None,
                "sector": quote.get("sector") or "--",
                "industry": quote.get("industry") or "--",
                "currency": quote.get("currency") or "USD",
            }
        )

    total = len(rows)
    sliced = rows[offset : offset + size]

    return {
        "rows": sliced,
        "offset": offset,
        "size": size,
        "total": total,
        "has_more": offset + size < total,
        "sector": sector,
        "industry": industry,
        "market": market or "us",
    }
