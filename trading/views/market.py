from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Iterable, Mapping
import re

import pandas as pd
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie

from django.utils.translation import gettext as _, gettext_lazy as _lazy

from .. import screener
from ..cache_utils import build_cache_key, cache_memoize
from .. import market_data
from ..observability import ensure_request_id, record_metric, track_latency
from ..rate_limit import check_rate_limit, rate_limit_key
from ..models import UserProfile
from ..alpaca_data import resolve_alpaca_credentials, fetch_news, fetch_stock_snapshots
from ..realtime.alpaca import fetch_assets as fetch_alpaca_assets
from ..realtime.market_stream import request_symbol as request_market_symbol, request_symbols as request_market_symbols
from ..realtime.storage import read_state, write_state

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yf = None  # type: ignore


@dataclass(slots=True)
class Timeframe:
    key: str
    label: str
    label_en: str
    period: str
    interval: str


@dataclass(slots=True)
class DetailTimeframe:
    key: str
    label: str
    label_en: str
    period: str
    interval: str
    resample: str | None = None
    limit: int = 360


TIMEFRAMES: dict[str, Timeframe] = {
    "1d": Timeframe("1d", _lazy("近1日"), "1D", "5d", "15m"),
    "5d": Timeframe("5d", _lazy("近5日"), "5D", "10d", "60m"),
    "1mo": Timeframe("1mo", _lazy("近1月"), "1M", "2mo", "1d"),
    "6mo": Timeframe("6mo", _lazy("近6月"), "6M", "1y", "1d"),
}

DEFAULT_TIMEFRAME = TIMEFRAMES["1mo"]
DETAIL_TIMEFRAMES: dict[str, DetailTimeframe] = {
    "1m": DetailTimeframe("1m", _lazy("1分钟"), "1m", "5d", "1m", None, 720),
    "2m": DetailTimeframe("2m", _lazy("2分钟"), "2m", "5d", "1m", "2min", 720),
    "3m": DetailTimeframe("3m", _lazy("3分钟"), "3m", "5d", "1m", "3min", 720),
    "5m": DetailTimeframe("5m", _lazy("5分钟"), "5m", "5d", "5m", None, 900),
    "10m": DetailTimeframe("10m", _lazy("10分钟"), "10m", "5d", "5m", "10min", 900),
    "15m": DetailTimeframe("15m", _lazy("15分钟"), "15m", "5d", "15m", None, 900),
    "30m": DetailTimeframe("30m", _lazy("30分钟"), "30m", "10d", "30m", None, 900),
    "45m": DetailTimeframe("45m", _lazy("45分钟"), "45m", "10d", "15m", "45min", 900),
    "1h": DetailTimeframe("1h", _lazy("1小时"), "1H", "1mo", "1h", None, 900),
    "2h": DetailTimeframe("2h", _lazy("2小时"), "2H", "1mo", "1h", "2h", 900),
    "4h": DetailTimeframe("4h", _lazy("4小时"), "4H", "3mo", "1h", "4h", 900),
    "1d": DetailTimeframe("1d", _lazy("1日"), "1D", "1d", "1m", None, 720),
    "5d": DetailTimeframe("5d", _lazy("5日"), "5D", "5d", "30m", None, 720),
    "1mo": DetailTimeframe("1mo", _lazy("1月"), "1M", "1mo", "1d", None, 400),
    "6mo": DetailTimeframe("6mo", _lazy("6月"), "6M", "6mo", "1d", None, 400),
}
DEFAULT_DETAIL_TIMEFRAME = DETAIL_TIMEFRAMES["1d"]
TOP_SYMBOLS = screener.CORE_TICKERS_US[:80]
MAX_SERIES_POINTS = 60
WINDOW_LENGTHS = {
    "1d": 26,
    "5d": 50,
    "1mo": 30,
    "6mo": 130,
}
_QUERY_SANITIZER = re.compile(r"[^A-Z0-9\.\-]")
MARKET_REQUEST_TIMEOUT = max(5, getattr(settings, "MARKET_DATA_TIMEOUT_SECONDS", 25))
MARKET_MAX_WORKERS = max(1, getattr(settings, "MARKET_DATA_MAX_WORKERS", 20))
MARKET_RATE_WINDOW = max(10, getattr(settings, "MARKET_DATA_RATE_WINDOW_SECONDS", 90))
MARKET_RATE_MAX_CALLS = max(1, getattr(settings, "MARKET_DATA_RATE_MAX_CALLS", 45))
MARKET_RATE_CACHE_ALIAS = getattr(settings, "MARKET_DATA_RATE_CACHE_ALIAS", "default")
MARKET_PROFILE_CACHE_TTL = max(120, getattr(settings, "MARKET_PROFILE_CACHE_TTL", 900))
MARKET_NEWS_CACHE_TTL = max(120, getattr(settings, "MARKET_NEWS_CACHE_TTL", 300))
MARKET_ASSETS_CACHE_TTL = max(300, getattr(settings, "MARKET_ASSETS_CACHE_TTL", 6 * 3600))
MARKET_RANKINGS_CACHE_TTL = max(30, getattr(settings, "MARKET_RANKINGS_CACHE_TTL", 55))
MARKET_ASSETS_PAGE_DEFAULT = max(20, getattr(settings, "MARKET_ASSETS_PAGE_DEFAULT", 50))
MARKET_ASSETS_PAGE_MAX = max(50, getattr(settings, "MARKET_ASSETS_PAGE_MAX", 200))
_MARKET_EXECUTOR = ThreadPoolExecutor(max_workers=MARKET_MAX_WORKERS)


@dataclass(slots=True)
class MarketQueryParams:
    timeframe: Timeframe
    timeframe_clamped: bool
    query: str
    limit: int
    limit_clamped: bool


def _normalize_query(value: str | None) -> str:
    if not value:
        return ""
    upper = value.upper()
    sanitized = _QUERY_SANITIZER.sub("", upper)
    return sanitized[:16]


def _build_suggestions(prefix: str) -> list[str]:
    if not prefix:
        return TOP_SYMBOLS[:6]
    prefix = prefix.upper()
    primary = [sym for sym in TOP_SYMBOLS if sym.startswith(prefix)]
    secondary = [sym for sym in TOP_SYMBOLS if prefix in sym and sym not in primary]
    combined = primary + secondary
    return combined[:8] if combined else TOP_SYMBOLS[:5]


def _resolve_market_params(params: Mapping[str, object]) -> MarketQueryParams:
    timeframe_key_raw = params.get("timeframe", DEFAULT_TIMEFRAME.key)
    timeframe_key = timeframe_key_raw if isinstance(timeframe_key_raw, str) else str(timeframe_key_raw)
    timeframe = TIMEFRAMES.get(timeframe_key, DEFAULT_TIMEFRAME)
    timeframe_clamped = timeframe_key not in TIMEFRAMES

    query_raw = params.get("query")
    query = _normalize_query(query_raw if isinstance(query_raw, str) else str(query_raw or ""))
    if len(query) > 16:
        query = query[:16]

    requested_limit_raw = params.get("limit", "100") or 100
    try:
        requested_limit = int(requested_limit_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        requested_limit = 8
    limit = max(20, min(200, requested_limit))
    limit_clamped = limit != requested_limit

    return MarketQueryParams(
        timeframe=timeframe,
        timeframe_clamped=timeframe_clamped,
        query=query,
        limit=limit,
        limit_clamped=limit_clamped,
    )


def _resolve_list_type(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"gainers", "losers", "most_active"}:
        return text
    return "gainers"


def _resolve_detail_timeframe(value: object) -> DetailTimeframe:
    text = str(value or "").strip().lower()
    return DETAIL_TIMEFRAMES.get(text, DEFAULT_DETAIL_TIMEFRAME)


def _parse_json_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, dict)]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        if isinstance(parsed, list):
            return [entry for entry in parsed if isinstance(entry, dict)]
    return []


def _coerce_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _normalize_asset_search(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text[:40]


def _normalize_asset_letter(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text or text == "ALL":
        return "ALL"
    if len(text) == 1 and text.isalpha():
        return text
    return "ALL"


def _load_assets_master(user_id: str | None) -> list[dict[str, object]]:
    cache_key = build_cache_key("market-assets-master", user_id or "anon")

    def _load() -> list[dict[str, object]]:
        payload = read_state("assets_master.json", default={})
        assets = payload.get("assets") if isinstance(payload, dict) else None
        updated_at = payload.get("updated_at") if isinstance(payload, dict) else None
        now = time.time()
        is_fresh = isinstance(updated_at, (int, float)) and (now - float(updated_at) < MARKET_ASSETS_CACHE_TTL)
        if isinstance(assets, list) and assets and is_fresh:
            return assets

        fresh_assets = fetch_alpaca_assets(user_id=user_id, status="active", asset_class="us_equity") or []
        if fresh_assets:
            write_state(
                "assets_master.json",
                {
                    "updated_at": now,
                    "count": len(fresh_assets),
                    "assets": fresh_assets,
                    "filters": {"status": "active", "asset_class": "us_equity"},
                },
            )
            return fresh_assets
        if isinstance(assets, list) and assets:
            return assets
        return []

    result = cache_memoize(cache_key, _load, min(300, MARKET_ASSETS_CACHE_TTL))
    return result if isinstance(result, list) else []


def _normalize_assets(assets: list[dict[str, object]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        symbol = str(asset.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        status = str(asset.get("status") or "").strip().lower()
        if status and status != "active":
            continue
        asset_class = str(asset.get("class") or asset.get("asset_class") or "").strip().lower()
        if asset_class and asset_class != "us_equity":
            continue
        tradable = asset.get("tradable")
        if tradable is False:
            continue
        cleaned.append(
            {
                "symbol": symbol,
                "name": str(asset.get("name") or asset.get("asset_name") or "").strip(),
                "exchange": str(asset.get("exchange") or asset.get("exchange_code") or "").strip(),
            }
        )
    cleaned.sort(key=lambda item: item["symbol"])
    return cleaned


def _build_asset_meta_map(
    *,
    user_id: str | None,
    symbols: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    assets = _normalize_assets(_load_assets_master(user_id))
    if not assets:
        return {}
    wanted = {sym.upper() for sym in symbols} if symbols else None
    mapping: dict[str, dict[str, str]] = {}
    for asset in assets:
        symbol = asset.get("symbol") or ""
        if not symbol:
            continue
        if wanted is not None and symbol not in wanted:
            continue
        mapping[symbol] = asset
    return mapping


def _attach_asset_meta(
    items: list[dict[str, object]],
    *,
    meta_map: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    if not items or not meta_map:
        return items
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").upper()
        if not symbol:
            continue
        meta = meta_map.get(symbol)
        if not meta:
            continue
        item.setdefault("name", meta.get("name") or "")
        item.setdefault("exchange", meta.get("exchange") or "")
    return items


def _build_snapshot_rankings(user_id: str | None) -> list[dict[str, object]]:
    cache_key = build_cache_key("market-rankings-snapshots", user_id or "anon")

    def _load() -> list[dict[str, object]] | None:
        assets = _normalize_assets(_load_assets_master(user_id))
        if not assets:
            return None
        symbols = [asset["symbol"] for asset in assets if asset.get("symbol")]
        if not symbols:
            return None
        snapshots = fetch_stock_snapshots(symbols, feed="sip", user_id=user_id, timeout=MARKET_REQUEST_TIMEOUT)
        if not isinstance(snapshots, dict) or not snapshots:
            return None
        rows: list[dict[str, object]] = []
        for asset in assets:
            symbol = asset.get("symbol") or ""
            if not symbol:
                continue
            snapshot = snapshots.get(symbol) if isinstance(snapshots, dict) else None
            if not isinstance(snapshot, dict):
                continue
            latest_trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or {}
            latest_quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or {}
            daily_bar = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
            minute_bar = snapshot.get("minuteBar") or snapshot.get("minute_bar") or {}
            prev_bar = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar") or {}

            last_price = _coerce_number(
                (latest_trade or {}).get("p")
                or (minute_bar or {}).get("c")
                or (daily_bar or {}).get("c")
                or (latest_quote or {}).get("ap")
                or (latest_quote or {}).get("bp")
            )
            prev_close = _coerce_number((prev_bar or {}).get("c"))
            if last_price is None or prev_close in (None, 0):
                continue
            try:
                change_pct = (last_price / prev_close - 1.0) * 100.0
            except Exception:
                continue
            volume = _coerce_number((daily_bar or {}).get("v") or (minute_bar or {}).get("v"))

            rows.append(
                {
                    "symbol": symbol,
                    "name": asset.get("name") or "",
                    "exchange": asset.get("exchange") or "",
                    "price": last_price,
                    "change_pct_day": change_pct,
                    "change_pct_period": change_pct,
                    "volume": volume,
                }
            )
        return rows or None

    result = cache_memoize(cache_key, _load, MARKET_RANKINGS_CACHE_TTL)
    return result if isinstance(result, list) else []


def _load_universe_ranked() -> list[dict[str, object]]:
    payload = read_state("universe_ranked.json", default={})
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    rows: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        symbol = str(entry.get("symbol") or "").upper()
        if not symbol:
            continue
        price = _coerce_number(entry.get("price"))
        change_pct = _coerce_number(entry.get("change_pct"))
        volume = _coerce_number(entry.get("volume"))
        if price is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "price": price,
                "change_pct_day": change_pct,
                "change_pct_period": change_pct,
                "volume": volume,
            }
        )
    return rows


def _filter_assets(
    assets: list[dict[str, str]],
    *,
    letter: str,
    query: str,
) -> list[dict[str, str]]:
    filtered = assets
    if letter and letter != "ALL":
        filtered = [asset for asset in filtered if asset.get("symbol", "").startswith(letter)]
    if query:
        upper_query = query.upper()
        lower_query = query.lower()
        filtered = [
            asset
            for asset in filtered
            if upper_query in asset.get("symbol", "")
            or lower_query in asset.get("name", "").lower()
        ]
    return filtered


def _apply_screener_filters(rows: list[dict[str, object]], filters: list[dict[str, object]]) -> list[dict[str, object]]:
    if not filters:
        return rows
    allowed_fields = {"ticker", "name", "price", "change_pct", "volume", "sector", "industry", "currency"}
    filtered: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        matches = True
        for flt in filters:
            field = str(flt.get("field") or "").strip()
            if field not in allowed_fields:
                continue
            op = str(flt.get("type") or flt.get("operator") or "like").strip().lower()
            target = flt.get("value")
            value = row.get(field)

            if op in {"like", "contains"}:
                hay = str(value or "").lower()
                needle = str(target or "").lower()
                if needle and needle not in hay:
                    matches = False
                    break
                continue

            if op in {"=", "==", "eq"}:
                if str(value).lower() != str(target).lower():
                    matches = False
                    break
                continue

            if op in {"!=", "neq"}:
                if str(value).lower() == str(target).lower():
                    matches = False
                    break
                continue

            left = _coerce_number(value)
            right = _coerce_number(target)
            if left is None or right is None:
                matches = False
                break
            if op in {">", "gt"} and not (left > right):
                matches = False
                break
            if op in {"<", "lt"} and not (left < right):
                matches = False
                break
            if op in {">=", "gte"} and not (left >= right):
                matches = False
                break
            if op in {"<=", "lte"} and not (left <= right):
                matches = False
                break
        if matches:
            filtered.append(row)
    return filtered


def _apply_screener_sort(rows: list[dict[str, object]], sorters: list[dict[str, object]]) -> list[dict[str, object]]:
    if not sorters:
        return rows
    allowed_fields = {"ticker", "name", "price", "change_pct", "volume", "sector", "industry", "currency"}
    sorters = [s for s in sorters if str(s.get("field") or "") in allowed_fields]
    if not sorters:
        return rows

    def sort_key(item: dict[str, object]) -> tuple:
        key_parts: list[object] = []
        for sorter in sorters:
            field = str(sorter.get("field") or "")
            value = item.get(field)
            numeric = _coerce_number(value)
            key_parts.append(numeric if numeric is not None else str(value or "").lower())
        return tuple(key_parts)

    reverse = str(sorters[0].get("dir") or "asc").lower() == "desc"
    return sorted(rows, key=sort_key, reverse=reverse)


def _format_compact_number(value: object) -> str | None:
    num = _coerce_number(value)
    if num is None:
        return None
    abs_val = abs(num)
    if abs_val >= 1e12:
        return f"{num / 1e12:.2f}T"
    if abs_val >= 1e9:
        return f"{num / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{num / 1e6:.2f}M"
    if abs_val >= 1e3:
        return f"{num / 1e3:.2f}K"
    return f"{num:.0f}"


def _format_percent(value: object) -> str | None:
    num = _coerce_number(value)
    if num is None:
        return None
    return f"{num * 100:.2f}%" if abs(num) <= 2 else f"{num:.2f}%"


def _fetch_company_profile(symbol: str, *, user_id: str | None = None) -> dict[str, object]:
    if not symbol:
        return {}
    if yf is None:
        return {}

    cache_key = build_cache_key("market-profile", symbol.upper())

    def _download() -> dict[str, object]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
        except Exception:
            info = {}
        if not info:
            return {}
        return {
            "name": info.get("longName") or info.get("shortName") or symbol,
            "summary": info.get("longBusinessSummary") or "",
            "sector": info.get("sector") or "",
            "industry": info.get("industry") or "",
            "market_cap": _format_compact_number(info.get("marketCap")),
            "pe": _coerce_number(info.get("trailingPE") or info.get("forwardPE")),
            "pb": _coerce_number(info.get("priceToBook")),
            "beta": _coerce_number(info.get("beta")),
            "employees": info.get("fullTimeEmployees"),
            "dividend_yield": _format_percent(info.get("dividendYield")),
        }

    result = cache_memoize(cache_key, _download, MARKET_PROFILE_CACHE_TTL)
    return result if isinstance(result, dict) else {}


def _format_news_time(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            return ""
    text = str(value).strip()
    if not text:
        return ""
    if len(text) >= 10:
        return text[:10]
    return text


def _normalize_news_items(items: list[dict[str, object]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        title = entry.get("headline") or entry.get("title") or entry.get("summary") or ""
        url = entry.get("url") or entry.get("link") or entry.get("article_url") or ""
        source = entry.get("source") or entry.get("publisher") or entry.get("author") or ""
        raw_time = entry.get("created_at") or entry.get("createdAt") or entry.get("time") or entry.get("published_at")
        snippet = entry.get("summary") or entry.get("description") or ""
        normalized.append(
            {
                "title": str(title).strip(),
                "url": str(url).strip(),
                "source": str(source).strip(),
                "time": _format_news_time(raw_time),
                "summary": str(snippet).strip(),
            }
        )
    return normalized


def _fetch_symbol_news(symbol: str, *, user_id: str | None = None, limit: int = 6) -> list[dict[str, str]]:
    if not symbol:
        return []
    cache_key = build_cache_key("market-news", symbol.upper())

    def _download() -> list[dict[str, str]]:
        items = fetch_news(symbols=[symbol], limit=limit, user_id=user_id)
        if not items and yf is not None:
            try:
                ticker = yf.Ticker(symbol)
                items = ticker.news or []
            except Exception:
                items = []
        if not isinstance(items, list):
            return []
        return _normalize_news_items(items)[:limit]

    result = cache_memoize(cache_key, _download, MARKET_NEWS_CACHE_TTL)
    return result if isinstance(result, list) else []


def _extract_ohlc(frame: pd.DataFrame, symbol: str, *, limit: int = 360) -> list[dict[str, float | int]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    sym = symbol.upper()

    def _pull_field(field: str) -> pd.Series | None:
        columns = frame.columns
        if isinstance(columns, pd.MultiIndex):
            try:
                if field in columns.get_level_values(0):
                    series = frame.xs(field, level=0, axis=1)
                    if sym in series.columns:
                        return series[sym]
            except Exception:
                pass
            try:
                if field in columns.get_level_values(1):
                    series = frame.xs(field, level=1, axis=1)
                    if sym in series.columns:
                        return series[sym]
            except Exception:
                pass
            return None
        if field in frame.columns:
            return frame[field]
        if sym in frame.columns:
            sub = frame[sym]
            if isinstance(sub, pd.DataFrame) and field in sub.columns:
                return sub[field]
        return None

    open_series = _pull_field("Open")
    high_series = _pull_field("High")
    low_series = _pull_field("Low")
    close_series = _pull_field("Close")
    if open_series is None or high_series is None or low_series is None or close_series is None:
        return []

    ohlc_frame = pd.DataFrame(
        {
            "open": open_series,
            "high": high_series,
            "low": low_series,
            "close": close_series,
        }
    ).dropna()
    if ohlc_frame.empty:
        return []

    ohlc_frame = ohlc_frame.sort_index().tail(limit)
    bars: list[dict[str, float | int]] = []
    for ts, row in ohlc_frame.iterrows():
        try:
            stamp = pd.Timestamp(ts)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize(timezone.utc)
            else:
                stamp = stamp.tz_convert(timezone.utc)
            time_val = int(stamp.timestamp())
        except Exception:
            continue
        try:
            bars.append(
                {
                    "time": time_val,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
            )
        except Exception:
            continue
    return bars


def _resample_ohlc_frame(frame: pd.DataFrame, symbol: str, rule: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    sym = symbol.upper()

    def _pull_field(field: str) -> pd.Series | None:
        columns = frame.columns
        if isinstance(columns, pd.MultiIndex):
            try:
                if field in columns.get_level_values(0):
                    series = frame.xs(field, level=0, axis=1)
                    if sym in series.columns:
                        return series[sym]
            except Exception:
                pass
            try:
                if field in columns.get_level_values(1):
                    series = frame.xs(field, level=1, axis=1)
                    if sym in series.columns:
                        return series[sym]
            except Exception:
                pass
            return None
        if field in frame.columns:
            return frame[field]
        if sym in frame.columns:
            sub = frame[sym]
            if isinstance(sub, pd.DataFrame) and field in sub.columns:
                return sub[field]
        return None

    open_series = _pull_field("Open")
    high_series = _pull_field("High")
    low_series = _pull_field("Low")
    close_series = _pull_field("Close")
    if open_series is None or high_series is None or low_series is None or close_series is None:
        return pd.DataFrame()

    ohlc_frame = pd.DataFrame(
        {
            "open": open_series,
            "high": high_series,
            "low": low_series,
            "close": close_series,
        }
    ).dropna()
    if ohlc_frame.empty:
        return pd.DataFrame()

    if not isinstance(ohlc_frame.index, pd.DatetimeIndex):
        ohlc_frame.index = pd.to_datetime(ohlc_frame.index, errors="coerce")
        ohlc_frame = ohlc_frame.dropna()
    if ohlc_frame.empty:
        return pd.DataFrame()
    ohlc_frame = ohlc_frame.sort_index()

    resampled = (
        ohlc_frame.resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    if resampled.empty:
        return pd.DataFrame()
    resampled = resampled.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
        }
    )
    return resampled


def _extract_close_series(frame: pd.DataFrame, *, limit: int = 20) -> list[float]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    series = None
    if isinstance(frame.columns, pd.MultiIndex):
        for field in ("Close", "close", "Adj Close", "adj_close", "c"):
            if field in frame.columns.get_level_values(0):
                subset = frame.xs(field, level=0, axis=1)
                if isinstance(subset, pd.DataFrame) and not subset.empty:
                    series = subset.iloc[:, 0]
                else:
                    series = subset
                break
    else:
        for field in ("Close", "close", "Adj Close", "adj_close", "c"):
            if field in frame.columns:
                series = frame[field]
                break
    if series is None:
        return []
    try:
        series = series.dropna().tail(limit)
    except Exception:
        return []
    values: list[float] = []
    for value in series:
        try:
            num = float(value)
        except Exception:
            continue
        if pd.isna(num):
            continue
        values.append(num)
    return values


@login_required
@ensure_csrf_cookie
def market_insights(request: HttpRequest) -> HttpResponse:
    lang_prefix = (getattr(request, "LANGUAGE_CODE", "") or request.COOKIES.get("django_language", "zh-hans")).lower()[:2]
    return render(
        request,
        "trading/market_insights.html",
        {
            "timeframes": TIMEFRAMES,
            "default_timeframe": DEFAULT_TIMEFRAME,
            "lang_prefix": lang_prefix,
        },
    )


@login_required
@require_http_methods(["GET", "POST"])
def market_insights_data(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    key_id, secret = resolve_alpaca_credentials(user=request.user)
    if yf is None and not (key_id and secret):
        record_metric(
            "market.insights.error",
            request_id=request_id,
            user_id=request.user.id,
            error="missing_data_source",
        )
        return JsonResponse(
            {"error": _("当前环境缺少可用的市场数据源，无法加载市场数据。"), "request_id": request_id},
            status=503,
        )

    if request.method == "POST":
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            return JsonResponse(
                {"error": _("请求体解析失败。"), "request_id": request_id},
                status=400,
            )
        params = payload if isinstance(payload, dict) else {}
    else:
        raw = request.GET
        params = raw.dict() if hasattr(raw, "dict") else dict(raw)

    resolved = _resolve_market_params(params)
    list_type = _resolve_list_type(params.get("list") or params.get("list_type"))
    detail_mode = str(params.get("detail") or "").strip().lower() in {"1", "true", "yes"}
    detail_symbol = _normalize_query(params.get("symbol") or params.get("detail_symbol"))
    detail_timeframe = _resolve_detail_timeframe(params.get("range") or params.get("detail_timeframe"))
    subscribe_symbol = _normalize_query(params.get("subscribe") or params.get("subscribe_symbol"))
    subscribe_symbols_raw = params.get("subscribe_symbols") or params.get("subscribe_list") or ""
    quote_mode = str(params.get("quote") or params.get("snapshot") or "").strip().lower() in {"1", "true", "yes"}
    subscribe_only = str(params.get("subscribe_only") or "").strip().lower() in {"1", "true", "yes"}

    if subscribe_symbol:
        try:
            request_market_symbol(subscribe_symbol, user_id=str(request.user.id))
        except Exception:
            pass
        if subscribe_only:
            return JsonResponse(
                {"subscribed": subscribe_symbol, "request_id": request_id},
                json_dumps_params={"ensure_ascii": False},
            )
    if subscribe_symbols_raw:
        if isinstance(subscribe_symbols_raw, str):
            symbols_list = [
                _normalize_query(item)
                for item in subscribe_symbols_raw.split(",")
                if isinstance(item, str)
            ]
        elif isinstance(subscribe_symbols_raw, list):
            symbols_list = [_normalize_query(item) for item in subscribe_symbols_raw]
        else:
            symbols_list = []
        symbols_list = [sym for sym in symbols_list if sym]
        if symbols_list:
            try:
                request_market_symbols(symbols_list, user_id=str(request.user.id))
            except Exception:
                pass
            if subscribe_only:
                return JsonResponse(
                    {"subscribed": symbols_list, "request_id": request_id},
                    json_dumps_params={"ensure_ascii": False},
                )

    if quote_mode:
        if not detail_symbol:
            return JsonResponse(
                {"error": _("缺少股票代码。"), "request_id": request_id},
                status=400,
                json_dumps_params={"ensure_ascii": False},
            )
        snapshots = fetch_stock_snapshots(
            [detail_symbol],
            feed="sip",
            user_id=str(request.user.id),
            timeout=MARKET_REQUEST_TIMEOUT,
        )
        snapshot = snapshots.get(detail_symbol) if isinstance(snapshots, dict) else None
        if not isinstance(snapshot, dict):
            return JsonResponse(
                {"error": _("未能获取 %(symbol)s 的行情快照。") % {"symbol": detail_symbol}, "request_id": request_id},
                status=404,
                json_dumps_params={"ensure_ascii": False},
            )
        latest_trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or {}
        latest_quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or {}
        daily_bar = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
        minute_bar = snapshot.get("minuteBar") or snapshot.get("minute_bar") or {}
        prev_bar = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar") or {}

        last_price = _coerce_number(
            (latest_trade or {}).get("p")
            or (minute_bar or {}).get("c")
            or (daily_bar or {}).get("c")
            or (latest_quote or {}).get("ap")
            or (latest_quote or {}).get("bp")
        )
        prev_close = _coerce_number((prev_bar or {}).get("c"))
        if last_price is None:
            return JsonResponse(
                {"error": _("未能获取 %(symbol)s 的行情快照。") % {"symbol": detail_symbol}, "request_id": request_id},
                status=404,
                json_dumps_params={"ensure_ascii": False},
            )
        change_pct = None
        if prev_close not in (None, 0):
            try:
                change_pct = (last_price / prev_close - 1.0) * 100.0
            except Exception:
                change_pct = None
        return JsonResponse(
            {
                "symbol": detail_symbol,
                "price": last_price,
                "change_pct": change_pct,
                "data_source": "alpaca",
                "server_ts": time.time(),
                "request_id": request_id,
            },
            json_dumps_params={"ensure_ascii": False},
        )

    rate_state = check_rate_limit(
        cache_alias=MARKET_RATE_CACHE_ALIAS,
        key=f"market-insights:{rate_limit_key(request)}",
        window=MARKET_RATE_WINDOW,
        max_calls=MARKET_RATE_MAX_CALLS,
    )
    if rate_state.limited:
        record_metric(
            "market.insights.rate_limited",
            request_id=request_id,
            user_id=request.user.id,
            retry_after=rate_state.retry_after,
        )
        return JsonResponse(
            {
                "error": _("请求过于频繁，请稍后再试。"),
                "rate_limited": True,
                "retry_after_seconds": rate_state.retry_after,
                "request_id": request_id,
            },
            status=429,
        )

    if detail_mode:
        if not detail_symbol:
            return JsonResponse(
                {"error": _("缺少股票代码。"), "request_id": request_id},
                status=400,
                json_dumps_params={"ensure_ascii": False},
            )
        fallback_map = {
            "1m": ("2m", "3m", "5m", "10m", "15m", "30m", "1h", "1d"),
            "2m": ("3m", "5m", "10m", "15m", "30m", "1h", "1d"),
            "3m": ("5m", "10m", "15m", "30m", "1h", "1d"),
            "5m": ("10m", "15m", "30m", "1h", "1d"),
            "10m": ("15m", "30m", "1h", "1d"),
            "15m": ("30m", "1h", "1d"),
            "30m": ("1h", "1d"),
            "45m": ("1h", "1d"),
            "1h": ("2h", "4h", "1d"),
            "2h": ("4h", "1d"),
            "4h": ("1d",),
            "1d": ("5d", "1mo"),
            "5d": ("1mo", "6mo"),
            "1mo": ("6mo",),
            "6mo": (),
        }
        candidates = [detail_timeframe]
        for key in fallback_map.get(detail_timeframe.key, ()):
            candidates.append(DETAIL_TIMEFRAMES[key])

        effective_timeframe = detail_timeframe
        frame = None
        bars: list[dict[str, float | int]] = []
        for candidate in candidates:
            frame = market_data.fetch(
                [detail_symbol],
                period=candidate.period,
                interval=candidate.interval,
                cache=True,
                timeout=MARKET_REQUEST_TIMEOUT,
                ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
                cache_alias=getattr(settings, "MARKET_HISTORY_CACHE_ALIAS", None),
                user_id=str(request.user.id),
            )
            bar_frame = frame
            if candidate.resample:
                bar_frame = _resample_ohlc_frame(frame, detail_symbol, candidate.resample)
            bars = _extract_ohlc(bar_frame, detail_symbol, limit=candidate.limit)
            if bars:
                effective_timeframe = candidate
                break

        if not bars:
            return JsonResponse(
                {"error": _("未能获取 %(symbol)s 的行情数据。") % {"symbol": detail_symbol}, "request_id": request_id},
                status=404,
                json_dumps_params={"ensure_ascii": False},
            )

        profile_payload = _fetch_company_profile(detail_symbol, user_id=str(request.user.id))
        news_payload = _fetch_symbol_news(detail_symbol, user_id=str(request.user.id))
        return JsonResponse(
            {
                "symbol": detail_symbol,
                "timeframe": {
                    "key": effective_timeframe.key,
                    "label": effective_timeframe.label,
                    "label_en": effective_timeframe.label_en,
                },
                "requested_timeframe": {
                    "key": detail_timeframe.key,
                    "label": detail_timeframe.label,
                    "label_en": detail_timeframe.label_en,
                },
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "data_source": _infer_market_source(frame),
                "bars": bars,
                "profile": profile_payload,
                "news": news_payload,
                "request_id": request_id,
            },
            json_dumps_params={"ensure_ascii": False},
        )

    session = request.session
    recent_queries: list[str] = list(session.get("market_recent_queries", []))
    profile = None
    watchlist: list[str] = []
    if request.user.is_authenticated:
        profile, _created = UserProfile.objects.get_or_create(user=request.user)
        watchlist = list(profile.market_watchlist or [])
    else:
        watchlist = list(session.get("market_watchlist", []))
    normalized_watch = []
    for item in watchlist:
        normalized = _normalize_query(str(item))
        if normalized and normalized not in normalized_watch:
            normalized_watch.append(normalized)
    watchlist = normalized_watch

    mutations_enabled = request.method == "POST"
    recent_action = (params.get("recent") or "").lower() if mutations_enabled else ""
    recent_target = _normalize_query(params.get("recent_target")) if mutations_enabled else ""
    watch_action = (params.get("watch") or "").lower() if mutations_enabled else ""

    if mutations_enabled:
        session_changed = False
        if recent_action == "clear":
            recent_queries = []
            session_changed = True
        elif recent_action == "delete" and recent_target:
            recent_queries = [q for q in recent_queries if q != recent_target]
            session_changed = True

        if resolved.query:
            updated = [q for q in recent_queries if q != resolved.query]
            updated.insert(0, resolved.query)
            recent_queries = updated[:6]
            session_changed = True

        if resolved.query and watch_action in {"add", "remove"}:
            updated_watch = [q for q in watchlist if q != resolved.query]
            if watch_action == "add":
                updated_watch.insert(0, resolved.query)
            watchlist = updated_watch[:10]
            if profile:
                profile.market_watchlist = watchlist
                profile.save(update_fields=["market_watchlist", "updated_at"])
            else:
                session["market_watchlist"] = watchlist
                session_changed = True

        if session_changed:
            session["market_recent_queries"] = recent_queries
            session.modified = True

    symbols: list[str] = []
    restrict_to_query = False
    if resolved.query:
        symbols = [resolved.query]
        restrict_to_query = True

    list_items: list[dict[str, object]] = []
    active_list_type = list_type
    data_source = "unknown"
    gainers: list[dict[str, object]] = []
    losers: list[dict[str, object]] = []
    most_actives: list[dict[str, object]] = []

    if restrict_to_query:
        future = None
        try:
            with track_latency(
                "market.insights.fetch",
                user_id=request.user.id,
                request_id=request_id,
                timeframe=resolved.timeframe.key,
                restrict=restrict_to_query,
            ):
                future = _MARKET_EXECUTOR.submit(
                    _fetch_history,
                    symbols,
                    resolved.timeframe,
                    user_id=str(request.user.id),
                )
                series_map, data_source = future.result(timeout=MARKET_REQUEST_TIMEOUT)
        except FuturesTimeout:
            if future:
                future.cancel()
            record_metric(
                "market.insights.error",
                request_id=request_id,
                user_id=request.user.id,
                error="timeout",
            )
            return JsonResponse(
                {"error": _("市场数据请求超时，请稍后再试。"), "request_id": request_id},
                status=504,
            )
        if not series_map:
            record_metric(
                "market.insights.error",
                request_id=request_id,
                user_id=request.user.id,
                error="empty_series",
            )
            return JsonResponse(
                {"error": _("未能获取市场数据，请稍后再试。"), "request_id": request_id},
                status=502,
                json_dumps_params={"ensure_ascii": False},
            )

        ranked = _rank_symbols(series_map, resolved.timeframe, limit=resolved.limit if restrict_to_query else 20)
        if restrict_to_query:
            payload = [entry for entry in ranked if entry["symbol"].upper() == resolved.query.upper()]
            if not payload:
                return JsonResponse(
                    {"error": _("未找到 %(symbol)s 的有效行情数据。") % {"symbol": resolved.query}, "request_id": request_id},
                    status=404,
                    json_dumps_params={"ensure_ascii": False},
                )
            gainers = payload if payload[0]["change_pct_period"] >= 0 else []
            losers = payload if payload and payload[0]["change_pct_period"] < 0 else []
            if gainers:
                list_items = gainers
                active_list_type = "gainers"
            else:
                list_items = losers
                active_list_type = "losers"
        else:
            gainers = [entry for entry in ranked if entry["change_pct_period"] >= 0][: resolved.limit]
            losers = [entry for entry in ranked if entry["change_pct_period"] < 0][: resolved.limit]
            list_items = losers if list_type == "losers" else gainers
    else:
        if list_type in {"gainers", "losers"}:
            movers = market_data.fetch_market_movers(
                limit=resolved.limit,
                user_id=str(request.user.id),
                timeout=MARKET_REQUEST_TIMEOUT,
            )
            gainers = movers.get("gainers", []) if movers else []
            losers = movers.get("losers", []) if movers else []
            list_items = losers if list_type == "losers" else gainers
            data_source = "alpaca" if list_items else "unknown"
        else:
            list_items = market_data.fetch_most_actives(
                by="volume",
                limit=resolved.limit,
                user_id=str(request.user.id),
                timeout=MARKET_REQUEST_TIMEOUT,
            )
            most_actives = list_items
            volume_label = _("成交量")
            for entry in list_items:
                if isinstance(entry, dict):
                    entry.setdefault("period_label", volume_label)
                    entry.setdefault("period_label_en", "Volume")
            data_source = "alpaca" if list_items else "unknown"

    suggestions = _build_suggestions(resolved.query)

    meta_symbols: set[str] = set()
    for collection in (list_items, gainers, losers):
        for item in collection:
            if isinstance(item, dict) and item.get("symbol"):
                meta_symbols.add(str(item.get("symbol")).upper())
    meta_map = _build_asset_meta_map(user_id=str(request.user.id), symbols=meta_symbols)
    _attach_asset_meta(list_items, meta_map=meta_map)
    _attach_asset_meta(gainers, meta_map=meta_map)
    _attach_asset_meta(losers, meta_map=meta_map)

    response = {
        "timeframe": {
            "key": resolved.timeframe.key,
            "label": resolved.timeframe.label,
            "label_en": resolved.timeframe.label_en,
            "clamped": resolved.timeframe_clamped,
        },
        # Use timezone.utc for Python 3.13 compatibility (datetime.UTC removed)
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_source": data_source or "unknown",
        "query": resolved.query,
        "list_type": active_list_type,
        "items": list_items,
        "gainers": gainers,
        "losers": losers,
        "most_actives": most_actives,
        "limit_clamped": resolved.limit_clamped,
        "request_id": request_id,
        "suggestions": suggestions,
        "recent_queries": recent_queries,
        "watchlist": watchlist,
    }
    record_metric(
        "market.insights.response",
        request_id=request_id,
        user_id=request.user.id,
        query=bool(resolved.query),
        timeframe=resolved.timeframe.key,
        list_type=active_list_type,
        items=len(list_items),
        gainers=len(gainers),
        losers=len(losers),
    )
    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})


@login_required
@require_http_methods(["GET"])
def market_assets_data(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    user_id = str(request.user.id)
    letter = _normalize_asset_letter(request.GET.get("letter"))
    query = _normalize_asset_search(request.GET.get("query"))

    try:
        page = int(request.GET.get("page") or 1)
    except (TypeError, ValueError):
        page = 1
    page = max(1, page)

    try:
        size = int(request.GET.get("size") or MARKET_ASSETS_PAGE_DEFAULT)
    except (TypeError, ValueError):
        size = MARKET_ASSETS_PAGE_DEFAULT
    size = max(20, min(MARKET_ASSETS_PAGE_MAX, size))

    assets = _normalize_assets(_load_assets_master(user_id))
    filtered = _filter_assets(assets, letter=letter, query=query)
    total = len(filtered)
    last_page = max(1, (total + size - 1) // size)
    if page > last_page:
        page = last_page
    start = (page - 1) * size
    end = start + size
    page_assets = filtered[start:end]

    symbols = [asset["symbol"] for asset in page_assets if asset.get("symbol")]
    snapshots = (
        fetch_stock_snapshots(symbols, feed="sip", user_id=user_id, timeout=MARKET_REQUEST_TIMEOUT)
        if symbols
        else {}
    )

    items: list[dict[str, object]] = []
    for asset in page_assets:
        symbol = asset.get("symbol") or ""
        snapshot = snapshots.get(symbol, {}) if isinstance(snapshots, dict) else {}
        latest_trade = snapshot.get("latestTrade") if isinstance(snapshot, dict) else None
        latest_quote = snapshot.get("latestQuote") if isinstance(snapshot, dict) else None
        daily_bar = snapshot.get("dailyBar") if isinstance(snapshot, dict) else None
        minute_bar = snapshot.get("minuteBar") if isinstance(snapshot, dict) else None
        prev_bar = snapshot.get("prevDailyBar") if isinstance(snapshot, dict) else None

        last_price = _coerce_number(
            (latest_trade or {}).get("p")
            or (minute_bar or {}).get("c")
            or (daily_bar or {}).get("c")
            or (latest_quote or {}).get("ap")
            or (latest_quote or {}).get("bp")
        )
        prev_close = _coerce_number((prev_bar or {}).get("c"))
        change_pct = None
        if last_price is not None and prev_close not in (None, 0):
            try:
                change_pct = (last_price / prev_close - 1.0) * 100.0
            except Exception:
                change_pct = None

        items.append(
            {
                "symbol": symbol,
                "name": asset.get("name") or "",
                "exchange": asset.get("exchange") or "",
                "last": last_price,
                "change_pct": change_pct,
            }
        )

    return JsonResponse(
        {
            "items": items,
            "page": page,
            "size": size,
            "total": total,
            "total_pages": last_page,
            "letter": letter,
            "query": query,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "request_id": request_id,
        },
        json_dumps_params={"ensure_ascii": False},
    )


@login_required
@require_http_methods(["GET"])
def market_watchlist_snapshot(request: HttpRequest) -> JsonResponse:
    user_id = str(request.user.id)
    profile, _created = UserProfile.objects.get_or_create(user=request.user)
    watchlist = list(profile.market_watchlist or [])
    if not watchlist:
        watchlist = list(request.session.get("market_watchlist", []))
    normalized: list[str] = []
    for item in watchlist:
        symbol = _normalize_query(str(item))
        if symbol and symbol not in normalized:
            normalized.append(symbol)
    watchlist = normalized[:120]
    if not watchlist:
        return JsonResponse({"items": [], "count": 0}, json_dumps_params={"ensure_ascii": False})

    interval = str(request.GET.get("interval") or "1d").lower()
    try:
        limit = int(request.GET.get("limit") or 20)
    except (TypeError, ValueError):
        limit = 20
    limit = max(8, min(60, limit))

    frames = market_data.fetch_recent_window(watchlist, interval=interval, limit=limit, user_id=user_id)
    items: list[dict[str, object]] = []
    for symbol in watchlist:
        frame = frames.get(symbol)
        series = _extract_close_series(frame, limit=limit) if isinstance(frame, pd.DataFrame) else []
        price = series[-1] if series else None
        change_pct = None
        if len(series) >= 2 and series[-2] not in (0, None):
            try:
                change_pct = (series[-1] / series[-2] - 1.0) * 100.0
            except Exception:
                change_pct = None
        items.append(
            {
                "symbol": symbol,
                "price": price,
                "change_pct": change_pct,
                "series": series[-limit:],
            }
        )
    return JsonResponse(
        {"items": items, "count": len(items), "interval": interval, "limit": limit},
        json_dumps_params={"ensure_ascii": False},
    )


@login_required
@require_http_methods(["GET"])
def market_screener_data(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    params = request.GET
    try:
        page = int(params.get("page") or 1)
    except (TypeError, ValueError):
        page = 1
    try:
        size = int(params.get("size") or 25)
    except (TypeError, ValueError):
        size = 25

    page = max(1, page)
    size = max(10, min(100, size))
    search_text = str(params.get("search") or params.get("query") or "").strip()
    search_lower = search_text.lower()

    filters = _parse_json_list(params.get("filters"))
    sorters = _parse_json_list(params.get("sorters"))

    min_volume = _coerce_number(params.get("min_volume"))
    if min_volume is not None:
        filters.append({"field": "volume", "type": ">=", "value": min_volume})

    base = screener.fetch_page(offset=0, size=5000, user_id=str(request.user.id))
    rows = base.get("rows", [])
    if not isinstance(rows, list):
        rows = []

    if search_lower:
        rows = [
            row
            for row in rows
            if isinstance(row, dict)
            and (
                search_lower in str(row.get("ticker") or "").lower()
                or search_lower in str(row.get("name") or "").lower()
            )
        ]

    rows = _apply_screener_filters(rows, filters)
    rows = _apply_screener_sort(rows, sorters)

    total = len(rows)
    last_page = max(1, (total + size - 1) // size)
    if page > last_page:
        page = last_page
    start = (page - 1) * size
    end = start + size
    page_rows = rows[start:end]

    return JsonResponse(
        {
            "data": page_rows,
            "page": page,
            "size": size,
            "total": total,
            "last_page": last_page,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "request_id": request_id,
        },
        json_dumps_params={"ensure_ascii": False},
    )


def _download_history(
    symbols: Iterable[str],
    timeframe: Timeframe,
    *,
    user_id: str | None = None,
) -> tuple[dict[str, pd.Series], str]:
    unique = [sym for sym in dict.fromkeys(symbols) if sym]
    if not unique:
        return {}, "unknown"

    history: dict[str, pd.Series] = {}
    data = market_data.fetch(
        unique,
        period=timeframe.period,
        interval=timeframe.interval,
        cache=True,
        timeout=MARKET_REQUEST_TIMEOUT,
        ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
        cache_alias=getattr(settings, "MARKET_HISTORY_CACHE_ALIAS", None),
        user_id=user_id,
    )
    source = _infer_market_source(data)

    def _extract_panel(frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.DataFrame()
        for field in ("Adj Close", "Close"):
            for level in (0, 1):
                try:
                    panel = frame.xs(field, level=level, axis=1)
                except Exception:
                    continue
                if isinstance(panel, pd.Series):
                    panel = panel.to_frame()
                if isinstance(panel, pd.DataFrame) and not panel.empty:
                    return panel
        return pd.DataFrame()

    try:
        if isinstance(data.columns, pd.MultiIndex):
            panel = _extract_panel(data)
            if not panel.empty:
                for sym in unique:
                    try:
                        close = panel.get(sym)
                        if isinstance(close, pd.Series):
                            close = close.dropna()
                            if not close.empty:
                                history[sym] = close
                    except Exception:
                        continue
        else:
            close = None
            if isinstance(data, pd.DataFrame):
                if "Adj Close" in data.columns:
                    close = data.get("Adj Close")
                else:
                    close = data.get("Close")
            if isinstance(close, pd.Series):
                close = close.dropna()
                if not close.empty:
                    history[unique[0]] = close
    except Exception:
        pass
    return history, source


def _fetch_history(
    symbols: Iterable[str],
    timeframe: Timeframe,
    *,
    user_id: str | None = None,
) -> tuple[dict[str, pd.Series], str]:
    cache_key = build_cache_key("market-history", timeframe.key, sorted(symbols))
    result = cache_memoize(
        cache_key,
        lambda: _download_history(symbols, timeframe, user_id=user_id),
        getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
    )
    if isinstance(result, tuple) and len(result) == 2:
        series_map, source = result
        return series_map or {}, source or "unknown"
    if isinstance(result, dict):
        return result, "unknown"
    return {}, "unknown"


def _infer_market_source(data: pd.DataFrame | None) -> str:
    fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    if not isinstance(data, pd.DataFrame) or data.empty:
        return "unknown"
    columns = data.columns
    if isinstance(columns, pd.MultiIndex):
        level0 = set(columns.get_level_values(0))
        if level0 & fields:
            return "alpaca"
        level1 = set(columns.get_level_values(1))
        if level1 & fields:
            return "yfinance"
        return "unknown"
    if set(columns) & fields:
        return "yfinance"
    return "unknown"


def _rank_symbols(
    series_map: dict[str, pd.Series],
    timeframe: Timeframe,
    limit: int = 20,
) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for sym, series in series_map.items():
        window = _slice_series(series, timeframe)
        if len(window) < 2:
            continue
        start_price = float(window.iloc[0])
        end_price = float(window.iloc[-1])
        prev_price = float(window.iloc[-2])
        if not start_price or not prev_price:
            continue
        period_change = ((end_price / start_price) - 1.0) * 100.0
        day_change = ((end_price / prev_price) - 1.0) * 100.0

        normalized = _normalize_series(window)
        timestamps = [ts.strftime("%Y-%m-%d") for ts in window.index]

        ranked.append(
            {
                "symbol": sym,
                "price": round(end_price, 2),
                "change_pct_period": round(period_change, 2),
                "change_pct_day": round(day_change, 2),
                "period_label": timeframe.label,
                "period_label_en": timeframe.label_en,
                "series": normalized,
                "timestamps": timestamps,
            }
        )

    ranked.sort(key=lambda item: item["change_pct_period"], reverse=True)
    return ranked[: limit * 2]


def _slice_series(series: pd.Series, timeframe: Timeframe) -> pd.Series:
    cleaned = series.dropna()
    if cleaned.empty:
        return cleaned
    window = WINDOW_LENGTHS.get(timeframe.key, MAX_SERIES_POINTS)
    return cleaned.tail(min(window, MAX_SERIES_POINTS))


def _normalize_series(series: pd.Series) -> list[float]:
    values = series.astype(float).tolist()
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [0.5 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]
