from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import time
from typing import Callable, Iterable, Mapping
import re
from zoneinfo import ZoneInfo

import requests

import pandas as pd
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie

from django.utils.translation import gettext as _, gettext_lazy as _lazy

from .. import screener
from ..cache_utils import build_cache_key, cache_get_object, cache_set_object, cache_memoize
from .. import market_data
from ..observability import ensure_request_id, record_metric, track_latency
from ..network import get_requests_session, resolve_retry_config, retry_call_result
from ..rate_limit import check_rate_limit, rate_limit_key
from ..models import UserProfile, RealtimeProfile
from ..alpaca_data import (
    DEFAULT_FEED,
    resolve_alpaca_data_credentials,
    fetch_news,
    fetch_stock_snapshots,
    fetch_stock_trades,
)
from ..realtime.alpaca import fetch_assets as fetch_alpaca_assets
from ..realtime.market_stream import request_symbol as request_market_symbol, request_symbols as request_market_symbols
from ..realtime.lock import InstanceLock
from ..realtime.storage import read_state, write_state, resolve_state_dir
from ..realtime.manual_orders import submit_manual_order
from ..realtime.config import DEFAULT_CONFIG_NAME, load_realtime_config_from_payload
from ..realtime.schema import RealtimePayloadError, validate_realtime_payload
from ..market_aggregation import (
    aggregate_trades_to_tick_bars,
    aggregate_trades_to_time_bars,
)


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


@dataclass(slots=True)
class ChartInterval:
    key: str
    unit: str
    value: int
    label: str


TIMEFRAMES: dict[str, Timeframe] = {
    "1d": Timeframe("1d", _lazy("实时榜"), "Realtime", "5d", "15m"),
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


def _setting_int(name: str, default: int) -> int:
    try:
        return int(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default


def _setting_float(name: str, default: float) -> float:
    try:
        return float(getattr(settings, name, default))
    except (TypeError, ValueError):
        return float(default)


def _setting_bool(name: str, default: bool) -> bool:
    value = getattr(settings, name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _setting_list(value: object, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or list(default)
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or list(default)
    return list(default)


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

CHART_MAX_TICK_RANGE_SECONDS = _setting_int("MARKET_CHART_MAX_TICK_RANGE_SECONDS", 2 * 3600)
CHART_MAX_SECOND_RANGE_SECONDS = _setting_int("MARKET_CHART_MAX_SECOND_RANGE_SECONDS", 24 * 3600)
CHART_MAX_TICK_BARS = _setting_int("MARKET_CHART_MAX_TICK_BARS", 1000)
CHART_MAX_TIME_BARS = _setting_int("MARKET_CHART_MAX_TIME_BARS", 1600)
CHART_TRADES_PAGE_LIMIT = _setting_int("MARKET_CHART_TRADES_PAGE_LIMIT", 1000)
CHART_TRADES_MAX_PAGES = _setting_int("MARKET_CHART_TRADES_MAX_PAGES", 6)
MARKET_PROFILE_CACHE_TTL = max(120, getattr(settings, "MARKET_PROFILE_CACHE_TTL", 900))
MARKET_NEWS_CACHE_TTL = max(120, getattr(settings, "MARKET_NEWS_CACHE_TTL", 300))
MARKET_ASSETS_CACHE_TTL = max(300, getattr(settings, "MARKET_ASSETS_CACHE_TTL", 6 * 3600))
MARKET_RANKINGS_CACHE_TTL = max(30, getattr(settings, "MARKET_RANKINGS_CACHE_TTL", 55))
MARKET_RANKINGS_MIN_PRICE = max(0.0, _setting_float("MARKET_RANKINGS_MIN_PRICE", 1.0))
MARKET_RANKINGS_MIN_VOLUME = max(0, _setting_int("MARKET_RANKINGS_MIN_VOLUME", 50_000))
MARKET_RANKINGS_MIN_DOLLAR_VOLUME = max(0.0, _setting_float("MARKET_RANKINGS_MIN_DOLLAR_VOLUME", 2_000_000.0))
MARKET_RANKINGS_EXCLUDE_KEYWORDS = _setting_list(
    getattr(settings, "MARKET_RANKINGS_EXCLUDE_KEYWORDS", None),
    ["warrant", "warrants", "right", "rights", "unit", "units", "preferred", "depositary", "note", "notes"],
)
MARKET_RANKINGS_EXCLUDE_SUFFIXES = _setting_list(
    getattr(settings, "MARKET_RANKINGS_EXCLUDE_SUFFIXES", None),
    [".W", ".WS", ".WT", ".U", ".R", ".RT"],
)
MARKET_RANKINGS_EXCLUDE_EXCHANGES = set(
    item.upper()
    for item in _setting_list(getattr(settings, "MARKET_RANKINGS_EXCLUDE_EXCHANGES", None), ["OTC"])
)
MARKET_UNIVERSE_RANKINGS_CACHE_TTL = max(120, _setting_int("MARKET_UNIVERSE_RANKINGS_CACHE_TTL", 900))
MARKET_UNIVERSE_CHUNK_SIZE = max(50, min(400, _setting_int("MARKET_UNIVERSE_CHUNK_SIZE", 200)))
MARKET_UNIVERSE_MAX_SYMBOLS = max(0, _setting_int("MARKET_UNIVERSE_MAX_SYMBOLS", 0))
MARKET_UNIVERSE_CHUNK_WORKERS = max(1, min(8, _setting_int("MARKET_UNIVERSE_CHUNK_WORKERS", 4)))
MARKET_ASSETS_PAGE_DEFAULT = max(20, getattr(settings, "MARKET_ASSETS_PAGE_DEFAULT", 50))
MARKET_ASSETS_PAGE_MAX = max(50, getattr(settings, "MARKET_ASSETS_PAGE_MAX", 200))
MARKET_RANKINGS_REFRESH_SECONDS = max(60, _setting_int("MARKET_RANKINGS_REFRESH_SECONDS", 300))
MARKET_RANKINGS_REFRESH_MARGIN_SECONDS = max(0, _setting_int("MARKET_RANKINGS_REFRESH_MARGIN_SECONDS", 5))
MARKET_RANKINGS_SNAPSHOT_CHUNK_SIZE = max(1, min(200, _setting_int("MARKET_RANKINGS_SNAPSHOT_CHUNK_SIZE", 200)))
MARKET_RANKINGS_SNAPSHOT_TTL = max(60, _setting_int("MARKET_RANKINGS_SNAPSHOT_TTL", MARKET_RANKINGS_REFRESH_SECONDS))
MARKET_RANKINGS_BACKGROUND_ONLY = _setting_bool("MARKET_RANKINGS_BACKGROUND_ONLY", True)
MARKET_RANKINGS_DISABLE_FILTERS = _setting_bool("MARKET_RANKINGS_DISABLE_FILTERS", True)
MARKET_RANKINGS_ALLOW_STALE_SNAPSHOTS = _setting_bool("MARKET_RANKINGS_ALLOW_STALE_SNAPSHOTS", True)
MARKET_RANKINGS_TIMEFRAME_KEYS = [
    key
    for key in _setting_list(
        getattr(settings, "MARKET_RANKINGS_TIMEFRAME_KEYS", None),
        [key for key in TIMEFRAMES if key != "1d"],
    )
    if key in TIMEFRAMES and key != "1d"
]
MARKET_RANKINGS_TIMEFRAME_SNAPSHOT_TTL = max(
    60,
    _setting_int("MARKET_RANKINGS_TIMEFRAME_SNAPSHOT_TTL", MARKET_RANKINGS_REFRESH_SECONDS),
)
MARKET_RANKINGS_DAILY_WINDOW_5D = max(2, _setting_int("MARKET_RANKINGS_DAILY_WINDOW_5D", 6))
MARKET_RANKINGS_DAILY_WINDOW_1MO = max(2, _setting_int("MARKET_RANKINGS_DAILY_WINDOW_1MO", 30))
MARKET_RANKINGS_DAILY_WINDOW_6MO = max(2, _setting_int("MARKET_RANKINGS_DAILY_WINDOW_6MO", 130))
SNAPSHOT_RANKINGS_STATE = "market_rankings_snapshot.json"
SNAPSHOT_RANKINGS_PROGRESS_STATE = "market_rankings_snapshot_progress.json"
SNAPSHOT_RANKINGS_LOCK = "market_rankings_snapshot.pid"
_MARKET_EXECUTOR = ThreadPoolExecutor(max_workers=MARKET_MAX_WORKERS)
MARKET_CLOCK_CACHE_TTL = max(15, getattr(settings, "MARKET_CLOCK_CACHE_TTL", 60))
MARKET_CLOCK_TIMEOUT = max(2, getattr(settings, "MARKET_CLOCK_TIMEOUT_SECONDS", 5))
ALPACA_TRADING_REST_URL = getattr(settings, "ALPACA_TRADING_REST_URL", "https://paper-api.alpaca.markets").rstrip("/")
LIVE_TRADING_URL = "https://api.alpaca.markets"
try:
    _MARKET_TZ = ZoneInfo("America/New_York")
except Exception:
    _MARKET_TZ = timezone.utc


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


def _parse_offset(value: object) -> int:
    try:
        offset = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, offset)


def _fetch_market_clock(user_id: str | None) -> bool | None:
    key_id, secret = resolve_alpaca_data_credentials(user_id=user_id)
    if not key_id or not secret:
        return None
    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }
    config = resolve_retry_config(timeout=MARKET_CLOCK_TIMEOUT)
    session = get_requests_session(config.timeout)

    def _should_retry(response: requests.Response) -> bool:
        return response.status_code in {408, 429} or response.status_code >= 500

    bases = [ALPACA_TRADING_REST_URL]
    if ALPACA_TRADING_REST_URL.rstrip("/") != LIVE_TRADING_URL:
        bases.append(LIVE_TRADING_URL)

    for base in bases:
        url = f"{base.rstrip('/')}/v2/clock"

        def _call():
            return session.get(url, headers=headers, timeout=config.timeout)

        try:
            response = retry_call_result(
                _call,
                config=config,
                exceptions=(requests.RequestException,),
                should_retry=_should_retry,
            )
        except Exception:
            continue
        if response is None or response.status_code >= 400:
            if response is not None and response.status_code in {401, 403, 404} and base != bases[-1]:
                continue
            return None
        try:
            payload = response.json()
        except ValueError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("is_open"), bool):
            return bool(payload.get("is_open"))
    return None


def _is_market_open(user_id: str | None) -> bool:
    cache_key = build_cache_key("market-clock")

    def _builder():
        return _fetch_market_clock(user_id)

    is_open = cache_memoize(cache_key, _builder, MARKET_CLOCK_CACHE_TTL)
    if isinstance(is_open, bool):
        return is_open

    now = datetime.now(_MARKET_TZ)
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= now < close_time


def _resolve_list_type(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {
        "gainers",
        "losers",
        "most_active",
        "top_turnover",
    }:
        return text
    return "gainers"


def _resolve_detail_timeframe(value: object) -> DetailTimeframe:
    text = str(value or "").strip().lower()
    return DETAIL_TIMEFRAMES.get(text, DEFAULT_DETAIL_TIMEFRAME)


_CHART_INTERVAL_PATTERN = re.compile(r"^(\d+)(t|s|m|h|d)$")


def _resolve_chart_interval(value: object) -> ChartInterval | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    match = _CHART_INTERVAL_PATTERN.match(text)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2)
    if amount <= 0:
        return None
    unit_map = {
        "t": ("tick", "ticks"),
        "s": ("second", "seconds"),
        "m": ("minute", "minutes"),
        "h": ("hour", "hours"),
        "d": ("day", "days"),
    }
    unit_name, unit_label = unit_map.get(unit, ("", ""))
    if not unit_name:
        return None
    label = f"{amount} {unit_label}" if amount != 1 else f"{amount} {unit_name}"
    return ChartInterval(key=f"{amount}{unit}", unit=unit_name, value=amount, label=label)


def _resolve_range_window(range_key: str) -> tuple[datetime, datetime, float]:
    key = (range_key or "").strip().lower()
    now = datetime.now(timezone.utc)
    if key.endswith("mo") and key[:-2].isdigit():
        months = int(key[:-2])
        days = max(1, months) * 30
    elif key.endswith("d") and key[:-1].isdigit():
        days = int(key[:-1])
    else:
        days = 1
    delta_seconds = float(days * 86400)
    start = now - timedelta(seconds=delta_seconds)
    return start, now, delta_seconds


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


def _is_excluded_symbol(symbol: str) -> bool:
    if MARKET_RANKINGS_DISABLE_FILTERS:
        return False
    upper = symbol.upper()
    for suffix in MARKET_RANKINGS_EXCLUDE_SUFFIXES:
        if not suffix:
            continue
        if upper.endswith(str(suffix).upper()):
            return True
    return False


def _is_excluded_name(name: str) -> bool:
    if MARKET_RANKINGS_DISABLE_FILTERS:
        return False
    lowered = name.lower()
    for keyword in MARKET_RANKINGS_EXCLUDE_KEYWORDS:
        if keyword and keyword in lowered:
            return True
    return False


def _is_rankable_asset(asset: dict[str, str]) -> bool:
    symbol = str(asset.get("symbol") or "").upper()
    if not symbol:
        return False
    if MARKET_RANKINGS_DISABLE_FILTERS:
        return True
    if _is_excluded_symbol(symbol):
        return False
    exchange = str(asset.get("exchange") or "").upper()
    if exchange and exchange in MARKET_RANKINGS_EXCLUDE_EXCHANGES:
        return False
    name = str(asset.get("name") or "")
    if name and _is_excluded_name(name):
        return False
    return True


def _filter_rankable_assets(assets: list[dict[str, str]]) -> list[dict[str, str]]:
    if MARKET_RANKINGS_DISABLE_FILTERS:
        return [asset for asset in assets if asset.get("symbol")]
    return [asset for asset in assets if _is_rankable_asset(asset)]


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


def _filter_by_meta(
    items: list[dict[str, object]],
    *,
    meta_map: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    if not items:
        return items
    filtered: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").upper()
        if not symbol:
            continue
        meta = meta_map.get(symbol)
        if meta and not _is_rankable_asset(meta):
            continue
        filtered.append(item)
    return filtered


def _snapshot_refresh_payload_summary(payload: dict[str, object]) -> dict[str, object]:
    summary_keys = (
        "status",
        "generated_at",
        "generated_ts",
        "total_symbols",
        "chunk_size",
        "api_calls",
        "api_calls_per_minute",
        "duration_seconds",
        "source",
    )
    summary: dict[str, object] = {}
    for key in summary_keys:
        if key in payload:
            summary[key] = payload.get(key)
    return summary


def _snapshot_timeframe_state_name(timeframe_key: str) -> str:
    return f"market_rankings_snapshot_{timeframe_key}.json"


def _snapshot_timeframe_payload_summary(payload: dict[str, object]) -> dict[str, object]:
    summary = _snapshot_refresh_payload_summary(payload)
    if "timeframe" in payload:
        summary["timeframe"] = payload.get("timeframe")
    return summary


def _load_timeframe_snapshot_rows(timeframe_key: str) -> list[dict[str, object]] | None:
    payload = read_state(_snapshot_timeframe_state_name(timeframe_key), default={})
    if not isinstance(payload, dict) or not payload:
        return None
    if payload.get("status") != "complete":
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    if not rows:
        return None
    ts = payload.get("generated_ts")
    if isinstance(ts, (int, float)):
        if time.time() - float(ts) > MARKET_RANKINGS_TIMEFRAME_SNAPSHOT_TTL:
            return rows if MARKET_RANKINGS_ALLOW_STALE_SNAPSHOTS else None
    else:
        return None
    return rows


def _load_snapshot_rankings_latest_rows() -> list[dict[str, object]] | None:
    payload = read_state(SNAPSHOT_RANKINGS_STATE, default={})
    if not isinstance(payload, dict) or not payload:
        return None
    if payload.get("status") != "complete":
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    if not rows:
        return None
    ts = payload.get("generated_ts")
    if isinstance(ts, (int, float)):
        if time.time() - float(ts) > MARKET_RANKINGS_SNAPSHOT_TTL:
            return rows if MARKET_RANKINGS_ALLOW_STALE_SNAPSHOTS else None
    else:
        return None
    return rows


def _snapshot_refresh_meta() -> dict[str, object] | None:
    latest = read_state(SNAPSHOT_RANKINGS_STATE, default={})
    progress = read_state(SNAPSHOT_RANKINGS_PROGRESS_STATE, default={})
    meta: dict[str, object] = {}
    if isinstance(latest, dict) and latest:
        meta["latest"] = _snapshot_refresh_payload_summary(latest)
    if isinstance(progress, dict) and progress:
        progress_keys = (
            "status",
            "started_at",
            "started_ts",
            "total_symbols",
            "chunk_size",
            "total_chunks",
            "chunks_completed",
            "api_calls",
            "elapsed_seconds",
            "target_seconds",
        )
        if progress.get("status") == "running":
            meta["progress"] = {key: progress.get(key) for key in progress_keys if key in progress}
        elif progress.get("status") == "error":
            error_payload = {key: progress.get(key) for key in progress_keys if key in progress}
            if "error" in progress:
                error_payload["error"] = progress.get("error")
            meta["error"] = error_payload
    timeframe_meta: dict[str, dict[str, object]] = {}
    for key in MARKET_RANKINGS_TIMEFRAME_KEYS:
        payload = read_state(_snapshot_timeframe_state_name(key), default={})
        if isinstance(payload, dict) and payload:
            timeframe_meta[key] = _snapshot_timeframe_payload_summary(payload)
    if timeframe_meta:
        meta["timeframes"] = timeframe_meta
    return meta or None


def _build_snapshot_rankings(user_id: str | None) -> list[dict[str, object]]:
    cached_rows = _load_snapshot_rankings_latest_rows()
    if cached_rows is not None:
        return cached_rows
    if MARKET_RANKINGS_BACKGROUND_ONLY:
        return []
    cache_key = build_cache_key("market-rankings-snapshots", user_id or "anon")

    def _load() -> list[dict[str, object]] | None:
        assets = _filter_rankable_assets(_normalize_assets(_load_assets_master(user_id)))
        if not assets:
            return None
        symbols = [asset["symbol"] for asset in assets if asset.get("symbol")]
        if not symbols:
            return None
        snapshots = fetch_stock_snapshots(symbols, feed=DEFAULT_FEED, user_id=user_id, timeout=MARKET_REQUEST_TIMEOUT)
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
            dollar_volume = None
            if volume is not None:
                dollar_volume = last_price * volume
            open_price = _coerce_number((daily_bar or {}).get("o"))
            high_price = _coerce_number((daily_bar or {}).get("h"))
            low_price = _coerce_number((daily_bar or {}).get("l"))
            prev_volume = _coerce_number((prev_bar or {}).get("v"))
            gap_pct = None
            range_pct = None
            volume_ratio = None
            if prev_close not in (None, 0):
                if open_price is not None:
                    gap_pct = (open_price / prev_close - 1.0) * 100.0
                if high_price is not None and low_price is not None:
                    range_pct = ((high_price - low_price) / prev_close) * 100.0
            if volume is not None and prev_volume not in (None, 0):
                volume_ratio = volume / prev_volume

            if not MARKET_RANKINGS_DISABLE_FILTERS:
                if MARKET_RANKINGS_MIN_PRICE and last_price < MARKET_RANKINGS_MIN_PRICE:
                    continue
                if volume is not None and MARKET_RANKINGS_MIN_VOLUME and volume < MARKET_RANKINGS_MIN_VOLUME:
                    continue
                if (
                    dollar_volume is not None
                    and MARKET_RANKINGS_MIN_DOLLAR_VOLUME
                    and dollar_volume < MARKET_RANKINGS_MIN_DOLLAR_VOLUME
                ):
                    continue

            rows.append(
                {
                    "symbol": symbol,
                    "name": asset.get("name") or "",
                    "exchange": asset.get("exchange") or "",
                    "price": last_price,
                    "change_pct_day": change_pct,
                    "change_pct_period": change_pct,
                    "volume": volume,
                    "dollar_volume": dollar_volume,
                    "gap_pct": gap_pct,
                    "range_pct": range_pct,
                    "volume_ratio": volume_ratio,
                }
            )
        return rows or None

    result = cache_memoize(cache_key, _load, MARKET_RANKINGS_CACHE_TTL)
    return result if isinstance(result, list) else []


def refresh_snapshot_rankings(*, user_id: str | None = None) -> dict[str, object]:
    lock = InstanceLock(resolve_state_dir() / SNAPSHOT_RANKINGS_LOCK)
    if not lock.acquire():
        return {"status": "locked"}
    started_ts = time.time()
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    progress_payload = {
        "status": "running",
        "started_at": started_at,
        "started_ts": started_ts,
    }
    write_state(SNAPSHOT_RANKINGS_PROGRESS_STATE, progress_payload)

    try:
        key_id, secret = resolve_alpaca_data_credentials(user_id=user_id)
        if not (key_id and secret):
            error_payload = {
                "status": "error",
                "started_at": started_at,
                "started_ts": started_ts,
                "error": "missing_credentials",
            }
            write_state(SNAPSHOT_RANKINGS_PROGRESS_STATE, error_payload)
            return error_payload

        assets = _filter_rankable_assets(_normalize_assets(_load_assets_master(user_id)))
        asset_map = {asset.get("symbol"): asset for asset in assets if asset.get("symbol")}
        symbols = list(asset_map.keys())
        total_symbols = len(symbols)
        chunk_size = min(200, max(1, MARKET_RANKINGS_SNAPSHOT_CHUNK_SIZE))
        chunks = list(_iter_chunks(symbols, chunk_size))
        timeframes = [TIMEFRAMES[key] for key in MARKET_RANKINGS_TIMEFRAME_KEYS if key in TIMEFRAMES]
        timeframe_batches = 1 if timeframes else 0
        total_chunks = len(chunks) * (1 + timeframe_batches)
        rows: list[dict[str, object]] = []
        api_calls = 0
        completed_chunks = 0

        target_seconds = float(MARKET_RANKINGS_REFRESH_SECONDS - MARKET_RANKINGS_REFRESH_MARGIN_SECONDS)
        if total_chunks <= 0:
            target_seconds = 0.0
        else:
            target_seconds = max(0.0, target_seconds)

        def _update_progress() -> None:
            elapsed = time.time() - started_ts
            progress_payload = {
                "status": "running",
                "started_at": started_at,
                "started_ts": started_ts,
                "total_symbols": total_symbols,
                "chunk_size": chunk_size,
                "total_chunks": total_chunks,
                "chunks_completed": completed_chunks,
                "api_calls": api_calls,
                "elapsed_seconds": round(elapsed, 2),
                "target_seconds": target_seconds,
            }
            write_state(SNAPSHOT_RANKINGS_PROGRESS_STATE, progress_payload)

            if total_chunks > 0 and target_seconds > 0 and completed_chunks < total_chunks:
                target_elapsed = target_seconds * completed_chunks / total_chunks
                sleep_for = target_elapsed - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        for chunk in chunks:
            snapshots = fetch_stock_snapshots(
                chunk,
                feed=DEFAULT_FEED,
                user_id=user_id,
                timeout=MARKET_REQUEST_TIMEOUT,
            )
            api_calls += 1
            if isinstance(snapshots, dict):
                for symbol in chunk:
                    asset = asset_map.get(symbol)
                    if not asset:
                        continue
                    snapshot = snapshots.get(symbol)
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
                    dollar_volume = None
                    if volume is not None:
                        dollar_volume = last_price * volume
                    open_price = _coerce_number((daily_bar or {}).get("o"))
                    high_price = _coerce_number((daily_bar or {}).get("h"))
                    low_price = _coerce_number((daily_bar or {}).get("l"))
                    prev_volume = _coerce_number((prev_bar or {}).get("v"))
                    gap_pct = None
                    range_pct = None
                    volume_ratio = None
                    if prev_close not in (None, 0):
                        if open_price is not None:
                            gap_pct = (open_price / prev_close - 1.0) * 100.0
                        if high_price is not None and low_price is not None:
                            range_pct = ((high_price - low_price) / prev_close) * 100.0
                    if volume is not None and prev_volume not in (None, 0):
                        volume_ratio = volume / prev_volume

                    if not MARKET_RANKINGS_DISABLE_FILTERS:
                        if MARKET_RANKINGS_MIN_PRICE and last_price < MARKET_RANKINGS_MIN_PRICE:
                            continue
                        if volume is not None and MARKET_RANKINGS_MIN_VOLUME and volume < MARKET_RANKINGS_MIN_VOLUME:
                            continue
                        if (
                            dollar_volume is not None
                            and MARKET_RANKINGS_MIN_DOLLAR_VOLUME
                            and dollar_volume < MARKET_RANKINGS_MIN_DOLLAR_VOLUME
                        ):
                            continue

                    rows.append(
                        {
                        "symbol": symbol,
                        "name": asset.get("name") or "",
                        "exchange": asset.get("exchange") or "",
                        "price": last_price,
                        "change_pct_day": change_pct,
                        "change_pct_period": change_pct,
                        "volume": volume,
                        "dollar_volume": dollar_volume,
                        "gap_pct": gap_pct,
                        "range_pct": range_pct,
                        "volume_ratio": volume_ratio,
                        }
                    )

            completed_chunks += 1
            _update_progress()

        if timeframes:
            timeframe_rows_map: dict[str, list[dict[str, object]]] = {tf.key: [] for tf in timeframes}
            timeframe_started = time.time()
            timeframe_error: str | None = None
            timeframe_api_calls = 0
            max_timeframe = max(timeframes, key=lambda item: _daily_window_length(item) or 0)
            daily_timeframe = _resolve_daily_timeframe(max_timeframe)
            try:
                for chunk in chunks:
                    series_map, _source = _download_history(chunk, daily_timeframe, user_id=user_id)
                    api_calls += 1
                    timeframe_api_calls += 1
                    if series_map:
                        for timeframe in timeframes:
                            timeframe_rows_map[timeframe.key].extend(_rank_symbols_daily(series_map, timeframe))
                    completed_chunks += 1
                    _update_progress()
            except Exception as exc:
                timeframe_error = str(exc)

            duration = time.time() - timeframe_started
            for timeframe in timeframes:
                timeframe_rows = timeframe_rows_map.get(timeframe.key, [])
                if timeframe_rows:
                    timeframe_rows.sort(key=lambda item: item.get("change_pct_period") or 0, reverse=True)
                timeframe_payload: dict[str, object] = {
                    "status": "error" if timeframe_error else "complete",
                    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "generated_ts": time.time(),
                    "source": "alpaca",
                    "timeframe": timeframe.key,
                    "total_symbols": total_symbols,
                    "chunk_size": chunk_size,
                    "api_calls": timeframe_api_calls,
                    "duration_seconds": round(duration, 2),
                    "rows": timeframe_rows,
                }
                if timeframe_error:
                    timeframe_payload["error"] = timeframe_error
                write_state(_snapshot_timeframe_state_name(timeframe.key), timeframe_payload)

        duration_seconds = time.time() - started_ts
        api_calls_per_minute = None
        if duration_seconds > 0:
            api_calls_per_minute = api_calls / (duration_seconds / 60.0)

        payload = {
            "status": "complete",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "generated_ts": time.time(),
            "source": "alpaca",
            "total_symbols": total_symbols,
            "chunk_size": chunk_size,
            "api_calls": api_calls,
            "api_calls_per_minute": api_calls_per_minute,
            "duration_seconds": round(duration_seconds, 2),
            "rows": rows,
        }
        write_state(SNAPSHOT_RANKINGS_STATE, payload)
        write_state(SNAPSHOT_RANKINGS_PROGRESS_STATE, _snapshot_refresh_payload_summary(payload))
        return payload
    except Exception as exc:
        error_payload = {
            "status": "error",
            "started_at": started_at,
            "started_ts": started_ts,
            "error": str(exc),
        }
        write_state(SNAPSHOT_RANKINGS_PROGRESS_STATE, error_payload)
        return error_payload
    finally:
        lock.release()


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
        dollar_volume = _coerce_number(entry.get("dollar_volume"))
        if price is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "price": price,
                "change_pct_day": change_pct,
                "change_pct_period": change_pct,
                "volume": volume,
                "dollar_volume": dollar_volume,
            }
        )
    return rows


def _resolve_universe_rankings(user_id: str | None) -> tuple[list[dict[str, object]], str]:
    rankings = _build_snapshot_rankings(user_id)
    if rankings:
        return rankings, "alpaca"
    rankings = _load_universe_ranked()
    if rankings:
        symbols = {item.get("symbol") for item in rankings if isinstance(item, dict)}
        meta_map = _build_asset_meta_map(user_id=user_id, symbols={sym for sym in symbols if isinstance(sym, str)})
        filtered: list[dict[str, object]] = []
        for entry in rankings:
            if not isinstance(entry, dict):
                continue
            symbol = str(entry.get("symbol") or "").upper()
            if not symbol or _is_excluded_symbol(symbol):
                continue
            meta = meta_map.get(symbol)
            if meta and not _is_rankable_asset(meta):
                continue
            if not _passes_rank_filters(entry):
                continue
            filtered.append(entry)
        if filtered:
            return filtered, "cache"
    return [], "unknown"


def _extract_change_value(entry: dict[str, object]) -> float | None:
    value = _coerce_number(entry.get("change_pct_period"))
    if value is None:
        value = _coerce_number(entry.get("change_pct_day"))
    return value


def _passes_rank_filters(entry: dict[str, object]) -> bool:
    if MARKET_RANKINGS_DISABLE_FILTERS:
        return True
    symbol = str(entry.get("symbol") or "").upper()
    if symbol and _is_excluded_symbol(symbol):
        return False
    price = _coerce_number(entry.get("price"))
    if price is not None and MARKET_RANKINGS_MIN_PRICE and price < MARKET_RANKINGS_MIN_PRICE:
        return False
    volume = _coerce_number(entry.get("volume"))
    dollar_volume = _coerce_number(entry.get("dollar_volume"))
    if dollar_volume is None and price is not None and volume is not None:
        dollar_volume = price * volume
        entry["dollar_volume"] = dollar_volume
    if volume is not None and MARKET_RANKINGS_MIN_VOLUME and volume < MARKET_RANKINGS_MIN_VOLUME:
        return False
    if dollar_volume is not None and MARKET_RANKINGS_MIN_DOLLAR_VOLUME and dollar_volume < MARKET_RANKINGS_MIN_DOLLAR_VOLUME:
        return False
    return True


def _split_rankings(
    rows: list[dict[str, object]],
    limit: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ranked: list[tuple[float, dict[str, object]]] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        if not _passes_rank_filters(entry):
            continue
        change_value = _extract_change_value(entry)
        if change_value is None:
            continue
        ranked.append((change_value, entry))
    ranked.sort(key=lambda item: item[0], reverse=True)
    gainers = [row for change, row in ranked if change >= 0][:limit]
    losers = [row for change, row in sorted(ranked, key=lambda item: item[0]) if change < 0][:limit]
    return gainers, losers


def _sort_rows_by_metric(
    rows: list[dict[str, object]],
    metric_key: str,
    *,
    reverse: bool = True,
    predicate: Callable[[float], bool] | None = None,
) -> list[dict[str, object]]:
    ranked: list[tuple[float, dict[str, object]]] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        value = _coerce_number(entry.get(metric_key))
        if value is None:
            continue
        if predicate and not predicate(value):
            continue
        ranked.append((value, entry))
    ranked.sort(key=lambda item: item[0], reverse=reverse)
    return [row for _value, row in ranked]


def _paginate_items(
    items: list[dict[str, object]],
    *,
    offset: int,
    limit: int,
) -> tuple[list[dict[str, object]], int, int | None]:
    total = len(items)
    if total == 0:
        return [], 0, None
    safe_offset = min(max(offset, 0), total)
    safe_limit = max(1, limit)
    end = min(safe_offset + safe_limit, total)
    page_items = items[safe_offset:end]
    next_offset = end if end < total else None
    return page_items, total, next_offset


def _rank_symbols_light(
    series_map: dict[str, pd.Series],
    timeframe: Timeframe,
) -> list[dict[str, object]]:
    label = str(timeframe.label)
    ranked: list[dict[str, object]] = []
    for sym, series in series_map.items():
        window = _slice_series(series, timeframe)
        if len(window) < 2:
            continue
        try:
            start_price = float(window.iloc[0])
            end_price = float(window.iloc[-1])
            prev_price = float(window.iloc[-2])
        except Exception:
            continue
        if not start_price or not prev_price:
            continue
        period_change = ((end_price / start_price) - 1.0) * 100.0
        day_change = ((end_price / prev_price) - 1.0) * 100.0
        ranked.append(
            {
                "symbol": sym,
                "price": round(end_price, 2),
                "change_pct_period": round(period_change, 2),
                "change_pct_day": round(day_change, 2),
                "period_label": label,
                "period_label_en": timeframe.label_en,
            }
        )
    ranked.sort(key=lambda item: item.get("change_pct_period") or 0, reverse=True)
    return ranked


def _resolve_daily_timeframe(timeframe: Timeframe) -> Timeframe:
    if timeframe.interval == "1d":
        return timeframe
    return Timeframe(
        timeframe.key,
        timeframe.label,
        timeframe.label_en,
        timeframe.period,
        "1d",
    )


def _daily_window_length(timeframe: Timeframe) -> int | None:
    if timeframe.key == "5d":
        return MARKET_RANKINGS_DAILY_WINDOW_5D
    if timeframe.key == "1mo":
        return MARKET_RANKINGS_DAILY_WINDOW_1MO
    if timeframe.key == "6mo":
        return MARKET_RANKINGS_DAILY_WINDOW_6MO
    return None


def _rank_symbols_daily(
    series_map: dict[str, pd.Series],
    timeframe: Timeframe,
) -> list[dict[str, object]]:
    label = str(timeframe.label)
    ranked: list[dict[str, object]] = []
    window_len = _daily_window_length(timeframe)
    for sym, series in series_map.items():
        window = series.dropna()
        if window.empty:
            continue
        if window_len:
            window = window.tail(window_len)
        else:
            window = _slice_series(window, timeframe)
        if len(window) < 2:
            continue
        try:
            start_price = float(window.iloc[0])
            end_price = float(window.iloc[-1])
            prev_price = float(window.iloc[-2])
        except Exception:
            continue
        if not start_price or not prev_price:
            continue
        period_change = ((end_price / start_price) - 1.0) * 100.0
        day_change = ((end_price / prev_price) - 1.0) * 100.0
        ranked.append(
            {
                "symbol": sym,
                "price": round(end_price, 2),
                "change_pct_period": round(period_change, 2),
                "change_pct_day": round(day_change, 2),
                "period_label": label,
                "period_label_en": timeframe.label_en,
            }
        )
    ranked.sort(key=lambda item: item.get("change_pct_period") or 0, reverse=True)
    return ranked


def _iter_chunks(items: list[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        yield items
        return
    for idx in range(0, len(items), size):
        chunk = items[idx : idx + size]
        if chunk:
            yield chunk


def _fetch_history_chunked(
    symbols: list[str],
    timeframe: Timeframe,
    *,
    user_id: str | None,
) -> tuple[dict[str, pd.Series], str]:
    series_map: dict[str, pd.Series] = {}
    source = "unknown"
    chunks = list(_iter_chunks(symbols, MARKET_UNIVERSE_CHUNK_SIZE))
    if not chunks:
        return {}, "unknown"
    if MARKET_UNIVERSE_CHUNK_WORKERS <= 1 or len(chunks) == 1:
        for chunk in chunks:
            history, chunk_source = _download_history(chunk, timeframe, user_id=user_id)
            if history:
                series_map.update(history)
            if chunk_source and chunk_source != "unknown":
                source = chunk_source
        return series_map, source
    with ThreadPoolExecutor(max_workers=MARKET_UNIVERSE_CHUNK_WORKERS) as executor:
        futures = [executor.submit(_download_history, chunk, timeframe, user_id=user_id) for chunk in chunks]
        for future in futures:
            try:
                history, chunk_source = future.result()
            except Exception:
                continue
            if history:
                series_map.update(history)
            if chunk_source and chunk_source != "unknown":
                source = chunk_source
    return series_map, source


def _resolve_universe_timeframe_rankings(
    timeframe: Timeframe,
    *,
    user_id: str | None,
) -> tuple[list[dict[str, object]], str]:
    if timeframe.key == "1d":
        rows, source = _resolve_universe_rankings(user_id)
        for entry in rows:
            if isinstance(entry, dict):
                entry.setdefault("period_label", timeframe.label)
                entry.setdefault("period_label_en", timeframe.label_en)
        return rows, source
    cache_key = build_cache_key("market-universe-timeframe", "alpaca", timeframe.key, user_id or "anon")

    def _load():
        assets = _filter_rankable_assets(_normalize_assets(_load_assets_master(user_id)))
        if not assets:
            return {"rows": [], "source": "unknown"}
        symbols = [asset["symbol"] for asset in assets if asset.get("symbol")]
        if not symbols:
            return {"rows": [], "source": "unknown"}
        if MARKET_UNIVERSE_MAX_SYMBOLS and len(symbols) > MARKET_UNIVERSE_MAX_SYMBOLS:
            symbols = symbols[:MARKET_UNIVERSE_MAX_SYMBOLS]
        series_map, source = _fetch_history_chunked(symbols, timeframe, user_id=user_id)
        if not series_map:
            return {"rows": [], "source": source or "unknown"}
        rows = _rank_symbols_light(series_map, timeframe)
        return {"rows": rows, "source": source or "unknown"}

    payload = cache_memoize(cache_key, _load, MARKET_UNIVERSE_RANKINGS_CACHE_TTL)
    if isinstance(payload, dict):
        rows = payload.get("rows")
        source = payload.get("source")
        return rows if isinstance(rows, list) else [], source if isinstance(source, str) else "unknown"
    if isinstance(payload, list):
        return payload, "unknown"
    return [], "unknown"


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
    cache_key = build_cache_key("market-profile", "alpaca", symbol.upper())

    def _download() -> dict[str, object]:
        meta_map = _build_asset_meta_map(user_id=user_id, symbols={symbol.upper()})
        meta = meta_map.get(symbol.upper())
        if not meta:
            return {}
        name = meta.get("name") or symbol
        return {
            "name": name,
            "shortName": name,
            "exchange": meta.get("exchange") or "",
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
        title = entry.get("headline") or entry.get("title") or entry.get("summary") or entry.get("content") or ""
        url = entry.get("url") or entry.get("link") or entry.get("article_url") or ""
        source = entry.get("source") or entry.get("publisher") or entry.get("author") or ""
        raw_time = (
            entry.get("created_at")
            or entry.get("createdAt")
            or entry.get("time")
            or entry.get("published_at")
            or entry.get("published")
            or entry.get("providerPublishTime")
        )
        snippet = entry.get("summary") or entry.get("description") or entry.get("content") or ""
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


def _infer_news_symbols(symbol: str, *, user_id: str | None = None) -> list[str]:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return []
    symbols = [symbol]
    meta_map = _build_asset_meta_map(user_id=user_id, symbols={symbol})
    name = str(meta_map.get(symbol, {}).get("name") or "").lower()
    if any(key in name for key in ("warrant", "unit")):
        for suffix in ("WS", "W", "U"):
            if symbol.endswith(suffix) and len(symbol) > len(suffix):
                base = symbol[: -len(suffix)].rstrip(".-")
                if base and base not in symbols:
                    symbols.append(base)
                break
    return symbols


def _fetch_yfinance_news(symbol: str, *, limit: int) -> list[dict[str, str]]:
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return []
    try:
        ticker = yf.Ticker(symbol)
        items = ticker.news or []
    except Exception:
        return []
    if not isinstance(items, list):
        return []
    normalized = _normalize_news_items([entry for entry in items if isinstance(entry, dict)])
    return normalized[:limit]


def _fetch_symbol_news(symbol: str, *, user_id: str | None = None, limit: int = 6) -> list[dict[str, str]]:
    if not symbol:
        return []
    cache_key = build_cache_key("market-news", "alpaca", symbol.upper())

    cached = cache_get_object(cache_key)
    if isinstance(cached, list) and cached:
        return cached

    symbols = _infer_news_symbols(symbol, user_id=user_id)
    items = fetch_news(symbols=symbols, limit=limit, user_id=user_id)
    normalized: list[dict[str, str]] = []
    if isinstance(items, list) and items:
        normalized = _normalize_news_items(items)[:limit]
    if not normalized:
        fallback_symbol = symbols[-1] if symbols else symbol
        normalized = _fetch_yfinance_news(fallback_symbol, limit=limit)
    if normalized:
        cache_set_object(cache_key, normalized, MARKET_NEWS_CACHE_TTL)
    return normalized


def _build_ai_summary(
    profile_payload: dict[str, object],
    news_payload: list[dict[str, str]],
    *,
    lang_prefix: str,
) -> str:
    summary = ""
    if isinstance(news_payload, list) and news_payload:
        first = news_payload[0] if isinstance(news_payload[0], dict) else {}
        title = str(first.get("title") or "").strip()
        source = str(first.get("source") or "").strip()
        if title:
            if lang_prefix == "zh":
                summary = f"{source} 最新消息：{title}" if source else f"最新消息：{title}"
            else:
                summary = f"Latest headline{f' from {source}' if source else ''}: {title}"
    if not summary and isinstance(profile_payload, dict):
        raw_summary = str(profile_payload.get("summary") or "").strip()
        if raw_summary:
            summary = raw_summary
    if not summary:
        summary = "暂无可用的 AI 摘要。" if lang_prefix == "zh" else "No AI summary available yet."
    if len(summary) > 160:
        summary = summary[:160].rstrip() + "..."
    return summary


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
    key_id, secret = resolve_alpaca_data_credentials(user=request.user)
    if not (key_id and secret):
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
    page_offset = _parse_offset(params.get("offset"))
    page_size = resolved.limit
    page_stop = max(page_size + 1, page_offset + page_size + 1)
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
            feed=DEFAULT_FEED,
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
        lang_prefix = (getattr(request, "LANGUAGE_CODE", "") or request.COOKIES.get("django_language", "zh-hans")).lower()[:2]
        ai_summary = _build_ai_summary(profile_payload, news_payload, lang_prefix=lang_prefix)
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
                "ai_summary": ai_summary,
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
    ranking_timeframe: Timeframe | None = None
    universe_rankings: list[dict[str, object]] | None = None
    universe_source = "unknown"
    timeframe_rankings: list[dict[str, object]] | None = None
    timeframe_source = "unknown"

    if restrict_to_query:
        page_offset = 0
        page_stop = page_size
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
            gainers = [entry for entry in ranked if entry["change_pct_period"] >= 0][: page_stop]
            losers = [entry for entry in ranked if entry["change_pct_period"] < 0][: page_stop]
            list_items = losers if list_type == "losers" else gainers
    else:
        user_id = str(request.user.id) if request.user.is_authenticated else None
        used_timeframe_rankings = False
        snapshot_rankings: list[dict[str, object]] | None = None
        timeframe_snapshot: list[dict[str, object]] | None = None
        if list_type in {"gainers", "losers"}:
            if resolved.timeframe.key == "1d":
                movers_limit = min(50, max(page_stop, page_size))
                movers = market_data.fetch_market_movers(
                    limit=movers_limit,
                    user_id=user_id,
                    timeout=MARKET_REQUEST_TIMEOUT,
                )
                raw_gainers = movers.get("gainers", []) if movers else []
                raw_losers = movers.get("losers", []) if movers else []
                gainers = [
                    entry for entry in raw_gainers if isinstance(entry, dict) and _passes_rank_filters(entry)
                ][:page_stop]
                losers = [entry for entry in raw_losers if isinstance(entry, dict) and _passes_rank_filters(entry)][
                    :page_stop
                ]
                if gainers or losers:
                    list_items = losers if list_type == "losers" else gainers
                    data_source = "alpaca"
                    ranking_timeframe = resolved.timeframe
                    used_timeframe_rankings = True

            if not used_timeframe_rankings and resolved.timeframe.key != "1d":
                timeframe_snapshot = _load_timeframe_snapshot_rows(resolved.timeframe.key)
                if timeframe_snapshot:
                    gainers, losers = _split_rankings(timeframe_snapshot, page_stop)
                    if gainers or losers:
                        list_items = losers if list_type == "losers" else gainers
                        data_source = "alpaca"
                        ranking_timeframe = resolved.timeframe
                        used_timeframe_rankings = True
                elif MARKET_RANKINGS_BACKGROUND_ONLY:
                    if snapshot_rankings is None:
                        snapshot_rankings = _build_snapshot_rankings(user_id)
                    if snapshot_rankings:
                        gainers, losers = _split_rankings(snapshot_rankings, page_stop)
                        if gainers or losers:
                            list_items = losers if list_type == "losers" else gainers
                            data_source = "alpaca"
                            ranking_timeframe = TIMEFRAMES["1d"]
                            used_timeframe_rankings = True

            if not used_timeframe_rankings and resolved.timeframe.key == "1d":
                if snapshot_rankings is None:
                    snapshot_rankings = _build_snapshot_rankings(user_id)
                if snapshot_rankings:
                    gainers, losers = _split_rankings(snapshot_rankings, page_stop)
                    if gainers or losers:
                        list_items = losers if list_type == "losers" else gainers
                        data_source = "alpaca"
                        ranking_timeframe = resolved.timeframe
                        used_timeframe_rankings = True

            if not used_timeframe_rankings:
                if timeframe_rankings is None and not MARKET_RANKINGS_BACKGROUND_ONLY:
                    timeframe_rankings, timeframe_source = _resolve_universe_timeframe_rankings(
                        resolved.timeframe,
                        user_id=user_id,
                    )
                if timeframe_rankings:
                    gainers, losers = _split_rankings(timeframe_rankings, page_stop)
                    if gainers or losers:
                        list_items = losers if list_type == "losers" else gainers
                        data_source = timeframe_source or data_source
                        ranking_timeframe = resolved.timeframe
                        used_timeframe_rankings = True

        if not used_timeframe_rankings:
            if list_type == "most_active":
                if resolved.timeframe.key == "1d":
                    active_limit = min(100, max(page_stop, page_size))
                    list_items = market_data.fetch_most_actives(
                        by="volume",
                        limit=active_limit,
                        user_id=user_id,
                        timeout=MARKET_REQUEST_TIMEOUT,
                    )
                    list_items = [
                        entry for entry in list_items if isinstance(entry, dict) and _passes_rank_filters(entry)
                    ][:page_stop]
                    most_actives = list_items
                    volume_label = _("成交量")
                    for entry in list_items:
                        if isinstance(entry, dict):
                            entry.setdefault("period_label", volume_label)
                            entry.setdefault("period_label_en", "Volume")
                    data_source = "alpaca" if list_items else "unknown"
                if not list_items:
                    if snapshot_rankings is None:
                        snapshot_rankings = _build_snapshot_rankings(user_id)
                    if snapshot_rankings:
                        list_items = _sort_rows_by_metric(
                            snapshot_rankings,
                            "volume",
                            reverse=True,
                            predicate=lambda value: value > 0,
                        )[:page_stop]
                        most_actives = list_items
                        volume_label = _("成交量")
                        for entry in list_items:
                            if isinstance(entry, dict):
                                entry.setdefault("period_label", volume_label)
                                entry.setdefault("period_label_en", "Volume")
                        data_source = "alpaca" if list_items else "unknown"
                    elif not MARKET_RANKINGS_BACKGROUND_ONLY:
                        list_items = market_data.fetch_most_actives(
                            by="volume",
                            limit=page_stop,
                            user_id=user_id,
                            timeout=MARKET_REQUEST_TIMEOUT,
                        )
                        list_items = [
                            entry for entry in list_items if isinstance(entry, dict) and _passes_rank_filters(entry)
                        ]
                        most_actives = list_items
                        volume_label = _("成交量")
                        for entry in list_items:
                            if isinstance(entry, dict):
                                entry.setdefault("period_label", volume_label)
                                entry.setdefault("period_label_en", "Volume")
                        data_source = "alpaca" if list_items else "unknown"
            elif list_type == "top_turnover":
                if snapshot_rankings is None:
                    snapshot_rankings = _build_snapshot_rankings(user_id)
                source_rows = snapshot_rankings
                if not snapshot_rankings:
                    if universe_rankings is None:
                        universe_rankings, universe_source = _resolve_universe_rankings(user_id)
                    source_rows = universe_rankings
                    data_source = universe_source or data_source
                else:
                    data_source = "alpaca" if snapshot_rankings else data_source
                list_items = _sort_rows_by_metric(
                    source_rows,
                    "dollar_volume",
                    reverse=True,
                    predicate=lambda value: value > 0,
                )[:page_stop]

    if list_type in {"gainers", "losers"} and not list_items:
        user_id = str(request.user.id) if request.user.is_authenticated else None
        if timeframe_rankings is None and not MARKET_RANKINGS_BACKGROUND_ONLY:
            timeframe_rankings, timeframe_source = _resolve_universe_timeframe_rankings(
                resolved.timeframe,
                user_id=user_id,
            )
        if timeframe_rankings:
            gainers, losers = _split_rankings(timeframe_rankings, page_stop)
            if gainers or losers:
                list_items = losers if list_type == "losers" else gainers
                data_source = timeframe_source or data_source
                if ranking_timeframe is None:
                    ranking_timeframe = resolved.timeframe

        if not list_items:
            if universe_rankings is None:
                universe_rankings, universe_source = _resolve_universe_rankings(user_id)
            if universe_rankings:
                gainers, losers = _split_rankings(universe_rankings, page_stop)
                if gainers or losers:
                    list_items = losers if list_type == "losers" else gainers
                    data_source = universe_source or data_source
                    if ranking_timeframe is None:
                        ranking_timeframe = TIMEFRAMES["1d"]

        if not list_items and not MARKET_RANKINGS_BACKGROUND_ONLY:
            series_map, fallback_source = _fetch_history(
                TOP_SYMBOLS,
                resolved.timeframe,
                user_id=str(request.user.id),
            )
            ranked = _rank_symbols(series_map, resolved.timeframe, limit=page_stop)
            if ranked:
                gainers = [entry for entry in ranked if entry["change_pct_period"] >= 0][: page_stop]
                losers = [entry for entry in ranked if entry["change_pct_period"] < 0][: page_stop]
                list_items = losers if list_type == "losers" else gainers
                data_source = fallback_source or data_source
                if ranking_timeframe is None:
                    ranking_timeframe = resolved.timeframe

    if list_type == "most_active" and not list_items:
        if universe_rankings is None:
            universe_rankings, universe_source = _resolve_universe_rankings(
                str(request.user.id) if request.user.is_authenticated else None
            )
        if universe_rankings:
            ranked_by_volume: list[tuple[float, dict[str, object]]] = []
            for entry in universe_rankings:
                if not isinstance(entry, dict):
                    continue
                if not _passes_rank_filters(entry):
                    continue
                volume_value = _coerce_number(entry.get("volume"))
                if volume_value is None:
                    continue
                ranked_by_volume.append((volume_value, entry))
            ranked_by_volume.sort(key=lambda item: item[0], reverse=True)
            most_actives = [row for _volume, row in ranked_by_volume][: page_stop]
            list_items = most_actives
            volume_label = _("成交量")
            for entry in list_items:
                if isinstance(entry, dict):
                    entry.setdefault("period_label", volume_label)
                    entry.setdefault("period_label_en", "Volume")
            data_source = universe_source or data_source

    suggestions = _build_suggestions(resolved.query)

    meta_symbols: set[str] = set()
    for collection in (list_items, gainers, losers, most_actives):
        for item in collection:
            if isinstance(item, dict) and item.get("symbol"):
                meta_symbols.add(str(item.get("symbol")).upper())
    meta_map = _build_asset_meta_map(user_id=str(request.user.id), symbols=meta_symbols)
    _attach_asset_meta(list_items, meta_map=meta_map)
    _attach_asset_meta(gainers, meta_map=meta_map)
    _attach_asset_meta(losers, meta_map=meta_map)
    _attach_asset_meta(most_actives, meta_map=meta_map)
    list_items = _filter_by_meta(list_items, meta_map=meta_map)
    gainers = _filter_by_meta(gainers, meta_map=meta_map)
    losers = _filter_by_meta(losers, meta_map=meta_map)
    most_actives = _filter_by_meta(most_actives, meta_map=meta_map)
    if active_list_type == "losers":
        list_items = losers
    elif active_list_type == "most_active":
        list_items = most_actives
    elif active_list_type == "gainers":
        list_items = gainers

    page_items, total_items, next_offset = _paginate_items(
        list_items,
        offset=page_offset,
        limit=page_size,
    )
    list_items = page_items
    has_more = next_offset is not None
    if active_list_type == "gainers":
        gainers = list_items
    elif active_list_type == "losers":
        losers = list_items
    elif active_list_type == "most_active":
        most_actives = list_items

    response = {
        "timeframe": {
            "key": resolved.timeframe.key,
            "label": resolved.timeframe.label,
            "label_en": resolved.timeframe.label_en,
            "clamped": resolved.timeframe_clamped,
        },
        "ranking_timeframe": (
            {
                "key": ranking_timeframe.key,
                "label": ranking_timeframe.label,
                "label_en": ranking_timeframe.label_en,
            }
            if ranking_timeframe
            else None
        ),
        # Use timezone.utc for Python 3.13 compatibility (datetime.UTC removed)
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_source": data_source or "unknown",
        "query": resolved.query,
        "list_type": active_list_type,
        "items": list_items,
        "gainers": gainers,
        "losers": losers,
        "most_actives": most_actives,
        "offset": page_offset,
        "limit": page_size,
        "next_offset": next_offset,
        "has_more": has_more,
        "total": total_items,
        "limit_clamped": resolved.limit_clamped,
        "request_id": request_id,
        "suggestions": suggestions,
        "recent_queries": recent_queries,
        "watchlist": watchlist,
        "snapshot_refresh": _snapshot_refresh_meta(),
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
def market_chart_data(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    key_id, secret = resolve_alpaca_data_credentials(user=request.user)
    if not (key_id and secret):
        return JsonResponse(
            {"error": _("当前环境缺少可用的市场数据源。"), "request_id": request_id},
            status=503,
            json_dumps_params={"ensure_ascii": False},
        )

    params = request.GET
    symbol = _normalize_query(params.get("symbol") or params.get("ticker"))
    if not symbol:
        return JsonResponse(
            {"error": _("缺少股票代码。"), "request_id": request_id},
            status=400,
            json_dumps_params={"ensure_ascii": False},
        )
    range_key = str(params.get("range") or "1d").strip().lower()
    interval_key = str(params.get("interval") or params.get("interval_key") or "1m").strip().lower()
    interval = _resolve_chart_interval(interval_key) or _resolve_chart_interval("1m")
    if interval is None:
        return JsonResponse(
            {"error": _("无效的时间粒度。"), "request_id": request_id},
            status=400,
            json_dumps_params={"ensure_ascii": False},
        )

    cache_alias = getattr(settings, "MARKET_HISTORY_CACHE_ALIAS", None)
    cache_ttl = 10 if interval.unit in {"tick", "second"} else 60 if interval.unit == "minute" else 300
    cache_key = build_cache_key("market-chart", symbol, range_key, interval.key, DEFAULT_FEED, request.user.id)
    cached = cache_get_object(cache_key, cache_alias=cache_alias)
    if isinstance(cached, dict) and cached.get("bars"):
        cached["request_id"] = request_id
        return JsonResponse(cached, json_dumps_params={"ensure_ascii": False})

    start, end, range_seconds = _resolve_range_window(range_key)
    downgrade_to: str | None = None
    downgrade_message: str | None = None
    window_limited = False

    if interval.unit in {"tick", "second"}:
        max_range = CHART_MAX_TICK_RANGE_SECONDS if interval.unit == "tick" else CHART_MAX_SECOND_RANGE_SECONDS
        if range_seconds > max_range:
            downgrade_to = "1m"
            downgrade_message = _("高频粒度仅支持近期窗口，已自动切换为 1m K 线。")
            interval = _resolve_chart_interval(downgrade_to) or interval
        else:
            trades, next_token = fetch_stock_trades(
                symbol,
                start=start,
                end=end,
                feed=DEFAULT_FEED,
                limit=CHART_TRADES_PAGE_LIMIT,
                max_pages=CHART_TRADES_MAX_PAGES,
                user=request.user,
                timeout=MARKET_REQUEST_TIMEOUT,
            )
            window_limited = True
            if interval.unit == "tick":
                bars = aggregate_trades_to_tick_bars(
                    trades,
                    ticks_per_bar=interval.value,
                    max_bars=CHART_MAX_TICK_BARS,
                )
            else:
                bars = aggregate_trades_to_time_bars(
                    trades,
                    interval_seconds=interval.value,
                    max_bars=CHART_MAX_TIME_BARS,
                )
            if not bars:
                return JsonResponse(
                    {"error": _("未能获取 %(symbol)s 的逐笔成交。") % {"symbol": symbol}, "request_id": request_id},
                    status=404,
                    json_dumps_params={"ensure_ascii": False},
                )
            response = {
                "symbol": symbol,
                "range": range_key,
                "interval": {
                    "key": interval.key,
                    "unit": interval.unit,
                    "value": interval.value,
                    "label": interval.label,
                },
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "data_source": "alpaca",
                "bars": bars,
                "window_limited": window_limited,
                "next_page_token": next_token,
            }
            if downgrade_to:
                response["downgrade_to"] = downgrade_to
                response["downgrade_message"] = downgrade_message
            cache_set_object(cache_key, response, cache_ttl, cache_alias=cache_alias)
            response["request_id"] = request_id
            return JsonResponse(response, json_dumps_params={"ensure_ascii": False})

    interval_unit = interval.unit
    value = interval.value
    base_interval = "1m"
    resample_rule = None
    if interval_unit == "minute":
        if value in {1, 5, 15, 30}:
            base_interval = f"{value}m"
        else:
            base_interval = "1m"
            resample_rule = f"{value}min"
    elif interval_unit == "hour":
        base_interval = "1h"
        if value > 1:
            resample_rule = f"{value}h"
    elif interval_unit == "day":
        base_interval = "1d"
        if value > 1:
            resample_rule = f"{value}d"

    frame = market_data.fetch(
        [symbol],
        period=range_key,
        interval=base_interval,
        cache=True,
        timeout=MARKET_REQUEST_TIMEOUT,
        ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
        cache_alias=cache_alias,
        user_id=str(request.user.id),
    )
    bar_frame = frame
    if resample_rule:
        bar_frame = _resample_ohlc_frame(frame, symbol, resample_rule)
    bars = _extract_ohlc(bar_frame, symbol, limit=CHART_MAX_TIME_BARS)
    if not bars:
        return JsonResponse(
            {"error": _("未能获取 %(symbol)s 的行情数据。") % {"symbol": symbol}, "request_id": request_id},
            status=404,
            json_dumps_params={"ensure_ascii": False},
        )

    response = {
        "symbol": symbol,
        "range": range_key,
        "interval": {
            "key": interval.key,
            "unit": interval.unit,
            "value": interval.value,
            "label": interval.label,
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_source": _infer_market_source(frame),
        "bars": bars,
        "window_limited": window_limited,
    }
    if downgrade_to:
        response["downgrade_to"] = downgrade_to
        response["downgrade_message"] = downgrade_message
    cache_set_object(cache_key, response, cache_ttl, cache_alias=cache_alias)
    response["request_id"] = request_id
    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})


@login_required
@require_http_methods(["POST"])
def market_manual_order(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"ok": False, "message": _("请求体解析失败。"), "request_id": request_id}, status=400)
    if not isinstance(payload, dict):
        return JsonResponse({"ok": False, "message": _("请求体解析失败。"), "request_id": request_id}, status=400)
    symbol = str(payload.get("symbol") or "").strip()
    side = str(payload.get("side") or "").strip()
    try:
        qty = float(payload.get("qty") or 0)
    except (TypeError, ValueError):
        qty = 0.0
    try:
        notional = float(payload.get("notional") or 0)
    except (TypeError, ValueError):
        notional = 0.0
    result = submit_manual_order(user=request.user, symbol=symbol, side=side, qty=qty, notional=notional)
    result["request_id"] = request_id
    if not result.get("ok"):
        return JsonResponse(result, status=400)
    return JsonResponse(result)


@login_required
@require_http_methods(["GET", "POST"])
def market_trading_mode(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    profile = RealtimeProfile.objects.filter(user=request.user, is_active=True).first()
    if not profile:
        normalized = validate_realtime_payload({})
        profile = RealtimeProfile.objects.create(
            user=request.user,
            name="Realtime Profile",
            description="",
            payload=normalized,
            is_active=True,
        )
        write_state(DEFAULT_CONFIG_NAME, normalized)

    payload = profile.payload if isinstance(profile.payload, dict) else {}
    config = load_realtime_config_from_payload(payload)

    if request.method == "GET":
        return JsonResponse(
            {
                "ok": True,
                "mode": config.trading.mode,
                "trading_enabled": config.trading.enabled,
                "execution_enabled": config.trading.execution.enabled,
                "request_id": request_id,
            }
        )

    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"ok": False, "message": _("请求体解析失败。"), "request_id": request_id}, status=400)
    if not isinstance(body, dict):
        return JsonResponse({"ok": False, "message": _("请求体解析失败。"), "request_id": request_id}, status=400)
    mode = str(body.get("mode") or body.get("value") or "").strip().lower()
    if mode not in {"paper", "live"}:
        return JsonResponse({"ok": False, "message": _("交易模式无效。"), "request_id": request_id}, status=400)

    trading_payload = payload.get("trading")
    if not isinstance(trading_payload, dict):
        trading_payload = {}
    payload["trading"] = trading_payload
    execution_payload = trading_payload.get("execution")
    if not isinstance(execution_payload, dict):
        execution_payload = {}
    trading_payload["execution"] = execution_payload

    trading_payload["enabled"] = True
    execution_payload["enabled"] = True
    if mode == "live":
        trading_payload["mode"] = "live"
        execution_payload["dry_run"] = False
    else:
        trading_payload["mode"] = "paper"
        execution_payload["dry_run"] = False

    try:
        normalized = validate_realtime_payload(payload)
    except RealtimePayloadError as exc:
        return JsonResponse({"ok": False, "message": f"{exc}", "request_id": request_id}, status=400)

    profile.payload = normalized
    profile.save(update_fields=["payload", "updated_at"])
    write_state(DEFAULT_CONFIG_NAME, normalized)
    config = load_realtime_config_from_payload(normalized)

    return JsonResponse(
        {
            "ok": True,
            "mode": config.trading.mode,
            "trading_enabled": config.trading.enabled,
            "execution_enabled": config.trading.execution.enabled,
            "message": _("交易模式已更新。"),
            "request_id": request_id,
        }
    )


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
        fetch_stock_snapshots(symbols, feed=DEFAULT_FEED, user_id=user_id, timeout=MARKET_REQUEST_TIMEOUT)
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
    cache_key = build_cache_key("market-history", "alpaca", timeframe.key, sorted(symbols))
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
        return "alpaca" if level0 & fields else "unknown"
    return "alpaca" if set(columns) & fields else "unknown"


def _rank_symbols(
    series_map: dict[str, pd.Series],
    timeframe: Timeframe,
    limit: int = 20,
) -> list[dict[str, object]]:
    label = str(timeframe.label)
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
                "period_label": label,
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
