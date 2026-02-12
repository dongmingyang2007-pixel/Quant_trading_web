from __future__ import annotations

import hashlib
import json
import logging
import os
import ast
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import time
from typing import Any, Callable, Iterable, Mapping
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
from ..massive_flatfiles import fetch_historical_bars as fetch_massive_flatfile_historical_bars
from ..error_contract import json_error, log_sanitized_exception
from ..observability import ensure_request_id, record_metric, track_latency
from ..network import get_requests_session, resolve_retry_config, retry_call_result
from ..rate_limit import check_rate_limit, rate_limit_key
from ..models import UserProfile, RealtimeProfile
from ..profile import resolve_api_credential
from .. import bailian_ai
from ..llm import run_llm_chat, LLMIntegrationError
from ..alpaca_data import (
    DEFAULT_FEED,
    resolve_alpaca_data_credentials,
)
from ..market_provider import (
    fetch_company_overview,
    fetch_news,
    fetch_stock_snapshots,
    fetch_stock_trades,
    has_market_data_credentials,
    resolve_market_provider,
    resolve_news_provider,
)
from ..realtime.alpaca import fetch_assets as fetch_alpaca_assets
from ..realtime.market_stream import request_symbol as request_market_symbol, request_symbols as request_market_symbols
from ..realtime.lock import InstanceLock
from ..realtime.storage import read_state, write_state, resolve_state_dir
from ..realtime.manual_orders import submit_manual_order
from ..realtime.config import DEFAULT_CONFIG_NAME, load_realtime_config_from_payload
from ..realtime.schema import RealtimePayloadError, validate_realtime_payload
from ..realtime.chart_store import get_trades as chart_get_trades, get_latest_trade as chart_get_latest_trade
from ..realtime.bars import parse_timestamp
from ..screen_patterns import PATTERN_KEYS, analyze_price_series
from ..screen_training import (
    load_model_meta as load_screen_model_meta,
    load_samples as load_screen_samples,
    save_sample as save_screen_sample,
    train_model as train_screen_model,
)
from ..market_aggregation import (
    aggregate_trades_to_tick_bars,
    aggregate_trades_to_time_bars,
)
from .market_auth import market_auth_debug as _market_auth_debug

LOGGER = logging.getLogger(__name__)


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
LIST_TYPES = (
    "gainers",
    "losers",
    "most_active",
    "top_turnover",
)
LIST_TIMEFRAME_SUPPORT_DEFAULT: dict[str, tuple[str, ...]] = {
    list_type: tuple(TIMEFRAMES.keys())
    for list_type in LIST_TYPES
}


def _resolve_list_timeframe_support() -> dict[str, tuple[str, ...]]:
    configured = getattr(settings, "MARKET_LIST_TIMEFRAME_SUPPORT", None)
    if not isinstance(configured, Mapping):
        return {key: tuple(value) for key, value in LIST_TIMEFRAME_SUPPORT_DEFAULT.items()}
    resolved: dict[str, tuple[str, ...]] = {}
    for list_type, defaults in LIST_TIMEFRAME_SUPPORT_DEFAULT.items():
        raw_value = configured.get(list_type, defaults)
        candidates = [key for key in _setting_list(raw_value, list(defaults)) if key in TIMEFRAMES]
        normalized = list(dict.fromkeys(candidates))
        resolved[list_type] = tuple(normalized or defaults)
    return resolved


def _market_capabilities_payload() -> dict[str, object]:
    support = _resolve_list_timeframe_support()
    return {
        "supported_timeframes_by_list": {
            list_type: list(keys)
            for list_type, keys in support.items()
        }
    }


def _is_list_timeframe_supported(list_type: str, timeframe_key: str) -> bool:
    support = _resolve_list_timeframe_support()
    allowed = support.get(list_type)
    if not allowed:
        return False
    return timeframe_key in allowed


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

MISSING_REASON_SOURCE_NOT_PROVIDED = "source_not_provided"
MISSING_REASON_NOT_APPLICABLE_FUND = "not_applicable_fund"
MISSING_REASON_INSUFFICIENT_WINDOW = "insufficient_window"
MISSING_REASON_TIMEFRAME_SNAPSHOT_PENDING = "timeframe_snapshot_pending"
MISSING_REASON_CODES = {
    MISSING_REASON_SOURCE_NOT_PROVIDED,
    MISSING_REASON_NOT_APPLICABLE_FUND,
    MISSING_REASON_INSUFFICIENT_WINDOW,
    MISSING_REASON_TIMEFRAME_SNAPSHOT_PENDING,
}
RANKING_REASON_FIELDS = ("volume", "dollar_volume", "range_pct", "prev_close", "open")
PROFILE_REASON_FIELDS = ("sector", "industry", "ceo", "hq", "market_cap")
FUND_LIKE_QUOTE_TYPES = {
    "ETF",
    "ETN",
    "MUTUALFUND",
    "MUTUAL FUND",
    "INDEX",
    "CLOSED_END_FUND",
    "CLOSED END FUND",
    "FUND",
}
FUND_LIKE_NAME_KEYWORDS = (
    " etf",
    " fund",
    " trust",
    " index",
    " spdr",
    " ishares",
    " invesco",
    " vanguard",
    " proshares",
)
_QUERY_SANITIZER = re.compile(r"[^A-Z0-9\.\-]")
MARKET_REQUEST_TIMEOUT = max(5, getattr(settings, "MARKET_DATA_TIMEOUT_SECONDS", 25))
MARKET_MAX_WORKERS = max(1, getattr(settings, "MARKET_DATA_MAX_WORKERS", 20))
MARKET_RATE_WINDOW = max(1, getattr(settings, "MARKET_DATA_RATE_WINDOW_SECONDS", 90))
MARKET_RATE_MAX_CALLS = max(1, getattr(settings, "MARKET_DATA_RATE_MAX_CALLS", 45))
MARKET_RATE_CACHE_ALIAS = getattr(settings, "MARKET_DATA_RATE_CACHE_ALIAS", "default")
MARKET_RATE_PROFILE = str(getattr(settings, "MARKET_DATA_RATE_PROFILE", "balanced")).strip().lower()
if MARKET_RATE_PROFILE in {"high", "high_throughput", "turbo"}:
    MARKET_RATE_WINDOW = 60
    MARKET_RATE_MAX_CALLS = max(MARKET_RATE_MAX_CALLS, 12000)

CHART_MAX_TICK_RANGE_SECONDS = _setting_int("MARKET_CHART_MAX_TICK_RANGE_SECONDS", 2 * 3600)
CHART_MAX_SECOND_RANGE_SECONDS = _setting_int("MARKET_CHART_MAX_SECOND_RANGE_SECONDS", 24 * 3600)
CHART_MAX_TICK_BARS = _setting_int("MARKET_CHART_MAX_TICK_BARS", 1000)
CHART_MAX_TIME_BARS = _setting_int("MARKET_CHART_MAX_TIME_BARS", 1600)
CHART_TRADES_PAGE_LIMIT = _setting_int("MARKET_CHART_TRADES_PAGE_LIMIT", 1000)
CHART_TRADES_MAX_PAGES = _setting_int("MARKET_CHART_TRADES_MAX_PAGES", 6)
MARKET_PROFILE_CACHE_TTL = max(120, getattr(settings, "MARKET_PROFILE_CACHE_TTL", 900))
MARKET_NEWS_CACHE_TTL = max(120, getattr(settings, "MARKET_NEWS_CACHE_TTL", 300))
MARKET_NEWS_CACHE_MAX_ITEMS = max(12, _setting_int("MARKET_NEWS_CACHE_MAX_ITEMS", 120))
MARKET_NEWS_PAGE_DEFAULT = max(
    6,
    min(MARKET_NEWS_CACHE_MAX_ITEMS, _setting_int("MARKET_NEWS_PAGE_DEFAULT", 10)),
)
MARKET_NEWS_PAGE_MAX = max(
    MARKET_NEWS_PAGE_DEFAULT,
    min(MARKET_NEWS_CACHE_MAX_ITEMS, _setting_int("MARKET_NEWS_PAGE_MAX", 30)),
)
MARKET_NEWS_SENTIMENT_CACHE_TTL = max(60, _setting_int("MARKET_NEWS_SENTIMENT_CACHE_TTL", 300))
MARKET_NEWS_SENTIMENT_TIMEOUT_SECONDS = max(3, _setting_int("MARKET_NEWS_SENTIMENT_TIMEOUT_SECONDS", 8))
MARKET_AI_SUMMARY_CACHE_TTL = max(120, _setting_int("MARKET_AI_SUMMARY_CACHE_TTL", 600))
MARKET_AI_SUMMARY_TIMEOUT_SECONDS = max(3, _setting_int("MARKET_AI_SUMMARY_TIMEOUT_SECONDS", 8))
MARKET_AI_SUMMARY_MAX_TOKENS = max(240, _setting_int("MARKET_AI_SUMMARY_MAX_TOKENS", 480))
MARKET_AI_SUMMARY_MAX_CHARS = max(240, _setting_int("MARKET_AI_SUMMARY_MAX_CHARS", 420))
MARKET_52W_CACHE_TTL = max(300, _setting_int("MARKET_52W_CACHE_TTL", 6 * 3600))
MARKET_FUNDAMENTALS_CACHE_TTL = max(300, _setting_int("MARKET_FUNDAMENTALS_CACHE_TTL", 6 * 3600))
MARKET_ASSETS_CACHE_TTL = max(300, getattr(settings, "MARKET_ASSETS_CACHE_TTL", 6 * 3600))
MARKET_RANKINGS_CACHE_TTL = max(30, getattr(settings, "MARKET_RANKINGS_CACHE_TTL", 70))
MARKET_RANKINGS_PAGE_CACHE_TTL = max(3, _setting_int("MARKET_RANKINGS_PAGE_CACHE_TTL", 15))
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
MARKET_UNIVERSE_RANKINGS_CACHE_TTL = max(120, _setting_int("MARKET_UNIVERSE_RANKINGS_CACHE_TTL", 1200))
MARKET_UNIVERSE_CHUNK_SIZE = max(50, min(400, _setting_int("MARKET_UNIVERSE_CHUNK_SIZE", 260)))
MARKET_UNIVERSE_MAX_SYMBOLS = max(0, _setting_int("MARKET_UNIVERSE_MAX_SYMBOLS", 0))
MARKET_UNIVERSE_CHUNK_WORKERS = max(1, min(8, _setting_int("MARKET_UNIVERSE_CHUNK_WORKERS", 8)))
MARKET_UNIVERSE_WINDOW_MAX_SYMBOLS = max(0, _setting_int("MARKET_UNIVERSE_WINDOW_MAX_SYMBOLS", 900))
MARKET_ASSETS_PAGE_DEFAULT = max(20, getattr(settings, "MARKET_ASSETS_PAGE_DEFAULT", 50))
MARKET_ASSETS_PAGE_MAX = max(50, getattr(settings, "MARKET_ASSETS_PAGE_MAX", 200))
MARKET_CHART_ANALYZER_NAMESPACE = str(
    getattr(settings, "MARKET_CHART_ANALYZER_NAMESPACE", "market_chart_analyzer")
).strip() or "market_chart_analyzer"
MARKET_CHART_ANALYZER_MIN_POINTS = max(12, _setting_int("MARKET_CHART_ANALYZER_MIN_POINTS", 24))
MARKET_CHART_ANALYZER_MAX_POINTS = max(
    MARKET_CHART_ANALYZER_MIN_POINTS,
    _setting_int("MARKET_CHART_ANALYZER_MAX_POINTS", 360),
)
MARKET_CHART_ANALYZER_TRAIN_MIN_SAMPLES = max(
    12,
    _setting_int("MARKET_CHART_ANALYZER_TRAIN_MIN_SAMPLES", 18),
)
MARKET_CHART_ANALYZER_SERIES_MODES = {"close", "hlc3", "ohlc4"}
MARKET_CHART_ANALYZER_DEFAULT_SERIES_MODE = str(
    getattr(settings, "MARKET_CHART_ANALYZER_DEFAULT_SERIES_MODE", "close")
).strip().lower()
if MARKET_CHART_ANALYZER_DEFAULT_SERIES_MODE not in MARKET_CHART_ANALYZER_SERIES_MODES:
    MARKET_CHART_ANALYZER_DEFAULT_SERIES_MODE = "close"
MARKET_CHART_ANALYZER_DEFAULT_SMOOTHING_WINDOW = max(
    1,
    _setting_int("MARKET_CHART_ANALYZER_DEFAULT_SMOOTHING_WINDOW", 1),
)
MARKET_CHART_ANALYZER_MAX_SMOOTHING_WINDOW = max(
    3,
    _setting_int("MARKET_CHART_ANALYZER_MAX_SMOOTHING_WINDOW", 11),
)
MARKET_RANKINGS_REFRESH_SECONDS = max(60, _setting_int("MARKET_RANKINGS_REFRESH_SECONDS", 300))
MARKET_RANKINGS_REFRESH_MARGIN_SECONDS = max(0, _setting_int("MARKET_RANKINGS_REFRESH_MARGIN_SECONDS", 5))
MARKET_RANKINGS_LOOP_DELAY_SECONDS = max(1, _setting_int("MARKET_RANKINGS_LOOP_DELAY_SECONDS", 3))
MARKET_RANKINGS_STALLED_SECONDS = max(
    120,
    _setting_int("MARKET_RANKINGS_STALLED_SECONDS", MARKET_RANKINGS_LOOP_DELAY_SECONDS * 3),
)
MARKET_RANKINGS_SNAPSHOT_CHUNK_SIZE = max(1, min(300, _setting_int("MARKET_RANKINGS_SNAPSHOT_CHUNK_SIZE", 200)))
MARKET_RANKINGS_TIMEFRAME_CHUNK_SIZE_MASSIVE = max(
    5,
    min(100, _setting_int("MARKET_RANKINGS_TIMEFRAME_CHUNK_SIZE_MASSIVE", 25)),
)
MARKET_RANKINGS_SNAPSHOT_TTL = max(60, _setting_int("MARKET_RANKINGS_SNAPSHOT_TTL", MARKET_RANKINGS_REFRESH_SECONDS))
MARKET_RANKINGS_BACKGROUND_ONLY = _setting_bool("MARKET_RANKINGS_BACKGROUND_ONLY", True)
MARKET_RANKINGS_DISABLE_FILTERS = _setting_bool("MARKET_RANKINGS_DISABLE_FILTERS", True)
MARKET_RANKINGS_ALLOW_STALE_SNAPSHOTS = _setting_bool("MARKET_RANKINGS_ALLOW_STALE_SNAPSHOTS", True)
MARKET_RANKINGS_ALLOW_CROSS_PROVIDER_SNAPSHOT_FALLBACK = _setting_bool(
    "MARKET_RANKINGS_ALLOW_CROSS_PROVIDER_SNAPSHOT_FALLBACK",
    True,
)
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
SNAPSHOT_RANKINGS_BUILDING_STATE = "market_rankings_snapshot_building.json"
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


def _parse_news_limit(value: object, *, default: int = MARKET_NEWS_PAGE_DEFAULT) -> int:
    try:
        limit = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        limit = default
    return max(1, min(MARKET_NEWS_PAGE_MAX, limit))


def _resolve_news_paging(limit_value: object, offset_value: object) -> tuple[int, int]:
    return _parse_news_limit(limit_value), _parse_offset(offset_value)


def _parse_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


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
    if text in LIST_TYPES:
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


def _coerce_timestamp(value: object) -> float | None:
    numeric = _coerce_number(value)
    if numeric is not None:
        return numeric
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return None
    return None


def _coerce_positive_price(value: object) -> float | None:
    number = _coerce_number(value)
    if number is None:
        return None
    if number <= 0:
        return None
    return number


def _resolve_snapshot_last_price(
    snapshot: Mapping[str, object] | None,
    *,
    prev_close: float | None = None,
    allow_quote_fallback: bool = False,
) -> float | None:
    if not isinstance(snapshot, Mapping):
        return None
    latest_trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or {}
    daily_bar = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
    minute_bar = snapshot.get("minuteBar") or snapshot.get("minute_bar") or {}
    latest_quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or {}

    # Match chart semantics first: prefer bar closes over quote-side prices.
    minute_close = _coerce_positive_price((minute_bar or {}).get("c"))
    daily_close = _coerce_positive_price((daily_bar or {}).get("c"))
    trade_price = _coerce_positive_price((latest_trade or {}).get("p") or (latest_trade or {}).get("price"))
    for candidate in (minute_close, daily_close, trade_price):
        if candidate is not None:
            return candidate
    if not allow_quote_fallback:
        return None

    bid = _coerce_positive_price((latest_quote or {}).get("bp"))
    ask = _coerce_positive_price((latest_quote or {}).get("ap"))
    if bid is not None and ask is not None and ask >= bid:
        midpoint = (bid + ask) / 2.0
        if midpoint > 0:
            spread_ratio = (ask - bid) / midpoint
            if spread_ratio <= 0.5:
                return midpoint
            if prev_close not in (None, 0):
                # Wide spreads are common on illiquid warrants; pick the side closer to prev close.
                return bid if abs(bid - float(prev_close)) <= abs(ask - float(prev_close)) else ask
        return bid
    if bid is not None:
        return bid
    if ask is not None:
        if prev_close not in (None, 0):
            ratio = ask / float(prev_close)
            if ratio < 0.1 or ratio > 10.0:
                return None
        return ask
    return None


def _downsample_indices(length: int, max_points: int) -> list[int]:
    if length <= 0:
        return []
    if max_points <= 0 or length <= max_points:
        return list(range(length))
    if max_points == 1:
        return [length - 1]
    step = (length - 1) / float(max_points - 1)
    indices: list[int] = []
    prev = -1
    for idx in range(max_points):
        candidate = int(round(idx * step))
        if candidate <= prev:
            candidate = prev + 1
        if candidate >= length:
            candidate = length - 1
        indices.append(candidate)
        prev = candidate
    indices[-1] = length - 1
    return indices


def _normalize_analyzer_series_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode in MARKET_CHART_ANALYZER_SERIES_MODES:
        return mode
    return MARKET_CHART_ANALYZER_DEFAULT_SERIES_MODE


def _normalize_analyzer_smoothing_window(value: object) -> int:
    try:
        window = int(value)
    except (TypeError, ValueError):
        window = MARKET_CHART_ANALYZER_DEFAULT_SMOOTHING_WINDOW
    window = max(1, min(MARKET_CHART_ANALYZER_MAX_SMOOTHING_WINDOW, window))
    if window > 1 and window % 2 == 0:
        window = window - 1
    return max(1, window)


def _smooth_numeric_values(values: list[float], window: int) -> list[float]:
    if not values or window <= 1 or len(values) < window:
        return list(values)
    half_window = window // 2
    smoothed: list[float] = []
    for idx in range(len(values)):
        total = 0.0
        count = 0
        for offset in range(-half_window, half_window + 1):
            target_index = min(len(values) - 1, max(0, idx + offset))
            value = values[target_index]
            if not isinstance(value, (int, float)):
                continue
            total += float(value)
            count += 1
        smoothed.append(total / count if count else float(values[idx]))
    return smoothed


def _resolve_bar_series_value(bar: Mapping[str, object], series_mode: str) -> float | None:
    close_value = _coerce_number(bar.get("close"))
    if close_value is None:
        return None
    if series_mode == "hlc3":
        high_value = _coerce_number(bar.get("high"))
        low_value = _coerce_number(bar.get("low"))
        high = high_value if high_value is not None else close_value
        low = low_value if low_value is not None else close_value
        return float((high + low + close_value) / 3.0)
    if series_mode == "ohlc4":
        open_value = _coerce_number(bar.get("open"))
        high_value = _coerce_number(bar.get("high"))
        low_value = _coerce_number(bar.get("low"))
        open_price = open_value if open_value is not None else close_value
        high = high_value if high_value is not None else close_value
        low = low_value if low_value is not None else close_value
        return float((open_price + high + low + close_value) / 4.0)
    return float(close_value)


def _extract_chart_analyzer_series(
    payload: Mapping[str, object],
    *,
    max_points: int,
) -> tuple[list[float], list[int], int, dict[str, object]] | None:
    series_mode = _normalize_analyzer_series_mode(payload.get("series_mode"))
    smoothing_window = _normalize_analyzer_smoothing_window(payload.get("smoothing_window"))
    values: list[float] = []
    source = "series"
    raw_series = payload.get("series")
    if isinstance(raw_series, list):
        for entry in raw_series:
            number = _coerce_number(entry)
            if number is None:
                continue
            values.append(float(number))
    elif isinstance(payload.get("bars"), list):
        source = "bars"
        for bar in payload.get("bars", []):  # type: ignore[arg-type]
            if not isinstance(bar, Mapping):
                continue
            number = _resolve_bar_series_value(bar, series_mode)
            if number is None:
                continue
            values.append(float(number))
    else:
        return None
    if not values:
        return None
    original_length = len(values)
    working_values = _smooth_numeric_values(values, smoothing_window)
    indices = _downsample_indices(original_length, max_points)
    sampled = [working_values[idx] for idx in indices]
    return sampled, indices, original_length, {
        "series_mode": series_mode,
        "smoothing_window": smoothing_window,
        "series_source": source,
    }


def _normalize_reason_code(value: object) -> str:
    code = str(value or "").strip()
    if code in MISSING_REASON_CODES:
        return code
    return MISSING_REASON_SOURCE_NOT_PROVIDED


def _normalize_missing_reason_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, reason in value.items():
        field = str(key or "").strip()
        if not field:
            continue
        normalized[field] = _normalize_reason_code(reason)
    return normalized


def _set_item_missing_reason(item: dict[str, object], field: str, reason_code: str) -> None:
    reasons = _normalize_missing_reason_map(item.get("missing_reasons"))
    reasons[field] = _normalize_reason_code(reason_code)
    item["missing_reasons"] = reasons


def _apply_common_market_item_shape(
    item: dict[str, object],
    *,
    timeframe_pending: bool = False,
) -> dict[str, object]:
    period_value = _coerce_number(item.get("change_pct_period"))
    if period_value is None:
        period_value = _coerce_number(item.get("change_pct"))
    if period_value is None:
        period_value = _coerce_number(item.get("change_pct_day"))
    day_value = _coerce_number(item.get("change_pct_day"))
    if day_value is None:
        day_value = _coerce_number(item.get("change_pct"))
    if day_value is None:
        day_value = period_value
    item["change_pct_period"] = period_value
    item["change_pct_day"] = day_value
    item["change_pct"] = period_value

    reasons = _normalize_missing_reason_map(item.get("missing_reasons"))
    for field in RANKING_REASON_FIELDS:
        if _coerce_number(item.get(field)) is None:
            reasons.setdefault(field, MISSING_REASON_SOURCE_NOT_PROVIDED)
        else:
            reasons.pop(field, None)
    if timeframe_pending:
        reasons["timeframe"] = MISSING_REASON_TIMEFRAME_SNAPSHOT_PENDING
    else:
        reasons.pop("timeframe", None)
    if reasons:
        item["missing_reasons"] = reasons
    elif "missing_reasons" in item:
        item.pop("missing_reasons", None)
    return item


def _extract_window_bars(
    frame: pd.DataFrame,
    symbol: str,
    *,
    timeframe: Timeframe,
) -> list[dict[str, float | int]]:
    bars = _extract_ohlc(frame, symbol, limit=420)
    if not bars:
        return []
    if timeframe.interval == "1d":
        window_len = _daily_window_length(timeframe)
        if window_len:
            bars = bars[-window_len:]
    return bars


def _compute_window_metrics_from_bars(
    bars: list[dict[str, float | int]],
    *,
    timeframe: Timeframe | None = None,
) -> tuple[dict[str, float | None], dict[str, str]]:
    metrics: dict[str, float | None] = {
        "price": None,
        "change_pct_period": None,
        "change_pct_day": None,
        "open": None,
        "prev_close": None,
        "volume": None,
        "dollar_volume": None,
        "range_pct": None,
    }
    reasons: dict[str, str] = {}
    parsed: list[dict[str, float | None]] = []

    for bar in bars:
        if not isinstance(bar, dict):
            continue
        close = _coerce_positive_price(bar.get("close"))
        if close is None:
            continue
        open_value = _coerce_positive_price(bar.get("open"))
        high_value = _coerce_positive_price(bar.get("high"))
        low_value = _coerce_positive_price(bar.get("low"))
        parsed.append(
            {
                "open": open_value,
                "high": high_value,
                "low": low_value,
                "close": close,
                "volume": _coerce_number(bar.get("volume")),
            }
        )

    if not parsed:
        reasons = {
            "open": MISSING_REASON_SOURCE_NOT_PROVIDED,
            "prev_close": MISSING_REASON_SOURCE_NOT_PROVIDED,
            "volume": MISSING_REASON_SOURCE_NOT_PROVIDED,
            "dollar_volume": MISSING_REASON_SOURCE_NOT_PROVIDED,
            "range_pct": MISSING_REASON_SOURCE_NOT_PROVIDED,
            "change_pct_period": MISSING_REASON_SOURCE_NOT_PROVIDED,
            "change_pct_day": MISSING_REASON_SOURCE_NOT_PROVIDED,
        }
        return metrics, reasons

    first = parsed[0]
    last = parsed[-1]
    first_open = first.get("open")
    first_close = first.get("close")
    price = last.get("close")
    open_price = first_open if first_open is not None else first_close
    prev_close = parsed[-2].get("close") if len(parsed) >= 2 else None

    high_values = [bar.get("high") if bar.get("high") is not None else bar.get("close") for bar in parsed]
    low_values = [bar.get("low") if bar.get("low") is not None else bar.get("close") for bar in parsed]
    highs = [value for value in high_values if value is not None]
    lows = [value for value in low_values if value is not None]
    max_high = max(highs) if highs else None
    min_low = min(lows) if lows else None

    volume_points = [bar for bar in parsed if bar.get("volume") is not None]
    volume_sum = sum(float(bar.get("volume") or 0.0) for bar in volume_points) if volume_points else None
    dollar_volume = (
        sum(float(bar.get("close") or 0.0) * float(bar.get("volume") or 0.0) for bar in volume_points)
        if volume_points
        else None
    )

    period_base = open_price
    if timeframe is not None and timeframe.key != "1d":
        period_base = first_close if first_close not in (None, 0) else open_price
    period_change = None
    if price is not None and period_base not in (None, 0):
        period_change = (float(price) / float(period_base) - 1.0) * 100.0
    day_change = None
    if price is not None and prev_close not in (None, 0):
        day_change = (float(price) / float(prev_close) - 1.0) * 100.0
    range_base = prev_close if prev_close not in (None, 0) else open_price
    range_pct = None
    if range_base not in (None, 0) and max_high is not None and min_low is not None:
        range_pct = ((float(max_high) - float(min_low)) / float(range_base)) * 100.0

    metrics.update(
        {
            "price": float(price) if price is not None else None,
            "change_pct_period": float(period_change) if period_change is not None else None,
            "change_pct_day": float(day_change) if day_change is not None else None,
            "open": float(open_price) if open_price is not None else None,
            "prev_close": float(prev_close) if prev_close is not None else None,
            "volume": float(volume_sum) if volume_sum is not None else None,
            "dollar_volume": float(dollar_volume) if dollar_volume is not None else None,
            "range_pct": float(range_pct) if range_pct is not None else None,
        }
    )

    if metrics["open"] is None:
        reasons["open"] = MISSING_REASON_SOURCE_NOT_PROVIDED
    if metrics["prev_close"] is None:
        reasons["prev_close"] = (
            MISSING_REASON_INSUFFICIENT_WINDOW if len(parsed) < 2 else MISSING_REASON_SOURCE_NOT_PROVIDED
        )
    if metrics["volume"] is None:
        reasons["volume"] = MISSING_REASON_SOURCE_NOT_PROVIDED
    if metrics["dollar_volume"] is None:
        reasons["dollar_volume"] = MISSING_REASON_SOURCE_NOT_PROVIDED
    if metrics["range_pct"] is None:
        reasons["range_pct"] = (
            MISSING_REASON_INSUFFICIENT_WINDOW if len(parsed) < 2 else MISSING_REASON_SOURCE_NOT_PROVIDED
        )
    if metrics["change_pct_period"] is None:
        reasons["change_pct_period"] = MISSING_REASON_SOURCE_NOT_PROVIDED
    if metrics["change_pct_day"] is None:
        reasons["change_pct_day"] = (
            MISSING_REASON_INSUFFICIENT_WINDOW if len(parsed) < 2 else MISSING_REASON_SOURCE_NOT_PROVIDED
        )

    return metrics, reasons


def _resolve_symbol_window_metrics(
    symbols: Iterable[str],
    *,
    timeframe: Timeframe,
    user_id: str | None,
) -> tuple[dict[str, dict[str, object]], str]:
    unique = [sym for sym in dict.fromkeys(symbols) if sym]
    if not unique:
        return {}, "unknown"

    def _resolve_chunk(chunk_symbols: list[str]) -> tuple[dict[str, dict[str, object]], str]:
        frame = market_data.fetch(
            chunk_symbols,
            period=timeframe.period,
            interval=timeframe.interval,
            cache=True,
            timeout=MARKET_REQUEST_TIMEOUT,
            ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
            cache_alias=getattr(settings, "MARKET_HISTORY_CACHE_ALIAS", None),
            user_id=user_id,
        )
        chunk_source = _infer_market_source(frame)
        chunk_map: dict[str, dict[str, object]] = {}
        for symbol in chunk_symbols:
            bars = _extract_window_bars(frame, symbol, timeframe=timeframe)
            metrics, missing = _compute_window_metrics_from_bars(bars, timeframe=timeframe)
            payload: dict[str, object] = {**metrics}
            if missing:
                payload["missing_reasons"] = missing
            chunk_map[symbol] = payload
        return chunk_map, chunk_source

    def _merge_source(current: str, incoming: str) -> str:
        if incoming and incoming != "unknown":
            if current == "unknown":
                return incoming
            if incoming == "alpaca":
                return incoming
        return current

    chunks = list(_iter_chunks(unique, MARKET_UNIVERSE_CHUNK_SIZE))
    if not chunks:
        return {}, "unknown"

    metrics_map: dict[str, dict[str, object]] = {}
    source = "unknown"
    if MARKET_UNIVERSE_CHUNK_WORKERS <= 1 or len(chunks) == 1:
        for chunk_symbols in chunks:
            chunk_map, chunk_source = _resolve_chunk(chunk_symbols)
            metrics_map.update(chunk_map)
            source = _merge_source(source, chunk_source)
        return metrics_map, source

    with ThreadPoolExecutor(max_workers=MARKET_UNIVERSE_CHUNK_WORKERS) as executor:
        futures = [
            executor.submit(_resolve_chunk, chunk_symbols)
            for chunk_symbols in chunks
        ]
        for future in futures:
            try:
                chunk_map, chunk_source = future.result()
            except Exception:
                continue
            metrics_map.update(chunk_map)
            source = _merge_source(source, chunk_source)
    return metrics_map, source


def _snapshot_metrics_from_payload(snapshot: Mapping[str, object]) -> dict[str, float | None]:
    daily_bar = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
    minute_bar = snapshot.get("minuteBar") or snapshot.get("minute_bar") or {}
    prev_bar = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar") or {}
    prev_close = _coerce_positive_price((prev_bar or {}).get("c"))
    last_price = _resolve_snapshot_last_price(snapshot, prev_close=prev_close, allow_quote_fallback=False)
    open_price = _coerce_positive_price((daily_bar or {}).get("o"))
    high_price = _coerce_number((daily_bar or {}).get("h"))
    low_price = _coerce_number((daily_bar or {}).get("l"))
    volume = _coerce_number((daily_bar or {}).get("v") or (minute_bar or {}).get("v"))

    change_pct = None
    if last_price is not None and prev_close not in (None, 0):
        change_pct = (last_price / prev_close - 1.0) * 100.0
    dollar_volume = None
    if last_price is not None and volume is not None:
        dollar_volume = last_price * volume
    range_pct = None
    if prev_close not in (None, 0) and high_price is not None and low_price is not None:
        range_pct = ((high_price - low_price) / prev_close) * 100.0

    return {
        "price": last_price,
        "change_pct_period": change_pct,
        "change_pct_day": change_pct,
        "open": open_price,
        "prev_close": prev_close,
        "volume": volume,
        "dollar_volume": dollar_volume,
        "range_pct": range_pct,
    }


def _enrich_rows_with_snapshot_metrics(
    rows: list[dict[str, object]],
    *,
    user_id: str | None,
    provider: str | None,
) -> list[dict[str, object]]:
    if not rows:
        return rows
    symbols = [
        str(item.get("symbol") or "").strip().upper()
        for item in rows
        if isinstance(item, dict) and item.get("symbol")
    ]
    symbols = [sym for sym in dict.fromkeys(symbols) if sym]
    if not symbols:
        return rows
    snapshots = fetch_stock_snapshots(
        symbols,
        feed=DEFAULT_FEED,
        user_id=user_id,
        timeout=MARKET_REQUEST_TIMEOUT,
        provider=provider,
    )
    if not isinstance(snapshots, dict) or not snapshots:
        return rows

    for item in rows:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        snapshot = snapshots.get(symbol)
        if not isinstance(snapshot, Mapping):
            continue
        metrics = _snapshot_metrics_from_payload(snapshot)
        reasons = _normalize_missing_reason_map(item.get("missing_reasons"))
        for field in ("price", "change_pct_period", "change_pct_day", "volume", "dollar_volume", "range_pct", "open", "prev_close"):
            value = _coerce_number(item.get(field))
            if value is None:
                snapshot_value = _coerce_number(metrics.get(field))
                if snapshot_value is not None:
                    item[field] = snapshot_value
                    if field == "price":
                        item["last"] = snapshot_value
                        if _coerce_number(item.get("change_pct")) is None:
                            change_val = _coerce_number(metrics.get("change_pct_period"))
                            if change_val is not None:
                                item["change_pct"] = change_val
            if _coerce_number(item.get(field)) is not None:
                reasons.pop(field, None)
        if reasons:
            item["missing_reasons"] = reasons
        else:
            item.pop("missing_reasons", None)
    return rows


def _enrich_rows_with_window_metrics(
    rows: list[dict[str, object]],
    *,
    timeframe: Timeframe,
    user_id: str | None,
) -> list[dict[str, object]]:
    if not rows or timeframe.key == "1d":
        return rows
    symbols = [
        str(item.get("symbol") or "").strip().upper()
        for item in rows
        if isinstance(item, dict) and item.get("symbol")
    ]
    symbols = [sym for sym in dict.fromkeys(symbols) if sym]
    if not symbols:
        return rows
    metrics_map, _source = _resolve_symbol_window_metrics(symbols, timeframe=timeframe, user_id=user_id)
    if not isinstance(metrics_map, dict) or not metrics_map:
        return rows
    for item in rows:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        metrics = metrics_map.get(symbol)
        if not isinstance(metrics, Mapping):
            continue
        reasons = _normalize_missing_reason_map(item.get("missing_reasons"))
        metric_reasons = _normalize_missing_reason_map(metrics.get("missing_reasons"))
        for field in ("price", "change_pct_period", "change_pct_day", "volume", "dollar_volume", "range_pct", "open", "prev_close"):
            if _coerce_number(item.get(field)) is None:
                metric_value = _coerce_number(metrics.get(field))
                if metric_value is not None:
                    item[field] = metric_value
                    if field == "price":
                        item["last"] = metric_value
                    if field == "change_pct_period" and _coerce_number(item.get("change_pct")) is None:
                        item["change_pct"] = metric_value
            if _coerce_number(item.get(field)) is None:
                reason_code = metric_reasons.get(field)
                if reason_code:
                    reasons[field] = reason_code
            else:
                reasons.pop(field, None)
        if reasons:
            item["missing_reasons"] = reasons
        else:
            item.pop("missing_reasons", None)
    return rows


def _is_fund_like_profile(profile: Mapping[str, object]) -> bool:
    quote_type = str(profile.get("quote_type") or profile.get("quoteType") or "").strip().upper()
    if quote_type in FUND_LIKE_QUOTE_TYPES:
        return True
    name_parts = [
        str(profile.get("name") or ""),
        str(profile.get("shortName") or ""),
        str(profile.get("symbol") or ""),
    ]
    lowered = " ".join(name_parts).lower()
    return any(keyword in lowered for keyword in FUND_LIKE_NAME_KEYWORDS)


def _profile_field_value(profile: Mapping[str, object], field: str) -> object:
    if field == "hq":
        hq = profile.get("hq") or profile.get("headquarters") or profile.get("location")
        if hq:
            return hq
        city = profile.get("city")
        state = profile.get("state")
        country = profile.get("country")
        joined = ", ".join(str(part) for part in (city, state, country) if part)
        return joined
    return profile.get(field)


def _build_profile_missing_reasons(profile: Mapping[str, object]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    if not profile:
        return {field: MISSING_REASON_SOURCE_NOT_PROVIDED for field in PROFILE_REASON_FIELDS}
    fund_like = _is_fund_like_profile(profile)
    for field in PROFILE_REASON_FIELDS:
        value = _profile_field_value(profile, field)
        if value not in (None, "", [], {}):
            continue
        reasons[field] = (
            MISSING_REASON_NOT_APPLICABLE_FUND if fund_like else MISSING_REASON_SOURCE_NOT_PROVIDED
        )
    return reasons


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


def _snapshot_timeframe_building_state_name(timeframe_key: str) -> str:
    return f"market_rankings_snapshot_{timeframe_key}_building.json"


def _snapshot_timeframe_payload_summary(payload: dict[str, object]) -> dict[str, object]:
    summary = _snapshot_refresh_payload_summary(payload)
    if "timeframe" in payload:
        summary["timeframe"] = payload.get("timeframe")
    return summary


def _snapshot_payload_source(payload: Mapping[str, object], *, fallback: str = "unknown") -> str:
    source = str(payload.get("source") or "").strip().lower()
    if source:
        return source
    fallback_source = str(fallback or "").strip().lower()
    return fallback_source or "unknown"


def _load_timeframe_snapshot_payload(
    timeframe_key: str,
    *,
    provider: str | None = None,
) -> dict[str, object] | None:
    payload = read_state(_snapshot_timeframe_state_name(timeframe_key), default={})
    if not isinstance(payload, dict) or not payload:
        return None
    if payload.get("status") != "complete":
        return None
    if provider:
        payload_source = _snapshot_payload_source(payload, fallback="")
        if payload_source and payload_source != provider:
            if not MARKET_RANKINGS_ALLOW_CROSS_PROVIDER_SNAPSHOT_FALLBACK:
                return None
            payload = dict(payload)
            payload["provider_expected"] = provider
            payload["provider_actual"] = payload_source
            payload["provider_mismatch"] = True
    ts = payload.get("generated_ts")
    if isinstance(ts, (int, float)):
        if time.time() - float(ts) > MARKET_RANKINGS_TIMEFRAME_SNAPSHOT_TTL:
            return payload if MARKET_RANKINGS_ALLOW_STALE_SNAPSHOTS else None
    else:
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    if not rows:
        return None
    return payload


def _load_timeframe_snapshot_rows(timeframe_key: str, *, provider: str | None = None) -> list[dict[str, object]] | None:
    payload = _load_timeframe_snapshot_payload(timeframe_key, provider=provider)
    if not isinstance(payload, dict):
        return None
    rows = payload.get("rows")
    return rows if isinstance(rows, list) and rows else None


def _load_snapshot_rankings_latest_payload(*, provider: str | None = None) -> dict[str, object] | None:
    payload = read_state(SNAPSHOT_RANKINGS_STATE, default={})
    if not isinstance(payload, dict) or not payload:
        return None
    if payload.get("status") != "complete":
        return None
    if provider:
        payload_source = _snapshot_payload_source(payload, fallback="")
        if payload_source and payload_source != provider:
            if not MARKET_RANKINGS_ALLOW_CROSS_PROVIDER_SNAPSHOT_FALLBACK:
                return None
            payload = dict(payload)
            payload["provider_expected"] = provider
            payload["provider_actual"] = payload_source
            payload["provider_mismatch"] = True
    ts = payload.get("generated_ts")
    if isinstance(ts, (int, float)):
        if time.time() - float(ts) > MARKET_RANKINGS_SNAPSHOT_TTL:
            return payload if MARKET_RANKINGS_ALLOW_STALE_SNAPSHOTS else None
    else:
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return None
    if not rows:
        return None
    return payload


def _load_snapshot_rankings_latest_rows(*, provider: str | None = None) -> list[dict[str, object]] | None:
    payload = _load_snapshot_rankings_latest_payload(provider=provider)
    if not isinstance(payload, dict):
        return None
    rows = payload.get("rows")
    return rows if isinstance(rows, list) and rows else None


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


def _snapshot_progress_percent(progress: object) -> int:
    if not isinstance(progress, Mapping):
        return 0
    percent = _coerce_number(progress.get("percent"))
    if percent is not None:
        try:
            return max(0, min(100, int(round(percent))))
        except Exception:
            return 0
    completed = _coerce_number(progress.get("chunks_completed"))
    total = _coerce_number(progress.get("total_chunks"))
    if total and total > 0 and completed is not None:
        try:
            return max(0, min(100, int(round((completed / total) * 100))))
        except Exception:
            return 0
    return 0


def _resolve_active_snapshot_payload(
    timeframe_key: str,
    *,
    provider: str | None,
) -> dict[str, object] | None:
    if timeframe_key == "1d":
        return _load_snapshot_rankings_latest_payload(provider=provider)
    return _load_timeframe_snapshot_payload(timeframe_key, provider=provider)


def _resolve_building_snapshot_payload(
    timeframe_key: str,
    *,
    provider: str | None,
) -> dict[str, object] | None:
    state_name = SNAPSHOT_RANKINGS_BUILDING_STATE
    if timeframe_key != "1d":
        state_name = _snapshot_timeframe_building_state_name(timeframe_key)
    payload = read_state(state_name, default={})
    if not isinstance(payload, dict) or not payload:
        return None
    if provider:
        payload_source = str(payload.get("source") or "").strip().lower()
        if payload_source and payload_source != provider:
            return None
    payload_status = str(payload.get("status") or "").strip().lower()
    if payload_status == "running":
        now_ts = time.time()
        updated_ts = _coerce_timestamp(payload.get("updated_ts"))
        if updated_ts is None:
            updated_ts = _coerce_timestamp(payload.get("updated_at"))
        if updated_ts is None:
            updated_ts = _coerce_timestamp(payload.get("started_ts"))
        if updated_ts is None:
            updated_ts = _coerce_timestamp(payload.get("started_at"))

        chunks_completed = _coerce_number(payload.get("chunks_completed"))
        total_chunks = _coerce_number(payload.get("total_chunks"))
        looks_finished = (
            chunks_completed is not None
            and total_chunks is not None
            and total_chunks > 0
            and chunks_completed >= total_chunks
        )
        stale_running = updated_ts is not None and now_ts - float(updated_ts) > MARKET_RANKINGS_STALLED_SECONDS
        finished_but_not_closed = (
            looks_finished
            and updated_ts is not None
            and now_ts - float(updated_ts) > max(10, MARKET_RANKINGS_LOOP_DELAY_SECONDS * 2)
        )
        if stale_running or finished_but_not_closed:
            stalled_payload = dict(payload)
            stalled_payload["status"] = "stalled"
            stalled_payload["stalled"] = True
            stalled_payload["stalled_at"] = now_ts
            stalled_payload["stalled_ts"] = now_ts
            return stalled_payload
    return payload


def _snapshot_stale_seconds(payload: object) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    generated_ts = _coerce_number(payload.get("generated_ts"))
    if generated_ts is None:
        generated_ts = _coerce_number(payload.get("generated_at"))
    if generated_ts is None:
        return None
    return max(0, int(time.time() - generated_ts))


def _snapshot_ttl_for_timeframe(timeframe_key: str) -> int:
    return MARKET_RANKINGS_SNAPSHOT_TTL if timeframe_key == "1d" else MARKET_RANKINGS_TIMEFRAME_SNAPSHOT_TTL


def _snapshot_state_payload(
    *,
    timeframe_key: str,
    provider: str,
    used_active_snapshot: bool,
    list_type: str | None = None,
) -> dict[str, object]:
    active_payload = _resolve_active_snapshot_payload(timeframe_key, provider=provider)
    stale_seconds = _snapshot_stale_seconds(active_payload)
    building_payload = _resolve_building_snapshot_payload(timeframe_key, provider=provider)
    progress: Mapping[str, object] | None = None
    building = False
    build_state = "idle"
    build_stalled = False
    if isinstance(building_payload, Mapping):
        payload_status = str(building_payload.get("status") or "").strip().lower()
        if payload_status in {"running", "error", "idle", "stalled"}:
            build_state = payload_status
        building = payload_status == "running"
        build_stalled = payload_status == "stalled" or bool(building_payload.get("stalled"))
        payload_progress = building_payload.get("progress")
        if isinstance(payload_progress, Mapping):
            progress = payload_progress
        elif payload_status == "running":
            progress = {
                "status": "running",
                "chunks_completed": building_payload.get("chunks_completed"),
                "total_chunks": building_payload.get("total_chunks"),
            }
    if progress is None and timeframe_key == "1d":
        refresh_meta = _snapshot_refresh_meta() or {}
        refresh_progress = refresh_meta.get("progress")
        if isinstance(refresh_progress, Mapping):
            progress = refresh_progress
            building = building or str(refresh_progress.get("status") or "").lower() == "running"
            if build_state == "idle" and building:
                build_state = "running"
    building_progress = _snapshot_progress_percent(progress) if build_state == "running" else 0
    if used_active_snapshot and isinstance(active_payload, Mapping):
        served_from = "building_fallback" if building else "active"
    else:
        served_from = "none"
    active_generated_at = None
    if isinstance(active_payload, Mapping):
        active_generated_at = active_payload.get("generated_at") or active_payload.get("generated_ts")
    active_schema_valid = None
    if isinstance(active_payload, Mapping) and list_type:
        active_rows = active_payload.get("rows")
        active_schema_valid = _snapshot_rows_schema_valid_for_list(active_rows, list_type=list_type)
    return {
        "served_from": served_from,
        "active_generated_at": active_generated_at,
        "building_progress": building_progress,
        "building": bool(building),
        "build_state": build_state,
        "build_stalled": bool(build_stalled),
        "stale_seconds": stale_seconds,
        "stale_threshold_seconds": _snapshot_ttl_for_timeframe(timeframe_key),
        "provider": provider,
        "active_schema_valid": active_schema_valid,
    }


def _resolve_market_data_state(
    *,
    items: list[dict[str, object]],
    snapshot_state: Mapping[str, object],
) -> str:
    has_items = bool(items)
    served_from = str(snapshot_state.get("served_from") or "none")
    building = bool(snapshot_state.get("building"))
    active_schema_valid = snapshot_state.get("active_schema_valid")
    stale_seconds = _coerce_number(snapshot_state.get("stale_seconds"))
    stale_threshold = _coerce_number(snapshot_state.get("stale_threshold_seconds"))
    if has_items:
        if served_from == "building_fallback":
            return "stale"
        if stale_seconds is not None and stale_threshold is not None and stale_seconds > stale_threshold:
            return "stale"
        return "ready"
    if building:
        return "building"
    if active_schema_valid is False:
        return "limited"
    if served_from == "active":
        return "stale"
    return "limited"


def _cached_rankings_payload_usable(payload: Mapping[str, object]) -> bool:
    """Guard against stale/invalid page-cache payloads after snapshot files are cleared."""
    items = payload.get("items")
    has_items = isinstance(items, list) and bool(items)
    # Ranking page cache should never serve empty items, otherwise the UI can be
    # stuck in an empty state even after snapshots become available again.
    return bool(has_items)


def _maybe_trigger_snapshot_refresh_nonblocking(
    *,
    request: HttpRequest,
    timeframe_key: str,
    list_type: str,
    snapshot_state: Mapping[str, object],
    data_state: str,
) -> None:
    if timeframe_key == "1d":
        return
    if list_type not in {"gainers", "losers", "most_active", "top_turnover"}:
        return
    is_building = bool(snapshot_state.get("building"))
    is_stalled = bool(snapshot_state.get("build_stalled"))
    is_error = str(snapshot_state.get("build_state") or "").lower() == "error"
    stale_seconds = _coerce_number(snapshot_state.get("stale_seconds"))
    stale_threshold = _coerce_number(snapshot_state.get("stale_threshold_seconds"))
    served_from = str(snapshot_state.get("served_from") or "none")
    should_trigger = False
    if data_state in {"building", "limited"}:
        should_trigger = True
    elif is_stalled:
        should_trigger = True
    elif is_error:
        should_trigger = True
    elif not is_building and stale_seconds is not None and stale_threshold and stale_seconds >= stale_threshold:
        should_trigger = True
    elif not is_building and served_from != "active":
        should_trigger = True
    if not should_trigger:
        return
    user_id = str(request.user.id) if getattr(request, "user", None) and request.user.is_authenticated else None
    try:
        from ..tasks import trigger_market_snapshot_refresh

        trigger_market_snapshot_refresh(user_id=user_id, prefer_thread=True)
    except Exception:
        LOGGER.debug("failed to trigger snapshot refresh in background", exc_info=True)


def _build_snapshot_rankings(user_id: str | None) -> list[dict[str, object]]:
    market_source = resolve_market_provider(user_id=user_id)
    cached_rows = _load_snapshot_rankings_latest_rows(provider=market_source)
    if cached_rows is not None:
        return cached_rows
    if MARKET_RANKINGS_BACKGROUND_ONLY:
        return []
    cache_key = build_cache_key("market-rankings-snapshots", market_source, user_id or "anon")

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
            daily_bar = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
            minute_bar = snapshot.get("minuteBar") or snapshot.get("minute_bar") or {}
            prev_bar = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar") or {}
            prev_close = _coerce_positive_price((prev_bar or {}).get("c"))
            last_price = _resolve_snapshot_last_price(snapshot, prev_close=prev_close, allow_quote_fallback=False)
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
            open_price = _coerce_positive_price((daily_bar or {}).get("o"))
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
                    "open": open_price,
                    "prev_close": prev_close,
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
    started_at = float(started_ts)
    progress_payload = {
        "status": "running",
        "started_at": started_at,
        "started_ts": started_ts,
        "updated_at": started_at,
        "updated_ts": started_ts,
    }
    write_state(SNAPSHOT_RANKINGS_PROGRESS_STATE, progress_payload)
    write_state(
        SNAPSHOT_RANKINGS_BUILDING_STATE,
        {
            "status": "running",
            "started_at": started_at,
            "started_ts": started_ts,
            "updated_at": started_at,
            "updated_ts": started_ts,
        },
    )

    try:
        market_source = resolve_market_provider(user_id=user_id)
        if market_source == "alpaca":
            key_id, secret = resolve_alpaca_data_credentials(user_id=user_id)
            has_credentials = bool(key_id and secret)
        else:
            has_credentials = has_market_data_credentials(user_id=user_id, provider=market_source)
        if not has_credentials:
            now_ts = time.time()
            error_payload = {
                "status": "error",
                "started_at": started_at,
                "started_ts": started_ts,
                "updated_at": now_ts,
                "updated_ts": now_ts,
                "error": "missing_credentials",
                "source": market_source,
            }
            write_state(SNAPSHOT_RANKINGS_PROGRESS_STATE, error_payload)
            write_state(SNAPSHOT_RANKINGS_BUILDING_STATE, error_payload)
            return error_payload

        assets = _filter_rankable_assets(_normalize_assets(_load_assets_master(user_id)))
        asset_map = {asset.get("symbol"): asset for asset in assets if asset.get("symbol")}
        symbols = list(asset_map.keys())
        total_symbols = len(symbols)
        chunk_size = min(300, max(1, MARKET_RANKINGS_SNAPSHOT_CHUNK_SIZE))
        timeframes = [TIMEFRAMES[key] for key in MARKET_RANKINGS_TIMEFRAME_KEYS if key in TIMEFRAMES]
        snapshot_chunks = list(_iter_chunks(symbols, chunk_size))

        api_calls = 0
        preloaded_snapshots: dict[str, object] | None = None
        timeframe_symbols = symbols
        if market_source == "massive" and symbols:
            try:
                snapshot_payload = fetch_stock_snapshots(
                    symbols,
                    feed=DEFAULT_FEED,
                    user_id=user_id,
                    timeout=MARKET_REQUEST_TIMEOUT,
                    provider=market_source,
                )
            except Exception:
                snapshot_payload = {}
            if isinstance(snapshot_payload, dict) and snapshot_payload:
                preloaded_snapshots = snapshot_payload
                api_calls += 1
                if timeframes:
                    candidate_rows: list[dict[str, object]] = []
                    for symbol in symbols:
                        snapshot_entry = preloaded_snapshots.get(symbol)
                        if not isinstance(snapshot_entry, Mapping):
                            continue
                        metrics = _snapshot_metrics_from_payload(snapshot_entry)
                        candidate_rows.append(
                            {
                                "symbol": symbol,
                                "dollar_volume": _coerce_number(metrics.get("dollar_volume")),
                                "volume": _coerce_number(metrics.get("volume")),
                            }
                        )
                    ranked_timeframe_symbols = _sort_rows_by_metric(
                        candidate_rows,
                        "dollar_volume",
                        reverse=True,
                        predicate=lambda value: value > 0,
                    )
                    if not ranked_timeframe_symbols:
                        ranked_timeframe_symbols = _sort_rows_by_metric(
                            candidate_rows,
                            "volume",
                            reverse=True,
                            predicate=lambda value: value > 0,
                        )
                    max_timeframe_symbols = max(100, MARKET_UNIVERSE_WINDOW_MAX_SYMBOLS)
                    timeframe_symbols = [
                        str(item.get("symbol") or "").strip().upper()
                        for item in ranked_timeframe_symbols[:max_timeframe_symbols]
                        if isinstance(item, Mapping) and item.get("symbol")
                    ]
                    if not timeframe_symbols:
                        timeframe_symbols = symbols[:max_timeframe_symbols]

        timeframe_chunk_size = chunk_size
        if market_source == "massive":
            timeframe_chunk_size = max(5, min(chunk_size, MARKET_RANKINGS_TIMEFRAME_CHUNK_SIZE_MASSIVE))
        timeframe_chunks = list(_iter_chunks(timeframe_symbols, timeframe_chunk_size)) if timeframes else []
        timeframe_total_chunks = len(timeframe_chunks)
        total_chunks = len(snapshot_chunks) + timeframe_total_chunks
        rows: list[dict[str, object]] = []
        completed_chunks = 0
        timeframe_completed_chunks = 0

        has_active_1d_snapshot = _load_snapshot_rankings_latest_payload(provider=market_source) is not None
        has_active_timeframe_snapshots = all(
            _load_timeframe_snapshot_payload(timeframe.key, provider=market_source) is not None
            for timeframe in timeframes
        ) if timeframes else True
        cold_start_refresh = not has_active_1d_snapshot or not has_active_timeframe_snapshots

        target_seconds = float(MARKET_RANKINGS_REFRESH_SECONDS - MARKET_RANKINGS_REFRESH_MARGIN_SECONDS)
        if cold_start_refresh or total_chunks <= 0:
            target_seconds = 0.0
        else:
            target_seconds = max(0.0, target_seconds)

        def _update_progress() -> None:
            elapsed = time.time() - started_ts
            now_ts = time.time()
            progress_payload = {
                "status": "running",
                "started_at": started_at,
                "started_ts": started_ts,
                "updated_at": now_ts,
                "updated_ts": now_ts,
                "total_symbols": total_symbols,
                "chunk_size": chunk_size,
                "total_chunks": total_chunks,
                "chunks_completed": completed_chunks,
                "api_calls": api_calls,
                "elapsed_seconds": round(elapsed, 2),
                "target_seconds": target_seconds,
            }
            write_state(SNAPSHOT_RANKINGS_PROGRESS_STATE, progress_payload)
            write_state(
                SNAPSHOT_RANKINGS_BUILDING_STATE,
                {
                    "status": "running",
                    "started_at": started_at,
                    "started_ts": started_ts,
                    "updated_at": now_ts,
                    "updated_ts": now_ts,
                    "source": market_source,
                    "total_chunks": total_chunks,
                    "chunks_completed": completed_chunks,
                    "api_calls": api_calls,
                    "progress": {
                        "status": "running",
                        "chunks_completed": completed_chunks,
                        "total_chunks": total_chunks,
                        "percent": _snapshot_progress_percent(
                            {"chunks_completed": completed_chunks, "total_chunks": total_chunks}
                        ),
                    },
                },
            )

            if total_chunks > 0 and target_seconds > 0 and completed_chunks < total_chunks:
                target_elapsed = target_seconds * completed_chunks / total_chunks
                sleep_for = target_elapsed - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        def _write_timeframe_building_progress() -> None:
            if not timeframes:
                return
            total = max(1, timeframe_total_chunks)
            completed_global = max(0, timeframe_completed_chunks)
            percent = max(0, min(100, int(round((completed_global / total) * 100))))
            now_ts = time.time()
            for timeframe in timeframes:
                write_state(
                    _snapshot_timeframe_building_state_name(timeframe.key),
                    {
                        "status": "running",
                        "started_at": started_at,
                        "started_ts": started_ts,
                        "updated_at": now_ts,
                        "updated_ts": now_ts,
                        "source": market_source,
                        "timeframe": timeframe.key,
                        "chunks_completed": completed_global,
                        "total_chunks": total,
                        "progress": {
                            "status": "running",
                            "chunks_completed": completed_global,
                            "total_chunks": total,
                            "percent": percent,
                        },
                    },
                )

        if timeframes:
            _write_timeframe_building_progress()

        if timeframes:
            timeframe_rows_map: dict[str, list[dict[str, object]]] = {tf.key: [] for tf in timeframes}
            timeframe_started = time.time()
            timeframe_error: str | None = None
            timeframe_api_calls = 0
            max_timeframe = max(timeframes, key=lambda item: _daily_window_length(item) or 0)
            daily_timeframe = _resolve_daily_timeframe(max_timeframe)
            try:
                for chunk in timeframe_chunks:
                    frame = market_data.fetch(
                        chunk,
                        period=daily_timeframe.period,
                        interval=daily_timeframe.interval,
                        cache=True,
                        timeout=MARKET_REQUEST_TIMEOUT,
                        ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
                        cache_alias=getattr(settings, "MARKET_HISTORY_CACHE_ALIAS", None),
                        user_id=user_id,
                    )
                    api_calls += 1
                    timeframe_api_calls += 1
                    if isinstance(frame, pd.DataFrame) and not frame.empty:
                        for symbol in chunk:
                            asset = asset_map.get(symbol)
                            if not asset:
                                continue
                            bars_full = _extract_ohlc(frame, symbol, limit=420)
                            if not bars_full:
                                continue
                            for timeframe in timeframes:
                                bars = bars_full
                                window_len = _daily_window_length(timeframe)
                                if window_len and len(bars_full) > window_len:
                                    bars = bars_full[-window_len:]
                                metrics, missing = _compute_window_metrics_from_bars(bars, timeframe=timeframe)
                                row: dict[str, object] = {
                                    "symbol": symbol,
                                    "name": asset.get("name") or "",
                                    "exchange": asset.get("exchange") or "",
                                    "price": _coerce_number(metrics.get("price")),
                                    "change_pct_day": _coerce_number(metrics.get("change_pct_day")),
                                    "change_pct_period": _coerce_number(metrics.get("change_pct_period")),
                                    "volume": _coerce_number(metrics.get("volume")),
                                    "dollar_volume": _coerce_number(metrics.get("dollar_volume")),
                                    "open": _coerce_number(metrics.get("open")),
                                    "prev_close": _coerce_number(metrics.get("prev_close")),
                                    "range_pct": _coerce_number(metrics.get("range_pct")),
                                    # Convert lazy translation proxies to plain strings
                                    # before persisting state JSON.
                                    "period_label": str(timeframe.label),
                                    "period_label_en": str(timeframe.label_en),
                                }
                                if not MARKET_RANKINGS_DISABLE_FILTERS and not _passes_rank_filters(row):
                                    continue
                                if missing:
                                    row["missing_reasons"] = missing
                                timeframe_rows_map[timeframe.key].append(row)
                    completed_chunks += 1
                    timeframe_completed_chunks += 1
                    _write_timeframe_building_progress()
                    _update_progress()
            except Exception:
                LOGGER.exception(
                    "Snapshot timeframe refresh failed user_id=%s timeframe=%s",
                    user_id,
                    max_timeframe.key,
                )
                timeframe_error = "timeframe_refresh_failed"

            duration = time.time() - timeframe_started
            for timeframe in timeframes:
                timeframe_rows = timeframe_rows_map.get(timeframe.key, [])
                if timeframe_rows:
                    timeframe_rows.sort(key=lambda item: item.get("change_pct_period") or 0, reverse=True)
                schema_error = None
                if timeframe_rows:
                    has_volume_metric = _snapshot_rows_schema_valid_for_list(
                        timeframe_rows,
                        list_type="most_active",
                    )
                    has_turnover_metric = _snapshot_rows_schema_valid_for_list(
                        timeframe_rows,
                        list_type="top_turnover",
                    )
                    if not has_volume_metric or not has_turnover_metric:
                        schema_error = "invalid_snapshot_schema"
                timeframe_status_error = timeframe_error or schema_error
                generated_ts = time.time()
                timeframe_payload: dict[str, object] = {
                    "status": "error" if timeframe_status_error else "complete",
                    "generated_at": generated_ts,
                    "generated_ts": generated_ts,
                    "source": market_source,
                    "timeframe": timeframe.key,
                    "total_symbols": total_symbols,
                    "chunk_size": chunk_size,
                    "api_calls": timeframe_api_calls,
                    "duration_seconds": round(duration, 2),
                    "rows": timeframe_rows,
                }
                if timeframe_status_error:
                    timeframe_payload["error"] = timeframe_status_error
                    write_state(_snapshot_timeframe_building_state_name(timeframe.key), timeframe_payload)
                else:
                    write_state(_snapshot_timeframe_state_name(timeframe.key), timeframe_payload)
                    write_state(
                        _snapshot_timeframe_building_state_name(timeframe.key),
                        {
                            "status": "idle",
                            "completed_at": generated_ts,
                            "completed_ts": generated_ts,
                            "updated_at": generated_ts,
                            "updated_ts": generated_ts,
                            "source": market_source,
                            "timeframe": timeframe.key,
                        },
                    )

        for chunk in snapshot_chunks:
            if preloaded_snapshots is not None:
                snapshots = {
                    symbol: preloaded_snapshots.get(symbol)
                    for symbol in chunk
                    if isinstance(preloaded_snapshots.get(symbol), dict)
                }
            else:
                try:
                    snapshots = fetch_stock_snapshots(
                        chunk,
                        feed=DEFAULT_FEED,
                        user_id=user_id,
                        timeout=MARKET_REQUEST_TIMEOUT,
                        provider=market_source,
                    )
                except Exception:
                    LOGGER.exception(
                        "snapshot chunk fetch failed user_id=%s provider=%s chunk_size=%s",
                        user_id,
                        market_source,
                        len(chunk),
                    )
                    snapshots = {}
                api_calls += 1
            if isinstance(snapshots, dict):
                for symbol in chunk:
                    asset = asset_map.get(symbol)
                    if not asset:
                        continue
                    snapshot = snapshots.get(symbol)
                    if not isinstance(snapshot, dict):
                        continue
                    daily_bar = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
                    minute_bar = snapshot.get("minuteBar") or snapshot.get("minute_bar") or {}
                    prev_bar = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar") or {}
                    prev_close = _coerce_positive_price((prev_bar or {}).get("c"))
                    last_price = _resolve_snapshot_last_price(snapshot, prev_close=prev_close, allow_quote_fallback=False)
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
                    open_price = _coerce_positive_price((daily_bar or {}).get("o"))
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
                            "open": open_price,
                            "prev_close": prev_close,
                            "gap_pct": gap_pct,
                            "range_pct": range_pct,
                            "volume_ratio": volume_ratio,
                        }
                    )

            completed_chunks += 1
            _update_progress()

        duration_seconds = time.time() - started_ts
        api_calls_per_minute = None
        if duration_seconds > 0:
            api_calls_per_minute = api_calls / (duration_seconds / 60.0)

        generated_ts = time.time()
        payload = {
            "status": "complete",
            "generated_at": generated_ts,
            "generated_ts": generated_ts,
            "source": market_source,
            "total_symbols": total_symbols,
            "chunk_size": chunk_size,
            "api_calls": api_calls,
            "api_calls_per_minute": api_calls_per_minute,
            "duration_seconds": round(duration_seconds, 2),
            "rows": rows,
        }
        write_state(SNAPSHOT_RANKINGS_STATE, payload)
        write_state(SNAPSHOT_RANKINGS_PROGRESS_STATE, _snapshot_refresh_payload_summary(payload))
        write_state(
            SNAPSHOT_RANKINGS_BUILDING_STATE,
            {
                "status": "idle",
                "completed_at": generated_ts,
                "completed_ts": generated_ts,
                "updated_at": generated_ts,
                "updated_ts": generated_ts,
                "source": market_source,
            },
        )
        return payload
    except Exception:
        LOGGER.exception("Snapshot rankings refresh failed user_id=%s", user_id)
        failed_ts = time.time()
        current_source = str(locals().get("market_source") or "unknown")
        error_payload = {
            "status": "error",
            "started_at": started_at,
            "started_ts": started_ts,
            "updated_at": failed_ts,
            "updated_ts": failed_ts,
            "error": "snapshot_refresh_failed",
            "source": current_source,
        }
        write_state(SNAPSHOT_RANKINGS_PROGRESS_STATE, error_payload)
        write_state(SNAPSHOT_RANKINGS_BUILDING_STATE, error_payload)
        for timeframe in MARKET_RANKINGS_TIMEFRAME_KEYS:
            write_state(
                _snapshot_timeframe_building_state_name(timeframe),
                {
                    "status": "error",
                    "timeframe": timeframe,
                    "source": current_source,
                    "started_at": started_at,
                    "started_ts": started_ts,
                    "updated_at": failed_ts,
                    "updated_ts": failed_ts,
                    "error": "snapshot_refresh_failed",
                },
            )
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
    market_source = resolve_market_provider(user_id=user_id)
    rankings = _build_snapshot_rankings(user_id)
    if rankings:
        return rankings, market_source
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


def _rows_have_metric(
    rows: list[dict[str, object]],
    metric_key: str,
    *,
    min_valid_rows: int = 1,
) -> bool:
    if not rows:
        return False
    valid = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _coerce_number(row.get(metric_key)) is not None:
            valid += 1
            if valid >= max(1, min_valid_rows):
                return True
    return False


def _snapshot_rows_schema_valid_for_list(
    rows: list[dict[str, object]] | None,
    *,
    list_type: str,
    min_valid_rows: int = 3,
) -> bool:
    if not isinstance(rows, list) or not rows:
        return False
    if list_type in {"gainers", "losers"}:
        return _rows_have_metric(rows, "change_pct_period", min_valid_rows=min_valid_rows)
    if list_type == "most_active":
        return _rows_have_metric(rows, "volume", min_valid_rows=min_valid_rows)
    if list_type == "top_turnover":
        return _rows_have_metric(rows, "dollar_volume", min_valid_rows=min_valid_rows)
    return True


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
                "price": float(end_price),
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
                "price": float(end_price),
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
    market_source = resolve_market_provider(user_id=user_id)
    cache_key = build_cache_key("market-universe-timeframe", market_source, timeframe.key, user_id or "anon")

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


def _resolve_universe_window_rankings(
    timeframe: Timeframe,
    *,
    user_id: str | None,
) -> tuple[list[dict[str, object]], str]:
    if timeframe.key == "1d":
        rows, source = _resolve_universe_rankings(user_id)
        normalized_rows = [
            _apply_common_market_item_shape(dict(entry))
            for entry in rows
            if isinstance(entry, dict)
        ]
        return normalized_rows, source
    market_source = resolve_market_provider(user_id=user_id)
    cache_key = build_cache_key("market-universe-window-v2", market_source, timeframe.key, user_id or "anon")

    def _load() -> dict[str, object]:
        assets = _filter_rankable_assets(_normalize_assets(_load_assets_master(user_id)))
        if not assets:
            return {"rows": [], "source": "unknown"}
        selected_assets = list(assets)
        if MARKET_UNIVERSE_WINDOW_MAX_SYMBOLS and len(selected_assets) > MARKET_UNIVERSE_WINDOW_MAX_SYMBOLS:
            asset_map = {
                str(asset.get("symbol") or "").strip().upper(): asset
                for asset in selected_assets
                if asset.get("symbol")
            }
            ranked_rows, _ranked_source = _resolve_universe_rankings(user_id)
            ranked_symbols: list[str] = []
            seen: set[str] = set()
            liquidity_rows = sorted(
                [row for row in ranked_rows if isinstance(row, dict)],
                key=lambda row: (
                    _coerce_number(row.get("dollar_volume")) or 0.0,
                    _coerce_number(row.get("volume")) or 0.0,
                ),
                reverse=True,
            )
            for row in liquidity_rows:
                symbol = str(row.get("symbol") or "").strip().upper()
                if not symbol or symbol in seen or symbol not in asset_map:
                    continue
                seen.add(symbol)
                ranked_symbols.append(symbol)
                if len(ranked_symbols) >= MARKET_UNIVERSE_WINDOW_MAX_SYMBOLS:
                    break
            if len(ranked_symbols) < MARKET_UNIVERSE_WINDOW_MAX_SYMBOLS:
                for asset in selected_assets:
                    symbol = str(asset.get("symbol") or "").strip().upper()
                    if not symbol or symbol in seen:
                        continue
                    seen.add(symbol)
                    ranked_symbols.append(symbol)
                    if len(ranked_symbols) >= MARKET_UNIVERSE_WINDOW_MAX_SYMBOLS:
                        break
            selected_assets = [asset_map[symbol] for symbol in ranked_symbols if symbol in asset_map]
        if MARKET_UNIVERSE_MAX_SYMBOLS and len(selected_assets) > MARKET_UNIVERSE_MAX_SYMBOLS:
            selected_assets = selected_assets[:MARKET_UNIVERSE_MAX_SYMBOLS]
        symbols = [asset.get("symbol") for asset in selected_assets if asset.get("symbol")]
        metrics_map, source = _resolve_symbol_window_metrics(symbols, timeframe=timeframe, user_id=user_id)
        label = str(timeframe.label)
        rows: list[dict[str, object]] = []
        for asset in selected_assets:
            symbol = str(asset.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            metrics = metrics_map.get(symbol, {})
            row: dict[str, object] = {
                "symbol": symbol,
                "name": asset.get("name") or "",
                "exchange": asset.get("exchange") or "",
                "price": metrics.get("price"),
                "change_pct_period": metrics.get("change_pct_period"),
                "change_pct_day": metrics.get("change_pct_day"),
                "change_pct": metrics.get("change_pct_period"),
                "volume": metrics.get("volume"),
                "dollar_volume": metrics.get("dollar_volume"),
                "range_pct": metrics.get("range_pct"),
                "open": metrics.get("open"),
                "prev_close": metrics.get("prev_close"),
                "period_label": label,
                "period_label_en": timeframe.label_en,
            }
            missing = _normalize_missing_reason_map(metrics.get("missing_reasons"))
            if missing:
                row["missing_reasons"] = missing
            rows.append(_apply_common_market_item_shape(row))
        return {"rows": rows, "source": source or "unknown"}

    payload = cache_memoize(cache_key, _load, MARKET_UNIVERSE_RANKINGS_CACHE_TTL)
    if isinstance(payload, dict):
        rows = payload.get("rows")
        source = payload.get("source")
        normalized_rows = [
            _apply_common_market_item_shape(dict(entry))
            for entry in (rows if isinstance(rows, list) else [])
            if isinstance(entry, dict)
        ]
        return normalized_rows, source if isinstance(source, str) else "unknown"
    if isinstance(payload, list):
        normalized_rows = [
            _apply_common_market_item_shape(dict(entry))
            for entry in payload
            if isinstance(entry, dict)
        ]
        return normalized_rows, "unknown"
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


def _normalize_fundamentals_payload(raw: Mapping[str, object]) -> dict[str, object]:
    payload = {
        "market_cap": raw.get("market_cap"),
        "sector": raw.get("sector"),
        "industry": raw.get("industry"),
        "ceo": raw.get("ceo"),
        "hq": raw.get("hq"),
        "city": raw.get("city"),
        "state": raw.get("state"),
        "country": raw.get("country"),
        "quote_type": raw.get("quote_type"),
    }
    has_hq = payload.get("hq")
    if not has_hq:
        parts = [payload.get("city"), payload.get("state"), payload.get("country")]
        joined = ", ".join(str(part) for part in parts if part not in (None, "", []))
        if joined:
            payload["hq"] = joined
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }


def _merge_missing_fields(primary: dict[str, object], fallback: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(primary, dict):
        primary = {}
    if not isinstance(fallback, Mapping):
        return primary
    for key, value in fallback.items():
        if value in (None, "", [], {}):
            continue
        if primary.get(key) in (None, "", [], {}):
            primary[key] = value
    return primary


def _download_yfinance_profile_fallback(symbol: str) -> tuple[dict[str, object] | None, dict[str, object]]:
    debug: dict[str, object] = {"symbol": symbol, "source": "yfinance"}
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:
        debug["error"] = f"import:{exc}"
        return None, debug
    try:
        ticker = yf.Ticker(symbol)
    except Exception as exc:
        debug["error"] = f"ticker:{exc}"
        return None, debug
    try:
        info = ticker.get_info() or {}
    except Exception:
        try:
            info = ticker.info or {}
        except Exception as exc:
            debug["error"] = f"info:{exc}"
            return None, debug
    if not isinstance(info, dict):
        debug["error"] = "info:invalid"
        return None, debug

    city = info.get("city")
    state = info.get("state")
    country = info.get("country")
    hq_parts = [part for part in (city, state, country) if part not in (None, "")]
    hq = ", ".join(str(part) for part in hq_parts) if hq_parts else None

    ceo = None
    officers = info.get("companyOfficers")
    if isinstance(officers, list):
        for officer in officers:
            if not isinstance(officer, dict):
                continue
            title = str(officer.get("title") or "").lower()
            if "ceo" in title or "chief executive" in title:
                ceo = officer.get("name") or officer.get("fullName") or officer.get("full_name")
                if ceo:
                    break

    market_cap = info.get("marketCap")
    if market_cap in (None, "", 0):
        try:
            fast_info = ticker.fast_info or {}
        except Exception:
            fast_info = {}
        if isinstance(fast_info, dict):
            market_cap = fast_info.get("market_cap") or fast_info.get("marketCap") or market_cap

    name = (
        info.get("longName")
        or info.get("shortName")
        or info.get("displayName")
        or info.get("name")
        or symbol
    )
    short_name = info.get("shortName") or info.get("longName") or info.get("displayName") or name
    payload = {
        "name": name,
        "shortName": short_name,
        "exchange": info.get("fullExchangeName") or info.get("exchangeName") or info.get("exchange"),
        "market_cap": market_cap,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "ceo": ceo,
        "hq": hq,
        "city": city,
        "state": state,
        "country": country,
        "quote_type": info.get("quoteType") or info.get("quote_type"),
    }
    cleaned = {
        key: value
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }
    debug["keys"] = list(cleaned.keys())
    debug["has_values"] = bool(cleaned)
    return (cleaned or None), debug


def _fetch_company_profile(
    symbol: str,
    *,
    user_id: str | None = None,
    provider: str | None = None,
) -> dict[str, object]:
    if not symbol:
        return {}
    selected_provider = str(provider or resolve_market_provider(user_id=user_id) or "alpaca").strip().lower() or "alpaca"
    cache_key = build_cache_key("market-profile", selected_provider, symbol.upper())

    def _download() -> dict[str, object]:
        profile: dict[str, object] = {}
        if selected_provider == "massive":
            overview = fetch_company_overview(
                symbol,
                user_id=user_id,
                timeout=MARKET_REQUEST_TIMEOUT,
                provider=selected_provider,
            )
            if isinstance(overview, dict):
                name = overview.get("name") or symbol
                profile.update(
                    {
                        "name": name,
                        "shortName": overview.get("shortName") or name,
                        "exchange": overview.get("exchange") or "",
                    }
                )
                for key, value in _normalize_fundamentals_payload(overview).items():
                    profile.setdefault(key, value)
                quote_type = overview.get("quote_type")
                if quote_type not in (None, "", [], {}):
                    profile["quote_type"] = quote_type
        needs_fallback = any(
            profile.get(key) in (None, "", [], {})
            for key in ("name", "shortName", "exchange", "market_cap", "sector", "industry", "ceo", "hq")
        )
        if needs_fallback:
            yf_payload, _yf_debug = _download_yfinance_profile_fallback(symbol)
            if isinstance(yf_payload, dict):
                _merge_missing_fields(profile, yf_payload)
        if not profile.get("name"):
            meta_map = _build_asset_meta_map(user_id=user_id, symbols={symbol.upper()})
            meta = meta_map.get(symbol.upper())
            if isinstance(meta, dict):
                name = meta.get("name") or symbol
                profile.setdefault("name", name)
                profile.setdefault("shortName", name)
                profile.setdefault("exchange", meta.get("exchange") or "")
        return profile

    result = cache_memoize(cache_key, _download, MARKET_PROFILE_CACHE_TTL)
    return result if isinstance(result, dict) else {}


def _fetch_52w_stats(symbol: str, *, user_id: str | None = None) -> dict[str, float | None]:
    if not symbol:
        return {"high_52w": None, "low_52w": None, "as_of": None}
    market_source = resolve_market_provider(user_id=user_id)
    cache_key = build_cache_key("market-52w", market_source, symbol.upper())

    def _download() -> dict[str, float | None]:
        frame = market_data.fetch(
            [symbol],
            period="1y",
            interval="1d",
            cache=True,
            ttl=MARKET_52W_CACHE_TTL,
            timeout=MARKET_REQUEST_TIMEOUT,
            user_id=user_id,
        )
        bars = _extract_ohlc(frame, symbol, limit=420)
        if not bars:
            return {"high_52w": None, "low_52w": None, "as_of": None}
        highs: list[float] = []
        lows: list[float] = []
        for bar in bars:
            if not isinstance(bar, dict):
                continue
            high = _coerce_number(bar.get("high"))
            low = _coerce_number(bar.get("low"))
            if high is not None:
                highs.append(float(high))
            if low is not None:
                lows.append(float(low))
        if not highs or not lows:
            return {"high_52w": None, "low_52w": None, "as_of": None}
        last = bars[-1] if isinstance(bars[-1], dict) else {}
        as_of = _coerce_number(last.get("time"))
        return {
            "high_52w": max(highs),
            "low_52w": min(lows),
            "as_of": float(as_of) if as_of is not None else None,
        }

    result = cache_memoize(cache_key, _download, MARKET_52W_CACHE_TTL)
    if isinstance(result, dict):
        return result  # type: ignore[return-value]
    return {"high_52w": None, "low_52w": None, "as_of": None}


def _download_yfinance_fundamentals(
    symbol: str,
    *,
    user_id: str | None = None,
    provider: str | None = None,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    # Kept function name for compatibility with existing tests/callers.
    selected_provider = str(provider or resolve_market_provider(user_id=user_id) or "alpaca").strip().lower() or "alpaca"
    debug: dict[str, object] = {"symbol": symbol, "provider": selected_provider}
    primary_payload: dict[str, object] = {}
    try:
        overview = fetch_company_overview(
            symbol,
            user_id=user_id,
            timeout=MARKET_REQUEST_TIMEOUT,
            provider=selected_provider,
        )
    except Exception as exc:
        debug["primary_error"] = f"overview:{exc}"
        overview = {}
    if isinstance(overview, dict) and overview:
        primary_payload = _normalize_fundamentals_payload(overview)
        if primary_payload:
            debug["primary_source"] = "provider"
            debug["provider_keys"] = list(primary_payload.keys())

    needs_fallback = not primary_payload or any(
        key not in primary_payload
        for key in ("market_cap", "sector", "industry", "ceo", "hq", "quote_type")
    )
    if needs_fallback:
        fallback_payload, fallback_debug = _download_yfinance_profile_fallback(symbol)
        if isinstance(fallback_payload, dict) and fallback_payload:
            fallback_fundamentals = _normalize_fundamentals_payload(fallback_payload)
            _merge_missing_fields(primary_payload, fallback_fundamentals)
            debug["fallback_source"] = "yfinance"
            debug["fallback_keys"] = list(fallback_fundamentals.keys())
        elif isinstance(fallback_debug, dict) and fallback_debug.get("error"):
            debug["fallback_error"] = fallback_debug.get("error")

    debug["keys"] = list(primary_payload.keys())
    debug["has_values"] = bool(primary_payload)
    return (primary_payload or None), debug


def _fetch_yfinance_fundamentals(
    symbol: str,
    *,
    user_id: str | None = None,
    provider: str | None = None,
) -> dict[str, object]:
    if not symbol:
        return {}
    selected_provider = str(provider or resolve_market_provider(user_id=user_id) or "alpaca").strip().lower() or "alpaca"
    cache_key = build_cache_key("market-fundamentals", selected_provider, "v5", symbol.upper())

    def _download() -> dict[str, object] | None:
        payload, _debug = _download_yfinance_fundamentals(symbol, user_id=user_id, provider=selected_provider)
        return payload

    result = cache_memoize(cache_key, _download, MARKET_FUNDAMENTALS_CACHE_TTL)
    return result if isinstance(result, dict) else {}


def _fetch_yfinance_fundamentals_debug(
    symbol: str,
    *,
    user_id: str | None = None,
    provider: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    selected_provider = str(provider or resolve_market_provider(user_id=user_id) or "alpaca").strip().lower() or "alpaca"
    cache_key = build_cache_key("market-fundamentals", selected_provider, "v5", symbol.upper())
    cached = cache_get_object(cache_key)
    if isinstance(cached, dict):
        keys = [key for key, value in cached.items() if value not in (None, "", [], {})]
        return cached, {
            "symbol": symbol,
            "provider": selected_provider,
            "cached": True,
            "keys": keys,
            "has_values": bool(keys),
        }
    payload, debug = _download_yfinance_fundamentals(symbol, user_id=user_id, provider=selected_provider)
    debug["cached"] = False
    if payload is not None:
        cache_set_object(cache_key, payload, MARKET_FUNDAMENTALS_CACHE_TTL)
    return payload or {}, debug


def _format_news_time(value: object) -> float | None:
    ts = parse_timestamp(value)
    return float(ts) if ts is not None else None


def _normalize_news_related_symbols(entry: dict[str, object], content: dict[str, object] | None) -> list[str]:
    candidates: list[object] = [
        entry.get("symbols"),
        entry.get("relatedTickers"),
    ]
    if isinstance(content, dict):
        candidates.extend(
            [
                content.get("symbols"),
                content.get("relatedTickers"),
                content.get("tickers"),
            ]
        )
    result: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if not isinstance(value, list):
            continue
        for raw in value:
            symbol = _normalize_query(raw)
            if not symbol:
                continue
            if symbol in seen:
                continue
            seen.add(symbol)
            result.append(symbol)
    return result


def _normalize_news_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content")
        content_title = content.get("title") if isinstance(content, dict) else None
        content_summary = content.get("summary") if isinstance(content, dict) else None
        content_desc = content.get("description") if isinstance(content, dict) else None

        def _coerce_text(value: object) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float)):
                return str(value)
            return ""

        def _pick_text(*candidates: object) -> str:
            for candidate in candidates:
                text = _coerce_text(candidate).strip()
                if text:
                    return text
            return ""

        title = _pick_text(
            entry.get("headline"),
            entry.get("title"),
            content_title,
            entry.get("summary"),
            content_summary,
            content if isinstance(content, str) else None,
        )
        snippet = _pick_text(
            entry.get("summary"),
            entry.get("description"),
            content_summary,
            content_desc,
            content if isinstance(content, str) else None,
        )
        url = (
            entry.get("url")
            or entry.get("link")
            or entry.get("article_url")
            or (content.get("canonicalUrl", {}).get("url") if isinstance(content, dict) and isinstance(content.get("canonicalUrl"), dict) else None)
            or (content.get("clickThroughUrl", {}).get("url") if isinstance(content, dict) and isinstance(content.get("clickThroughUrl"), dict) else None)
            or ""
        )
        if not url and isinstance(entry.get("canonicalUrl"), dict):
            url = entry.get("canonicalUrl", {}).get("url")  # type: ignore[assignment]
        if not url and isinstance(entry.get("clickThroughUrl"), dict):
            url = entry.get("clickThroughUrl", {}).get("url")  # type: ignore[assignment]
        source = entry.get("source") or entry.get("publisher") or entry.get("author") or ""
        if not source and isinstance(entry.get("provider"), dict):
            source = entry.get("provider", {}).get("displayName") or entry.get("provider", {}).get("name")  # type: ignore[assignment]
        if not source and isinstance(content, dict) and isinstance(content.get("provider"), dict):
            provider = content.get("provider", {})
            source = provider.get("displayName") or provider.get("name") or provider.get("url")  # type: ignore[assignment]
        related_symbols = _normalize_news_related_symbols(entry, content if isinstance(content, dict) else None)
        raw_time = (
            entry.get("created_at")
            or entry.get("createdAt")
            or entry.get("updated_at")
            or entry.get("updatedAt")
            or entry.get("time")
            or entry.get("published_at")
            or entry.get("published")
            or entry.get("providerPublishTime")
            or entry.get("pubDate")
            or entry.get("displayTime")
            or (content.get("pubDate") if isinstance(content, dict) else None)
            or (content.get("displayTime") if isinstance(content, dict) else None)
        )
        normalized.append(
            {
                "title": _coerce_text(title).strip(),
                "url": _coerce_text(url).strip(),
                "source": _coerce_text(source).strip(),
                "time": _format_news_time(raw_time),
                "summary": _coerce_text(snippet).strip(),
                "related_symbols": related_symbols,
            }
        )
    return normalized


def _sort_and_dedupe_news_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    deduped: list[dict[str, object]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "").strip().lower()
        title = str(entry.get("title") or "").strip().lower()
        summary = str(entry.get("summary") or "").strip().lower()
        time_value = _coerce_number(entry.get("time"))
        if url:
            dedupe_key = f"url:{url}"
        elif title and time_value is not None:
            dedupe_key = f"title_time:{title}|{int(time_value)}"
        elif title:
            dedupe_key = f"title:{title}"
        elif summary:
            dedupe_key = f"summary:{summary[:120]}"
        else:
            continue
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(entry)

    return sorted(
        deduped,
        key=lambda item: (_coerce_number(item.get("time")) or 0.0, str(item.get("title") or "").lower()),
        reverse=True,
    )


_NEWS_SENTIMENT_POSITIVE = {
    "beat",
    "beats",
    "surge",
    "surged",
    "upgrade",
    "upgraded",
    "record",
    "growth",
    "profit",
    "profits",
    "rally",
    "raises",
    "raised",
    "strong",
    "outperform",
    "buyback",
    "bullish",
    "expands",
    "expansion",
    "accelerate",
    "accelerates",
    "上调",
    "增长",
    "盈利",
    "创新高",
    "强劲",
    "利好",
    "回购",
    "看多",
}

_NEWS_SENTIMENT_NEGATIVE = {
    "miss",
    "missed",
    "downgrade",
    "downgraded",
    "falls",
    "fall",
    "drop",
    "drops",
    "lawsuit",
    "cuts",
    "cut",
    "weak",
    "warns",
    "warning",
    "bearish",
    "decline",
    "plunge",
    "slump",
    "下调",
    "下跌",
    "亏损",
    "利空",
    "警告",
    "看空",
}


def _infer_sentiment_rule(text: str) -> str:
    lowered = (text or "").lower()
    if not lowered:
        return "neutral"
    score = 0
    for word in _NEWS_SENTIMENT_POSITIVE:
        if word in lowered:
            score += 1
    for word in _NEWS_SENTIMENT_NEGATIVE:
        if word in lowered:
            score -= 1
    if score > 0:
        return "bullish"
    if score < 0:
        return "bearish"
    return "neutral"


def _normalize_sentiment_label(value: str) -> str:
    raw = str(value or "").strip()
    lowered = raw.lower()
    if lowered in {"bullish", "positive", "pos", "up"}:
        return "bullish"
    if lowered in {"bearish", "negative", "neg", "down"}:
        return "bearish"
    if lowered in {"neutral", "mixed"}:
        return "neutral"
    if any(token in raw for token in ("利好", "看多", "上涨", "正面")):
        return "bullish"
    if any(token in raw for token in ("利空", "看空", "下跌", "负面")):
        return "bearish"
    return "neutral"


def _news_sentiment_cache_key(symbol: str, items: list[dict[str, object]]) -> str:
    digest_items: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        digest_items.append(
            {
                "title": str(item.get("title") or "").strip(),
                "summary": str(item.get("summary") or "").strip(),
                "source": str(item.get("source") or "").strip(),
            }
        )
    raw = json.dumps({"symbol": symbol.upper(), "items": digest_items}, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return build_cache_key("market-news-sentiment", symbol.upper(), digest)


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


def _fetch_symbol_news_pool(symbol: str, *, user_id: str | None = None) -> list[dict[str, object]]:
    if not symbol:
        return []
    provider = resolve_news_provider(user_id=user_id)
    cache_key = build_cache_key("market-news", provider, "v4-unbounded", symbol.upper())

    cached = cache_get_object(cache_key)
    if isinstance(cached, list):
        cached_items = _sort_and_dedupe_news_items(_normalize_news_items([entry for entry in cached if isinstance(entry, dict)]))
        if cached_items:
            return cached_items

    symbols = _infer_news_symbols(symbol, user_id=user_id)
    items = fetch_news(symbols=symbols, limit=None, user_id=user_id, provider=provider)
    normalized: list[dict[str, object]] = []
    if isinstance(items, list) and items:
        normalized = _sort_and_dedupe_news_items(_normalize_news_items(items))
    if normalized:
        cache_set_object(cache_key, normalized, MARKET_NEWS_CACHE_TTL)
    return normalized


def _fetch_symbol_news_page(
    symbol: str,
    *,
    user_id: str | None = None,
    limit: int = MARKET_NEWS_PAGE_DEFAULT,
    offset: int = 0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    safe_limit = _parse_news_limit(limit, default=MARKET_NEWS_PAGE_DEFAULT)
    safe_offset = _parse_offset(offset)
    all_items = _fetch_symbol_news_pool(symbol, user_id=user_id)
    page_items = all_items[safe_offset : safe_offset + safe_limit]
    next_offset = safe_offset + len(page_items)
    has_more = next_offset < len(all_items)
    return (
        page_items,
        {
            "offset": safe_offset,
            "limit": safe_limit,
            "count": len(page_items),
            "has_more": has_more,
            "next_offset": next_offset if has_more else None,
            "total_cached": len(all_items),
        },
    )


def _fetch_symbol_news(
    symbol: str,
    *,
    user_id: str | None = None,
    limit: int = MARKET_NEWS_PAGE_DEFAULT,
    offset: int = 0,
) -> list[dict[str, object]]:
    page_items, _meta = _fetch_symbol_news_page(symbol, user_id=user_id, limit=limit, offset=offset)
    return page_items


def _build_ai_summary(
    profile_payload: dict[str, object],
    news_payload: list[dict[str, object]],
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


def _ai_summary_cache_key(
    symbol: str,
    profile_payload: dict[str, object],
    news_payload: list[dict[str, str]],
    *,
    lang_prefix: str,
) -> str:
    safe_profile: dict[str, object] = {}
    if isinstance(profile_payload, dict):
        for key in (
            "name",
            "shortName",
            "summary",
            "exchange",
            "sector",
            "industry",
            "marketCap",
            "market_cap",
            "market_capitalization",
            "currency",
            "country",
        ):
            value = profile_payload.get(key)
            if value:
                safe_profile[key] = value
    safe_news: list[dict[str, object]] = []
    if isinstance(news_payload, list):
        for item in news_payload[:6]:
            if not isinstance(item, dict):
                continue
            safe_news.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "source": str(item.get("source") or "").strip(),
                    "time": item.get("time"),
                    "summary": str(item.get("summary") or "").strip(),
                }
            )
    digest_payload = {
        "symbol": symbol.upper(),
        "lang": lang_prefix,
        "profile": safe_profile,
        "news": safe_news,
    }
    raw = json.dumps(digest_payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return build_cache_key("market-ai-summary", symbol.upper(), lang_prefix, digest)


def _ai_summary_struct_cache_key(
    symbol: str,
    profile_payload: dict[str, object],
    news_payload: list[dict[str, str]],
    *,
    lang_prefix: str,
) -> str:
    digest = _ai_summary_cache_key(symbol, profile_payload, news_payload, lang_prefix=lang_prefix).split(":")[-1]
    return build_cache_key("market-ai-summary-struct", symbol.upper(), lang_prefix, digest)


def _parse_ai_json(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    def _strip_code_fence(value: str) -> str:
        content = value.strip()
        if content.startswith("```") and content.endswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        if content.lower().startswith("json\n"):
            return content[5:].strip()
        return content

    def _json_candidates(value: str) -> list[str]:
        candidates: list[str] = []
        primary = _strip_code_fence(value)
        if primary:
            candidates.append(primary)
        start = primary.find("{")
        end = primary.rfind("}")
        if start >= 0 and end > start:
            snippet = primary[start : end + 1].strip()
            if snippet:
                candidates.append(snippet)
        normalized = primary.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        if normalized and normalized not in candidates:
            candidates.append(normalized)
        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    for candidate in _json_candidates(raw):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def _normalize_ai_struct(payload: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(payload, dict) or not payload:
        return None

    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple)):
            parts = [str(item).strip() for item in value if str(item or "").strip()]
            return " ".join(parts).strip()
        if isinstance(value, dict):
            parts = []
            for inner in value.values():
                text = str(inner or "").strip()
                if text:
                    parts.append(text)
            return " ".join(parts).strip()
        return str(value).strip()

    mapping = {
        "event": "event",
        "key_event": "event",
        "core_event": "event",
        "headline": "event",
        "main_event": "event",
        "核心事件": "event",
        "事件": "event",
        "impact": "impact",
        "market_impact": "impact",
        "impact_analysis": "impact",
        "effect": "impact",
        "市场影响": "impact",
        "影响": "impact",
        "implication": "implication",
        "trading_implication": "implication",
        "trade_implication": "implication",
        "trading_hint": "implication",
        "trading_signal": "implication",
        "suggestion": "implication",
        "交易暗示": "implication",
        "交易建议": "implication",
        "交易启示": "implication",
    }
    normalized: dict[str, str] = {}
    for key, target in mapping.items():
        if key not in payload:
            continue
        value = _coerce_text(payload.get(key))
        if value:
            normalized[target] = value
    if not normalized:
        summary_text = _coerce_text(payload.get("summary"))
        if summary_text:
            normalized["event"] = summary_text
    if not normalized:
        return None
    return normalized


def _struct_from_plain_text(answer: str) -> dict[str, str] | None:
    text = str(answer or "").strip()
    if not text:
        return None
    lines = [line.strip(" -*\t") for line in text.splitlines() if line and line.strip()]
    label_patterns = [
        ("event", (r"event", r"key\s*event", r"core\s*event", r"核心事件", r"事件")),
        ("impact", (r"impact", r"market\s*impact", r"影响", r"市场影响")),
        ("implication", (r"implication", r"trading\s*hint", r"trading\s*implication", r"交易暗示", r"交易建议")),
    ]
    extracted: dict[str, str] = {}
    for line in lines:
        for target, aliases in label_patterns:
            if target in extracted:
                continue
            for alias in aliases:
                if re.match(rf"^(?:{alias})\s*[:：\-]\s*(.+)$", line, flags=re.IGNORECASE):
                    value = re.sub(rf"^(?:{alias})\s*[:：\-]\s*", "", line, count=1, flags=re.IGNORECASE).strip()
                    if value:
                        extracted[target] = value
                    break
    if extracted:
        return extracted
    sentences = [segment.strip() for segment in re.split(r"[。\n!?]+", text) if segment and segment.strip()]
    if not sentences:
        return None
    event = sentences[0]
    impact = sentences[1] if len(sentences) > 1 else ""
    implication = " ".join(sentences[2:]).strip() if len(sentences) > 2 else ""
    return {
        "event": event,
        "impact": impact,
        "implication": implication,
    }


def _contains_cjk(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _struct_matches_lang(structured: dict[str, str], lang_prefix: str) -> bool:
    values = [value for value in structured.values() if isinstance(value, str) and value.strip()]
    if not values:
        return True
    if lang_prefix == "zh":
        return any(_contains_cjk(value) for value in values)
    if lang_prefix == "en":
        return not any(_contains_cjk(value) for value in values)
    return True


def _translate_ai_struct(
    structured: dict[str, str],
    *,
    lang_prefix: str,
    model: str,
    api_key: str,
) -> dict[str, str] | None:
    if not structured:
        return None
    if lang_prefix == "zh":
        system_prompt = (
            "你是专业翻译助手。请把 JSON 中的 event/impact/implication 翻译成中文，"
            "保持 JSON 结构不变，只输出 JSON。公司名/股票代码可保留英文。"
        )
    else:
        system_prompt = (
            "You are a professional translator. Translate event/impact/implication into English, "
            "keep JSON structure, output JSON only. Keep tickers/proper nouns as-is."
        )
    user_prompt = json.dumps(structured, ensure_ascii=False)
    try:
        response = bailian_ai.chat(
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout_seconds=MARKET_AI_SUMMARY_TIMEOUT_SECONDS,
            api_key=api_key,
            response_format={"type": "json_object"},
            extra_params={"temperature": 0.1, "max_tokens": MARKET_AI_SUMMARY_MAX_TOKENS},
        )
    except Exception:
        return None
    answer = str(response.get("answer") or "").strip()
    parsed = _parse_ai_json(answer)
    translated = _normalize_ai_struct(parsed)
    if not translated:
        return None
    if not _struct_matches_lang(translated, lang_prefix):
        return None
    return translated


def _summary_from_struct(structured: dict[str, str], *, lang_prefix: str) -> str:
    parts = [structured.get("event"), structured.get("impact"), structured.get("implication")]
    parts = [part for part in parts if part]
    if not parts:
        return ""
    sep = "。" if lang_prefix == "zh" else ". "
    text = sep.join(parts)
    if lang_prefix == "en" and text and not text.endswith((".", "!", "?")):
        text = f"{text}."
    return text.strip()


def _resolve_bailian_model(preferred: str | None) -> str | None:
    if not preferred:
        return None
    text = str(preferred).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith(("bailian:", "dashscope:", "aliyun:")):
        return text.split(":", 1)[1].strip() or None
    if lowered.startswith("qwen"):
        return text
    return None


def _build_ai_summary_llm(
    profile_payload: dict[str, object],
    news_payload: list[dict[str, str]],
    *,
    lang_prefix: str,
    symbol: str,
    user: Any,
    api_key: str | None = None,
    raise_on_error: bool = False,
) -> str | None:
    api_key = api_key or resolve_api_credential(user, "bailian_api_key") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("BAILIAN_API_KEY")
    if not api_key:
        return None
    cache_key = _ai_summary_cache_key(symbol, profile_payload, news_payload, lang_prefix=lang_prefix)
    cached = cache_get_object(cache_key)
    if isinstance(cached, str) and cached:
        return cached
    preferred_model = resolve_api_credential(user, "ai_model")
    model = (
        _resolve_bailian_model(preferred_model)
        or os.environ.get("BAILIAN_MODEL")
        or os.environ.get("DASHSCOPE_MODEL")
        or "qwen-max"
    )

    name = ""
    exchange = ""
    profile_summary = ""
    if isinstance(profile_payload, dict):
        name = str(profile_payload.get("name") or profile_payload.get("shortName") or "").strip()
        exchange = str(profile_payload.get("exchange") or "").strip()
        profile_summary = str(profile_payload.get("summary") or "").strip()

    news_lines: list[str] = []
    if isinstance(news_payload, list):
        for item in news_payload[:6]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            source = str(item.get("source") or "").strip()
            summary = str(item.get("summary") or "").strip()
            if source:
                news_line = f"- {title} ({source})"
            else:
                news_line = f"- {title}"
            if summary:
                news_line = f"{news_line} 摘要：{summary}" if lang_prefix == "zh" else f"{news_line} Summary: {summary}"
            news_lines.append(news_line)

    if lang_prefix == "zh":
        system_prompt = (
            "你是谨慎的金融摘要助手。只根据提供的信息生成 4-6 句中文摘要，"
            "覆盖核心事件、驱动因素、市场反应/影响、交易关注点（如信息不足请说明）。"
            "每句尽量包含具体信息（如数值、时间或来源），"
            "不做推测、不提供投资建议。"
        )
        user_prompt_lines = [
            f"股票代码：{symbol}",
        ]
        if name:
            user_prompt_lines.append(f"公司名称：{name}")
        if exchange:
            user_prompt_lines.append(f"交易所：{exchange}")
        if profile_summary:
            user_prompt_lines.append(f"公司简介：{profile_summary}")
        if news_lines:
            user_prompt_lines.append("相关新闻：")
            user_prompt_lines.extend(news_lines)
        else:
            user_prompt_lines.append("相关新闻：暂无")
        user_prompt_lines.append(
            f"请输出 4-6 句中文摘要，不超过 {MARKET_AI_SUMMARY_MAX_CHARS} 个中文字符。"
        )
    else:
        system_prompt = (
            "You are a cautious market summary assistant. Summarize in 4-6 sentences using only the provided info, "
            "covering event, key drivers, market reaction/impact, and what to watch if possible. "
            "Each sentence should include concrete details (numbers, dates, or sources) when available. "
            "No speculation or advice; if info is insufficient, say so."
        )
        user_prompt_lines = [
            f"Symbol: {symbol}",
        ]
        if name:
            user_prompt_lines.append(f"Company: {name}")
        if exchange:
            user_prompt_lines.append(f"Exchange: {exchange}")
        if profile_summary:
            user_prompt_lines.append(f"Profile: {profile_summary}")
        if news_lines:
            user_prompt_lines.append("News:")
            user_prompt_lines.extend(news_lines)
        else:
            user_prompt_lines.append("News: none")
        user_prompt_lines.append(f"Write 4-6 sentences within {MARKET_AI_SUMMARY_MAX_CHARS} characters.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_prompt_lines)},
    ]
    try:
        response = bailian_ai.chat(
            model,
            messages,
            timeout_seconds=MARKET_AI_SUMMARY_TIMEOUT_SECONDS,
            api_key=api_key,
            extra_params={"temperature": 0.2, "max_tokens": MARKET_AI_SUMMARY_MAX_TOKENS},
        )
    except Exception:
        if raise_on_error:
            raise
        return None
    answer = str(response.get("answer") or "").strip()
    if not answer:
        return None
    if len(answer) > MARKET_AI_SUMMARY_MAX_CHARS:
        answer = answer[:MARKET_AI_SUMMARY_MAX_CHARS].rstrip() + "..."
    cache_set_object(cache_key, answer, MARKET_AI_SUMMARY_CACHE_TTL)
    return answer


def _build_ai_summary_struct_llm(
    profile_payload: dict[str, object],
    news_payload: list[dict[str, str]],
    *,
    lang_prefix: str,
    symbol: str,
    user: Any,
    api_key: str | None = None,
    raise_on_error: bool = False,
) -> dict[str, str] | None:
    api_key = api_key or resolve_api_credential(user, "bailian_api_key") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("BAILIAN_API_KEY")
    if not api_key:
        return None
    preferred_model = resolve_api_credential(user, "ai_model")
    model = (
        _resolve_bailian_model(preferred_model)
        or os.environ.get("BAILIAN_MODEL")
        or os.environ.get("DASHSCOPE_MODEL")
        or "qwen-max"
    )
    cache_key = _ai_summary_struct_cache_key(symbol, profile_payload, news_payload, lang_prefix=lang_prefix)
    cached = cache_get_object(cache_key)
    if isinstance(cached, dict) and cached:
        cached_struct = _normalize_ai_struct(cached)
        if cached_struct:
            if _struct_matches_lang(cached_struct, lang_prefix):
                return cached_struct
            translated = _translate_ai_struct(cached_struct, lang_prefix=lang_prefix, model=model, api_key=api_key)
            if translated:
                cache_set_object(cache_key, translated, MARKET_AI_SUMMARY_CACHE_TTL)
                return translated

    name = ""
    exchange = ""
    profile_summary = ""
    if isinstance(profile_payload, dict):
        name = str(profile_payload.get("name") or profile_payload.get("shortName") or "").strip()
        exchange = str(profile_payload.get("exchange") or "").strip()
        profile_summary = str(profile_payload.get("summary") or "").strip()

    news_lines: list[str] = []
    if isinstance(news_payload, list):
        for item in news_payload[:6]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            source = str(item.get("source") or "").strip()
            summary = str(item.get("summary") or "").strip()
            news_line = f"- {title}"
            if source:
                news_line = f"{news_line} ({source})"
            if summary:
                news_line = f"{news_line} 摘要：{summary}" if lang_prefix == "zh" else f"{news_line} Summary: {summary}"
            news_lines.append(news_line)

    if lang_prefix == "zh":
        system_prompt = (
            "你是谨慎的金融摘要助手。只根据提供的信息输出严格 JSON，"
            "不得猜测或给出投资建议。必须输出 JSON 对象，键为 event/impact/implication。"
            "每个字段写 2-3 句中文（至少 2 句），尽量包含具体信息/数值/来源；"
            "信息不足可写空字符串，但必须保留键。"
            "只输出 JSON，不要任何额外文字。"
        )
        user_prompt_lines = [
            f"股票代码：{symbol}",
        ]
        if name:
            user_prompt_lines.append(f"公司：{name}")
        if exchange:
            user_prompt_lines.append(f"交易所：{exchange}")
        if profile_summary:
            user_prompt_lines.append(f"公司简介：{profile_summary}")
        if news_lines:
            user_prompt_lines.append("相关新闻：")
            user_prompt_lines.extend(news_lines)
        else:
            user_prompt_lines.append("相关新闻：暂无")
        user_prompt_lines.append(
            "请输出 JSON：{\"event\":\"...\",\"impact\":\"...\",\"implication\":\"...\"}，不要输出其它文字。"
        )
    else:
        system_prompt = (
            "You are a cautious market summary assistant. Output strict JSON only with keys "
            "event/impact/implication. Use only the provided info. No speculation or advice. "
            "Write 2-3 sentences per field (at least 2) and include concrete details when available; "
            "if insufficient, use empty string but keep the key. "
            "Respond in English only and output JSON only."
        )
        user_prompt_lines = [
            f"Symbol: {symbol}",
        ]
        if name:
            user_prompt_lines.append(f"Company: {name}")
        if exchange:
            user_prompt_lines.append(f"Exchange: {exchange}")
        if profile_summary:
            user_prompt_lines.append(f"Profile: {profile_summary}")
        if news_lines:
            user_prompt_lines.append("News:")
            user_prompt_lines.extend(news_lines)
        else:
            user_prompt_lines.append("News: none")
        user_prompt_lines.append('Return JSON only: {"event":"...","impact":"...","implication":"..."}')

    preferred_model = resolve_api_credential(user, "ai_model")
    model = (
        _resolve_bailian_model(preferred_model)
        or os.environ.get("BAILIAN_MODEL")
        or os.environ.get("DASHSCOPE_MODEL")
        or "qwen-max"
    )
    def _call_llm(prompt_messages: list[dict[str, str]], *, temperature: float) -> dict[str, str] | None:
        try:
            response = bailian_ai.chat(
                model,
                prompt_messages,
                timeout_seconds=MARKET_AI_SUMMARY_TIMEOUT_SECONDS,
                api_key=api_key,
                response_format={"type": "json_object"},
                extra_params={"temperature": temperature, "max_tokens": MARKET_AI_SUMMARY_MAX_TOKENS},
            )
        except Exception:
            if raise_on_error:
                raise
            return None
        answer = str(response.get("answer") or "").strip()
        parsed = _parse_ai_json(answer)
        normalized = _normalize_ai_struct(parsed)
        if not normalized:
            normalized = _struct_from_plain_text(answer)
        if not normalized:
            return None
        if not _struct_matches_lang(normalized, lang_prefix):
            translated = _translate_ai_struct(normalized, lang_prefix=lang_prefix, model=model, api_key=api_key)
            if translated:
                return translated
            return None
        return normalized

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_prompt_lines)},
    ]
    structured = _call_llm(messages, temperature=0.2)
    if not structured:
        retry_prompt = (
            system_prompt
            + (" 必须严格 JSON，不要代码块。" if lang_prefix == "zh" else " Output strict JSON without code fences.")
        )
        retry_messages = [
            {"role": "system", "content": retry_prompt},
            {"role": "user", "content": "\n".join(user_prompt_lines)},
        ]
        structured = _call_llm(retry_messages, temperature=0.1)
    if not structured:
        return None
    cache_set_object(cache_key, structured, MARKET_AI_SUMMARY_CACHE_TTL)
    return structured


def _build_ai_summary_with_llm(
    profile_payload: dict[str, object],
    news_payload: list[dict[str, str]],
    *,
    lang_prefix: str,
    symbol: str,
    user: Any,
) -> str:
    fallback = _build_ai_summary(profile_payload, news_payload, lang_prefix=lang_prefix)
    summary = _build_ai_summary_llm(profile_payload, news_payload, lang_prefix=lang_prefix, symbol=symbol, user=user)
    return summary or fallback


def _build_ai_summary_with_meta(
    profile_payload: dict[str, object],
    news_payload: list[dict[str, str]],
    *,
    lang_prefix: str,
    symbol: str,
    user: Any,
) -> tuple[str, dict[str, str], dict[str, str] | None]:
    fallback = _build_ai_summary(profile_payload, news_payload, lang_prefix=lang_prefix)
    api_key = resolve_api_credential(user, "bailian_api_key") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("BAILIAN_API_KEY")
    if not api_key:
        message = "AI 未配置百炼 API Key，当前显示新闻摘要。" if lang_prefix == "zh" else "AI API key is missing; showing fallback summary."
        return fallback, {"status": "missing_key", "message": message, "source": "fallback"}, None
    try:
        structured = _build_ai_summary_struct_llm(
            profile_payload,
            news_payload,
            lang_prefix=lang_prefix,
            symbol=symbol,
            user=user,
            api_key=api_key,
            raise_on_error=True,
        )
    except Exception:
        message = "AI 暂时不可用，已显示回退摘要。" if lang_prefix == "zh" else "AI is unavailable right now; showing fallback summary."
        return fallback, {"status": "error", "message": message, "source": "fallback"}, None
    if structured:
        summary = _summary_from_struct(structured, lang_prefix=lang_prefix)
        return summary or fallback, {"status": "llm", "message": "", "source": "bailian"}, structured
    text_summary = _build_ai_summary_llm(
        profile_payload,
        news_payload,
        lang_prefix=lang_prefix,
        symbol=symbol,
        user=user,
        api_key=api_key,
        raise_on_error=False,
    )
    if text_summary:
        parsed_struct = _struct_from_plain_text(text_summary)
        return text_summary, {"status": "llm", "message": "", "source": "bailian"}, parsed_struct
    message = "AI 返回格式异常，已显示回退摘要。" if lang_prefix == "zh" else "AI returned an invalid format; showing fallback summary."
    return fallback, {"status": "fallback", "message": message, "source": "fallback"}, None


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
    
    def _pull_first(fields: tuple[str, ...]) -> pd.Series | None:
        for field in fields:
            series = _pull_field(field)
            if series is not None:
                return series
        return None

    open_series = _pull_first(("Open", "open"))
    high_series = _pull_first(("High", "high"))
    low_series = _pull_first(("Low", "low"))
    close_series = _pull_first(("Close", "close"))
    if open_series is None or high_series is None or low_series is None or close_series is None:
        return []
    volume_series = _pull_first(("Volume", "volume", "V", "v"))

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
    if volume_series is not None:
        ohlc_frame["volume"] = volume_series

    ohlc_frame = ohlc_frame.sort_index().tail(limit)
    bars: list[dict[str, float | int]] = []
    for ts, row in ohlc_frame.iterrows():
        try:
            stamp = pd.Timestamp(ts)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize(timezone.utc)
            else:
                stamp = stamp.tz_convert(timezone.utc)
            time_val = float(stamp.timestamp())
        except Exception:
            continue
        try:
            entry = {
                "time": time_val,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            if "volume" in row and row["volume"] is not None:
                try:
                    volume_val = float(row["volume"])
                except Exception:
                    volume_val = None
                if volume_val is not None and not pd.isna(volume_val):
                    entry["volume"] = volume_val
            bars.append(entry)
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
    
    def _pull_first(fields: tuple[str, ...]) -> pd.Series | None:
        for field in fields:
            series = _pull_field(field)
            if series is not None:
                return series
        return None

    open_series = _pull_first(("Open", "open"))
    high_series = _pull_first(("High", "high"))
    low_series = _pull_first(("Low", "low"))
    close_series = _pull_first(("Close", "close"))
    if open_series is None or high_series is None or low_series is None or close_series is None:
        return pd.DataFrame()
    volume_series = _pull_first(("Volume", "volume", "V", "v"))

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
    if volume_series is not None:
        ohlc_frame["volume"] = volume_series

    if not isinstance(ohlc_frame.index, pd.DatetimeIndex):
        ohlc_frame.index = pd.to_datetime(ohlc_frame.index, errors="coerce")
        ohlc_frame = ohlc_frame.dropna()
    if ohlc_frame.empty:
        return pd.DataFrame()
    ohlc_frame = ohlc_frame.sort_index()

    agg_map = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in ohlc_frame.columns:
        agg_map["volume"] = "sum"
    resampled = ohlc_frame.resample(rule, label="right", closed="right").agg(agg_map)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    if resampled.empty:
        return pd.DataFrame()
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
    }
    if "volume" in resampled.columns:
        rename_map["volume"] = "Volume"
    resampled = resampled.rename(columns=rename_map)
    return resampled


def _merge_ohlc_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    normalized: list[pd.DataFrame] = []
    for frame in frames:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        expected = {"Open", "High", "Low", "Close"}
        if not expected.issubset(set(frame.columns)):
            continue
        local = frame.copy()
        if not isinstance(local.index, pd.DatetimeIndex):
            local.index = pd.to_datetime(local.index, utc=True, errors="coerce")
        else:
            if local.index.tz is None:
                local.index = local.index.tz_localize(timezone.utc)
            else:
                local.index = local.index.tz_convert(timezone.utc)
        local = local.sort_index()
        normalized.append(local)
    if not normalized:
        return pd.DataFrame()
    merged = pd.concat(normalized, axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


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
    market_source = resolve_market_provider(user=request.user)
    if market_source == "alpaca":
        key_id, secret = resolve_alpaca_data_credentials(user=request.user)
        has_credentials = bool(key_id and secret)
    else:
        has_credentials = has_market_data_credentials(user=request.user, provider=market_source)
    if not has_credentials:
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
    include_ai = _parse_bool(params.get("include_ai"), default=True)
    ai_only = _parse_bool(params.get("ai_only"), default=False)
    include_bars = _parse_bool(params.get("include_bars"), default=True)
    news_only = _parse_bool(params.get("news_only"), default=False)
    news_limit, news_offset = _resolve_news_paging(params.get("news_limit"), params.get("news_offset"))
    if ai_only:
        include_ai = True
        include_bars = False
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
        prev_bar = snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar") or {}
        prev_close = _coerce_positive_price((prev_bar or {}).get("c"))
        last_price = _resolve_snapshot_last_price(snapshot, prev_close=prev_close, allow_quote_fallback=True)
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
                "data_source": market_source,
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

        if news_only:
            news_payload, news_meta = _fetch_symbol_news_page(
                detail_symbol,
                user_id=str(request.user.id),
                limit=news_limit,
                offset=news_offset,
            )
            return JsonResponse(
                {
                    "symbol": detail_symbol,
                    "news": news_payload,
                    "news_meta": news_meta,
                    "request_id": request_id,
                },
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
        if include_bars:
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

        debug_requested = str(params.get("debug") or "").strip().lower() in {"1", "true", "yes"}
        debug_flag = debug_requested and bool(getattr(settings, "DEBUG", False)) and bool(getattr(request.user, "is_staff", False))
        user_id = str(request.user.id)
        profile_payload = _fetch_company_profile(detail_symbol, user_id=user_id, provider=market_source)
        fundamentals_debug = None
        if debug_flag:
            fundamentals, fundamentals_debug = _fetch_yfinance_fundamentals_debug(
                detail_symbol,
                user_id=user_id,
                provider=market_source,
            )
        else:
            fundamentals = _fetch_yfinance_fundamentals(
                detail_symbol,
                user_id=user_id,
                provider=market_source,
            )
        if isinstance(profile_payload, dict) and isinstance(fundamentals, dict):
            for key, value in fundamentals.items():
                if value in (None, "", []):
                    continue
                if profile_payload.get(key):
                    continue
                profile_payload[key] = value
        profile_missing_reasons = _build_profile_missing_reasons(profile_payload if isinstance(profile_payload, dict) else {})
        news_payload, news_meta = _fetch_symbol_news_page(
            detail_symbol,
            user_id=str(request.user.id),
            limit=news_limit,
            offset=news_offset,
        )
        key_stats = _fetch_52w_stats(detail_symbol, user_id=str(request.user.id)) if not ai_only else {"high_52w": None, "low_52w": None, "as_of": None}
        raw_lang = str(params.get("lang") or "").strip().lower()
        if raw_lang.startswith("zh"):
            lang_prefix = "zh"
        elif raw_lang.startswith("en"):
            lang_prefix = "en"
        else:
            lang_prefix = (getattr(request, "LANGUAGE_CODE", "") or request.COOKIES.get("django_language", "zh-hans")).lower()[:2]
        if include_ai:
            ai_summary, ai_summary_meta, ai_summary_struct = _build_ai_summary_with_meta(
                profile_payload,
                news_payload,
                lang_prefix=lang_prefix,
                symbol=detail_symbol,
                user=request.user,
            )
        else:
            ai_summary = ""
            ai_summary_struct = None
            ai_summary_meta = {"status": "pending", "message": "", "source": "bailian"}
        debug_payload = None
        if debug_flag:
            fundamentals_info = fundamentals_debug or {
                "symbol": detail_symbol,
                "keys": sorted(fundamentals.keys()) if isinstance(fundamentals, dict) else [],
                "has_values": bool(
                    isinstance(fundamentals, dict)
                    and any(value not in (None, "", [], {}) for value in fundamentals.values())
                ),
            }
            debug_payload = {
                "auth": {
                    "is_authenticated": bool(getattr(request.user, "is_authenticated", False)),
                    "user_id": getattr(request.user, "id", None),
                    "session_key_present": bool(getattr(getattr(request, "session", None), "session_key", None)),
                    "has_session_cookie": "sessionid" in request.COOKIES,
                    "cookie_names": sorted(request.COOKIES.keys()),
                    "secure_cookie_required": bool(getattr(settings, "SESSION_COOKIE_SECURE", False)),
                    "scheme": request.scheme,
                    "host": request.get_host(),
                },
                "fundamentals": fundamentals_info,
                "lang": {"param": raw_lang, "resolved": lang_prefix},
                "options": {
                    "include_ai": include_ai,
                    "ai_only": ai_only,
                    "include_bars": include_bars,
                },
            }
        if ai_only:
            response_payload = {
                "symbol": detail_symbol,
                "generated_at": time.time(),
                "ai_summary": ai_summary,
                "ai_summary_meta": ai_summary_meta,
                "ai_summary_struct": ai_summary_struct,
                "profile_missing_reasons": profile_missing_reasons,
                "request_id": request_id,
            }
            if debug_payload:
                response_payload["debug"] = debug_payload
            return JsonResponse(response_payload, json_dumps_params={"ensure_ascii": False})
        data_source = _infer_market_source(frame) if frame is not None else "alpaca"
        response_payload = {
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
            "generated_at": time.time(),
            "data_source": data_source,
            "bars": bars,
            "key_stats": key_stats,
            "profile": profile_payload,
            "profile_missing_reasons": profile_missing_reasons,
            "news": news_payload,
            "news_meta": news_meta,
            "ai_summary": ai_summary,
            "ai_summary_meta": ai_summary_meta,
            "ai_summary_struct": ai_summary_struct,
            "request_id": request_id,
        }
        if debug_payload:
            response_payload["debug"] = debug_payload
        return JsonResponse(response_payload, json_dumps_params={"ensure_ascii": False})

    capabilities_payload = _market_capabilities_payload()
    if not _is_list_timeframe_supported(list_type, resolved.timeframe.key):
        message = _("该榜单暂不支持所选区间。")
        return JsonResponse(
            {
                "error": message,
                "message": message,
                "error_code": "timeframe_not_supported",
                "list_type": list_type,
                "timeframe": {
                    "key": resolved.timeframe.key,
                    "label": resolved.timeframe.label,
                    "label_en": resolved.timeframe.label_en,
                },
                "capabilities": capabilities_payload,
                "request_id": request_id,
            },
            status=400,
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

    rankings_page_cache_key: str | None = None
    if request.method == "GET" and not resolved.query and list_type in LIST_TYPES:
        rankings_page_cache_key = build_cache_key(
            "market-rankings-page",
            market_source,
            str(request.user.id),
            resolved.timeframe.key,
            list_type,
            page_offset,
            page_size,
            str(getattr(request, "LANGUAGE_CODE", "") or ""),
        )
        cached_rankings_page = cache_get_object(rankings_page_cache_key)
        if isinstance(cached_rankings_page, dict):
            if not _cached_rankings_payload_usable(cached_rankings_page):
                cached_rankings_page = None
            else:
                cached_payload = dict(cached_rankings_page)
                cached_payload["generated_at"] = time.time()
                cached_payload["request_id"] = request_id
                return JsonResponse(cached_payload, json_dumps_params={"ensure_ascii": False})
        if rankings_page_cache_key and cached_rankings_page is None:
            cache_set_object(rankings_page_cache_key, None, 1)

    symbols: list[str] = []
    restrict_to_query = False
    if resolved.query:
        symbols = [resolved.query]
        restrict_to_query = True

    list_items: list[dict[str, object]] = []
    active_list_type = list_type
    data_source = market_source
    user_id = str(request.user.id) if request.user.is_authenticated else None
    gainers: list[dict[str, object]] = []
    losers: list[dict[str, object]] = []
    most_actives: list[dict[str, object]] = []
    top_turnovers: list[dict[str, object]] = []
    ranking_timeframe: Timeframe | None = None
    universe_rankings: list[dict[str, object]] | None = None
    universe_source = "unknown"
    timeframe_rankings: list[dict[str, object]] | None = None
    timeframe_source = "unknown"
    timeframe_fallback_pending = False
    used_active_snapshot = False
    background_only_non_1d = resolved.timeframe.key != "1d" and MARKET_RANKINGS_BACKGROUND_ONLY

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
        if background_only_non_1d:
            timeframe_snapshot = _load_timeframe_snapshot_rows(
                resolved.timeframe.key,
                provider=market_source,
            )
        if list_type in {"gainers", "losers"}:
            if resolved.timeframe.key == "1d":
                if market_source == "alpaca":
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
                        data_source = market_source
                        ranking_timeframe = resolved.timeframe
                        used_timeframe_rankings = True

            if not used_timeframe_rankings and resolved.timeframe.key != "1d" and background_only_non_1d:
                timeframe_snapshot = _load_timeframe_snapshot_rows(
                    resolved.timeframe.key,
                    provider=market_source,
                )
                if _snapshot_rows_schema_valid_for_list(timeframe_snapshot, list_type=list_type):
                    gainers, losers = _split_rankings(timeframe_snapshot, page_stop)
                    if gainers or losers:
                        list_items = losers if list_type == "losers" else gainers
                        data_source = market_source
                        ranking_timeframe = resolved.timeframe
                        used_timeframe_rankings = True
                        used_active_snapshot = True

            if not used_timeframe_rankings and resolved.timeframe.key == "1d":
                if snapshot_rankings is None:
                    snapshot_rankings = _build_snapshot_rankings(user_id)
                if snapshot_rankings:
                    gainers, losers = _split_rankings(snapshot_rankings, page_stop)
                    if gainers or losers:
                        list_items = losers if list_type == "losers" else gainers
                        data_source = market_source
                        ranking_timeframe = resolved.timeframe
                        used_timeframe_rankings = True
                        used_active_snapshot = True

            if not used_timeframe_rankings and not background_only_non_1d:
                if timeframe_rankings is None:
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
                timeframe_rows: list[dict[str, object]] = []
                timeframe_rows_source = "unknown"
                timeframe_rows_from_snapshot = False
                if resolved.timeframe.key != "1d":
                    if background_only_non_1d:
                        if timeframe_snapshot is None:
                            timeframe_snapshot = _load_timeframe_snapshot_rows(
                                resolved.timeframe.key,
                                provider=market_source,
                            )
                        if _snapshot_rows_schema_valid_for_list(timeframe_snapshot, list_type="most_active"):
                            timeframe_rows = timeframe_snapshot
                            timeframe_rows_source = market_source
                            timeframe_rows_from_snapshot = True
                            used_active_snapshot = True
                    if timeframe_rows_from_snapshot and not _snapshot_rows_schema_valid_for_list(
                        timeframe_rows,
                        list_type="most_active",
                    ):
                        timeframe_rows = []
                        timeframe_rows_source = "unknown"
                        timeframe_rows_from_snapshot = False
                        used_active_snapshot = False
                    if not timeframe_rows and not background_only_non_1d:
                        try:
                            timeframe_rows, timeframe_rows_source = _resolve_universe_window_rankings(
                                resolved.timeframe,
                                user_id=user_id,
                            )
                        except Exception:
                            LOGGER.exception(
                                "failed to resolve most_active window rankings [timeframe=%s user_id=%s]",
                                resolved.timeframe.key,
                                user_id,
                            )
                            record_metric(
                                "market.insights.error",
                                request_id=request_id,
                                user_id=request.user.id,
                                error="most_active_window_rankings_failed",
                                timeframe=resolved.timeframe.key,
                            )
                            timeframe_rows, timeframe_rows_source = [], "unknown"
                if timeframe_rows:
                    list_items = _sort_rows_by_metric(
                        timeframe_rows,
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
                    data_source = timeframe_rows_source or data_source
                    ranking_timeframe = resolved.timeframe
                elif resolved.timeframe.key == "1d":
                    if market_source == "alpaca":
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
                        data_source = market_source if list_items else "unknown"
                        if list_items:
                            ranking_timeframe = resolved.timeframe
                if not list_items and resolved.timeframe.key == "1d":
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
                        data_source = market_source if list_items else "unknown"
                        if list_items and ranking_timeframe is None:
                            ranking_timeframe = TIMEFRAMES["1d"] if resolved.timeframe.key != "1d" else resolved.timeframe
                        used_active_snapshot = True
                    elif not MARKET_RANKINGS_BACKGROUND_ONLY:
                        if market_source == "alpaca":
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
                            data_source = market_source if list_items else "unknown"
                            if list_items and ranking_timeframe is None:
                                ranking_timeframe = TIMEFRAMES["1d"] if resolved.timeframe.key != "1d" else resolved.timeframe
            elif list_type == "top_turnover":
                timeframe_rows: list[dict[str, object]] = []
                timeframe_rows_source = "unknown"
                timeframe_rows_from_snapshot = False
                if resolved.timeframe.key != "1d":
                    if background_only_non_1d:
                        if timeframe_snapshot is None:
                            timeframe_snapshot = _load_timeframe_snapshot_rows(
                                resolved.timeframe.key,
                                provider=market_source,
                            )
                        if _snapshot_rows_schema_valid_for_list(timeframe_snapshot, list_type="top_turnover"):
                            timeframe_rows = timeframe_snapshot
                            timeframe_rows_source = market_source
                            timeframe_rows_from_snapshot = True
                            used_active_snapshot = True
                    if timeframe_rows_from_snapshot and not _snapshot_rows_schema_valid_for_list(
                        timeframe_rows,
                        list_type="top_turnover",
                    ):
                        timeframe_rows = []
                        timeframe_rows_source = "unknown"
                        timeframe_rows_from_snapshot = False
                        used_active_snapshot = False
                    if not timeframe_rows and not background_only_non_1d:
                        try:
                            timeframe_rows, timeframe_rows_source = _resolve_universe_window_rankings(
                                resolved.timeframe,
                                user_id=user_id,
                            )
                        except Exception:
                            LOGGER.exception(
                                "failed to resolve top_turnover window rankings [timeframe=%s user_id=%s]",
                                resolved.timeframe.key,
                                user_id,
                            )
                            record_metric(
                                "market.insights.error",
                                request_id=request_id,
                                user_id=request.user.id,
                                error="top_turnover_window_rankings_failed",
                                timeframe=resolved.timeframe.key,
                            )
                            timeframe_rows, timeframe_rows_source = [], "unknown"
                if timeframe_rows:
                    list_items = _sort_rows_by_metric(
                        timeframe_rows,
                        "dollar_volume",
                        reverse=True,
                        predicate=lambda value: value > 0,
                    )[:page_stop]
                    if list_items:
                        data_source = timeframe_rows_source or data_source
                        ranking_timeframe = resolved.timeframe
                if not list_items and resolved.timeframe.key == "1d":
                    if snapshot_rankings is None:
                        snapshot_rankings = _build_snapshot_rankings(user_id)
                    source_rows = snapshot_rankings
                    if not snapshot_rankings:
                        if universe_rankings is None:
                            universe_rankings, universe_source = _resolve_universe_rankings(user_id)
                        source_rows = universe_rankings
                        data_source = universe_source or data_source
                    else:
                        data_source = market_source if snapshot_rankings else data_source
                    list_items = _sort_rows_by_metric(
                        source_rows,
                        "dollar_volume",
                        reverse=True,
                        predicate=lambda value: value > 0,
                    )[:page_stop]
                    if list_items and ranking_timeframe is None:
                        ranking_timeframe = TIMEFRAMES["1d"] if resolved.timeframe.key != "1d" else resolved.timeframe
                    if snapshot_rankings and list_items:
                        used_active_snapshot = True

    if list_type in {"gainers", "losers"} and not list_items and not background_only_non_1d:
        user_id = str(request.user.id) if request.user.is_authenticated else None
        if timeframe_rankings is None:
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

    if list_type == "most_active" and not list_items and resolved.timeframe.key == "1d":
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
            if ranking_timeframe is None:
                ranking_timeframe = TIMEFRAMES["1d"] if resolved.timeframe.key != "1d" else resolved.timeframe

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

    if (
        active_list_type in {"most_active", "top_turnover"}
        and resolved.timeframe.key != "1d"
        and ranking_timeframe is not None
        and ranking_timeframe.key != resolved.timeframe.key
    ):
        timeframe_fallback_pending = True

    if resolved.timeframe.key == "1d":
        gainers = _enrich_rows_with_snapshot_metrics(gainers, user_id=user_id, provider=market_source)
        losers = _enrich_rows_with_snapshot_metrics(losers, user_id=user_id, provider=market_source)
        most_actives = _enrich_rows_with_snapshot_metrics(most_actives, user_id=user_id, provider=market_source)
        list_items = _enrich_rows_with_snapshot_metrics(list_items, user_id=user_id, provider=market_source)
    else:
        gainers = _enrich_rows_with_window_metrics(gainers, timeframe=resolved.timeframe, user_id=user_id)
        losers = _enrich_rows_with_window_metrics(losers, timeframe=resolved.timeframe, user_id=user_id)
        most_actives = _enrich_rows_with_window_metrics(most_actives, timeframe=resolved.timeframe, user_id=user_id)
        list_items = _enrich_rows_with_window_metrics(list_items, timeframe=resolved.timeframe, user_id=user_id)

    gainers = [
        _apply_common_market_item_shape(entry)
        for entry in gainers
        if isinstance(entry, dict)
    ]
    losers = [
        _apply_common_market_item_shape(entry)
        for entry in losers
        if isinstance(entry, dict)
    ]
    most_actives = [
        _apply_common_market_item_shape(entry)
        for entry in most_actives
        if isinstance(entry, dict)
    ]
    list_items = [
        _apply_common_market_item_shape(entry, timeframe_pending=timeframe_fallback_pending)
        for entry in list_items
        if isinstance(entry, dict)
    ]

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
    elif active_list_type == "top_turnover":
        top_turnovers = list_items

    snapshot_state = _snapshot_state_payload(
        timeframe_key=resolved.timeframe.key,
        provider=market_source,
        used_active_snapshot=used_active_snapshot,
        list_type=active_list_type,
    )
    data_state = _resolve_market_data_state(items=list_items, snapshot_state=snapshot_state)
    _maybe_trigger_snapshot_refresh_nonblocking(
        request=request,
        timeframe_key=resolved.timeframe.key,
        list_type=active_list_type,
        snapshot_state=snapshot_state,
        data_state=data_state,
    )

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
        "generated_at": time.time(),
        "data_source": (
            data_source
            if isinstance(data_source, str) and data_source and data_source.lower() != "unknown"
            else (market_source or "unknown")
        ),
        "query": resolved.query,
        "list_type": active_list_type,
        "items": list_items,
        "gainers": gainers,
        "losers": losers,
        "most_actives": most_actives,
        "top_turnover": top_turnovers,
        "top_turnovers": top_turnovers,
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
        "snapshot_state": snapshot_state,
        "data_state": data_state,
        "capabilities": capabilities_payload,
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
    if rankings_page_cache_key:
        cache_set_object(rankings_page_cache_key, response, MARKET_RANKINGS_PAGE_CACHE_TTL)
    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})


def _build_rankings_status_items(*, provider: str) -> list[dict[str, object]]:
    support = _resolve_list_timeframe_support()
    global_building_payload = _resolve_building_snapshot_payload("1d", provider=provider)
    global_build_state = "idle"
    global_build_progress = 0
    if isinstance(global_building_payload, Mapping):
        state = str(global_building_payload.get("status") or "").strip().lower()
        if state in {"running", "error", "idle", "stalled"}:
            global_build_state = state
        progress_payload = global_building_payload.get("progress")
        if isinstance(progress_payload, Mapping):
            global_build_progress = _snapshot_progress_percent(progress_payload)
        elif global_build_state == "running":
            global_build_progress = _snapshot_progress_percent(
                {
                    "chunks_completed": global_building_payload.get("chunks_completed"),
                    "total_chunks": global_building_payload.get("total_chunks"),
                }
            )

    items: list[dict[str, object]] = []
    for list_type in LIST_TYPES:
        allowed = support.get(list_type, ())
        for timeframe_key in allowed:
            timeframe = TIMEFRAMES.get(timeframe_key)
            if timeframe is None:
                continue
            snapshot_state = _snapshot_state_payload(
                timeframe_key=timeframe_key,
                provider=provider,
                used_active_snapshot=True,
                list_type=list_type,
            )
            stale_seconds = _coerce_number(snapshot_state.get("stale_seconds"))
            stale_threshold_seconds = _coerce_number(snapshot_state.get("stale_threshold_seconds"))
            cycle_progress = 0
            if (
                stale_seconds is not None
                and stale_seconds >= 0
                and stale_threshold_seconds is not None
                and stale_threshold_seconds > 0
            ):
                try:
                    cycle_progress = max(0, min(100, int(round((stale_seconds / stale_threshold_seconds) * 100))))
                except Exception:
                    cycle_progress = 0
            building_payload = _resolve_building_snapshot_payload(timeframe_key, provider=provider)
            build_state = "idle"
            build_started_at = None
            build_updated_at = None
            build_error_code = None
            if isinstance(building_payload, Mapping):
                payload_status = str(building_payload.get("status") or "").strip().lower()
                if payload_status in {"running", "error", "idle", "stalled"}:
                    build_state = payload_status
                elif bool(snapshot_state.get("building")):
                    build_state = "running"
                build_started_at = building_payload.get("started_at") or building_payload.get("started_ts")
                build_updated_at = building_payload.get("updated_at") or building_payload.get("updated_ts")
                build_error_code = building_payload.get("error")
            elif bool(snapshot_state.get("building")):
                build_state = "running"

            if timeframe_key == "1d" and global_build_state == "running":
                build_state = "running"
                build_progress = global_build_progress
            else:
                build_progress = int(snapshot_state.get("building_progress") or 0) if build_state == "running" else 0
            building = build_state == "running"
            items.append(
                {
                    "list_type": list_type,
                    "timeframe": {
                        "key": timeframe.key,
                        "label": timeframe.label,
                        "label_en": timeframe.label_en,
                    },
                    "active_generated_at": snapshot_state.get("active_generated_at"),
                    "building": bool(building),
                    "progress": build_progress,
                    "build_progress": build_progress,
                    "cycle_progress": cycle_progress,
                    "build_state": build_state,
                    "build_started_at": build_started_at,
                    "build_updated_at": build_updated_at,
                    "build_error_code": build_error_code,
                    "provider": snapshot_state.get("provider") or provider,
                    "stale_seconds": stale_seconds,
                    "stale_threshold_seconds": stale_threshold_seconds,
                    "active_schema_valid": snapshot_state.get("active_schema_valid"),
                }
            )
    return items


def _build_rankings_status_groups(
    *,
    provider: str,
    items: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    support = _resolve_list_timeframe_support()
    flat_items = items if isinstance(items, list) else _build_rankings_status_items(provider=provider)
    by_pair: dict[tuple[str, str], dict[str, object]] = {}
    for item in flat_items:
        if not isinstance(item, Mapping):
            continue
        list_type = str(item.get("list_type") or "")
        timeframe = item.get("timeframe")
        if not isinstance(timeframe, Mapping):
            continue
        timeframe_key = str(timeframe.get("key") or "")
        if not list_type or not timeframe_key:
            continue
        by_pair[(list_type, timeframe_key)] = dict(item)

    groups: list[dict[str, object]] = []
    for timeframe_key, timeframe in TIMEFRAMES.items():
        group_items: list[dict[str, object]] = []
        for list_type in LIST_TYPES:
            supported = timeframe_key in support.get(list_type, ())
            row = by_pair.get((list_type, timeframe_key))
            if not isinstance(row, Mapping):
                row = {
                    "list_type": list_type,
                    "timeframe": {
                        "key": timeframe.key,
                        "label": timeframe.label,
                        "label_en": timeframe.label_en,
                    },
                    "active_generated_at": None,
                    "building": False,
                    "progress": 0,
                    "build_progress": 0,
                    "cycle_progress": 0,
                    "build_state": "idle",
                    "build_started_at": None,
                    "build_updated_at": None,
                    "build_error_code": None,
                    "provider": provider,
                    "stale_seconds": None,
                    "stale_threshold_seconds": _snapshot_ttl_for_timeframe(timeframe.key),
                    "active_schema_valid": None,
                }
            payload = dict(row)
            payload["supported"] = bool(supported)
            payload.setdefault(
                "timeframe",
                {"key": timeframe.key, "label": timeframe.label, "label_en": timeframe.label_en},
            )
            payload.setdefault("provider", provider)
            payload.setdefault("build_progress", int(payload.get("progress") or 0))
            payload.setdefault("cycle_progress", 0)
            payload.setdefault("build_started_at", None)
            payload.setdefault("build_updated_at", None)
            payload.setdefault("build_error_code", None)
            payload.setdefault("build_state", "running" if payload.get("building") else "idle")
            group_items.append(payload)
        groups.append(
            {
                "timeframe": {
                    "key": timeframe.key,
                    "label": timeframe.label,
                    "label_en": timeframe.label_en,
                },
                "items": group_items,
            }
        )
    return groups


@login_required
@require_http_methods(["GET"])
def market_rankings_status(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    provider = resolve_market_provider(user=request.user)
    items = _build_rankings_status_items(provider=provider)
    groups = _build_rankings_status_groups(provider=provider, items=items)
    # Self-heal path: when snapshots are missing/stalled/over-TTL and nothing is
    # actively running, kick a non-blocking rebuild so observer/menu does not
    # stay stale forever.
    try:
        has_active = any(
            isinstance(item, Mapping) and bool(item.get("active_generated_at"))
            for item in items
        )
        has_running = any(
            isinstance(item, Mapping) and str(item.get("build_state") or "").lower() == "running"
            for item in items
        )
        has_stalled_or_error = any(
            isinstance(item, Mapping)
            and str(item.get("build_state") or "").lower() in {"stalled", "error"}
            for item in items
        )
        has_over_ttl_stale = any(
            isinstance(item, Mapping)
            and (_coerce_number(item.get("stale_threshold_seconds")) or 0) > 0
            and (_coerce_number(item.get("stale_seconds")) or 0)
            >= (_coerce_number(item.get("stale_threshold_seconds")) or 0)
            for item in items
        )
        should_trigger = (not has_active and not has_running) or has_stalled_or_error or (
            has_over_ttl_stale and not has_running
        )
        if should_trigger:
            from ..tasks import trigger_market_snapshot_refresh

            user_id = str(request.user.id) if request.user.is_authenticated else None
            trigger_market_snapshot_refresh(user_id=user_id, prefer_thread=True)
    except Exception:
        LOGGER.debug("market_rankings_status self-heal trigger failed", exc_info=True)
    return JsonResponse(
        {
            "provider": provider,
            "items": items,
            "groups": groups,
            "generated_at": time.time(),
            "request_id": request_id,
        },
        json_dumps_params={"ensure_ascii": False},
    )


@require_http_methods(["GET"])
def market_auth_debug(request: HttpRequest) -> JsonResponse:
    return _market_auth_debug(request)


@login_required
@require_http_methods(["POST"])
def market_news_sentiment(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"error": _("请求体解析失败。"), "request_id": request_id}, status=400)

    if not isinstance(payload, dict):
        payload = {}
    symbol = str(payload.get("symbol") or "").strip().upper()
    raw_items = payload.get("news") or []
    if not isinstance(raw_items, list) or not raw_items:
        return JsonResponse({"labels": [], "request_id": request_id}, json_dumps_params={"ensure_ascii": False})

    items = _normalize_news_items([item for item in raw_items if isinstance(item, dict)])
    if not items:
        return JsonResponse({"labels": [], "request_id": request_id}, json_dumps_params={"ensure_ascii": False})

    cache_key = _news_sentiment_cache_key(symbol or "GLOBAL", items)
    cached = cache_get_object(cache_key)
    if isinstance(cached, list) and cached:
        return JsonResponse({"labels": cached, "request_id": request_id}, json_dumps_params={"ensure_ascii": False})

    prompt_lines = []
    for idx, item in enumerate(items, start=1):
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        source = str(item.get("source") or "").strip()
        parts = [title]
        if summary:
            parts.append(summary)
        if source:
            parts.append(f"来源:{source}")
        prompt_lines.append(f"{idx}. " + " | ".join(parts))

    system_prompt = (
        "你是金融新闻情绪分类器。只输出JSON结果，不要解释。"
        "标签只能是 bullish / bearish / neutral。"
        "如果信息不足，返回 neutral。"
    )
    user_prompt = "请按顺序为以下新闻打标签，输出 JSON：\n" + "\n".join(prompt_lines)
    response_schema = {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["labels"],
    }

    labels: list[str] = []
    model_used = ""
    provider_used = ""
    try:
        resp = run_llm_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            user=request.user,
            timeout_seconds=MARKET_NEWS_SENTIMENT_TIMEOUT_SECONDS,
            response_schema=response_schema,
        )
        answer = (resp.get("answer") or "").strip() if isinstance(resp, dict) else ""
        model_used = str(resp.get("model") or "").strip() if isinstance(resp, dict) else ""
        provider_used = str(resp.get("provider") or "").strip() if isinstance(resp, dict) else ""
        parsed = None
        if answer:
            try:
                parsed = json.loads(answer)
            except Exception:
                match = re.search(r"(\{.*\}|\[.*\])", answer, flags=re.S)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                    except Exception:
                        parsed = None
        if isinstance(parsed, dict):
            candidate = parsed.get("labels") or parsed.get("sentiments") or parsed.get("result")
            if isinstance(candidate, list):
                labels = [str(item) for item in candidate]
        elif isinstance(parsed, list):
            labels = [str(item) for item in parsed]
    except LLMIntegrationError:
        labels = []
    except Exception:
        labels = []

    if not labels:
        labels = [_infer_sentiment_rule(f"{item.get('title', '')} {item.get('summary', '')}") for item in items]

    normalized_labels: list[str] = []
    for label in labels[: len(items)]:
        normalized_labels.append(_normalize_sentiment_label(label))
    while len(normalized_labels) < len(items):
        normalized_labels.append("neutral")

    cache_set_object(cache_key, normalized_labels, MARKET_NEWS_SENTIMENT_CACHE_TTL)
    record_metric(
        "market.news.sentiment",
        request_id=request_id,
        user_id=request.user.id,
        provider=provider_used,
        model=model_used,
        count=len(normalized_labels),
    )
    return JsonResponse(
        {"labels": normalized_labels, "request_id": request_id},
        json_dumps_params={"ensure_ascii": False},
    )


@login_required
@require_http_methods(["GET"])
def market_chart_data(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    market_source = resolve_market_provider(user=request.user)
    if market_source == "alpaca":
        key_id, secret = resolve_alpaca_data_credentials(user=request.user)
        has_credentials = bool(key_id and secret)
    else:
        has_credentials = has_market_data_credentials(user=request.user, provider=market_source)
    if not has_credentials:
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
    history_mode = str(params.get("history_mode") or "auto").strip().lower()
    if history_mode not in {"auto", "flatfiles", "rest"}:
        history_mode = "auto"
    flatfiles_enabled = _setting_bool("MASSIVE_FLATFILES_ENABLED", True)
    interval = _resolve_chart_interval(interval_key) or _resolve_chart_interval("1m")
    if interval is None:
        return JsonResponse(
            {"error": _("无效的时间粒度。"), "request_id": request_id},
            status=400,
            json_dumps_params={"ensure_ascii": False},
        )

    cache_alias = getattr(settings, "MARKET_HISTORY_CACHE_ALIAS", None)
    chart_cache_ttl = getattr(settings, "MARKET_CHART_CACHE_TTL", None)
    if chart_cache_ttl is None:
        if interval.unit in {"tick", "second"}:
            chart_cache_ttl = 0
        elif interval.unit == "minute":
            chart_cache_ttl = 5
        elif interval.unit == "hour":
            chart_cache_ttl = 10
        else:
            chart_cache_ttl = 30
    has_range_override = bool(params.get("start") or params.get("end"))
    cache_enabled = chart_cache_ttl > 0 and interval.unit not in {"tick", "second"} and not has_range_override
    cache_key = build_cache_key(
        "market-chart",
        symbol,
        range_key,
        interval.key,
        DEFAULT_FEED,
        market_source,
        history_mode,
        request.user.id,
    )
    if cache_enabled:
        cached = cache_get_object(cache_key, cache_alias=cache_alias)
        if isinstance(cached, dict) and cached.get("bars"):
            cached.setdefault("historical_source", "none")
            cached.setdefault("intraday_source", "none")
            cached.setdefault("history_coverage", {"start_ts": None, "end_ts": None, "complete": True})
            cached.setdefault("history_cursor_next", None)
            cached.setdefault("cache_meta", {"hit": 0, "miss": 0, "warmed_ranges": []})
            cached["request_id"] = request_id
            return JsonResponse(cached, json_dumps_params={"ensure_ascii": False})

    start, end, range_seconds = _resolve_range_window(range_key)
    start_override = parse_timestamp(params.get("start")) if params.get("start") else None
    end_override = parse_timestamp(params.get("end")) if params.get("end") else None
    if end_override is not None:
        end = datetime.fromtimestamp(end_override, tz=timezone.utc)
    if start_override is not None:
        start = datetime.fromtimestamp(start_override, tz=timezone.utc)
    if start_override is not None and end_override is not None:
        range_seconds = max(1.0, float(end_override - start_override))
    elif end_override is not None:
        start = end - timedelta(seconds=range_seconds)
    elif start_override is not None:
        end = start + timedelta(seconds=range_seconds)
    downgrade_to: str | None = None
    downgrade_message: str | None = None
    window_limited = False
    effective_range = range_key
    base_interval = "1m"
    resample_rule: str | None = None

    range_fallbacks: dict[str, tuple[str, ...]] = {
        "1d": ("5d", "1mo"),
        "5d": ("1mo", "6mo"),
        "1mo": ("6mo",),
    }

    historical_source = "none"
    intraday_source = "none"
    history_cursor_next: float | None = None
    history_coverage: dict[str, object] = {"start_ts": None, "end_ts": None, "complete": True}
    cache_meta: dict[str, object] = {"hit": 0, "miss": 0, "warmed_ranges": []}

    def _today_market_start_utc() -> datetime:
        try:
            now_et = datetime.now(ZoneInfo("America/New_York"))
            return now_et.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        except Exception:
            now_utc = datetime.now(timezone.utc)
            return now_utc.replace(hour=0, minute=0, second=0, microsecond=0)

    def _fetch_ohlc_bars(range_key_value: str) -> tuple[list[dict[str, float | int]], pd.DataFrame]:
        nonlocal historical_source, intraday_source, history_cursor_next, history_coverage, cache_meta
        frame_local = pd.DataFrame()
        if has_range_override:
            start_local = start
            end_local = end
        else:
            start_local, end_local, _window_seconds = _resolve_range_window(range_key_value)

        use_flatfiles = (
            market_source == "massive"
            and flatfiles_enabled
            and history_mode != "rest"
            and interval.unit in {"minute", "hour", "day"}
        )

        if use_flatfiles:
            flatfiles_frame, flatfiles_meta = fetch_massive_flatfile_historical_bars(
                symbol=symbol,
                start=start_local,
                end=end_local,
                interval=base_interval,
                user_id=str(request.user.id),
            )
            historical_source = str(flatfiles_meta.get("historical_source") or "massive_flatfiles")
            history_cursor_next = _coerce_number(flatfiles_meta.get("history_cursor_next"))
            meta_coverage = flatfiles_meta.get("history_coverage")
            if isinstance(meta_coverage, dict):
                history_coverage = {
                    "start_ts": _coerce_number(meta_coverage.get("start_ts")),
                    "end_ts": _coerce_number(meta_coverage.get("end_ts")),
                    "complete": bool(meta_coverage.get("complete", True)),
                }
            cache_meta = {
                "hit": int(_coerce_number(flatfiles_meta.get("cache_hit")) or 0),
                "miss": int(_coerce_number(flatfiles_meta.get("cache_miss")) or 0),
                "warmed_ranges": list(flatfiles_meta.get("warmed_days") or []),
            }

            intraday_frame = pd.DataFrame()
            today_start_utc = _today_market_start_utc()
            if end_local > today_start_utc:
                intraday_start = max(start_local, today_start_utc)
                intraday_frame = market_data.fetch(
                    [symbol],
                    start=intraday_start,
                    end=end_local,
                    interval=base_interval,
                    cache=True,
                    timeout=MARKET_REQUEST_TIMEOUT,
                    ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
                    cache_alias=cache_alias,
                    user_id=str(request.user.id),
                )
                if isinstance(intraday_frame, pd.DataFrame) and not intraday_frame.empty:
                    intraday_source = "massive_rest"

            frame_local = _merge_ohlc_frames(flatfiles_frame, intraday_frame)
            if frame_local.empty and history_mode in {"auto", "rest"}:
                if has_range_override:
                    frame_local = market_data.fetch(
                        [symbol],
                        start=start_local,
                        end=end_local,
                        interval=base_interval,
                        cache=True,
                        timeout=MARKET_REQUEST_TIMEOUT,
                        ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
                        cache_alias=cache_alias,
                        user_id=str(request.user.id),
                    )
                else:
                    frame_local = market_data.fetch(
                        [symbol],
                        period=range_key_value,
                        interval=base_interval,
                        cache=True,
                        timeout=MARKET_REQUEST_TIMEOUT,
                        ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
                        cache_alias=cache_alias,
                        user_id=str(request.user.id),
                    )
                if isinstance(frame_local, pd.DataFrame) and not frame_local.empty:
                    if historical_source == "none":
                        historical_source = "massive_rest"
                    if intraday_source == "none":
                        intraday_source = "massive_rest"
                    if history_coverage.get("start_ts") is None:
                        history_coverage["complete"] = False
        else:
            if has_range_override:
                frame_local = market_data.fetch(
                    [symbol],
                    start=start_local,
                    end=end_local,
                    interval=base_interval,
                    cache=True,
                    timeout=MARKET_REQUEST_TIMEOUT,
                    ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
                    cache_alias=cache_alias,
                    user_id=str(request.user.id),
                )
            else:
                frame_local = market_data.fetch(
                    [symbol],
                    period=range_key_value,
                    interval=base_interval,
                    cache=True,
                    timeout=MARKET_REQUEST_TIMEOUT,
                    ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
                    cache_alias=cache_alias,
                    user_id=str(request.user.id),
                )
            if market_source == "massive":
                historical_source = "massive_rest"
                intraday_source = "massive_rest"
            else:
                historical_source = "alpaca_rest"
                intraday_source = "alpaca_rest"

        if isinstance(frame_local, pd.DataFrame):
            frame_local.attrs["market_source"] = market_source
            if historical_source != "none":
                frame_local.attrs["historical_source"] = historical_source
            if intraday_source != "none":
                frame_local.attrs["intraday_source"] = intraday_source
            frame_local.attrs["history_coverage"] = history_coverage
            frame_local.attrs["history_cursor_next"] = history_cursor_next
            frame_local.attrs["cache_meta"] = cache_meta
        bar_frame_local = frame_local
        if resample_rule:
            bar_frame_local = _resample_ohlc_frame(frame_local, symbol, resample_rule)
        bars_local = _extract_ohlc(bar_frame_local, symbol, limit=CHART_MAX_TIME_BARS)
        return bars_local, frame_local

    server_ts = float(time.time())
    if interval.unit in {"tick", "second"}:
        max_range = CHART_MAX_TICK_RANGE_SECONDS if interval.unit == "tick" else CHART_MAX_SECOND_RANGE_SECONDS
        if range_seconds > max_range:
            start = end - timedelta(seconds=max_range)
            range_seconds = max_range
            window_limited = True
        start_ts = start.timestamp() if start else None
        end_ts = end.timestamp() if end else None
        cached_trades = chart_get_trades(
            symbol,
            start=start_ts,
            end=end_ts,
            limit=CHART_TRADES_PAGE_LIMIT * CHART_TRADES_MAX_PAGES,
        )
        trades = cached_trades
        next_token = None
        data_origin = "realtime"
        cached_min = cached_trades[0].get("ts") if cached_trades else None
        cached_max = cached_trades[-1].get("ts") if cached_trades else None
        needs_fetch = not cached_trades
        if not needs_fetch and start_ts is not None and cached_min is not None:
            if cached_min > start_ts + 1e-6:
                needs_fetch = True
        if not needs_fetch and end_ts is not None and cached_max is not None:
            if cached_max < end_ts - 1e-6:
                needs_fetch = True
        if needs_fetch:
            fetched, next_token, downgrade_to, downgrade_message = fetch_stock_trades(
                symbol,
                start=start,
                end=end,
                feed=DEFAULT_FEED,
                limit=CHART_TRADES_PAGE_LIMIT,
                max_pages=CHART_TRADES_MAX_PAGES,
                sort="desc",
                user=request.user,
                timeout=MARKET_REQUEST_TIMEOUT,
            )
            if fetched:
                trades = fetched
                data_origin = market_source
                if cached_trades:
                    fetched_max = max(
                        (parse_timestamp(item.get("ts") or item.get("t") or item.get("timestamp")) or 0.0)
                        for item in fetched
                        if isinstance(item, dict)
                    )
                    newer_cached = [
                        item
                        for item in cached_trades
                        if (parse_timestamp(item.get("ts") or item.get("t") or item.get("timestamp")) or 0.0)
                        > fetched_max + 1e-6
                    ]
                    if newer_cached:
                        seen: set[tuple[float, float, float]] = set()
                        merged: list[dict[str, Any]] = []
                        for item in trades + newer_cached:
                            if not isinstance(item, dict):
                                continue
                            ts_val = parse_timestamp(item.get("ts") or item.get("t") or item.get("timestamp"))
                            if ts_val is None:
                                continue
                            price_val = item.get("price") or item.get("p")
                            size_val = item.get("size") or item.get("s") or item.get("v") or 0
                            try:
                                price_num = float(price_val)
                            except (TypeError, ValueError):
                                continue
                            try:
                                size_num = float(size_val) if size_val is not None else 0.0
                            except (TypeError, ValueError):
                                size_num = 0.0
                            key = (float(ts_val), price_num, size_num)
                            if key in seen:
                                continue
                            seen.add(key)
                            merged.append(item)
                        merged.sort(key=lambda row: parse_timestamp(row.get("ts") or row.get("t") or row.get("timestamp")) or 0.0)
                        trades = merged
                        data_origin = "mixed"
                if next_token:
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
        if not bars and not has_range_override:
            latest_snapshot = market_data.fetch_latest_quote(symbol, interval="1m", user_id=str(request.user.id))
            latest_as_of = latest_snapshot.get("as_of") if isinstance(latest_snapshot, dict) else None
            latest_ts = parse_timestamp(latest_as_of)
            if latest_ts:
                anchored_end = datetime.fromtimestamp(latest_ts, tz=timezone.utc)
                anchored_start = anchored_end - timedelta(seconds=max_range)
                anchored_trades, next_token, anchored_feed, anchored_message = fetch_stock_trades(
                    symbol,
                    start=anchored_start,
                    end=anchored_end,
                    feed=DEFAULT_FEED,
                    limit=CHART_TRADES_PAGE_LIMIT,
                    max_pages=CHART_TRADES_MAX_PAGES,
                    sort="desc",
                    user=request.user,
                    timeout=MARKET_REQUEST_TIMEOUT,
                )
                if anchored_trades:
                    trades = anchored_trades
                    data_origin = market_source
                    if anchored_feed:
                        downgrade_to = anchored_feed
                    if anchored_message:
                        downgrade_message = anchored_message
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
                    if bars:
                        anchor_message = _("当前窗口无逐笔成交，已回溯到最近成交时段。")
                        downgrade_message = (
                            f"{downgrade_message} · {anchor_message}" if downgrade_message else anchor_message
                        )
        if not bars:
            fallback_interval = _resolve_chart_interval("1m")
            fallback_message = _("逐笔/秒级行情不可用，已改用分钟K线。")
            range_message = _("市场休市或数据稀疏，已展示更长周期历史。")
            if fallback_interval is not None:
                fallback_bars, fallback_frame = _fetch_ohlc_bars(range_key)
                if not fallback_bars and not has_range_override:
                    for candidate_range in range_fallbacks.get(range_key, ()):
                        candidate_bars, candidate_frame = _fetch_ohlc_bars(candidate_range)
                        if candidate_bars:
                            fallback_bars = candidate_bars
                            fallback_frame = candidate_frame
                            effective_range = candidate_range
                            downgrade_message = (
                                f"{downgrade_message} · {range_message}" if downgrade_message else range_message
                            )
                            break
                if fallback_bars:
                    messages = []
                    if downgrade_message:
                        messages.append(downgrade_message)
                    messages.append(fallback_message)
                    response = {
                        "symbol": symbol,
                        "range": range_key,
                        "range_fallback": effective_range if effective_range != range_key else None,
                        "interval": {
                            "key": fallback_interval.key,
                            "unit": fallback_interval.unit,
                            "value": fallback_interval.value,
                            "label": fallback_interval.label,
                        },
                        "generated_at": server_ts,
                        "data_source": _infer_market_source(fallback_frame),
                        "data_origin": "fallback",
                        "historical_source": str(
                            getattr(fallback_frame, "attrs", {}).get("historical_source")
                            or ("massive_rest" if market_source == "massive" else "alpaca_rest")
                        ),
                        "intraday_source": str(
                            getattr(fallback_frame, "attrs", {}).get("intraday_source")
                            or ("massive_rest" if market_source == "massive" else "alpaca_rest")
                        ),
                        "bars": fallback_bars,
                        "window_limited": False,
                        "latest_trade_ts": parse_timestamp(fallback_bars[-1]["time"] if fallback_bars else None),
                        "server_ts": server_ts,
                        "history_coverage": getattr(fallback_frame, "attrs", {}).get("history_coverage")
                        or {"start_ts": None, "end_ts": None, "complete": True},
                        "history_cursor_next": getattr(fallback_frame, "attrs", {}).get("history_cursor_next"),
                        "cache_meta": getattr(fallback_frame, "attrs", {}).get("cache_meta")
                        or {"hit": 0, "miss": 0, "warmed_ranges": []},
                        "downgrade_to": fallback_interval.key,
                        "downgrade_message": " · ".join(messages),
                        "request_id": request_id,
                    }
                    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})
            return JsonResponse(
                {"error": _("未能获取 %(symbol)s 的逐笔成交。") % {"symbol": symbol}, "request_id": request_id},
                status=404,
                json_dumps_params={"ensure_ascii": False},
            )
        latest_trade_ts: float | None = None
        if trades:
            last_trade = trades[-1]
            if isinstance(last_trade, dict):
                latest_trade_ts = parse_timestamp(
                    last_trade.get("ts") or last_trade.get("t") or last_trade.get("timestamp")
                )
        if latest_trade_ts is None:
            latest_trade = chart_get_latest_trade(symbol)
            latest_trade_ts = parse_timestamp(latest_trade.get("ts") if latest_trade else None)
        response = {
            "symbol": symbol,
            "range": range_key,
            "interval": {
                "key": interval.key,
                "unit": interval.unit,
                "value": interval.value,
                "label": interval.label,
            },
            "generated_at": server_ts,
            "data_source": market_source,
            "data_origin": data_origin,
            "historical_source": "none",
            "intraday_source": "massive_ws" if market_source == "massive" else "alpaca_ws",
            "bars": bars,
            "window_limited": window_limited,
            "next_page_token": next_token,
            "latest_trade_ts": latest_trade_ts,
            "server_ts": server_ts,
            "history_coverage": {"start_ts": None, "end_ts": None, "complete": True},
            "history_cursor_next": None,
            "cache_meta": {"hit": 0, "miss": 0, "warmed_ranges": []},
        }
        if downgrade_to:
            response["downgrade_to"] = downgrade_to
        if downgrade_message:
            response["downgrade_message"] = downgrade_message
        if cache_enabled:
            cache_set_object(cache_key, response, chart_cache_ttl, cache_alias=cache_alias)
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

    bars, frame = _fetch_ohlc_bars(range_key)
    if not bars and has_range_override and start_override is None and end_override is not None:
        range_message = _("市场休市或数据稀疏，已展示更长周期历史。")
        for candidate_range in range_fallbacks.get(range_key, ()):
            _candidate_start, _candidate_end, candidate_seconds = _resolve_range_window(candidate_range)
            candidate_start = end - timedelta(seconds=candidate_seconds)
            frame_candidate = market_data.fetch(
                [symbol],
                start=candidate_start,
                end=end,
                interval=base_interval,
                cache=True,
                timeout=MARKET_REQUEST_TIMEOUT,
                ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
                cache_alias=cache_alias,
                user_id=str(request.user.id),
            )
            bar_frame_candidate = frame_candidate
            if resample_rule:
                bar_frame_candidate = _resample_ohlc_frame(frame_candidate, symbol, resample_rule)
            candidate_bars = _extract_ohlc(bar_frame_candidate, symbol, limit=CHART_MAX_TIME_BARS)
            if candidate_bars:
                bars = candidate_bars
                frame = frame_candidate
                effective_range = candidate_range
                downgrade_message = (
                    f"{downgrade_message} · {range_message}" if downgrade_message else range_message
                )
                break
    if not bars and not has_range_override:
        range_message = _("市场休市或数据稀疏，已展示更长周期历史。")
        for candidate_range in range_fallbacks.get(range_key, ()):
            candidate_bars, candidate_frame = _fetch_ohlc_bars(candidate_range)
            if candidate_bars:
                bars = candidate_bars
                frame = candidate_frame
                effective_range = candidate_range
                downgrade_message = (
                    f"{downgrade_message} · {range_message}" if downgrade_message else range_message
                )
                break
    if not bars:
        return JsonResponse(
            {"error": _("未能获取 %(symbol)s 的行情数据。") % {"symbol": symbol}, "request_id": request_id},
            status=404,
            json_dumps_params={"ensure_ascii": False},
        )

    response = {
        "symbol": symbol,
        "range": range_key,
        "range_fallback": effective_range if effective_range != range_key else None,
        "interval": {
            "key": interval.key,
            "unit": interval.unit,
            "value": interval.value,
            "label": interval.label,
        },
        "generated_at": server_ts,
        "data_source": _infer_market_source(frame),
        "historical_source": str(getattr(frame, "attrs", {}).get("historical_source") or historical_source or "none"),
        "intraday_source": str(getattr(frame, "attrs", {}).get("intraday_source") or intraday_source or "none"),
        "bars": bars,
        "window_limited": window_limited,
        "latest_trade_ts": parse_timestamp(bars[-1]["time"] if bars else None),
        "server_ts": server_ts,
        "history_coverage": getattr(frame, "attrs", {}).get("history_coverage") or history_coverage,
        "history_cursor_next": getattr(frame, "attrs", {}).get("history_cursor_next") or history_cursor_next,
        "cache_meta": getattr(frame, "attrs", {}).get("cache_meta") or cache_meta,
    }
    if downgrade_to:
        response["downgrade_to"] = downgrade_to
    if downgrade_message:
        response["downgrade_message"] = downgrade_message
    if cache_enabled:
        cache_set_object(cache_key, response, chart_cache_ttl, cache_alias=cache_alias)
    response["request_id"] = request_id
    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})


@login_required
@require_http_methods(["POST"])
def market_chart_analyze(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return json_error(
            error_code="series_invalid",
            message=_("请求体解析失败。"),
            status_code=400,
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze",
        )
    if not isinstance(payload, dict):
        payload = {}

    series_payload = _extract_chart_analyzer_series(payload, max_points=MARKET_CHART_ANALYZER_MAX_POINTS)
    if not series_payload:
        return json_error(
            error_code="series_invalid",
            message=_("图表序列无效。"),
            status_code=400,
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze",
        )
    series_values, sample_index_map, original_length, series_meta = series_payload
    if len(series_values) < MARKET_CHART_ANALYZER_MIN_POINTS:
        return json_error(
            error_code="series_insufficient",
            message=_("可分析的数据点不足。"),
            status_code=400,
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze",
            extra={
                "required_points": MARKET_CHART_ANALYZER_MIN_POINTS,
                "points": len(series_values),
            },
        )

    symbol = _normalize_query(payload.get("symbol")) if isinstance(payload.get("symbol"), str) else ""
    range_key = str(payload.get("range") or payload.get("timeframe") or "").strip().lower()
    interval_key = str(payload.get("interval") or "").strip().lower()
    session_id = str(payload.get("session_id") or "").strip() or None
    include_fusion = _parse_bool(payload.get("include_fusion"), default=True)

    try:
        result = analyze_price_series(
            series_values,
            symbol=symbol or None,
            timeframe=range_key or None,
            analysis_mode=interval_key or None,
            include_features=False,
            include_waves=True,
            include_fusion=include_fusion,
            include_timings=False,
            session_id=session_id,
            min_points=MARKET_CHART_ANALYZER_MIN_POINTS,
            max_points=MARKET_CHART_ANALYZER_MAX_POINTS,
            training_namespace=MARKET_CHART_ANALYZER_NAMESPACE,
        )
    except Exception as exc:
        log_sanitized_exception(
            context="Market chart analyze failed",
            exc=exc,
            error_code="chart_analysis_failed",
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze",
            status_code=500,
        )
        return json_error(
            error_code="chart_analysis_failed",
            message=_("图表分析失败，请稍后重试。"),
            status_code=500,
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze",
        )

    if result.get("error"):
        error_code = str(result.get("error") or "series_invalid")
        status_code = 400
        if error_code not in {"series_invalid", "series_insufficient"}:
            error_code = "series_invalid"
        return json_error(
            error_code=error_code,
            message=_("图表序列暂不可分析。") if error_code == "series_invalid" else _("可分析的数据点不足。"),
            status_code=status_code,
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze",
            extra={
                "required_points": MARKET_CHART_ANALYZER_MIN_POINTS,
                "points": len(series_values),
            }
            if error_code == "series_insufficient"
            else None,
        )

    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, dict):
        diagnostics["sample_index_map"] = sample_index_map
        diagnostics["sample_points"] = len(series_values)
        diagnostics["series_original_length"] = original_length
        diagnostics["series_mode"] = series_meta.get("series_mode")
        diagnostics["smoothing_window"] = series_meta.get("smoothing_window")
        diagnostics["series_source"] = series_meta.get("series_source")

    response = {
        **result,
        "series_mode": series_meta.get("series_mode"),
        "smoothing_window": series_meta.get("smoothing_window"),
        "request_id": request_id,
    }
    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})


@login_required
@require_http_methods(["POST"])
def market_chart_analyze_sample(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return json_error(
            error_code="series_invalid",
            message=_("请求体解析失败。"),
            status_code=400,
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze.sample",
        )
    if not isinstance(payload, dict):
        payload = {}

    label = str(payload.get("label") or "").strip()
    if label not in PATTERN_KEYS:
        return json_error(
            error_code="invalid_label",
            message=_("标签无效。"),
            status_code=400,
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze.sample",
        )

    series_payload = _extract_chart_analyzer_series(payload, max_points=MARKET_CHART_ANALYZER_MAX_POINTS)
    if not series_payload:
        return json_error(
            error_code="series_invalid",
            message=_("图表序列无效。"),
            status_code=400,
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze.sample",
        )
    series_values, _sample_index_map, _original_length, series_meta = series_payload
    if len(series_values) < MARKET_CHART_ANALYZER_MIN_POINTS:
        return json_error(
            error_code="series_insufficient",
            message=_("可分析的数据点不足。"),
            status_code=400,
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze.sample",
            extra={
                "required_points": MARKET_CHART_ANALYZER_MIN_POINTS,
                "points": len(series_values),
            },
        )

    symbol = _normalize_query(payload.get("symbol")) if isinstance(payload.get("symbol"), str) else ""
    range_key = str(payload.get("range") or payload.get("timeframe") or "").strip().lower()
    interval_key = str(payload.get("interval") or "").strip().lower()

    try:
        analysis = analyze_price_series(
            series_values,
            symbol=symbol or None,
            timeframe=range_key or None,
            analysis_mode=interval_key or None,
            include_features=True,
            include_waves=False,
            include_fusion=False,
            include_timings=False,
            min_points=MARKET_CHART_ANALYZER_MIN_POINTS,
            max_points=MARKET_CHART_ANALYZER_MAX_POINTS,
            training_namespace=MARKET_CHART_ANALYZER_NAMESPACE,
        )
        features = analysis.get("features")
        if not isinstance(features, list) or not features:
            return json_error(
                error_code="series_invalid",
                message=_("特征提取失败。"),
                status_code=400,
                request_id=request_id,
                user_id=request.user.id,
                endpoint="api.market.chart.analyze.sample",
            )
        save_screen_sample(
            features,
            label=label,
            meta={
                "symbol": symbol,
                "range": range_key,
                "interval": interval_key,
                "series_mode": series_meta.get("series_mode"),
                "smoothing_window": series_meta.get("smoothing_window"),
                "lang": (getattr(request, "LANGUAGE_CODE", "") or "").lower()[:8],
                "user_id": request.user.id,
            },
            namespace=MARKET_CHART_ANALYZER_NAMESPACE,
        )
        total_samples = len(load_screen_samples(namespace=MARKET_CHART_ANALYZER_NAMESPACE))
    except Exception as exc:
        log_sanitized_exception(
            context="Market chart sample save failed",
            exc=exc,
            error_code="chart_sample_save_failed",
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze.sample",
            status_code=500,
        )
        return json_error(
            error_code="chart_sample_save_failed",
            message=_("样本保存失败。"),
            status_code=500,
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze.sample",
        )

    return JsonResponse(
        {
            "status": "saved",
            "label": label,
            "total_samples": total_samples,
            "request_id": request_id,
        },
        json_dumps_params={"ensure_ascii": False},
    )


@login_required
@require_http_methods(["POST"])
def market_chart_analyze_train(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    try:
        metrics = train_screen_model(
            min_samples=MARKET_CHART_ANALYZER_TRAIN_MIN_SAMPLES,
            namespace=MARKET_CHART_ANALYZER_NAMESPACE,
        )
    except RuntimeError:
        return json_error(
            error_code="training_state_invalid",
            message=_("训练状态无效，请补充样本后再试。"),
            status_code=400,
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze.train",
        )
    except Exception as exc:
        log_sanitized_exception(
            context="Market chart model train failed",
            exc=exc,
            error_code="chart_train_failed",
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze.train",
            status_code=500,
        )
        return json_error(
            error_code="chart_train_failed",
            message=_("训练失败，请稍后重试。"),
            status_code=500,
            request_id=request_id,
            user_id=request.user.id,
            endpoint="api.market.chart.analyze.train",
        )
    return JsonResponse(
        {
            "status": "trained",
            "total_samples": metrics.total_samples,
            "classes": metrics.classes,
            "accuracy": metrics.accuracy,
            "test_size": metrics.test_size,
            "override_threshold": metrics.override_threshold,
            "override_accuracy": metrics.override_accuracy,
            "override_coverage": metrics.override_coverage,
            "override_samples": metrics.override_samples,
            "override_source": metrics.override_source,
            "request_id": request_id,
        },
        json_dumps_params={"ensure_ascii": False},
    )


@login_required
@require_http_methods(["GET"])
def market_chart_analyze_meta(request: HttpRequest) -> JsonResponse:
    request_id = ensure_request_id(request)
    try:
        samples = load_screen_samples(namespace=MARKET_CHART_ANALYZER_NAMESPACE)
        model_meta = load_screen_model_meta(namespace=MARKET_CHART_ANALYZER_NAMESPACE) or {}
    except Exception:
        samples = []
        model_meta = {}

    class_counts: dict[str, int] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        label = sample.get("label")
        if not isinstance(label, str) or not label:
            continue
        class_counts[label] = class_counts.get(label, 0) + 1
    if isinstance(model_meta.get("classes"), Mapping):
        class_counts = {str(key): int(value) for key, value in model_meta["classes"].items()}  # type: ignore[index]

    return JsonResponse(
        {
            "namespace": MARKET_CHART_ANALYZER_NAMESPACE,
            "total_samples": len(samples),
            "classes": class_counts,
            "override_threshold": model_meta.get("override_threshold"),
            "override_source": model_meta.get("override_source"),
            "last_trained_at": model_meta.get("trained_at"),
            "request_id": request_id,
        },
        json_dumps_params={"ensure_ascii": False},
    )


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
    except RealtimePayloadError:
        return JsonResponse(
            {
                "ok": False,
                "error_code": "invalid_realtime_config",
                "message": _("交易配置校验失败。"),
                "request_id": request_id,
            },
            status=400,
        )

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
    timeframe_key_raw = str(request.GET.get("timeframe") or "1d").strip().lower()
    timeframe = TIMEFRAMES.get(timeframe_key_raw, TIMEFRAMES["1d"])
    timeframe_clamped = timeframe_key_raw not in TIMEFRAMES

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
    snapshots = fetch_stock_snapshots(symbols, feed=DEFAULT_FEED, user_id=user_id, timeout=MARKET_REQUEST_TIMEOUT) if symbols else {}
    window_metrics: dict[str, dict[str, object]] = {}
    window_source = "unknown"
    if symbols and timeframe.key != "1d":
        window_metrics, window_source = _resolve_symbol_window_metrics(
            symbols,
            timeframe=timeframe,
            user_id=user_id,
        )

    items: list[dict[str, object]] = []
    for asset in page_assets:
        symbol = asset.get("symbol") or ""
        snapshot = snapshots.get(symbol, {}) if isinstance(snapshots, dict) else {}
        daily_bar = (
            snapshot.get("dailyBar") or snapshot.get("daily_bar")
            if isinstance(snapshot, dict)
            else None
        )
        minute_bar = (
            snapshot.get("minuteBar") or snapshot.get("minute_bar")
            if isinstance(snapshot, dict)
            else None
        )
        prev_bar = (
            snapshot.get("prevDailyBar") or snapshot.get("prev_daily_bar")
            if isinstance(snapshot, dict)
            else None
        )
        snapshot_prev_close = _coerce_positive_price((prev_bar or {}).get("c"))
        snapshot_price = _resolve_snapshot_last_price(
            snapshot if isinstance(snapshot, dict) else None,
            prev_close=snapshot_prev_close,
            allow_quote_fallback=False,
        )
        snapshot_open = _coerce_positive_price((daily_bar or {}).get("o"))
        snapshot_high = _coerce_number((daily_bar or {}).get("h"))
        snapshot_low = _coerce_number((daily_bar or {}).get("l"))
        snapshot_volume = _coerce_number((daily_bar or {}).get("v") or (minute_bar or {}).get("v"))
        snapshot_dollar_volume = snapshot_price * snapshot_volume if snapshot_price is not None and snapshot_volume is not None else None

        snapshot_change = None
        if snapshot_price is not None and snapshot_prev_close not in (None, 0):
            snapshot_change = (snapshot_price / snapshot_prev_close - 1.0) * 100.0
        snapshot_range = None
        if snapshot_prev_close not in (None, 0) and snapshot_high is not None and snapshot_low is not None:
            snapshot_range = ((snapshot_high - snapshot_low) / snapshot_prev_close) * 100.0

        metrics = window_metrics.get(symbol, {}) if timeframe.key != "1d" else {}
        missing_reasons = _normalize_missing_reason_map(metrics.get("missing_reasons"))
        timeframe_pending = False
        if timeframe.key == "1d":
            item: dict[str, object] = {
                "symbol": symbol,
                "name": asset.get("name") or "",
                "exchange": asset.get("exchange") or "",
                "last": snapshot_price,
                "price": snapshot_price,
                "change_pct_period": snapshot_change,
                "change_pct_day": snapshot_change,
                "change_pct": snapshot_change,
                "open": snapshot_open,
                "prev_close": snapshot_prev_close,
                "volume": snapshot_volume,
                "dollar_volume": snapshot_dollar_volume,
                "range_pct": snapshot_range,
            }
        else:
            period_price = _coerce_number(metrics.get("price"))
            period_change = _coerce_number(metrics.get("change_pct_period"))
            day_change = _coerce_number(metrics.get("change_pct_day"))
            if period_price is None and snapshot_price is not None:
                period_price = snapshot_price
                timeframe_pending = True
            if period_change is None and snapshot_change is not None:
                timeframe_pending = True
            item = {
                "symbol": symbol,
                "name": asset.get("name") or "",
                "exchange": asset.get("exchange") or "",
                "last": period_price,
                "price": period_price,
                "change_pct_period": period_change,
                "change_pct_day": day_change,
                "change_pct": period_change,
                "open": _coerce_number(metrics.get("open")),
                "prev_close": _coerce_number(metrics.get("prev_close")),
                "volume": _coerce_number(metrics.get("volume")),
                "dollar_volume": _coerce_number(metrics.get("dollar_volume")),
                "range_pct": _coerce_number(metrics.get("range_pct")),
            }
            if missing_reasons:
                item["missing_reasons"] = missing_reasons
        items.append(_apply_common_market_item_shape(item, timeframe_pending=timeframe_pending))

    return JsonResponse(
        {
            "items": items,
            "page": page,
            "size": size,
            "total": total,
            "total_pages": last_page,
            "letter": letter,
            "query": query,
            "timeframe": {
                "key": timeframe.key,
                "label": timeframe.label,
                "label_en": timeframe.label_en,
                "clamped": timeframe_clamped,
            },
            "data_source": resolve_market_provider(user=request.user) if timeframe.key == "1d" else (window_source or "unknown"),
            "generated_at": time.time(),
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
            "generated_at": time.time(),
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
    market_source = resolve_market_provider(user_id=user_id)
    cache_key = build_cache_key("market-history", market_source, timeframe.key, sorted(symbols))
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
    attrs = getattr(data, "attrs", {})
    if isinstance(attrs, dict):
        source = str(attrs.get("market_source") or "").strip().lower()
        if source in {"alpaca", "massive"}:
            return source
    columns = data.columns
    if isinstance(columns, pd.MultiIndex):
        level0 = set(columns.get_level_values(0))
        if level0 & fields:
            return "alpaca"
        return "unknown"
    if set(columns) & fields:
        return "alpaca"
    return "unknown"


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
        timestamps: list[float] = []
        for ts in window.index:
            try:
                stamp = pd.Timestamp(ts)
                if stamp.tzinfo is None:
                    stamp = stamp.tz_localize(timezone.utc)
                else:
                    stamp = stamp.tz_convert(timezone.utc)
                timestamps.append(float(stamp.timestamp()))
            except Exception:
                continue

        ranked.append(
            {
                "symbol": sym,
                "price": float(end_price),
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
