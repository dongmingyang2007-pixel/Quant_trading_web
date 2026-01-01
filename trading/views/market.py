from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
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


TIMEFRAMES: dict[str, Timeframe] = {
    "1d": Timeframe("1d", _lazy("近1日"), "1D", "5d", "15m"),
    "5d": Timeframe("5d", _lazy("近5日"), "5D", "10d", "60m"),
    "1mo": Timeframe("1mo", _lazy("近1月"), "1M", "2mo", "1d"),
    "6mo": Timeframe("6mo", _lazy("近6月"), "6M", "1y", "1d"),
}

DEFAULT_TIMEFRAME = TIMEFRAMES["1mo"]
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
MARKET_MAX_WORKERS = max(1, getattr(settings, "MARKET_DATA_MAX_WORKERS", 4))
MARKET_RATE_WINDOW = max(10, getattr(settings, "MARKET_DATA_RATE_WINDOW_SECONDS", 90))
MARKET_RATE_MAX_CALLS = max(1, getattr(settings, "MARKET_DATA_RATE_MAX_CALLS", 45))
MARKET_RATE_CACHE_ALIAS = getattr(settings, "MARKET_DATA_RATE_CACHE_ALIAS", "default")
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

    requested_limit_raw = params.get("limit", "8") or 8
    try:
        requested_limit = int(requested_limit_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        requested_limit = 8
    limit = max(5, min(15, requested_limit))
    limit_clamped = limit != requested_limit

    return MarketQueryParams(
        timeframe=timeframe,
        timeframe_clamped=timeframe_clamped,
        query=query,
        limit=limit,
        limit_clamped=limit_clamped,
    )


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
    if yf is None:
        record_metric(
            "market.insights.error",
            request_id=request_id,
            user_id=request.user.id,
            error="missing_yfinance",
        )
        return JsonResponse(
            {"error": _("当前环境未安装 yfinance，无法加载市场数据。"), "request_id": request_id},
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

    symbols: list[str]
    restrict_to_query = False
    if resolved.query:
        symbols = [resolved.query]
        restrict_to_query = True
    else:
        symbols = TOP_SYMBOLS

    future = None
    try:
        with track_latency(
            "market.insights.fetch",
            user_id=request.user.id,
            request_id=request_id,
            timeframe=resolved.timeframe.key,
            restrict=restrict_to_query,
        ):
            future = _MARKET_EXECUTOR.submit(_fetch_history, symbols, resolved.timeframe)
            series_map = future.result(timeout=MARKET_REQUEST_TIMEOUT)
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
        payload = [
            entry for entry in ranked if entry["symbol"].upper() == resolved.query.upper()
        ]
        if not payload:
            return JsonResponse(
                {"error": _("未找到 %(symbol)s 的有效行情数据。") % {"symbol": resolved.query}, "request_id": request_id},
                status=404,
                json_dumps_params={"ensure_ascii": False},
            )
        gainers: list[dict[str, object]] = payload if payload[0]["change_pct_period"] >= 0 else []
        losers: list[dict[str, object]] = payload if payload and payload[0]["change_pct_period"] < 0 else []
    else:
        gainers = [entry for entry in ranked if entry["change_pct_period"] >= 0][: resolved.limit]
        losers = [entry for entry in ranked if entry["change_pct_period"] < 0][: resolved.limit]

    suggestions = _build_suggestions(resolved.query)

    response = {
        "timeframe": {
            "key": resolved.timeframe.key,
            "label": resolved.timeframe.label,
            "label_en": resolved.timeframe.label_en,
            "clamped": resolved.timeframe_clamped,
        },
        # Use timezone.utc for Python 3.13 compatibility (datetime.UTC removed)
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "query": resolved.query,
        "gainers": gainers,
        "losers": losers,
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
        gainers=len(gainers),
        losers=len(losers),
    )
    return JsonResponse(response, json_dumps_params={"ensure_ascii": False})


def _download_history(symbols: Iterable[str], timeframe: Timeframe) -> dict[str, pd.Series]:
    unique = [sym for sym in dict.fromkeys(symbols) if sym]
    if not unique:
        return {}

    history: dict[str, pd.Series] = {}
    data = market_data.fetch(
        unique,
        period=timeframe.period,
        interval=timeframe.interval,
        cache=True,
        timeout=MARKET_REQUEST_TIMEOUT,
        ttl=getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
        cache_alias=getattr(settings, "MARKET_HISTORY_CACHE_ALIAS", None),
    )

    try:
        if isinstance(data.columns, pd.MultiIndex):
            for sym in unique:
                try:
                    close = data[sym]["Close"].dropna()
                    if not close.empty:
                        history[sym] = close
                except Exception:
                    continue
        else:
            close = data.get("Close") if isinstance(data, pd.DataFrame) else None
            if isinstance(close, pd.Series):
                close = close.dropna()
                if not close.empty:
                    history[unique[0]] = close
    except Exception:
        pass
    return history


def _fetch_history(symbols: Iterable[str], timeframe: Timeframe) -> dict[str, pd.Series]:
    cache_key = build_cache_key("market-history", timeframe.key, sorted(symbols))
    return cache_memoize(
        cache_key,
        lambda: _download_history(symbols, timeframe),
        getattr(settings, "MARKET_HISTORY_CACHE_TTL", 300),
    ) or {}


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
