from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from .alpaca_data import bars_to_frame
from .api_usage import record_provider_api_call
from .network import get_requests_session, resolve_retry_config, retry_call_result
from .profile import load_api_credentials

DEFAULT_REST_URL = os.environ.get("MASSIVE_REST_URL", "https://api.polygon.io").rstrip("/")
DEFAULT_WS_URL = os.environ.get("MASSIVE_WS_URL", "wss://socket.polygon.io/stocks")
DEFAULT_PLAN = os.environ.get("MASSIVE_PLAN", "stocks_advanced")
FULL_SNAPSHOT_THRESHOLD = max(20, int(os.environ.get("MASSIVE_FULL_SNAPSHOT_THRESHOLD", "20")))
FULL_SNAPSHOT_CACHE_TTL_SECONDS = max(10, int(os.environ.get("MASSIVE_FULL_SNAPSHOT_CACHE_TTL", "30")))
FULL_SNAPSHOT_MAX_PAGES = max(1, int(os.environ.get("MASSIVE_FULL_SNAPSHOT_MAX_PAGES", "20")))
FULL_SNAPSHOT_PAGE_LIMIT = min(1000, max(100, int(os.environ.get("MASSIVE_FULL_SNAPSHOT_PAGE_LIMIT", "1000"))))

_FULL_SNAPSHOT_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def _normalize_symbols(symbols: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        if not raw:
            continue
        symbol = str(raw).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            normalized.append(symbol)
    return normalized


def _resolve_massive_api_key(*, user: Any | None = None, user_id: str | None = None) -> str | None:
    creds: dict[str, str] | None = None
    if user is not None and getattr(user, "is_authenticated", False):
        try:
            creds = load_api_credentials(str(user.id))
        except Exception:
            creds = None
    elif user_id:
        try:
            creds = load_api_credentials(str(user_id))
        except Exception:
            creds = None
    if isinstance(creds, dict):
        value = str(creds.get("massive_api_key") or "").strip()
        if value:
            return value
    env_value = str(os.environ.get("MASSIVE_API_KEY") or "").strip()
    return env_value or None


def resolve_massive_credentials(*, user: Any | None = None, user_id: str | None = None) -> tuple[str | None, str | None]:
    key = _resolve_massive_api_key(user=user, user_id=user_id)
    return key, None


def resolve_massive_s3_credentials(*, user: Any | None = None, user_id: str | None = None) -> tuple[str | None, str | None]:
    creds: dict[str, str] | None = None
    if user is not None and getattr(user, "is_authenticated", False):
        try:
            creds = load_api_credentials(str(user.id))
        except Exception:
            creds = None
    elif user_id:
        try:
            creds = load_api_credentials(str(user_id))
        except Exception:
            creds = None
    access_key_id = ""
    secret_access_key = ""
    if isinstance(creds, dict):
        access_key_id = str(creds.get("massive_s3_access_key_id") or "").strip()
        secret_access_key = str(creds.get("massive_s3_secret_access_key") or "").strip()
    if not access_key_id:
        access_key_id = str(os.environ.get("MASSIVE_S3_ACCESS_KEY_ID") or "").strip()
    if not secret_access_key:
        secret_access_key = str(os.environ.get("MASSIVE_S3_SECRET_ACCESS_KEY") or "").strip()
    return (access_key_id or None, secret_access_key or None)


def _resolve_rest_url(*, user: Any | None = None, user_id: str | None = None, base_url: str | None = None) -> str:
    if base_url:
        return str(base_url).strip().rstrip("/")
    creds: dict[str, str] | None = None
    if user is not None and getattr(user, "is_authenticated", False):
        try:
            creds = load_api_credentials(str(user.id))
        except Exception:
            creds = None
    elif user_id:
        try:
            creds = load_api_credentials(str(user_id))
        except Exception:
            creds = None
    if isinstance(creds, dict):
        configured = str(creds.get("massive_rest_url") or "").strip()
        if configured:
            return configured.rstrip("/")
    configured_env = str(os.environ.get("MASSIVE_REST_URL") or "").strip()
    if configured_env:
        return configured_env.rstrip("/")
    return DEFAULT_REST_URL


def resolve_massive_ws_url(*, user: Any | None = None, user_id: str | None = None) -> str:
    creds: dict[str, str] | None = None
    if user is not None and getattr(user, "is_authenticated", False):
        try:
            creds = load_api_credentials(str(user.id))
        except Exception:
            creds = None
    elif user_id:
        try:
            creds = load_api_credentials(str(user_id))
        except Exception:
            creds = None
    if isinstance(creds, dict):
        configured = str(creds.get("massive_ws_url") or "").strip()
        if configured:
            return configured
    return str(os.environ.get("MASSIVE_WS_URL") or DEFAULT_WS_URL)


def _to_datetime(value: date | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)


def _format_iso_ts(value: date | datetime | None) -> str | None:
    dt = _to_datetime(value)
    if dt is None:
        return None
    return dt.isoformat().replace("+00:00", "Z")


def _parse_timeframe(timeframe: str) -> tuple[int, str]:
    text = str(timeframe or "1Day").strip().lower()
    aliases = {
        "1min": (1, "minute"),
        "5min": (5, "minute"),
        "15min": (15, "minute"),
        "30min": (30, "minute"),
        "1hour": (1, "hour"),
        "1h": (1, "hour"),
        "1day": (1, "day"),
        "1d": (1, "day"),
    }
    if text in aliases:
        return aliases[text]
    digits = ""
    unit = ""
    for ch in text:
        if ch.isdigit() and not unit:
            digits += ch
        elif ch.isalpha():
            unit += ch
    try:
        multiplier = max(1, int(digits or "1"))
    except ValueError:
        multiplier = 1
    if unit.startswith("min") or unit == "m":
        timespan = "minute"
    elif unit.startswith("hour") or unit == "h":
        timespan = "hour"
    elif unit.startswith("week") or unit == "w":
        timespan = "week"
    elif unit.startswith("month") or unit in {"mo", "mon"}:
        timespan = "month"
    else:
        timespan = "day"
    return multiplier, timespan


def _default_start(end_dt: datetime, *, multiplier: int, timespan: str, limit: int | None) -> datetime:
    lookback = max(30, int(limit or 300))
    if timespan == "minute":
        delta = timedelta(minutes=multiplier * lookback)
    elif timespan == "hour":
        delta = timedelta(hours=multiplier * lookback)
    elif timespan == "week":
        delta = timedelta(weeks=multiplier * lookback)
    elif timespan == "month":
        delta = timedelta(days=30 * multiplier * lookback)
    else:
        delta = timedelta(days=multiplier * lookback)
    return end_dt - delta


def _request_json(
    path: str,
    *,
    params: dict[str, Any],
    user: Any | None,
    user_id: str | None,
    timeout: float | None,
    base_url: str | None,
) -> dict[str, Any] | None:
    api_key = _resolve_massive_api_key(user=user, user_id=user_id)
    if not api_key:
        return None
    rest_url = _resolve_rest_url(user=user, user_id=user_id, base_url=base_url)
    url = f"{rest_url.rstrip('/')}/{path.lstrip('/')}"
    request_params = dict(params)
    request_params.setdefault("apiKey", api_key)

    config = resolve_retry_config(timeout=timeout)
    session = get_requests_session(config.timeout)
    user_text = ""
    if user is not None and getattr(user, "is_authenticated", False):
        user_text = str(getattr(user, "id", "") or "")
    if not user_text:
        user_text = str(user_id or "")

    def _call():
        record_provider_api_call("massive", user_id=user_text)
        return session.get(url, params=request_params, timeout=config.timeout)

    try:
        response = retry_call_result(
            _call,
            config=config,
            exceptions=(requests.RequestException,),
            should_retry=lambda resp: resp.status_code in {408, 429} or resp.status_code >= 500,
        )
    except Exception:
        return None
    if response is None or response.status_code >= 400:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_bar_timestamp(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        unit = "ms"
        if raw > 10_000_000_000_000:
            unit = "ns"
        elif raw < 10_000_000_000:
            unit = "s"
        ts = pd.to_datetime(raw, unit=unit, utc=True, errors="coerce")
        if ts is None or pd.isna(ts):
            return None
        return ts.isoformat().replace("+00:00", "Z")
    text = str(raw).strip()
    if not text:
        return None
    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    return ts.isoformat().replace("+00:00", "Z")


def _normalize_snapshot_row(ticker: dict[str, Any]) -> dict[str, Any]:
    day = ticker.get("day") if isinstance(ticker.get("day"), dict) else {}
    prev_day = ticker.get("prevDay") if isinstance(ticker.get("prevDay"), dict) else {}
    minute = ticker.get("min") if isinstance(ticker.get("min"), dict) else {}
    last_trade = ticker.get("lastTrade") if isinstance(ticker.get("lastTrade"), dict) else {}
    last_quote = ticker.get("lastQuote") if isinstance(ticker.get("lastQuote"), dict) else {}
    return {
        "dailyBar": {
            "o": day.get("o"),
            "h": day.get("h"),
            "l": day.get("l"),
            "c": day.get("c"),
            "v": day.get("v"),
            "t": _normalize_bar_timestamp(day.get("t")),
        },
        "prevDailyBar": {
            "o": prev_day.get("o"),
            "h": prev_day.get("h"),
            "l": prev_day.get("l"),
            "c": prev_day.get("c"),
            "v": prev_day.get("v"),
            "t": _normalize_bar_timestamp(prev_day.get("t")),
        },
        "minuteBar": {
            "o": minute.get("o"),
            "h": minute.get("h"),
            "l": minute.get("l"),
            "c": minute.get("c"),
            "v": minute.get("v"),
            "t": _normalize_bar_timestamp(minute.get("t")),
        },
        "latestTrade": {
            "p": last_trade.get("p") or last_trade.get("price"),
            "s": last_trade.get("s") or last_trade.get("size"),
            "t": _normalize_bar_timestamp(last_trade.get("t") or last_trade.get("timestamp")),
        },
        "latestQuote": {
            "ap": last_quote.get("P") or last_quote.get("ask") or last_quote.get("ap"),
            "bp": last_quote.get("p") or last_quote.get("bid") or last_quote.get("bp"),
            "as": last_quote.get("S") or last_quote.get("as"),
            "bs": last_quote.get("s") or last_quote.get("bs"),
            "t": _normalize_bar_timestamp(last_quote.get("t") or last_quote.get("timestamp")),
        },
    }


def _extract_cursor_from_next_url(next_url: Any) -> str | None:
    if not next_url:
        return None
    parsed = urlparse(str(next_url))
    query = parse_qs(parsed.query)
    cursor_vals = query.get("cursor") or query.get("page_token")
    return cursor_vals[0] if cursor_vals else None


def _fetch_full_market_snapshots(
    *,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    api_key = _resolve_massive_api_key(user=user, user_id=user_id)
    if not api_key:
        return {}
    rest_url = _resolve_rest_url(user=user, user_id=user_id, base_url=base_url)
    cache_key = (api_key, rest_url)
    now = time.time()
    cached = _FULL_SNAPSHOT_CACHE.get(cache_key)
    if cached:
        cached_ts, cached_payload = cached
        if now - cached_ts <= FULL_SNAPSHOT_CACHE_TTL_SECONDS and isinstance(cached_payload, dict):
            return cached_payload

    merged: dict[str, Any] = {}
    cursor: str | None = None
    for _ in range(FULL_SNAPSHOT_MAX_PAGES):
        params: dict[str, Any] = {"limit": FULL_SNAPSHOT_PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        payload = _request_json(
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            params=params,
            user=user,
            user_id=user_id,
            timeout=timeout,
            base_url=rest_url,
        )
        if not isinstance(payload, dict):
            break
        rows = payload.get("tickers")
        if not isinstance(rows, list):
            rows = payload.get("results")
        if not isinstance(rows, list) or not rows:
            break
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("ticker") or "").strip().upper()
            if not symbol:
                continue
            merged[symbol] = _normalize_snapshot_row(item)
        next_cursor = _extract_cursor_from_next_url(payload.get("next_url"))
        if not next_cursor:
            break
        cursor = next_cursor

    if merged:
        _FULL_SNAPSHOT_CACHE[cache_key] = (now, merged)
    return merged


def fetch_stock_bars(
    symbols: Iterable[str],
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    timeframe: str = "1Day",
    feed: str | None = None,
    limit: int | None = None,
    adjustment: str | None = None,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    del feed, adjustment
    api_key = _resolve_massive_api_key(user=user, user_id=user_id)
    if not api_key:
        return {}
    normalized = _normalize_symbols(symbols)
    if not normalized:
        return {}

    multiplier, timespan = _parse_timeframe(timeframe)
    end_dt = _to_datetime(end) or datetime.now(timezone.utc)
    start_dt = _to_datetime(start) or _default_start(end_dt, multiplier=multiplier, timespan=timespan, limit=limit)
    from_date = start_dt.date().isoformat()
    to_date = end_dt.date().isoformat()

    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    capped_limit = min(50_000, max(1, int(limit or 5000)))

    for symbol in normalized:
        payload = _request_json(
            f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}",
            params={
                "adjusted": "true",
                "sort": "asc",
                "limit": capped_limit,
            },
            user=user,
            user_id=user_id,
            timeout=timeout,
            base_url=base_url,
        )
        if not isinstance(payload, dict):
            continue
        results = payload.get("results")
        if not isinstance(results, list):
            continue
        parsed_rows: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            ts = _normalize_bar_timestamp(item.get("t") or item.get("timestamp"))
            close = item.get("c")
            if ts is None or close is None:
                continue
            parsed_rows.append(
                {
                    "t": ts,
                    "o": item.get("o"),
                    "h": item.get("h"),
                    "l": item.get("l"),
                    "c": close,
                    "v": item.get("v"),
                    "n": item.get("n"),
                }
            )
        if parsed_rows:
            rows_by_symbol[symbol] = parsed_rows

    return rows_by_symbol


def fetch_stock_bars_frame(
    symbols: Iterable[str],
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    timeframe: str = "1Day",
    feed: str | None = None,
    limit: int | None = None,
    adjustment: str | None = None,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
) -> pd.DataFrame:
    bars = fetch_stock_bars(
        symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        feed=feed,
        limit=limit,
        adjustment=adjustment,
        user=user,
        user_id=user_id,
        timeout=timeout,
        base_url=base_url,
    )
    return bars_to_frame(bars)


def fetch_stock_snapshots(
    symbols: Iterable[str],
    *,
    feed: str | None = None,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    del feed
    api_key = _resolve_massive_api_key(user=user, user_id=user_id)
    if not api_key:
        return {}
    normalized = _normalize_symbols(symbols)
    if not normalized:
        return {}

    if len(normalized) >= FULL_SNAPSHOT_THRESHOLD:
        merged = _fetch_full_market_snapshots(
            user=user,
            user_id=user_id,
            timeout=timeout,
            base_url=base_url,
        )
        if isinstance(merged, dict) and merged:
            filtered = {
                symbol: snapshot
                for symbol in normalized
                if isinstance((snapshot := merged.get(symbol)), dict)
            }
            if filtered:
                return filtered

    snapshots: dict[str, Any] = {}
    for symbol in normalized:
        payload = _request_json(
            f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}",
            params={},
            user=user,
            user_id=user_id,
            timeout=timeout,
            base_url=base_url,
        )
        if not isinstance(payload, dict):
            continue
        ticker = payload.get("ticker") if isinstance(payload.get("ticker"), dict) else payload
        if not isinstance(ticker, dict):
            continue
        snapshots[symbol] = _normalize_snapshot_row(ticker)

    return snapshots


def fetch_stock_trades(
    symbol: str,
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    feed: str | None = None,
    limit: int | None = None,
    page_token: str | None = None,
    max_pages: int = 1,
    sort: str | None = None,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
) -> tuple[list[dict[str, Any]], str | None, str | None, str | None]:
    del feed
    api_key = _resolve_massive_api_key(user=user, user_id=user_id)
    if not api_key:
        return [], None, None, None
    symbol_text = str(symbol or "").strip().upper()
    if not symbol_text:
        return [], None, None, None

    params: dict[str, Any] = {
        "limit": min(50_000, max(1, int(limit or 5000))),
        "sort": "timestamp",
        "order": "desc" if str(sort or "asc").lower() == "desc" else "asc",
    }
    start_iso = _format_iso_ts(start)
    end_iso = _format_iso_ts(end)
    if start_iso:
        params["timestamp.gte"] = start_iso
    if end_iso:
        params["timestamp.lte"] = end_iso
    if page_token:
        params["cursor"] = page_token

    pages = max(1, int(max_pages))
    next_token = page_token
    all_trades: list[dict[str, Any]] = []

    for _ in range(pages):
        current = dict(params)
        if next_token:
            current["cursor"] = next_token
        payload = _request_json(
            f"/v3/trades/{symbol_text}",
            params=current,
            user=user,
            user_id=user_id,
            timeout=timeout,
            base_url=base_url,
        )
        if not isinstance(payload, dict):
            break
        results = payload.get("results")
        if not isinstance(results, list):
            break
        for item in results:
            if not isinstance(item, dict):
                continue
            all_trades.append(
                {
                    "t": _normalize_bar_timestamp(item.get("sip_timestamp") or item.get("participant_timestamp") or item.get("t")),
                    "p": item.get("price") if item.get("price") is not None else item.get("p"),
                    "s": item.get("size") if item.get("size") is not None else item.get("s"),
                    "x": item.get("exchange") if item.get("exchange") is not None else item.get("x"),
                }
            )
        next_url = payload.get("next_url")
        if not next_url:
            next_token = None
            break
        parsed = urlparse(str(next_url))
        query = parse_qs(parsed.query)
        cursor_vals = query.get("cursor") or query.get("page_token")
        next_token = cursor_vals[0] if cursor_vals else None
        if not next_token:
            break

    return all_trades, next_token, None, None


def _normalize_company_quote_type(raw_type: Any, *, company_name: str = "") -> str | None:
    text = str(raw_type or "").strip().upper()
    if text in {"ETF", "ETN", "ETP", "MUTUALFUND"}:
        return "ETF"
    if text in {"CS", "COMMONSTOCK", "STOCK"}:
        return "EQUITY"
    if text in {"ADRC", "ADR"}:
        return "ADR"
    lowered_name = company_name.lower()
    if "etf" in lowered_name or "trust" in lowered_name:
        return "ETF"
    return text or None


def fetch_company_overview(
    symbol: str,
    *,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    api_key = _resolve_massive_api_key(user=user, user_id=user_id)
    if not api_key:
        return {}
    symbol_text = str(symbol or "").strip().upper()
    if not symbol_text:
        return {}

    payload = _request_json(
        f"/v3/reference/tickers/{symbol_text}",
        params={},
        user=user,
        user_id=user_id,
        timeout=timeout,
        base_url=base_url,
    )
    if not isinstance(payload, dict):
        return {}
    result = payload.get("results")
    if not isinstance(result, dict):
        result = payload if isinstance(payload, dict) else {}
    if not isinstance(result, dict) or not result:
        return {}

    address = result.get("address") if isinstance(result.get("address"), dict) else {}
    city = address.get("city")
    state = address.get("state")
    country = address.get("country") or address.get("country_code")
    hq_parts = [part for part in (city, state, country) if part not in (None, "")]
    hq = ", ".join(str(part) for part in hq_parts) if hq_parts else None

    name = result.get("name") or result.get("ticker") or symbol_text
    sector = result.get("sector") or result.get("gics_sector")
    industry = result.get("industry") or result.get("gics_industry") or result.get("sic_description")
    quote_type = _normalize_company_quote_type(result.get("type"), company_name=str(name))

    normalized = {
        "symbol": symbol_text,
        "name": name,
        "shortName": name,
        "exchange": result.get("primary_exchange") or result.get("exchange"),
        "market_cap": result.get("market_cap"),
        "sector": sector,
        "industry": industry,
        "ceo": result.get("ceo"),
        "hq": hq,
        "city": city,
        "state": state,
        "country": country,
        "quote_type": quote_type,
        "description": result.get("description"),
        "source": "massive",
    }
    return {
        key: value
        for key, value in normalized.items()
        if value not in (None, "", [], {})
    }


def fetch_news(
    *,
    symbols: Iterable[str] | None = None,
    limit: int | None = None,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
    path: str | None = None,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    del path
    api_key = _resolve_massive_api_key(user=user, user_id=user_id)
    if not api_key:
        return []

    params: dict[str, Any] = {
        "order": "desc",
        "sort": "published_utc",
        "limit": min(1000, max(1, int(limit or 50))),
    }
    normalized = _normalize_symbols(symbols or [])
    if normalized:
        # Massive/Polygon ticker news relevance works better through ticker.any_of:
        # this includes stories where the symbol appears as a related ticker,
        # not only as a primary ticker field.
        params["ticker.any_of"] = ",".join(normalized)
    start_iso = _format_iso_ts(start)
    end_iso = _format_iso_ts(end)
    if start_iso:
        params["published_utc.gte"] = start_iso
    if end_iso:
        params["published_utc.lte"] = end_iso

    pages = max(1, int(max_pages or 5))
    aggregated: list[dict[str, Any]] = []
    cursor: str | None = None

    for _ in range(pages):
        current = dict(params)
        if cursor:
            current["cursor"] = cursor
        payload = _request_json(
            "/v2/reference/news",
            params=current,
            user=user,
            user_id=user_id,
            timeout=timeout,
            base_url=base_url,
        )
        if not isinstance(payload, dict):
            break
        items = payload.get("results")
        if not isinstance(items, list):
            break

        for item in items:
            if not isinstance(item, dict):
                continue
            publisher = item.get("publisher") if isinstance(item.get("publisher"), dict) else {}
            image_url = item.get("image_url") or ""
            aggregated.append(
                {
                    "headline": item.get("title") or item.get("headline") or "",
                    "summary": item.get("description") or item.get("summary") or "",
                    "url": item.get("article_url") or item.get("url") or "",
                    "source": publisher.get("name") or item.get("source") or "",
                    "created_at": item.get("published_utc") or item.get("created_at") or "",
                    "updated_at": item.get("updated_utc") or item.get("updated_at") or "",
                    "images": [{"url": image_url}] if image_url else [],
                }
            )

        next_url = payload.get("next_url")
        if not next_url:
            break
        parsed = urlparse(str(next_url))
        query = parse_qs(parsed.query)
        cursor_vals = query.get("cursor")
        cursor = cursor_vals[0] if cursor_vals else None
        if not cursor:
            break

        if limit and len(aggregated) >= int(limit):
            break

    if limit:
        return aggregated[: int(limit)]
    return aggregated
