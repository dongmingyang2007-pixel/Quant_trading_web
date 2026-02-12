from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any, Iterable

import pandas as pd
import requests

from .api_usage import record_provider_api_call
from .network import get_requests_session, resolve_retry_config, retry_call_result
from .profile import load_api_credentials

DEFAULT_DATA_URL = os.environ.get("ALPACA_DATA_REST_URL", "https://data.alpaca.markets").rstrip("/")
DEFAULT_FEED = os.environ.get("ALPACA_DATA_FEED", "sip")
DEFAULT_NEWS_PATH = os.environ.get("ALPACA_NEWS_PATH", "/v1beta1/news")
DEFAULT_NEWS_MAX_PAGES = max(1, int(os.environ.get("ALPACA_NEWS_MAX_PAGES", "500") or 500))
TradeFetchResult = tuple[list[dict[str, Any]], str | None, str | None, str | None]


def _resolve_trading_mode(creds: dict[str, str] | None) -> str:
    mode = ""
    if isinstance(creds, dict):
        mode = str(creds.get("alpaca_trading_mode") or "").strip().lower()
    if not mode:
        mode = str(os.environ.get("ALPACA_TRADING_MODE", "")).strip().lower()
    if mode not in {"paper", "live"}:
        mode = "paper"
    return mode


def resolve_alpaca_trading_mode(*, user: Any | None = None, user_id: str | None = None) -> str:
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
    return _resolve_trading_mode(creds)


def resolve_alpaca_credentials(
    *,
    user: Any | None = None,
    user_id: str | None = None,
    mode: str | None = None,
    strict_mode: bool = False,
) -> tuple[str | None, str | None]:
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

    resolved_mode = (mode or "").strip().lower()
    if resolved_mode not in {"paper", "live"}:
        resolved_mode = _resolve_trading_mode(creds)

    def _pick(keys: tuple[str, str]) -> tuple[str | None, str | None]:
        key_id_val, secret_val = None, None
        if isinstance(creds, dict):
            key_id_val = creds.get(keys[0])
            secret_val = creds.get(keys[1])
        if not key_id_val:
            key_id_val = os.environ.get(keys[0].upper()) or os.environ.get(keys[0])
        if not secret_val:
            secret_val = os.environ.get(keys[1].upper()) or os.environ.get(keys[1])
        return key_id_val, secret_val

    live_pair = ("alpaca_live_api_key_id", "alpaca_live_api_secret_key")
    paper_pair = ("alpaca_paper_api_key_id", "alpaca_paper_api_secret_key")
    legacy_pair = ("alpaca_api_key_id", "alpaca_api_secret_key")

    key_id = None
    secret = None
    if resolved_mode == "live":
        key_id, secret = _pick(live_pair)
        if (not key_id or not secret) and not strict_mode:
            key_id, secret = _pick(legacy_pair)
        if (not key_id or not secret) and not strict_mode:
            key_id, secret = _pick(paper_pair)
    else:
        key_id, secret = _pick(paper_pair)
        if (not key_id or not secret) and not strict_mode:
            key_id, secret = _pick(legacy_pair)
        if (not key_id or not secret) and not strict_mode:
            key_id, secret = _pick(live_pair)

    if not key_id or not secret:
        return None, None
    return key_id, secret


def resolve_alpaca_data_credentials(
    *,
    user: Any | None = None,
    user_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Prefer live keys for market data (SIP), then fall back to legacy/paper."""
    return resolve_alpaca_credentials(user=user, user_id=user_id, mode="live")


def _alpaca_headers(key_id: str, secret: str) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }


def _alpaca_get(
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str],
    timeout: float | None,
    user_id: str | None = None,
) -> requests.Response | None:
    config = resolve_retry_config(timeout=timeout)
    session = get_requests_session(config.timeout)

    def _call():
        record_provider_api_call("alpaca", user_id=user_id)
        return session.get(url, params=params, headers=headers, timeout=config.timeout)

    def _should_retry(response: requests.Response) -> bool:
        return response.status_code in {408, 429} or response.status_code >= 500

    try:
        response = retry_call_result(
            _call,
            config=config,
            exceptions=(requests.RequestException,),
            should_retry=_should_retry,
        )
    except Exception:
        return None
    if response is None or response.status_code >= 400:
        return None
    return response


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


def _to_datetime(value: date | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)


def _format_ts(value: date | datetime | None) -> str | None:
    dt = _to_datetime(value)
    if dt is None:
        return None
    return dt.isoformat().replace("+00:00", "Z")


def _resolve_usage_user_id(*, user: Any | None, user_id: str | None) -> str:
    if user is not None and getattr(user, "is_authenticated", False):
        resolved = str(getattr(user, "id", "") or "")
        if resolved:
            return resolved
    return str(user_id or "")


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
    usage_user_id = _resolve_usage_user_id(user=user, user_id=user_id)
    key_id, secret = resolve_alpaca_data_credentials(user=user, user_id=user_id)
    if not key_id or not secret:
        return {}
    normalized = _normalize_symbols(symbols)
    if not normalized:
        return {}
    params: dict[str, Any] = {
        "symbols": ",".join(normalized),
        "timeframe": timeframe,
    }
    if feed or DEFAULT_FEED:
        params["feed"] = feed or DEFAULT_FEED
    if adjustment:
        params["adjustment"] = adjustment
    if limit:
        params["limit"] = limit
    start_ts = _format_ts(start)
    end_ts = _format_ts(end)
    if start_ts:
        params["start"] = start_ts
    if end_ts:
        params["end"] = end_ts
    url = f"{(base_url or DEFAULT_DATA_URL).rstrip('/')}/v2/stocks/bars"
    headers = _alpaca_headers(key_id, secret)

    aggregated: dict[str, list[dict[str, Any]]] = {}
    page_token: str | None = None
    for _ in range(60):
        if page_token:
            params["page_token"] = page_token
        response = _alpaca_get(url, params=params, headers=headers, timeout=timeout, user_id=usage_user_id)
        if response is None:
            break
        try:
            payload = response.json()
        except ValueError:
            break
        data = payload.get("bars") or payload.get("data") or {}
        if isinstance(data, list):
            symbol = normalized[0]
            aggregated.setdefault(symbol, []).extend(data)
        elif isinstance(data, dict):
            for symbol, items in data.items():
                if isinstance(items, list):
                    aggregated.setdefault(symbol, []).extend(items)
        page_token = payload.get("next_page_token") or payload.get("next_page") or None
        if not page_token:
            break
    return aggregated


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
) -> TradeFetchResult:
    usage_user_id = _resolve_usage_user_id(user=user, user_id=user_id)
    # Contract: always return a 4-tuple so callers can safely unpack.
    key_id, secret = resolve_alpaca_data_credentials(user=user, user_id=user_id)
    if not key_id or not secret:
        return [], None, None, None
    symbol_text = str(symbol or "").strip().upper()
    if not symbol_text:
        return [], None, None, None

    params: dict[str, Any] = {}
    resolved_feed = feed or DEFAULT_FEED
    if resolved_feed:
        params["feed"] = resolved_feed
    if limit:
        params["limit"] = int(limit)
    start_ts = _format_ts(start)
    end_ts = _format_ts(end)
    if start_ts:
        params["start"] = start_ts
    if end_ts:
        params["end"] = end_ts
    sort_mode = (sort or "asc").strip().lower()
    if sort_mode not in {"asc", "desc"}:
        sort_mode = "asc"
    params["sort"] = sort_mode

    url = f"{(base_url or DEFAULT_DATA_URL).rstrip('/')}/v2/stocks/{symbol_text}/trades"
    headers = _alpaca_headers(key_id, secret)

    def _fetch_with_params(current_params: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
        aggregated: list[dict[str, Any]] = []
        next_token = page_token
        pages = max(1, int(max_pages))
        for _ in range(pages):
            if next_token:
                current_params["page_token"] = next_token
            response = _alpaca_get(
                url,
                params=current_params,
                headers=headers,
                timeout=timeout,
                user_id=usage_user_id,
            )
            if response is None:
                break
            try:
                payload = response.json()
            except ValueError:
                break
            trades = payload.get("trades") or payload.get("data") or []
            if isinstance(trades, list):
                aggregated.extend([t for t in trades if isinstance(t, dict)])
            next_token = payload.get("next_page_token") or payload.get("next_page") or None
            if not next_token:
                break
        return aggregated, next_token

    aggregated, next_token = _fetch_with_params(params.copy())
    downgrade_to = None
    downgrade_message = None
    if not aggregated and resolved_feed and resolved_feed.lower() != "iex":
        params_iex = params.copy()
        params_iex["feed"] = "iex"
        aggregated, next_token = _fetch_with_params(params_iex)
        if aggregated:
            downgrade_to = "iex"
            downgrade_message = "SIP feed unavailable. Falling back to IEX."
    return aggregated, next_token, downgrade_to, downgrade_message


def bars_to_frame(bars_by_symbol: dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for symbol, bars in (bars_by_symbol or {}).items():
        if not bars:
            continue
        for bar in bars:
            ts = bar.get("t") or bar.get("timestamp") or bar.get("time") or bar.get("start")
            if not ts:
                continue
            open_val = bar.get("o") or bar.get("open")
            high_val = bar.get("h") or bar.get("high")
            low_val = bar.get("l") or bar.get("low")
            close_val = bar.get("c") or bar.get("close")
            volume_val = bar.get("v") or bar.get("volume")
            if close_val is None:
                continue
            records.append(
                {
                    "timestamp": ts,
                    "symbol": str(symbol).upper(),
                    "Open": float(open_val) if open_val is not None else None,
                    "High": float(high_val) if high_val is not None else None,
                    "Low": float(low_val) if low_val is not None else None,
                    "Close": float(close_val),
                    "Adj Close": float(close_val),
                    "Volume": float(volume_val) if volume_val is not None else None,
                }
            )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return pd.DataFrame()
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for field in fields:
        if field not in df.columns:
            df[field] = None
    df = df[fields + ["symbol"]]
    pivoted = df.pivot_table(index="timestamp", columns="symbol", values=fields, aggfunc="last")
    if isinstance(pivoted.columns, pd.MultiIndex):
        pivoted = pivoted.reindex(fields, level=0, axis=1)
    return pivoted.sort_index()


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
    usage_user_id = _resolve_usage_user_id(user=user, user_id=user_id)
    key_id, secret = resolve_alpaca_data_credentials(user=user, user_id=user_id)
    if not key_id or not secret:
        return {}
    normalized = _normalize_symbols(symbols)
    if not normalized:
        return {}
    headers = _alpaca_headers(key_id, secret)
    url = f"{(base_url or DEFAULT_DATA_URL).rstrip('/')}/v2/stocks/snapshots"
    snapshots: dict[str, Any] = {}
    chunk_size = 200
    for idx in range(0, len(normalized), chunk_size):
        chunk = normalized[idx : idx + chunk_size]
        params: dict[str, Any] = {"symbols": ",".join(chunk)}
        if feed or DEFAULT_FEED:
            params["feed"] = feed or DEFAULT_FEED
        response = _alpaca_get(url, params=params, headers=headers, timeout=timeout, user_id=usage_user_id)
        if response is None:
            continue
        try:
            payload = response.json()
        except ValueError:
            continue
        data = payload.get("snapshots") if isinstance(payload, dict) else None
        if isinstance(data, dict):
            snapshots.update(data)
        elif isinstance(payload, dict):
            for symbol, value in payload.items():
                if isinstance(value, dict):
                    snapshots[str(symbol).upper()] = value
    return snapshots


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
    usage_user_id = _resolve_usage_user_id(user=user, user_id=user_id)
    key_id, secret = resolve_alpaca_data_credentials(user=user, user_id=user_id)
    if not key_id or not secret:
        return []
    params: dict[str, Any] = {}
    max_items: int | None = None
    if symbols:
        normalized = _normalize_symbols(symbols)
        if normalized:
            params["symbols"] = ",".join(normalized)
    if limit:
        max_items = max(1, int(limit))
    start_ts = _format_ts(start)
    end_ts = _format_ts(end)
    if start_ts:
        params["start"] = start_ts
    if end_ts:
        params["end"] = end_ts
    url = f"{(base_url or DEFAULT_DATA_URL).rstrip('/')}{path or DEFAULT_NEWS_PATH}"
    headers = _alpaca_headers(key_id, secret)
    aggregated: list[dict[str, Any]] = []
    page_token: str | None = None
    safe_max_pages = max(1, int(max_pages)) if max_pages is not None else DEFAULT_NEWS_MAX_PAGES
    seen_tokens: set[str] = set()
    for _ in range(safe_max_pages):
        request_params = dict(params)
        if max_items is not None:
            remaining = max_items - len(aggregated)
            if remaining <= 0:
                break
            request_params["limit"] = min(50, max(1, remaining))
        elif "limit" not in request_params:
            request_params["limit"] = 50
        if page_token:
            if page_token in seen_tokens:
                break
            seen_tokens.add(page_token)
            request_params["page_token"] = page_token
        response = _alpaca_get(
            url,
            params=request_params,
            headers=headers,
            timeout=timeout,
            user_id=usage_user_id,
        )
        if response is None:
            break
        try:
            payload = response.json()
        except ValueError:
            break
        page_items: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            for key in ("news", "items", "data"):
                items = payload.get(key)
                if isinstance(items, list):
                    page_items = [item for item in items if isinstance(item, dict)]
                    break
            page_token = payload.get("next_page_token") or payload.get("next_page") or None
        elif isinstance(payload, list):
            page_items = [item for item in payload if isinstance(item, dict)]
            page_token = None
        if page_items:
            aggregated.extend(page_items)
        if not page_token:
            break
        if not page_items:
            break
    if max_items is not None:
        return aggregated[:max_items]
    return aggregated
