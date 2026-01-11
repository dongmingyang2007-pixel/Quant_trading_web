from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any, Iterable

import pandas as pd
import requests

from .network import get_requests_session, resolve_retry_config, retry_call_result
from .profile import load_api_credentials, resolve_api_credential

DEFAULT_DATA_URL = os.environ.get("ALPACA_DATA_REST_URL", "https://data.alpaca.markets").rstrip("/")
DEFAULT_FEED = os.environ.get("ALPACA_DATA_FEED", "sip")
DEFAULT_NEWS_PATH = os.environ.get("ALPACA_NEWS_PATH", "/v1beta1/news")


def resolve_alpaca_credentials(
    *,
    user: Any | None = None,
    user_id: str | None = None,
) -> tuple[str | None, str | None]:
    key_id = None
    secret = None
    if user is not None:
        key_id = resolve_api_credential(user, "alpaca_api_key_id")
        secret = resolve_api_credential(user, "alpaca_api_secret_key")
    elif user_id:
        try:
            creds = load_api_credentials(str(user_id))
        except Exception:
            creds = {}
        if isinstance(creds, dict):
            key_id = creds.get("alpaca_api_key_id")
            secret = creds.get("alpaca_api_secret_key")
    if not key_id:
        key_id = os.environ.get("ALPACA_API_KEY_ID")
    if not secret:
        secret = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key_id or not secret:
        return None, None
    return key_id, secret


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
) -> requests.Response | None:
    config = resolve_retry_config(timeout=timeout)
    session = get_requests_session(config.timeout)

    def _call():
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
    key_id, secret = resolve_alpaca_credentials(user=user, user_id=user_id)
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
        response = _alpaca_get(url, params=params, headers=headers, timeout=timeout)
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
    key_id, secret = resolve_alpaca_credentials(user=user, user_id=user_id)
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
        response = _alpaca_get(url, params=params, headers=headers, timeout=timeout)
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
) -> list[dict[str, Any]]:
    key_id, secret = resolve_alpaca_credentials(user=user, user_id=user_id)
    if not key_id or not secret:
        return []
    params: dict[str, Any] = {}
    if symbols:
        normalized = _normalize_symbols(symbols)
        if normalized:
            params["symbols"] = ",".join(normalized)
    if limit:
        params["limit"] = int(limit)
    start_ts = _format_ts(start)
    end_ts = _format_ts(end)
    if start_ts:
        params["start"] = start_ts
    if end_ts:
        params["end"] = end_ts
    url = f"{(base_url or DEFAULT_DATA_URL).rstrip('/')}{path or DEFAULT_NEWS_PATH}"
    headers = _alpaca_headers(key_id, secret)
    response = _alpaca_get(url, params=params, headers=headers, timeout=timeout)
    if response is None:
        return []
    try:
        payload = response.json()
    except ValueError:
        return []
    for key in ("news", "items", "data"):
        items = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(items, list):
            return items
    if isinstance(payload, list):
        return payload
    return []
