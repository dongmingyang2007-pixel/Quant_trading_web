from __future__ import annotations

from typing import Any, Iterable
import os

import pandas as pd
import requests

from ...alpaca_data import fetch_stock_bars_frame, fetch_stock_snapshots, resolve_alpaca_credentials
from ...network import get_requests_session, resolve_retry_config, retry_call_result

DEFAULT_TRADING_URL = os.environ.get("ALPACA_TRADING_REST_URL", "https://paper-api.alpaca.markets").rstrip("/")


def fetch_snapshots(
    symbols: Iterable[str],
    *,
    feed: str | None,
    user_id: str | None,
    timeout: float | None = None,
) -> dict[str, Any]:
    return fetch_stock_snapshots(symbols, feed=feed, user_id=user_id, timeout=timeout)


def fetch_bars_frame(
    symbols: Iterable[str],
    *,
    timeframe: str,
    limit: int,
    feed: str | None,
    user_id: str | None,
    timeout: float | None = None,
) -> pd.DataFrame:
    return fetch_stock_bars_frame(
        symbols,
        timeframe=timeframe,
        limit=limit,
        feed=feed,
        user_id=user_id,
        timeout=timeout,
    )


def fetch_assets(
    *,
    user_id: str | None,
    status: str | None = "active",
    asset_class: str | None = "us_equity",
    timeout: float | None = None,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    key_id, secret = resolve_alpaca_credentials(user_id=user_id)
    if not key_id or not secret:
        return []
    url = f"{(base_url or DEFAULT_TRADING_URL).rstrip('/')}/v2/assets"
    params: dict[str, Any] = {}
    if status:
        params["status"] = status
    if asset_class:
        params["asset_class"] = asset_class
    headers = {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }
    config = resolve_retry_config(timeout=timeout)
    session = get_requests_session(config.timeout)

    def _call():
        return session.get(url, params=params, headers=headers, timeout=config.timeout)

    def _should_retry(response):
        try:
            return response.status_code >= 500
        except Exception:
            return False

    try:
        response = retry_call_result(
            _call,
            config=config,
            exceptions=(requests.RequestException,),
            should_retry=_should_retry,
        )
    except Exception:
        return []
    if response.status_code >= 400:
        return []
    try:
        payload = response.json()
    except ValueError:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        assets = payload.get("assets")
        if isinstance(assets, list):
            return assets
    return []
