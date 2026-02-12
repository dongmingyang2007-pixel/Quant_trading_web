from __future__ import annotations

import os
from typing import Any, Iterable

import pandas as pd

from . import alpaca_data, massive_data
from .profile import load_api_credentials

SUPPORTED_PROVIDERS = {"alpaca", "massive"}
SUPPORTED_NEWS_PROVIDERS = {"follow_data", "alpaca", "massive"}
DEFAULT_MARKET_PROVIDER = "alpaca"
DEFAULT_NEWS_PROVIDER = "follow_data"
DEFAULT_MASSIVE_PLAN = "stocks_advanced"


def _load_credentials(*, user: Any | None = None, user_id: str | None = None) -> dict[str, str]:
    if user is not None and getattr(user, "is_authenticated", False):
        user_id = str(user.id)
    if not user_id:
        return {}
    try:
        creds = load_api_credentials(str(user_id))
    except Exception:
        return {}
    return creds if isinstance(creds, dict) else {}


def _normalize_provider(value: Any, *, default: str = DEFAULT_MARKET_PROVIDER) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in SUPPORTED_PROVIDERS:
        return normalized
    return default


def _normalize_news_provider(value: Any, *, default: str = DEFAULT_NEWS_PROVIDER) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in SUPPORTED_NEWS_PROVIDERS:
        return normalized
    return default


def resolve_market_provider(*, user: Any | None = None, user_id: str | None = None) -> str:
    creds = _load_credentials(user=user, user_id=user_id)
    preferred = creds.get("market_data_provider")
    if preferred:
        return _normalize_provider(preferred)
    env_default = os.environ.get("MARKET_DATA_PROVIDER")
    if env_default:
        return _normalize_provider(env_default)
    return DEFAULT_MARKET_PROVIDER


def resolve_news_provider(*, user: Any | None = None, user_id: str | None = None) -> str:
    creds = _load_credentials(user=user, user_id=user_id)
    raw_preferred = str(creds.get("market_news_provider") or "").strip()
    if raw_preferred:
        preferred = _normalize_news_provider(raw_preferred)
        if preferred == "follow_data":
            return resolve_market_provider(user=user, user_id=user_id)
        return preferred
    env_raw = str(os.environ.get("MARKET_NEWS_PROVIDER") or "").strip()
    if env_raw:
        env_default = _normalize_news_provider(env_raw)
        if env_default == "follow_data":
            return resolve_market_provider(user=user, user_id=user_id)
        return env_default
    return resolve_market_provider(user=user, user_id=user_id)


def resolve_massive_plan(*, user: Any | None = None, user_id: str | None = None) -> str:
    creds = _load_credentials(user=user, user_id=user_id)
    plan = str(creds.get("massive_plan") or "").strip().lower()
    if plan:
        return plan
    return str(os.environ.get("MASSIVE_PLAN") or DEFAULT_MASSIVE_PLAN).strip().lower() or DEFAULT_MASSIVE_PLAN


def provider_caps(provider: str, plan: str | None = None) -> dict[str, Any]:
    key = _normalize_provider(provider)
    normalized_plan = str(plan or DEFAULT_MASSIVE_PLAN).strip().lower()
    if key == "massive":
        return {
            "provider": key,
            "plan": normalized_plan,
            "bars": True,
            "snapshots": True,
            "trades": True,
            "news": normalized_plan in {"stocks_advanced", "stocks_business", "stocks_enterprise"},
            "fundamentals": normalized_plan in {"stocks_advanced", "stocks_business", "stocks_enterprise"},
            "websocket": True,
            "partner": False,
        }
    return {
        "provider": "alpaca",
        "plan": "alpaca",
        "bars": True,
        "snapshots": True,
        "trades": True,
        "news": True,
        "fundamentals": False,
        "websocket": True,
        "partner": False,
    }


def resolve_market_data_credentials(
    *,
    user: Any | None = None,
    user_id: str | None = None,
    provider: str | None = None,
) -> tuple[str | None, str | None]:
    selected = _normalize_provider(provider, default=resolve_market_provider(user=user, user_id=user_id))
    if selected == "massive":
        return massive_data.resolve_massive_credentials(user=user, user_id=user_id)
    return alpaca_data.resolve_alpaca_data_credentials(user=user, user_id=user_id)


def has_market_data_credentials(*, user: Any | None = None, user_id: str | None = None, provider: str | None = None) -> bool:
    key, secret = resolve_market_data_credentials(user=user, user_id=user_id, provider=provider)
    if key:
        return True
    return bool(secret)


def resolve_market_context(*, user: Any | None = None, user_id: str | None = None) -> dict[str, str]:
    market_provider = resolve_market_provider(user=user, user_id=user_id)
    news_provider = resolve_news_provider(user=user, user_id=user_id)
    return {
        "market_data_source": market_provider,
        "news_source": news_provider,
        "execution_source": "alpaca",
        "massive_plan": resolve_massive_plan(user=user, user_id=user_id),
    }


def fetch_stock_bars(
    symbols: Iterable[str],
    *,
    start=None,
    end=None,
    timeframe: str = "1Day",
    feed: str | None = None,
    limit: int | None = None,
    adjustment: str | None = None,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
    provider: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    selected = _normalize_provider(provider, default=resolve_market_provider(user=user, user_id=user_id))
    if selected == "massive":
        return massive_data.fetch_stock_bars(
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
    return alpaca_data.fetch_stock_bars(
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


def fetch_stock_bars_frame(
    symbols: Iterable[str],
    *,
    start=None,
    end=None,
    timeframe: str = "1Day",
    feed: str | None = None,
    limit: int | None = None,
    adjustment: str | None = None,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
    provider: str | None = None,
) -> pd.DataFrame:
    selected = _normalize_provider(provider, default=resolve_market_provider(user=user, user_id=user_id))
    frame: pd.DataFrame
    if selected == "massive":
        frame = massive_data.fetch_stock_bars_frame(
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
    else:
        frame = alpaca_data.fetch_stock_bars_frame(
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
    if isinstance(frame, pd.DataFrame):
        attrs = getattr(frame, "attrs", None)
        if isinstance(attrs, dict):
            attrs["market_source"] = selected
        else:
            try:
                frame.attrs = {"market_source": selected}
            except Exception:
                pass
    return frame


def fetch_stock_snapshots(
    symbols: Iterable[str],
    *,
    feed: str | None = None,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    selected = _normalize_provider(provider, default=resolve_market_provider(user=user, user_id=user_id))
    if selected == "massive":
        return massive_data.fetch_stock_snapshots(
            symbols,
            feed=feed,
            user=user,
            user_id=user_id,
            timeout=timeout,
            base_url=base_url,
        )
    return alpaca_data.fetch_stock_snapshots(
        symbols,
        feed=feed,
        user=user,
        user_id=user_id,
        timeout=timeout,
        base_url=base_url,
    )


def fetch_stock_trades(
    symbol: str,
    *,
    start=None,
    end=None,
    feed: str | None = None,
    limit: int | None = None,
    page_token: str | None = None,
    max_pages: int = 1,
    sort: str | None = None,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
    provider: str | None = None,
):
    selected = _normalize_provider(provider, default=resolve_market_provider(user=user, user_id=user_id))
    if selected == "massive":
        return massive_data.fetch_stock_trades(
            symbol,
            start=start,
            end=end,
            feed=feed,
            limit=limit,
            page_token=page_token,
            max_pages=max_pages,
            sort=sort,
            user=user,
            user_id=user_id,
            timeout=timeout,
            base_url=base_url,
        )
    return alpaca_data.fetch_stock_trades(
        symbol,
        start=start,
        end=end,
        feed=feed,
        limit=limit,
        page_token=page_token,
        max_pages=max_pages,
        sort=sort,
        user=user,
        user_id=user_id,
        timeout=timeout,
        base_url=base_url,
    )


def fetch_company_overview(
    symbol: str,
    *,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    selected = _normalize_provider(provider, default=resolve_market_provider(user=user, user_id=user_id))
    if selected == "massive":
        return massive_data.fetch_company_overview(
            symbol,
            user=user,
            user_id=user_id,
            timeout=timeout,
            base_url=base_url,
        )
    return {}


def fetch_news(
    *,
    symbols: Iterable[str] | None = None,
    limit: int | None = None,
    start=None,
    end=None,
    user: Any | None = None,
    user_id: str | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
    path: str | None = None,
    max_pages: int | None = None,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    selected = _normalize_provider(provider, default=resolve_news_provider(user=user, user_id=user_id))
    if selected == "massive":
        return massive_data.fetch_news(
            symbols=symbols,
            limit=limit,
            start=start,
            end=end,
            user=user,
            user_id=user_id,
            timeout=timeout,
            base_url=base_url,
            path=path,
            max_pages=max_pages,
        )
    return alpaca_data.fetch_news(
        symbols=symbols,
        limit=limit,
        start=start,
        end=end,
        user=user,
        user_id=user_id,
        timeout=timeout,
        base_url=base_url,
        path=path,
        max_pages=max_pages,
    )
