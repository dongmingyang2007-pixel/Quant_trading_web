from __future__ import annotations

import os
import random
import time
from typing import Any

from .cache_utils import build_cache_key, cache_memoize
from .network import resolve_retry_config


def _normalize_item(raw: dict[str, Any]) -> dict[str, str]:
    title = raw.get("title") or raw.get("heading") or ""
    url = raw.get("url") or raw.get("href") or ""
    snippet = raw.get("body") or raw.get("snippet") or raw.get("excerpt") or ""
    source = raw.get("source") or raw.get("publisher") or raw.get("site") or ""
    published = raw.get("published") or raw.get("date") or raw.get("time") or ""
    return {
        "title": str(title).strip(),
        "url": str(url).strip(),
        "snippet": str(snippet).strip(),
        "source": str(source).strip(),
        "published": str(published).strip(),
    }


def _build_ddg_client():
    try:
        from ddgs import DDGS  # type: ignore

        return DDGS
    except Exception:
        try:
            from duckduckgo_search import DDGS  # type: ignore

            return DDGS
        except Exception:
            return None


def _search_ddg(query: str, *, max_results: int, mode: str) -> list[dict[str, str]]:
    ddg_cls = _build_ddg_client()
    if ddg_cls is None:
        return []
    region = os.environ.get("DDG_REGION", "wt-wt")
    safesearch = os.environ.get("DDG_SAFESEARCH", "off")
    proxy = os.environ.get("DDG_PROXY")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    retry_config = resolve_retry_config()
    timeout_seconds = max(1, int(retry_config.timeout))

    def _build_client(client_cls):
        try:
            return client_cls(proxies=proxies, timeout=timeout_seconds)  # type: ignore[arg-type]
        except TypeError:
            proxy_value = None
            if isinstance(proxies, dict):
                proxy_value = proxies.get("http") or proxies.get("https")
            elif isinstance(proxies, str):
                proxy_value = proxies
            try:
                return client_cls(proxy=proxy_value, timeout=timeout_seconds)  # type: ignore[arg-type]
            except TypeError:
                return client_cls()

    def _run_query(client, query_text: str, kind: str) -> list[dict[str, Any]]:
        if kind == "news":
            return list(
                client.news(  # type: ignore[attr-defined]
                    query_text,
                    region=region,
                    safesearch=safesearch,
                    max_results=max_results,
                )
            )
        return list(
            client.text(  # type: ignore[attr-defined]
                query_text,
                region=region,
                safesearch=safesearch,
                max_results=max_results,
            )
        )

    results: list[dict[str, str]] = []
    try:
        with _build_client(ddg_cls) as client:  # type: ignore
            for attempt in range(max(1, retry_config.retries + 1)):
                try:
                    raw = _run_query(client, query, mode)
                    if raw:
                        results = [_normalize_item(item) for item in raw]
                        break
                except Exception:
                    time.sleep(min(1.0, retry_config.backoff + random.random() * 0.4))
    except Exception:
        return []
    return [item for item in results if item.get("title") or item.get("snippet")]


def search_web(query: str, *, max_results: int = 6, mode: str = "news", cache_ttl: int | None = None) -> list[dict[str, str]]:
    query = (query or "").strip()
    if not query:
        return []
    ttl = cache_ttl if cache_ttl is not None else int(os.environ.get("WEB_SEARCH_CACHE_TTL", "120") or 120)
    key = build_cache_key("web-search", mode, max_results, query)
    return cache_memoize(key, lambda: _search_ddg(query, max_results=max_results, mode=mode), ttl) or []
