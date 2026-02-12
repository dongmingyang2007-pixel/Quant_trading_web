from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List

from django.conf import settings

from .market_provider import fetch_news, resolve_news_provider

from .file_utils import atomic_write_json

DATA_CACHE_DIR = settings.DATA_CACHE_DIR
DATA_CACHE_DIR.mkdir(exist_ok=True)

CACHE_PATH = DATA_CACHE_DIR / "global_headlines.json"
CACHE_TTL = timedelta(minutes=int(os.environ.get("HEADLINE_CACHE_MINUTES", "10") or 10))
FRESHNESS_THRESHOLD = timedelta(hours=48)
HEADLINE_MAX_AGE = timedelta(days=int(os.environ.get("HEADLINE_MAX_AGE_DAYS", "7") or 7))
MAX_HEADLINES = max(6, int(os.environ.get("GLOBAL_HEADLINES_MAX", "12") or 12))
MIN_HEADLINES = min(10, MAX_HEADLINES)
DDG_MAX_RESULTS = max(8, min(20, int(os.environ.get("DDG_HEADLINE_RESULTS", "12") or 12)))
AGGREGATE_TARGET = MAX_HEADLINES * 3

DEFAULT_HEADLINES: List[Dict[str, str]] = [
    {
        "id": "fallback-sp500",
        "title": "全球市场早盘速览：关注美国科技股与能源价格",
        "url": "https://finance.yahoo.com/",
        "snippet": "科技与能源双线拉动主要指数走势，市场聚焦本周财报与宏观数据。",
        "image": "",
        "source": "Market Watch",
        "published": "",
        "published_dt": "",
        "heat": 3,
    },
    {
        "id": "fallback-asia",
        "title": "亚洲股市集体走高，资金回流半导体与人工智能主题",
        "url": "https://www.bloomberg.com/markets",
        "snippet": "半导体、AI 与绿色能源板块领涨，投资者等待晚间美联储官员讲话。",
        "image": "",
        "source": "Bloomberg",
        "published": "",
        "published_dt": "",
        "heat": 2,
    },
    {
        "id": "fallback-europe",
        "title": "欧洲市场：能源价格回落提振航空与制造业",
        "url": "https://www.reuters.com/finance/markets",
        "snippet": "油气价格回落，航空、制造与消费板块获资金青睐，市场关注欧洲央行动向。",
        "image": "",
        "source": "Reuters",
        "published": "",
        "published_dt": "",
        "heat": 2,
    },
]


def _load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(payload: Dict[str, Any]) -> None:
    try:
        atomic_write_json(CACHE_PATH, payload, indent=2)
    except OSError:
        pass


def _parse_datetime(value: str) -> tuple[str, str]:
    value = (value or "").strip()
    if not value:
        return "", ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                dt = parsedate_to_datetime(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                return value, ""
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%d %H:%M UTC"), dt_utc.isoformat()


def _normalize_item(item: Dict[str, Any]) -> Dict[str, str]:
    title = item.get("title") or item.get("heading") or "今日市场动态"
    url = item.get("url") or item.get("href") or ""
    snippet = item.get("excerpt") or item.get("body") or item.get("snippet") or ""
    image = (
        item.get("image")
        or item.get("image_url")
        or item.get("img")
        or item.get("thumbnail")
        or ""
    )
    if isinstance(image, dict):
        image = image.get("src", "")
    if image.startswith("//"):
        image = "https:" + image
    source = item.get("source") or item.get("publisher") or item.get("site") or ""
    published = (
        item.get("published")
        or item.get("date")
        or item.get("time")
        or item.get("pubDate")
        or item.get("published_parsed")
        or ""
    )
    if isinstance(published, str):
        published_display, published_iso = _parse_datetime(published)
    else:
        published_display, published_iso = "", ""
    story_id = hashlib.sha256((url or title).encode("utf-8")).hexdigest()[:16]
    return {
        "id": story_id,
        "title": title.strip(),
        "url": url,
        "snippet": snippet.strip(),
        "image": image,
        "source": source.strip(),
        "published": published_display,
        "published_dt": published_iso,
        "readers": 0,
    }


def _score_item(item: Dict[str, str]) -> int:
    text = f"{item['title']} {item['snippet']}".lower()
    score = 0
    keywords = [
        "stock",
        "market",
        "equity",
        "指数",
        "指数",
        "科技",
        "earnings",
        "利率",
        "能源",
    ]
    for word in keywords:
        if word in text:
            score += 1
    if re.search(r"\b(nasdaq|s&p|dow|恒生|上证)\b", text):
        score += 2
    return score


def _ensure_readers_field(story: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(story)
    readers_value = normalized.get("readers")
    if isinstance(readers_value, str) and readers_value in {"高", "中", "低"}:
        return normalized
    if "readers" not in normalized or isinstance(readers_value, (int, float)):
        score = int(normalized.get("heat", 0) or 0)
        basis = normalized.get("id") or normalized.get("url") or normalized.get("title") or ""
        story_id = hashlib.sha256(str(basis).encode("utf-8")).hexdigest()[:16] if basis else "fallback000000"
        base = 12000 + max(score, 0) * 8000
        pseudo = int(story_id[:8], 16) % 7000
        readers = base + pseudo
        if readers >= 32000:
            label = "高"
        elif readers >= 24000:
            label = "中"
        else:
            label = "低"
        normalized["readers"] = label
    return normalized


def estimate_readers(story_id: str, score: int) -> str:
    base = 12000 + max(score, 0) * 8000
    pseudo = int(story_id[:8], 16) % 7000
    readers = base + pseudo
    if readers >= 32000:
        return "高"
    if readers >= 24000:
        return "中"
    return "低"


def _has_fresh_story(stories: List[Dict[str, Any]]) -> bool:
    now_utc = datetime.now(timezone.utc)
    for story in stories:
        published_iso = story.get("published_dt")
        if not published_iso:
            continue
        try:
            published_dt = datetime.fromisoformat(published_iso)
        except ValueError:
            continue
        if now_utc - published_dt <= FRESHNESS_THRESHOLD:
            return True
    return False


def _fetch_provider_headlines(user_id: str | None = None) -> List[Dict[str, str]]:
    symbols_raw = os.environ.get("ALPACA_NEWS_SYMBOLS", "SPY,QQQ,AAPL,MSFT,NVDA")
    symbols = [sym.strip() for sym in symbols_raw.split(",") if sym.strip()]
    try:
        limit = int(os.environ.get("ALPACA_NEWS_LIMIT", str(AGGREGATE_TARGET)))
    except ValueError:
        limit = AGGREGATE_TARGET
    provider = resolve_news_provider(user_id=user_id)
    items = fetch_news(symbols=symbols, limit=limit, user_id=user_id, provider=provider)
    if not items:
        return []
    aggregated: List[Dict[str, str]] = []
    seen_urls: set[str] = set()
    for entry in items:
        title = entry.get("headline") or entry.get("title") or entry.get("summary") or ""
        if not title:
            continue
        url = entry.get("url") or entry.get("link") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        snippet = entry.get("summary") or entry.get("description") or ""
        source = entry.get("source") or entry.get("publisher") or ""
        image = ""
        images = entry.get("images") or entry.get("image") or []
        if isinstance(images, list) and images:
            image = images[0].get("url") or images[0].get("image_url") or ""
        elif isinstance(images, dict):
            image = images.get("url") or images.get("image_url") or ""
        published_raw = entry.get("created_at") or entry.get("updated_at") or entry.get("published_at") or ""
        published_display, published_iso = _parse_datetime(str(published_raw)) if published_raw else ("", "")
        normalized = {
            "id": hashlib.sha256((url or title).encode("utf-8")).hexdigest()[:16],
            "title": str(title).strip(),
            "url": url,
            "snippet": str(snippet).strip(),
            "image": image,
            "source": str(source).strip(),
            "published": published_display,
            "published_dt": published_iso,
        }
        normalized["score"] = _score_item(normalized)
        aggregated.append(normalized)
    return aggregated


def _fetch_remote_headlines(user_id: str | None = None) -> tuple[List[Dict[str, str]], bool]:
    provider_items = _fetch_provider_headlines(user_id=user_id)
    if not provider_items:
        return (DEFAULT_HEADLINES, True)
    top_items = provider_items[:MAX_HEADLINES]
    for item in top_items:
        heat = int(item.pop("score", 0) or 0)
        story_id = item.get("id") or hashlib.sha256(item.get("title", "").encode("utf-8")).hexdigest()[:16]
        item["readers"] = estimate_readers(story_id, heat)
    return (top_items, False)


def get_global_headlines(refresh: bool = False, *, user_id: str | None = None) -> List[Dict[str, str]]:
    cache = _load_cache()
    now = datetime.now(timezone.utc)
    if not refresh and cache:
        timestamp = cache.get("timestamp")
        try:
            cached_at = datetime.fromisoformat(timestamp)
        except Exception:
            cached_at = None
        if cached_at and now - cached_at < CACHE_TTL:
            stories = [_ensure_readers_field(item) for item in cache.get("headlines") or []]
            if stories and _has_fresh_story(stories):
                return stories

    stories_raw, used_fallback = _fetch_remote_headlines(user_id=user_id)
    stories = [_ensure_readers_field(item) for item in stories_raw]

    if not used_fallback:
        payload = {
            "timestamp": now.isoformat(),
            "headlines": stories,
        }
        _save_cache(payload)

    return stories
