from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List

from django.conf import settings

from .network import get_requests_session, resolve_retry_config, retry_call

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


def _fetch_remote_headlines() -> tuple[List[Dict[str, str]], bool]:
    # 先尝试用 yfinance 自带新闻源，命中则可直接使用
    yf_headlines = _fetch_yfinance_headlines()
    if yf_headlines:
        return (yf_headlines, False)
    if os.environ.get("ENABLE_WEB_SEARCH", "1") == "0":
        return (DEFAULT_HEADLINES, True)
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return (DEFAULT_HEADLINES, True)

    queries = [
        "global stock market news today",
        "wall street market wrap",
        "asia markets opening news",
        "europe market close recap",
        "宏观 市场 今日 要闻 股票",
        "科技股 最新 动态",
    ]
    region = os.environ.get("DDG_REGION", "wt-wt")
    safesearch = os.environ.get("DDG_SAFESEARCH", "off")
    proxy = os.environ.get("DDG_PROXY")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    retry_config = resolve_retry_config()
    timeout_seconds = max(1, int(retry_config.timeout))

    def _build_ddg_client(client_cls):
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

    aggregated: List[Dict[str, str]] = []
    seen_urls: set[str] = set()
    try:
        with _build_ddg_client(DDGS) as ddgs:  # type: ignore
            for query in queries:
                try:
                    def _news_query():
                        return list(
                            ddgs.news(  # type: ignore
                                query,
                                region=region,
                                safesearch=safesearch,
                                max_results=DDG_MAX_RESULTS,
                            )
                        )

                    results = retry_call(_news_query, config=retry_config)
                except Exception:
                    try:
                        def _text_query():
                            return list(
                                ddgs.text(  # type: ignore
                                    query,
                                    region=region,
                                    safesearch=safesearch,
                                    max_results=DDG_MAX_RESULTS,
                                )
                            )

                        results = retry_call(_text_query, config=retry_config)
                    except Exception:
                        results = []
                for raw in results:
                    normalized = _normalize_item(raw)
                    url = normalized.get("url")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    normalized["score"] = _score_item(normalized)
                    aggregated.append(normalized)
                if len(aggregated) >= AGGREGATE_TARGET:
                    break
    except Exception:
        return (DEFAULT_HEADLINES, True)

    if not aggregated:
        return (DEFAULT_HEADLINES, True)

    aggregated.sort(key=lambda x: x.get("score", 0), reverse=True)

    now_utc = datetime.now(timezone.utc)
    max_age_cutoff = now_utc - HEADLINE_MAX_AGE
    fresh_items: List[Dict[str, str]] = []
    stale_items: List[Dict[str, str]] = []
    undated_items: List[Dict[str, str]] = []
    unique_by_title: Dict[str, Dict[str, str]] = {}
    for item in aggregated:
        key = item["title"]
        if key not in unique_by_title:
            unique_by_title[key] = item
    cutoff = now_utc - FRESHNESS_THRESHOLD
    for item in unique_by_title.values():
        published_iso = item.get("published_dt") or ""
        published_dt = None
        if published_iso:
            try:
                published_dt = datetime.fromisoformat(published_iso)
            except ValueError:
                published_dt = None
        if published_dt and published_dt < max_age_cutoff:
            continue  # 丢弃超过一周的旧新闻
        if published_dt and published_dt >= cutoff:
            fresh_items.append(item)
        elif published_dt:
            stale_items.append(item)
        else:
            undated_items.append(item)

    selected: List[Dict[str, str]] = list(fresh_items)
    if len(selected) < MIN_HEADLINES:
        for pool in (stale_items, undated_items):
            for entry in pool:
                if len(selected) >= MAX_HEADLINES:
                    break
                selected.append(entry)
            if len(selected) >= MAX_HEADLINES:
                break

    if not selected:
        return (DEFAULT_HEADLINES, True)

    top_items = selected[:MAX_HEADLINES]
    for item in top_items:
        heat = item.pop("score", 0)
        story_id = item.get("id") or hashlib.sha256(item.get("title", "").encode("utf-8")).hexdigest()[:16]
        item["readers"] = estimate_readers(story_id, heat)
    return (top_items, False)


def get_global_headlines(refresh: bool = False) -> List[Dict[str, str]]:
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

    stories_raw, used_fallback = _fetch_remote_headlines()
    stories = [_ensure_readers_field(item) for item in stories_raw]

    if not used_fallback:
        payload = {
            "timestamp": now.isoformat(),
            "headlines": stories,
        }
        _save_cache(payload)

    return stories
def _fetch_yfinance_headlines() -> List[Dict[str, str]]:
    """Use yfinance ticker news as a lightweight, dependency-free headline source."""
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return []

    tickers = [sym.strip() for sym in os.environ.get("HEADLINE_TICKERS", "SPY,QQQ,GLD,BTC-USD").split(",") if sym.strip()]
    aggregated: List[Dict[str, str]] = []
    seen_urls: set[str] = set()
    retry_config = resolve_retry_config()
    session = get_requests_session(retry_config.timeout)

    def _extract_image(thumbnail: Dict[str, Any] | None) -> str:
        if not thumbnail:
            return ""
        if thumbnail.get("originalUrl"):
            return thumbnail["originalUrl"]
        resolutions = thumbnail.get("resolutions") or []
        for res in resolutions:
            if res.get("url"):
                return res["url"]
        return ""

    for symbol in tickers:
        try:
            def _download_news():
                ticker = yf.Ticker(symbol, session=session)
                return ticker.news or []

            news_items = retry_call(_download_news, config=retry_config)
        except Exception:
            continue
        for entry in news_items:
            content = entry.get("content") or {}
            title = content.get("title")
            if not title:
                continue
            url = (
                (content.get("canonicalUrl") or {}).get("url")
                or (content.get("clickThroughUrl") or {}).get("url")
                or ""
            )
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            summary = content.get("summary") or content.get("description") or ""
            provider = (content.get("provider") or {}).get("displayName") or symbol.upper()
            published = content.get("pubDate") or content.get("displayTime") or ""
            published_display, published_iso = _parse_datetime(published)
            thumbnail = _extract_image(content.get("thumbnail"))
            story_id = entry.get("id") or hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            heat = int(entry.get("heat", 0) or 0)
            aggregated.append(
                {
                    "id": story_id,
                    "title": title.strip(),
                    "url": url,
                    "snippet": summary.strip(),
                    "image": thumbnail,
                    "source": provider,
                    "published": published_display,
                    "published_dt": published_iso,
                    "heat": heat,
                }
            )
            if len(aggregated) >= AGGREGATE_TARGET:
                break
        if len(aggregated) >= AGGREGATE_TARGET:
            break
    return aggregated
