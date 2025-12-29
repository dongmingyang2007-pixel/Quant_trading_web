from __future__ import annotations

import hashlib
import os
import random
import re
import time
from typing import Any

import yfinance as yf

from .config import StrategyInput
from ..headlines import estimate_readers


def fetch_market_context(params: StrategyInput) -> dict[str, Any]:
    context: dict[str, Any] = {
        "message": "",
        "news": [],
        "tickers": [],
        "analysis": "",
    }

    if os.environ.get("ENABLE_WEB_SEARCH", "1") == "0":
        context["message"] = (
            "当前运行环境未启用外部网络搜索。可在服务器设置 ENABLE_WEB_SEARCH=1 并配置代理后获取最新资讯。"
        )
        return context

    sector = ""
    industry = ""
    info: dict[str, Any] = {}
    try:
        ticker_info = yf.Ticker(params.ticker)
        info = ticker_info.info or {}
        sector = info.get("sector", "") or ""
        industry = info.get("industry", "") or ""
    except Exception:
        sector = ""
        industry = ""

    interest_terms: list[str] = []
    if params.interest_keywords:
        interest_terms.extend(params.interest_keywords)
    if sector:
        interest_terms.append(sector)
    if industry:
        interest_terms.append(industry)
    if params.benchmark_ticker:
        interest_terms.append(params.benchmark_ticker.upper())

    def normalize_term(term: str) -> str:
        if not term:
            return ""
        term = term.strip()
        return term

    interest_terms = list(dict.fromkeys(filter(None, (normalize_term(term) for term in interest_terms))))

    year = params.end_date.year
    ticker_upper = params.ticker.upper()
    base_queries = [
        f"{ticker_upper} 投资 前景 {year}",
        f"{ticker_upper} 新闻 {year}",
    ]
    english_queries = [
        f"{ticker_upper} stock news {year}",
        f"{ticker_upper} outlook {year}",
        f"{ticker_upper} earnings updates",
    ]
    base_queries.extend(english_queries)
    def quote_if_needed(term: str) -> str:
        if not term:
            return term
        if any(ch.isspace() for ch in term) or re.search(r"[\u4e00-\u9fff]", term):
            return f'"{term}"'
        return term

    if interest_terms:
        combined_terms = " ".join(quote_if_needed(t) for t in interest_terms[:2])
        base_queries.insert(0, f"{ticker_upper} {combined_terms} {year}")
        for term in interest_terms[:4]:
            base_queries.append(f"{quote_if_needed(term)} {ticker_upper} 最新 动态")
    if industry and industry not in interest_terms:
        base_queries.append(f"{industry} 行业 趋势 {year}")

    region = os.environ.get("DDG_REGION", "wt-wt")
    safesearch = os.environ.get("DDG_SAFESEARCH", "off")
    proxy = os.environ.get("DDG_PROXY")

    rate_limit_exceptions: tuple[type[Exception], ...] = ()
    ddg_client_cls: Any | None = None
    try:
        from ddgs import DDGS as _DDGS  # type: ignore

        ddg_client_cls = _DDGS
        try:
            from ddgs.exceptions import RatelimitException as _DDGRateLimit  # type: ignore

            rate_limit_exceptions = (_DDGRateLimit,)
        except Exception:
            rate_limit_exceptions = ()
    except ImportError:
        try:
            from duckduckgo_search import DDGS as _DDGS  # type: ignore

            ddg_client_cls = _DDGS
            try:
                from duckduckgo_search.exceptions import (  # type: ignore
                    RatelimitException as _DDGRateLimit,
                )

                rate_limit_exceptions = (_DDGRateLimit,)
            except Exception:
                rate_limit_exceptions = ()
        except ImportError:
            context["message"] = "缺少 duckduckgo-search / ddgs 依赖，无法执行在线搜索。"
            return context

    if ddg_client_cls is None:
        context["message"] = "未找到可用的 DuckDuckGo 搜索客户端。"
        return context

    def run_duck_query(ddgs_client: Any, query: str) -> list[dict[str, str]]:
        """Retry DuckDuckGo新闻/文本查询，自动处理限流。"""
        wait_seconds = 1.0
        for _ in range(3):
            try:
                news_items = list(
                    ddgs_client.news(
                        query,
                        region=region,
                        safesearch=safesearch,
                        max_results=6,
                    )
                )
                if news_items:
                    return news_items
            except Exception as exc:
                if rate_limit_exceptions and isinstance(exc, rate_limit_exceptions):
                    time.sleep(min(6.0, wait_seconds + random.random()))
                    wait_seconds *= 1.8
                    continue
                break
            time.sleep(0.5 + random.random() * 0.5)

        wait_seconds = 1.0
        for _ in range(2):
            try:
                text_items = list(
                    ddgs_client.text(
                        query,
                        region=region,
                        safesearch=safesearch,
                        max_results=6,
                    )
                )
                if text_items:
                    return text_items
            except Exception as exc:
                if rate_limit_exceptions and isinstance(exc, rate_limit_exceptions):
                    time.sleep(min(5.0, wait_seconds + random.random()))
                    wait_seconds *= 1.7
                    continue
                break
            time.sleep(0.4 + random.random() * 0.4)
        return []

    try:
        proxies = {"http": proxy, "https": proxy} if proxy else None
        aggregated: list[dict[str, str]] = []
        seen_queries: set[str] = set()
        with ddg_client_cls(proxies=proxies) as ddgs:  # type: ignore
            for q in base_queries:
                query_key = q.lower()
                if query_key in seen_queries:
                    continue
                seen_queries.add(query_key)
                search_results = run_duck_query(ddgs, q)
                aggregated.extend(search_results)
                if len(aggregated) >= 18:
                    break

        if not aggregated:
            context["message"] = "已尝试 DuckDuckGo 搜索，但未检索到相关新闻条目。"
            return context

        ticker_candidates: set[str] = set()
        unique_items: list[dict[str, str]] = []
        interest_lower = [term.lower() for term in interest_terms]
        interest_lower.append(params.ticker.lower())
        interest_lower = list(dict.fromkeys(filter(None, interest_lower)))
        ticker_pattern = re.compile(rf"\\b{re.escape(ticker_upper)}\\b", re.IGNORECASE)

        def _alias_variants(raw_alias: str) -> list[str]:
            alias = raw_alias.lower().strip()
            if not alias:
                return []
            variants = [alias]
            normalized = re.sub(r"[^\w\s]", " ", alias)
            normalized = re.sub(r"\s+", " ", normalized).strip()
            if normalized and normalized not in variants:
                variants.append(normalized)
            compact = re.sub(r"[^a-z0-9]", "", alias)
            if compact and compact not in variants:
                variants.append(compact)
            if normalized:
                first_token = normalized.split(" ")[0]
                if first_token and first_token not in variants:
                    variants.append(first_token)
            return [variant for variant in variants if variant]

        company_aliases: list[str] = []
        for alias_key in ("shortName", "longName"):
            alias_value = info.get(alias_key)
            if alias_value:
                company_aliases.extend(_alias_variants(str(alias_value)))
        if not company_aliases:
            company_aliases.append(params.ticker.lower())

        def normalize_story(item: dict[str, Any]) -> dict[str, Any] | None:
            title = item.get("title") or item.get("heading") or ""
            url = item.get("url") or item.get("href") or ""
            snippet = item.get("body") or item.get("excerpt") or item.get("snippet") or ""
            if not url:
                return None
            image_url = (
                item.get("image")
                or item.get("image_url")
                or item.get("img")
                or item.get("thumbnail")
                or ""
            )
            if image_url.startswith("//"):
                image_url = "https:" + image_url
            source = item.get("source") or item.get("publisher") or item.get("site") or ""
            published = item.get("published") or item.get("date") or item.get("time") or ""
            return {
                "title": title or "相关新闻",
                "url": url,
                "snippet": snippet,
                "image": image_url,
                "source": source,
                "published": published,
                "raw_score": item.get("score", 0) or 0,
            }

        normalized_items: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in aggregated:
            story = normalize_story(item)
            if not story:
                continue
            if story["url"] in seen_urls:
                continue
            seen_urls.add(story["url"])
            normalized_items.append(story)

        def _story_match_flags(story: dict[str, Any]) -> tuple[bool, bool, bool, str]:
            combined_text = f"{story['title']} {story['snippet']}"
            text_mix = combined_text.lower()
            matches_ticker = bool(ticker_pattern.search(combined_text))
            matches_alias = any(alias in text_mix for alias in company_aliases if alias)
            matches_interest = any(term and term in text_mix for term in interest_lower)
            return matches_ticker, matches_alias, matches_interest, text_mix

        for story in normalized_items:
            matches_ticker, matches_alias, matches_interest, text_mix = _story_match_flags(story)
            if not (matches_ticker or matches_alias or matches_interest):
                continue
            score = 0
            if matches_ticker:
                score += 3
            if matches_alias:
                score += 2
            if matches_interest:
                score += 1
            if industry and industry.lower() in text_mix:
                score += 1
            if sector and sector.lower() in text_mix:
                score += 1
            raw_score = story.get("raw_score", 0) or 0
            score += min(3, int(raw_score) // 10 if isinstance(raw_score, (int, float)) else 0)
            unique_items.append(
                {
                    "title": story["title"],
                    "url": story["url"],
                    "snippet": story["snippet"],
                    "score": score,
                    "image": story["image"],
                    "source": story["source"],
                    "published": story["published"],
                }
            )

            upper_words = {
                token
                for token in re.findall(r"[A-Z]{2,6}", f"{story['title']} {story['snippet']}")
                if not token.isdigit()
            }
            ticker_candidates.update(upper_words)

        if not unique_items and normalized_items:
            context[
                "message"
            ] = "DuckDuckGo 未返回足够精确的条目，以下结果为放宽条件后的最新资讯。"
            for story in normalized_items[:6]:
                unique_items.append(
                    {
                        "title": story["title"],
                        "url": story["url"],
                        "snippet": story["snippet"],
                        "score": story.get("raw_score", 0) or 0,
                        "image": story["image"],
                        "source": story["source"],
                        "published": story["published"],
                    }
                )

        unique_items.sort(key=lambda item: item.get("score", 0), reverse=True)
        filtered_items = [item for item in unique_items if item.get("score", 0) > 0]
        top_items = (filtered_items or unique_items)[:8]
        context["news"] = [
            {
                "title": item["title"],
                "url": item["url"],
                "snippet": item["snippet"],
                "image": item.get("image") or "",
                "source": item.get("source") or "",
                "published": item.get("published") or "",
            }
            for item in top_items
        ]
        focus_cards: list[dict[str, str]] = []
        for item in top_items[:6]:
            story_hash = hashlib.sha256(item["url"].encode("utf-8")).hexdigest()
            focus_cards.append(
                {
                    "id": story_hash[:12],
                    "title": item["title"],
                    "url": item["url"],
                    "snippet": item["snippet"],
                    "image": item.get("image") or "",
                    "source": item.get("source") or "",
                    "published": item.get("published") or "",
                    "readers": estimate_readers(story_hash[:16], item.get("score", 0)),
                }
            )
        context["focus_news"] = focus_cards

        context["tickers"] = sorted(ticker_candidates - {params.ticker.upper()})
        snippets = [item.get("snippet", "") for item in context["news"] if item.get("snippet")]
        analysis_parts: list[str] = []
        if sector:
            analysis_parts.append(f"所属行业：{sector}")
        if industry:
            analysis_parts.append(f"细分领域：{industry}")
        if snippets:
            analysis_parts.append("新闻摘要：" + " ".join(snippets[:2]))
        if interest_terms:
            analysis_parts.append("关注主题：" + "、".join(interest_terms[:4]))
        context["analysis"] = "；".join(analysis_parts)
        context["message"] = "以下为根据 DuckDuckGo 检索到的重点新闻摘要，可用于拓展与公司相关的标的。"
        context["interest_terms"] = interest_terms
    except Exception as exc:
        context["message"] = (
            "尝试调用外部搜索接口失败，请确认网络可达且 DuckDuckGo API 可用。"
            f" 错误详情：{exc}"
        )

    return context


# === Pipeline orchestration & summaries reinstated ===
