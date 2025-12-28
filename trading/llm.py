from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
import calendar
from typing import Any, Callable, Dict, List
from urllib.parse import urljoin, urlsplit
import time

from .http_client import http_client, HttpClientError


class LLMIntegrationError(Exception):
    """Raised when the local LLM could not return a response."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default

def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

def _default_enable_web(provider: str, user_pref: bool | None) -> bool:
    """
    Decide whether to enable web search by default.
    - user_pref has highest priority
    - gemini 默认开启（可通过 GEMINI_ENABLE_WEB_DEFAULT=0 关闭）
    - ollama 按原有环境变量 OLLAMA_ENABLE_WEB
    """
    if user_pref is not None:
        return user_pref
    if provider == "gemini":
        return _env_bool("GEMINI_ENABLE_WEB_DEFAULT", True)
    return _env_bool("OLLAMA_ENABLE_WEB", False)

def _resolve_provider(model_hint: str | None = None) -> str:
    """
    Pick the LLM provider based on environment or model name.

    Priority:
      1) AI_PROVIDER env (e.g. "ollama" / "gemini")
      2) model hint starting with "gemini" -> gemini
      3) default to ollama
    """
    env_provider = (os.environ.get("AI_PROVIDER") or "").strip().lower()
    if env_provider:
        return env_provider
    if model_hint and str(model_hint).lower().startswith("gemini"):
        return "gemini"
    return "ollama"

def _build_ollama_options(is_secondary: bool) -> dict[str, Any]:
    temperature = _env_float("OLLAMA_TEMPERATURE", 0.6)
    # Cap output tokens to avoid long blocking on large models
    default_primary_predict = 900
    default_secondary_predict = 600
    if is_secondary:
        num_predict = _env_int(
            "OLLAMA_SECONDARY_NUM_PREDICT",
            _env_int("OLLAMA_NUM_PREDICT", default_secondary_predict),
        )
        num_ctx = _env_int("OLLAMA_SECONDARY_NUM_CTX", _env_int("OLLAMA_NUM_CTX", 4096))
    else:
        num_predict = _env_int("OLLAMA_NUM_PREDICT", default_primary_predict)
        num_ctx = _env_int("OLLAMA_NUM_CTX", 4096)
    options: dict[str, Any] = {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx}
    return options

def _iter_web_search_candidates() -> list[str]:
    """Yield possible Ollama web search endpoints (env -> derived -> localhost -> cloud)."""
    candidates: list[str] = []
    env_url = os.environ.get("OLLAMA_WEB_SEARCH_URL")
    if env_url:
        candidates.append(env_url.strip())

    base_endpoint = os.environ.get("OLLAMA_ENDPOINT")
    if base_endpoint:
        base_endpoint = base_endpoint.strip().rstrip("/")
        if base_endpoint.endswith(("/chat", "/generate", "/embeddings", "/pull", "/push")):
            base_endpoint = base_endpoint.rsplit("/", 1)[0]
        if base_endpoint:
            candidates.append(f"{base_endpoint.rstrip('/')}/web_search")

    # Prefer localhost when running Ollama locally
    candidates.append("http://127.0.0.1:11434/api/web_search")
    # Fallback to Ollama Cloud
    candidates.append("https://ollama.com/api/web_search")

    deduped: list[str] = []
    for url in candidates:
        if not url:
            continue
        normalized = url.strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


# in-memory TTL cache for web search results
_WEB_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _now_utc_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _shift_months(anchor: date, months_back: int) -> date:
    year = anchor.year
    month = anchor.month - months_back
    while month <= 0:
        month += 12
        year -= 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _generate_month_labels(anchor: date, months: int) -> list[str]:
    labels: list[str] = []
    if months <= 0:
        return labels
    steps = min(6, max(1, months))
    interval = max(1, months // steps)
    for offset in range(0, months, interval):
        dt = _shift_months(anchor, offset)
        labels.append(f"{dt.year}年{dt.month}月")
    return list(dict.fromkeys(labels))


def _extract_timeframe_months(user_text: str | None) -> int | None:
    if not user_text:
        return None
    normalized = user_text.strip()
    patterns = [
        (r"(?:最近|近|过去)(\d+)\s*个?(?:月|个月)", lambda m: int(m.group(1))),
        (r"(?:最近|近|过去)半年|半年内", lambda _m: 6),
        (r"(?:最近|近|过去)一年|一年内", lambda _m: 12),
        (r"(?:最近|近|过去)三个月|3个月|季度", lambda _m: 3),
    ]
    for pattern, extractor in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            months = extractor(match)
            if months > 0:
                return min(24, months)
    return None


def _parse_date_safe(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
    return None


def _build_news_digest(items: list[dict[str, Any]], limit: int = 8) -> str:
    parts: list[str] = []
    for item in items[:limit]:
        ts = item.get("retrieved_at") or ""
        host = item.get("host") or urlsplit(item.get("url") or "").netloc or "未知来源"
        title = item.get("title") or "资讯"
        snippet = (item.get("snippet") or "").strip()
        parts.append(f"{ts} · {title}（{host}）{': ' + snippet if snippet else ''}")
    return "\n".join(parts)


def _sort_news(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _parse_ts(value: str | None) -> float:
        if not value:
            return 0.0
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M UTC").timestamp()
        except Exception:
            return 0.0

    return sorted(items, key=lambda item: _parse_ts(item.get("retrieved_at")), reverse=True)


def _prepend_web_digest(answer: str, web_results: list[dict[str, Any]] | None) -> str:
    if not web_results:
        return answer
    lines = []
    for item in web_results[:3]:
        host = item.get("host") or urlsplit(item.get("url") or "").netloc or "未知来源"
        ts = item.get("retrieved_at") or _now_utc_label()
        title = item.get("title") or "实时资讯"
        url = item.get("url") or ""
        snippet = (item.get("snippet") or "").replace("\n", " ").strip()
        if snippet:
            snippet = snippet[:220]
        link_markup = f"[访问链接]({url})" if url else ""
        extra = f" — {snippet}" if snippet else ""
        lines.append(f"- [{ts}] **{title}**（{host}） {link_markup}{extra}")
    digest = "实时资讯引用（系统已联网获取）：\n" + "\n".join(lines)
    if answer:
        return f"{digest}\n\n{answer}"
    return digest

def _ollama_web_search(
    query: str,
    *,
    max_results: int = 5,
    timeout: int = 20,
) -> tuple[list[dict[str, str]], str | None, str | None]:
    """Call a Web Search endpoint and return items (title/url/snippet).
    Requires OLLAMA_API_KEY for endpoints that use Bearer authentication.
    """
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not query.strip():
        return [], "未提供有效的检索关键词", None

    errors: list[str] = []
    for url in _iter_web_search_candidates():
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = http_client.post(
                url,
                json={"query": query, "max_results": max_results},
                headers=headers,
                timeout=timeout,
                retries=0,
            )
        except HttpClientError as exc:
            errors.append(f"{url} 连接失败：{exc}")
            continue

        try:
            data = resp.json() or {}
        except ValueError:
            errors.append(f"{url} 返回了无法解析的 JSON")
            continue

        raw_results = data.get("results") or data.get("data") or []
        items: list[dict[str, str]] = []
        retrieved_label = _now_utc_label()
        for r in raw_results[:max_results]:
            url = r.get("url") or ""
            host = urlsplit(url).netloc
            items.append(
                {
                    "title": (r.get("title") or r.get("heading") or "网页").strip(),
                    "url": url,
                    "snippet": (r.get("content") or r.get("body") or r.get("excerpt") or "")[:220].strip(),
                    "host": host,
                    "retrieved_at": retrieved_label,
                }
            )

        if items:
            return items, None, url
        errors.append(f"{url} 未返回有效结果")

    message = "；".join(errors[-3:]) if errors else "未获取到实时资讯"
    return [], message, None

def _build_market_context_from_web(query: str, *, max_results: int = 5) -> dict[str, Any]:
    news, error_msg, source = _ollama_web_search(query, max_results=max_results)
    retrieved_label = _now_utc_label()
    enriched: list[dict[str, Any]] = []
    for item in news:
        url = item.get("url") or ""
        host = item.get("host") or urlsplit(url).netloc
        enriched.append(
            {
                **item,
                "host": host,
                "retrieved_at": item.get("retrieved_at") or retrieved_label,
            }
        )
    payload: dict[str, Any] = {
        "news": enriched,
        "tickers": [],
        "interest_terms": [query],
        "source": source,
        "retrieved_at": retrieved_label,
    }
    if enriched:
        host = urlsplit(source or "").netloc if source else ""
        host_label = host or (source or "实时接口")
        payload["message"] = f"来自 {host_label} 的实时资讯"
    else:
        payload["message"] = error_msg or "未获取到外部资讯"
        payload["error"] = error_msg
    return payload


def _merge_market_context(base: dict[str, Any], part: dict[str, Any], max_results: int) -> dict[str, Any]:
    if not part:
        return base
    news_list = base.setdefault("news", [])
    seen_urls = {item.get("url") for item in news_list if item.get("url")}
    for item in part.get("news") or []:
        url = item.get("url")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        news_list.append(item)
    if max_results > 0 and len(news_list) > max_results:
        base["news"] = news_list[:max_results]
    base.setdefault("interest_terms", [])
    for term in part.get("interest_terms") or []:
        if term and term not in base["interest_terms"]:
            base["interest_terms"].append(term)
    if not base.get("message"):
        base["message"] = part.get("message")
    if not base.get("source"):
        base["source"] = part.get("source")
    if not base.get("retrieved_at"):
        base["retrieved_at"] = part.get("retrieved_at")
    return base


def _cached_web_search(q: str | list[str], *, max_results: int = 0) -> dict[str, Any]:
    ttl = _env_int("OLLAMA_WEB_TTL_SECONDS", 1800)
    queries = q if isinstance(q, (list, tuple)) else [q]
    normalized_queries = [query.strip() for query in queries if query and str(query).strip()]
    if not normalized_queries:
        return {"message": "缺少有效的联网关键词", "news": []}
    key = " || ".join(normalized_queries) + f"|{max_results}"
    now = time.time()
    hit = _WEB_CACHE.get(key)
    if hit and (now - hit[0] < ttl):
        return hit[1]
    aggregated: dict[str, Any] = {"news": [], "interest_terms": [], "message": "", "source": "", "retrieved_at": ""}
    messages: list[str] = []
    per_query_cap = max_results if max_results > 0 else 10
    for query in normalized_queries:
        part = _build_market_context_from_web(query, max_results=per_query_cap)
        if part.get("message") and not part.get("news"):
            messages.append(part.get("message", ""))
        aggregated = _merge_market_context(aggregated, part, max_results if max_results > 0 else 0)
    if not aggregated.get("message") and messages:
        aggregated["message"] = "；".join(filter(None, messages))
    if aggregated.get("news"):
        _WEB_CACHE[key] = (now, aggregated)
    else:
        _WEB_CACHE.pop(key, None)
    return aggregated


def _resolve_chat_endpoint(endpoint: str | None) -> str:
    """Derive the chat endpoint from the configured base endpoint."""
    if not endpoint:
        return "http://localhost:11434/api/chat"
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat"):
        return endpoint
    if endpoint.endswith("/generate"):
        base = endpoint.rsplit("/", 1)[0]
    else:
        base = endpoint
    return urljoin(base.rstrip("/") + "/", "chat")


def _build_advisor_system_prompt(result: dict[str, Any]) -> str:
    """Compose a system prompt grounded in the latest backtest context."""
    ticker = result.get("ticker") or "未指定标的"
    start = result.get("start_date")
    end = result.get("end_date")
    window = ""
    if start and end:
        window = f"{start} 至 {end}"
    capital = result.get("capital")
    risk_profile = result.get("risk_profile")
    goal = result.get("primary_goal_label") or result.get("primary_goal")
    horizon = result.get("investment_horizon_label") or result.get("investment_horizon")
    experience = result.get("experience_label") or result.get("experience_level")

    info_bits: list[str] = [f"标的 {ticker}"]
    if window:
        info_bits.append(f"区间 {window}")
    if capital:
        info_bits.append(f"资金 {capital:,.0f}")
    if risk_profile:
        info_bits.append(f"风险偏好 {risk_profile}")
    if horizon:
        info_bits.append(f"投资周期 {horizon}")
    if goal:
        info_bits.append(f"核心目标 {goal}")
    if experience:
        info_bits.append(f"经验 {experience}")

    metrics = result.get("metrics") or []
    metric_line = "；".join(f"{m.get('label')}: {m.get('value')}" for m in metrics[:4] if m)
    quick = result.get("quick_summary") or result.get("key_takeaways") or []
    quick_line = "；".join(str(item) for item in quick[:4])
    risk_alerts = result.get("risk_alerts") or []
    risk_line = "；".join(str(alert) for alert in risk_alerts[:3])
    scenario = result.get("scenario_simulation") or {}
    scenario_line = ""
    if scenario.get("available") and scenario.get("scenarios"):
        scenario_line = "、".join(f"{row.get('label')} {row.get('return')}" for row in scenario["scenarios"][:3])

    context_sections: list[str] = []
    if metric_line:
        context_sections.append(f"核心指标：{metric_line}")
    if quick_line:
        context_sections.append(f"亮点：{quick_line}")
    if scenario_line:
        context_sections.append(f"情景模拟：{scenario_line}")
    if risk_line:
        context_sections.append(f"风险提示：{risk_line}")

    # Macro & flows
    macro_bundle = result.get("macro_bundle") or {}
    macro_snippets: list[str] = []
    for entry in list(macro_bundle.values()):
        if not isinstance(entry, dict) or not entry.get("available"):
            continue
        short = entry.get("short") or entry.get("label")
        change = entry.get("change_21d")
        trend = entry.get("trend")
        if short:
            macro_snippets.append(f"{short} {trend or ''} (21日 {change}% )")
        if len(macro_snippets) >= 4:
            break
    if macro_snippets:
        context_sections.append("宏观脉搏：" + "；".join(macro_snippets))

    flows = result.get("capital_flows") or {}
    flow_summary = flows.get("_summary") or {}
    flow_parts = []
    if flow_summary.get("risk_appetite"):
        flow_parts.append(f"风险偏好 {flow_summary['risk_appetite']}")
    if flow_summary.get("growth_vs_value") is not None:
        flow_parts.append(f"成长-价值 {flow_summary['growth_vs_value']:.2f}%")
    if flow_summary.get("duration_vs_equity") is not None:
        flow_parts.append(f"久期-股票 {flow_summary['duration_vs_equity']:.2f}%")
    if flow_parts:
        context_sections.append("资金流向：" + "；".join(flow_parts))

    sentiment = result.get("news_sentiment") or {}
    if sentiment.get("available"):
        context_sections.append(
            f"新闻情绪：{sentiment.get('label')} (avg {sentiment.get('avg_score')}, n={sentiment.get('sample_size')})"
        )

    options = result.get("options_metrics") or {}
    if options.get("available"):
        context_sections.append(
            f"期权温度计：Call IV {options.get('call_iv')}%，Put IV {options.get('put_iv')}%，Put/Call {options.get('put_call_ratio')}"
        )

    global_macro = result.get("global_macro_context") or {}
    summary = global_macro.get("summary")
    if summary:
        summary_text = "；".join(f"{k}:{v}" for k, v in summary.items())
        context_sections.append("全球宏观摘要：" + summary_text)

    validation = result.get("validation_report") or {}
    if validation.get("mean_sharpe") is not None:
        context_sections.append(
            f"OOS 验证：平均Sharpe {validation['mean_sharpe']:.2f}，平均CAGR {validation.get('mean_cagr', 0.0):.2f}"
        )

    context_block = "\n".join(context_sections) if context_sections else "当前回测尚未生成更多要点。"

    return (
        "你是一位资深的中文金融顾问，熟悉资产配置、风险对冲与交易执行。"
        "请结合提供的回测上下文，给出简洁、可执行的建议，并在回答中体现专业判断。"
        "可以引用数据，但不要机械地逐条罗列；当结论不确定时应明确说明不确定性。"
        "\n\n"
        f"回测背景：{'；'.join(info_bits)}\n"
        f"{context_block}\n"
        "回答时使用自然语言，逻辑清晰，围绕收益前景、关键依据与风险控制展开。"
    )


def _prepare_chat_messages(
    result: dict[str, Any],
    history: list[dict[str, Any]] | None,
    user_message: str | None,
) -> list[dict[str, str]]:
    """Build Ollama-compatible chat messages with system context and short history."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _build_advisor_system_prompt(result)}
    ]

    trimmed_history: list[dict[str, str]] = []
    if history:
        for item in history[-8:]:
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                trimmed_history.append({"role": role, "content": content})
    messages.extend(trimmed_history)

    normalized_message = (user_message or "").strip()
    if normalized_message:
        if not trimmed_history or trimmed_history[-1]["role"] != "user" or trimmed_history[-1]["content"] != normalized_message:
            messages.append({"role": "user", "content": normalized_message})
    elif not trimmed_history:
        messages.append({"role": "user", "content": "请基于当前回测给我下一步的操作建议。"})

    return messages


def _call_gemini_chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMIntegrationError("缺少 GEMINI_API_KEY 环境变量，无法调用 Gemini。")
    base_url = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
    url = f"{base_url}/v1beta/models/{model}:generateContent?key={api_key}"

    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role") or "user"
        text = (msg.get("content") or "").strip()
        if not text:
            continue
        part = {"text": text}
        if role == "system":
            system_parts.append(part)
            continue
        normalized_role = "user" if role == "user" else "model"
        contents.append({"role": normalized_role, "parts": [part]})

    payload: dict[str, Any] = {"contents": contents}
    if system_parts:
        payload["system_instruction"] = {"parts": system_parts}

    generation_config: dict[str, Any] = {
        "temperature": _env_float("GEMINI_TEMPERATURE", 0.35),
    }
    max_tokens = _env_int("GEMINI_MAX_TOKENS", 900)
    if max_tokens > 0:
        generation_config["maxOutputTokens"] = max_tokens
    payload["generationConfig"] = generation_config

    try:
        response = http_client.post(url, json=payload, timeout=timeout_seconds, retries=0)
        data = response.json()
    except HttpClientError as exc:
        raise LLMIntegrationError(f"Gemini 请求失败：{exc}")
    except ValueError:
        raise LLMIntegrationError("Gemini 返回了无法解析的响应")

    candidates = data.get("candidates") or []
    texts: list[str] = []
    for candidate in candidates:
        parts = (candidate.get("content") or {}).get("parts") or []
        for part in parts:
            txt = (part.get("text") or "").strip()
            if txt:
                texts.append(txt)
        if texts:
            break

    answer = "\n\n".join(texts).strip()
    if not answer:
        raise LLMIntegrationError("Gemini 未返回有效内容，请稍后重试。")

    return {"status": "ok", "answer": answer, "raw": data, "thoughts": []}


def _call_gemini_generate(
    model: str,
    prompt: str,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    """
    Simpler单轮生成接口，用于非对话模式（报告生成）。
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMIntegrationError("缺少 GEMINI_API_KEY 环境变量，无法调用 Gemini。")
    base_url = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
    url = f"{base_url}/v1beta/models/{model}:generateContent?key={api_key}"
    payload: dict[str, Any] = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]},
        ],
        "generationConfig": {
            "temperature": _env_float("GEMINI_TEMPERATURE", 0.35),
            "maxOutputTokens": _env_int("GEMINI_MAX_TOKENS", 1200),
        },
    }
    try:
        response = http_client.post(url, json=payload, timeout=timeout_seconds, retries=0)
        data = response.json()
    except HttpClientError as exc:
        raise LLMIntegrationError(f"Gemini 请求失败：{exc}")
    except ValueError:
        raise LLMIntegrationError("Gemini 返回了无法解析的响应")

    candidates = data.get("candidates") or []
    texts: list[str] = []
    for candidate in candidates:
        parts = (candidate.get("content") or {}).get("parts") or []
        for part in parts:
            txt = (part.get("text") or "").strip()
            if txt:
                texts.append(txt)
        if texts:
            break
    answer = "\n\n".join(texts).strip()
    if not answer:
        raise LLMIntegrationError("Gemini 未返回有效内容，请稍后重试。")
    return {"status": "ok", "answer": answer, "raw": data, "thoughts": []}

def _build_cost_profile(start_ts: float, end_ts: float, *, streaming: bool = False) -> dict[str, Any]:
    return {
        "total_sec": round(max(0.0, end_ts - start_ts), 3),
        "streaming": streaming,
    }

def _usage_from_raw(raw: Any) -> dict[str, int]:
    """Extract token usage if provider返回 usageMetadata 或类似字段。"""
    if not isinstance(raw, dict):
        return {}
    usage = raw.get("usage") or raw.get("usageMetadata") or {}
    prompt = usage.get("promptTokenCount") or usage.get("prompt_tokens")
    output = usage.get("candidatesTokenCount") or usage.get("completion_tokens") or usage.get("output_tokens")
    total = usage.get("totalTokenCount") or usage.get("total_tokens")
    tokens: dict[str, int] = {}
    if isinstance(prompt, int):
        tokens["prompt"] = prompt
    if isinstance(output, int):
        tokens["output"] = output
    if isinstance(total, int):
        tokens["total"] = total
    return tokens

def _call_gemini_stream(
    model: str,
    messages: list[dict[str, str]],
    *,
    timeout_seconds: int,
) -> tuple[str, list[str]]:
    """
    Stream responses from Gemini (:streamGenerateContent). Returns (answer, thoughts).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMIntegrationError("缺少 GEMINI_API_KEY 环境变量，无法调用 Gemini。")
    base_url = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com").rstrip("/")
    url = f"{base_url}/v1beta/models/{model}:streamGenerateContent?key={api_key}"
    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role") or "user"
        text = (msg.get("content") or "").strip()
        if not text:
            continue
        part = {"text": text}
        if role == "system":
            system_parts.append(part)
            continue
        normalized_role = "user" if role == "user" else "model"
        contents.append({"role": normalized_role, "parts": [part]})
    payload: dict[str, Any] = {"contents": contents}
    if system_parts:
        payload["system_instruction"] = {"parts": system_parts}
    payload["generationConfig"] = {
        "temperature": _env_float("GEMINI_TEMPERATURE", 0.35),
        "maxOutputTokens": _env_int("GEMINI_MAX_TOKENS", 1200),
    }
    try:
        resp = http_client.post(url, json=payload, timeout=timeout_seconds, retries=0, stream=True)
    except HttpClientError as exc:
        raise LLMIntegrationError(f"Gemini 流式请求失败：{exc}")
    buffer: list[str] = []
    thoughts: list[str] = []
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        try:
            data = json.loads(raw_line)
        except Exception:
            continue
        candidates = data.get("candidates") or []
        for cand in candidates:
            parts = (cand.get("content") or {}).get("parts") or []
            for part in parts:
                txt = (part.get("text") or "").strip()
                if txt:
                    buffer.append(txt)
            if cand.get("thinking"):
                if isinstance(cand["thinking"], str):
                    thoughts.append(cand["thinking"])
                elif isinstance(cand["thinking"], list):
                    for t in cand["thinking"]:
                        if isinstance(t, str):
                            thoughts.append(t)
        # short-circuit if stop reason appears
        if buffer and (data.get("finishReason") or data.get("finish_reason")):
            break
    answer = "\n".join(buffer).strip()
    if not answer:
        raise LLMIntegrationError("Gemini 流式未返回有效内容，请稍后重试。")
    return answer, thoughts


def _emit_streaming_deltas(answer: str, progress_callback: Callable[[str, dict[str, Any]], None] | None, *, segment_chars: int = 200) -> None:
    """Split answer into small chunks and send as incremental deltas."""
    if not progress_callback or not answer:
        return
    text = answer.strip()
    if not text:
        return
    chunks = []
    idx = 0
    step = max(80, segment_chars)
    while idx < len(text):
        chunks.append(text[idx : idx + step])
        idx += step
    for chunk in chunks:
        try:
            progress_callback("delta", {"text": chunk})
        except Exception:
            break


def _call_ollama_chat(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any] | None,
    *,
    timeout_seconds: int,
    retries: int,
    keep_alive: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options
    if keep_alive:
        payload["keep_alive"] = keep_alive

    last_exc: Exception | None = None
    tries = max(1, retries)
    for attempt in range(tries):
        try:
            response = http_client.post(
                endpoint,
                json=payload,
                timeout=timeout_seconds,
                retries=0,
            )
            data = response.json()
            return {
                "status": "ok",
                "data": data,
                "message": data.get("message") or {},
            }
        except (HttpClientError, ValueError) as exc:
            last_exc = exc
        time.sleep(1.5 * (2 ** attempt))

    return {
        "status": f"error: {last_exc}" if last_exc else "error",
        "data": {},
        "message": {},
        "error": str(last_exc) if last_exc else "unknown error",
    }


def _parse_chat_message(message: dict[str, Any]) -> tuple[str, list[str]]:
    """Extract final answer text and reasoning segments from a chat response."""
    if not isinstance(message, dict):
        return "", []

    reasoning: list[str] = []
    outputs: list[str] = []
    content = message.get("content")

    def push_text(target: list[str], text: str) -> None:
        text = (text or "").strip()
        if text:
            target.append(text)

    if isinstance(content, str):
        push_text(outputs, content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                push_text(outputs, item)
            elif isinstance(item, dict):
                text = (item.get("text") or item.get("content") or "").strip()
                typ = (item.get("type") or "").lower()
                if typ in {"thinking", "thought", "reasoning"}:
                    push_text(reasoning, text)
                elif typ not in {"tool_call", "tool"}:
                    push_text(outputs, text)
    elif isinstance(content, dict):
        push_text(outputs, content.get("text") or content.get("content") or "")

    # Some models place reasoning separately
    extra_reasoning = message.get("thinking") or message.get("reasoning")
    if isinstance(extra_reasoning, str):
        push_text(reasoning, extra_reasoning)
    elif isinstance(extra_reasoning, list):
        for item in extra_reasoning:
            if isinstance(item, str):
                push_text(reasoning, item)
            elif isinstance(item, dict):
                push_text(reasoning, item.get("text") or item.get("content") or "")

    # If still no output, fall back to response/raw keys
    if not outputs:
        fallback = message.get("response") or message.get("raw") or ""
        push_text(outputs, fallback)

    return "\n\n".join(outputs).strip(), reasoning

def generate_ai_commentary(
    result: dict[str, Any],
    show_thoughts: bool = False,
    user_message: str | None = None,
    history: list[dict[str, str]] | None = None,
    enable_web: bool | None = None,
    web_query: str | None = None,
    web_max_results: int | None = None,
    profile: bool | None = None,
    model_name: str | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Run two-model Ollama analysis (Primary: DeepSeek, Secondary: Qwen3) and aggregate outputs.

    Defaults:
      - OLLAMA_MODEL=deepseek-r1:8b
      - OLLAMA_SECONDARY_MODEL=qwen3:8b
    """

    def emit(stage: str, **payload: Any) -> None:
        if progress_callback:
            try:
                progress_callback(stage, payload)
            except Exception:
                pass

    emit("stage", message="开始整理回测上下文")

    selected_model = (model_name or result.get("ai_model") or "").strip()
    provider = _resolve_provider(selected_model)
    if provider == "gemini":
        primary_model = selected_model or os.environ.get("GEMINI_MODEL", "gemini-3.0-pro")
        endpoint = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com")
        secondary_model = ""
        secondary_endpoint = ""
        primary_options = {}
        secondary_options = {}
    else:
        primary_model = selected_model or os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
        endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
        secondary_model = os.environ.get("OLLAMA_SECONDARY_MODEL", "qwen3:8b")
        secondary_endpoint = os.environ.get("OLLAMA_SECONDARY_ENDPOINT", endpoint)
        # Build options per model
        primary_options = _build_ollama_options(is_secondary=False)
        secondary_options = _build_ollama_options(is_secondary=True)

    if isinstance(result, dict):
        result["ai_model"] = primary_model
        choices = result.get("ai_model_choices")
        if isinstance(choices, list) and primary_model not in choices:
            choices.append(primary_model)

    thinking_log: List[Dict[str, Any]] = []
    history = history or []
    profile_final = profile if profile is not None else _env_bool("AI_ENABLE_PROFILE", False)

    # baseline start time
    t0 = time.time()

    # (Optional) Build web context using Web Search
    enable_web_final = _default_enable_web(provider, enable_web)
    web_note: str | None = None
    web_results: list[dict[str, Any]] = []
    timeframe_hint = _extract_timeframe_months(user_message or "")
    if enable_web_final:
        emit("stage", message="正在检索实时资讯")
        q_raw = (web_query or (user_message or "")).strip()
        if not q_raw:
            q_raw = " ".join([str(x) for x in (result.get('ticker'), result.get('end_date') or '', '最新 新闻') if x])
        max_results = web_max_results if web_max_results is not None else _env_int("OLLAMA_WEB_MAX_RESULTS", 0)
        search_queries: list[str] = [q_raw]
        anchor_date = _parse_date_safe(result.get("end_date")) or date.today()
        if timeframe_hint:
            search_queries.append(f"{q_raw} 近{timeframe_hint}个月 新闻")
            for label in _generate_month_labels(anchor_date, timeframe_hint):
                search_queries.append(f"{q_raw} {label}")
        query_bundle: list[str] = []
        for candidate in search_queries:
            candidate = candidate.strip()
            if candidate and candidate not in query_bundle:
                query_bundle.append(candidate)
        market_ctx = _cached_web_search(query_bundle, max_results=max_results)
        hits = len(market_ctx.get('news') or [])
        if hits:
            source = market_ctx.get("source") or ""
            host = urlsplit(source).netloc if source else ""
            src_label = host or (source or "实时接口")
            span_hint = f"，覆盖近 {timeframe_hint} 个月" if timeframe_hint else ""
            web_note = f"联网检索（{src_label}）已注入 {hits} 条资讯{span_hint}"
        else:
            detail = market_ctx.get("message") or market_ctx.get("error") or "未获取到实时资讯"
            web_note = f"联网检索失败：{detail}"
        web_results = _sort_news(market_ctx.get('news') or [])
        # merge into result
        result = dict(result)
        if timeframe_hint:
            result["web_timeframe_hint"] = timeframe_hint
        prev_ctx = result.get('market_context') or {}
        if isinstance(prev_ctx, dict) and prev_ctx:
            merged = dict(prev_ctx)
            if market_ctx.get('news'):
                merged['news'] = (prev_ctx.get('news') or []) + web_results
            merged.setdefault('interest_terms', []).append(q_raw)
            result['market_context'] = merged
        else:
            result['market_context'] = market_ctx
        if web_results:
            result["web_news_digest"] = _build_news_digest(web_results)
        thinking_log.append({
            "model": "web.search",
            "status": "ok" if (market_ctx.get("news")) else "empty",
            "thoughts": [],
            "answer": "",
            "raw": f"query={q_raw}; results={len(market_ctx.get('news', []))}",
        })
        t_web_end = time.time()
        emit("stage", message="实时资讯处理完成", hits=len(web_results))
    else:
        t_web_end = time.time()

    # Decide mode: report vs Q&A
    is_question = bool((user_message or "").strip())
    if is_question:
        messages = _prepare_chat_messages(result, history, user_message)
        t_chat0 = time.time()
        stream_segment_chars = _env_int("AI_STREAM_SEGMENT_CHARS", 200)
        chat_timeout_base = _env_int("OLLAMA_CHAT_TIMEOUT_SECONDS", _env_int("OLLAMA_TIMEOUT_SECONDS", 60))
        stream_timeout = _env_int("AI_STREAM_CHAT_TIMEOUT_SECONDS", chat_timeout_base)
        soft_timeout = _env_int("AI_STREAM_SOFT_TIMEOUT_SECONDS", max(5, min(stream_timeout, 15)))
        hard_timeout = _env_int("AI_STREAM_HARD_TIMEOUT_SECONDS", stream_timeout)

        def _run_chat_call(curr_provider: str, model: str, options: dict[str, Any] | None, streaming: bool = False) -> tuple[str, list[str], str, dict[str, Any]]:
            if curr_provider == "gemini":
                chat_timeout = _env_int("GEMINI_TIMEOUT_SECONDS", stream_timeout)
                if streaming and progress_callback:
                    try:
                        answer, thoughts = _call_gemini_stream(model, messages, timeout_seconds=chat_timeout)
                        return answer, thoughts, "ok", {}
                    except LLMIntegrationError as exc:
                        emit("progress", stage="fallback", message=f"Gemini 流式失败，改用非流式：{exc}")
                resp = _call_gemini_chat(model, messages, timeout_seconds=chat_timeout)
                answer = (resp.get("answer") or "").strip()
                return answer, resp.get("thoughts") or [], resp.get("status", "ok"), resp.get("raw", {})
            chat_endpoint_local = _resolve_chat_endpoint(endpoint)
            chat_timeout = stream_timeout
            chat_retries = _env_int("OLLAMA_CHAT_RETRIES", _env_int("OLLAMA_RETRIES", 2))
            keep_alive_val = os.environ.get("OLLAMA_KEEP_ALIVE")
            if streaming and progress_callback:
                answer, thoughts = _call_ollama_stream(
                    chat_endpoint_local,
                    model,
                    messages,
                    options or {},
                    timeout_seconds=chat_timeout,
                    keep_alive=keep_alive_val,
                )
                return answer, thoughts, "ok", {}
            resp = _call_ollama_chat(
                chat_endpoint_local,
                model,
                messages,
                options or {},
                timeout_seconds=chat_timeout,
                retries=chat_retries,
                keep_alive=keep_alive_val,
            )
            if resp.get("status") != "ok":
                raise LLMIntegrationError(resp.get("error") or "AI 服务暂不可用，请稍后重试。")
            msg_payload = resp.get("message") or {}
            answer, thoughts = _parse_chat_message(msg_payload)
            if not answer:
                fallback_response = (resp.get("data") or {}).get("response", "")
                answer = (fallback_response or "抱歉，目前无法给出明确的建议，请稍后再试。").strip()
            return answer, thoughts, resp.get("status", "ok"), resp.get("data", {})

        emit("stage", message="调用主模型", model=primary_model, provider=provider)
        try:
            answer_text, reasoning_segments, chat_status, raw_payload = _run_chat_call(provider, primary_model, primary_options, streaming=bool(progress_callback))
            t_chat1 = time.time()
        except LLMIntegrationError as exc:
            fallback_used = False
            if provider == "gemini":
                emit("stage", message="主模型失败，回退至本地模型", fallback=os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b"))
                fallback_model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
                try:
                    answer_text, reasoning_segments, chat_status, raw_payload = _run_chat_call("ollama", fallback_model, primary_options, streaming=bool(progress_callback))
                    t_chat1 = time.time()
                    fallback_used = True
                    provider = "ollama"
                    primary_model = fallback_model
                except LLMIntegrationError:
                    raise exc
            else:
                raise exc

        if not answer_text:
            raise LLMIntegrationError("AI 服务未返回有效内容，请稍后重试。")

        chat_entry = {
            "model": primary_model,
            "provider": provider,
            "status": chat_status,
            "thoughts": reasoning_segments,
            "answer": answer_text,
            "raw": raw_payload,
        }

        answer_text = _prepend_web_digest(answer_text, web_results)
        # 优先使用真实流式，否则退回分段推送
        if progress_callback:
            if chat_status == "ok" and answer_text:
                _emit_streaming_deltas(answer_text, progress_callback, segment_chars=stream_segment_chars)

        tokens = _usage_from_raw(raw_payload)
        ret: dict[str, Any] = {
            "answer": answer_text,
            "models": [
                {"name": primary_model, "status": chat_status, "provider": provider},
            ],
        }
        ret["selected_model"] = primary_model
        ret["web_used"] = bool(enable_web_final)
        if web_note:
            ret["web_note"] = web_note
        if web_results:
            ret["web_results"] = web_results
        if show_thoughts:
            ret["thinking"] = thinking_log + [chat_entry]
        else:
            ret["thinking"] = []

        if profile_final:
            ret["profile"] = {
                "primary_sec": round((t_chat1 - t_chat0), 3),
                "total_sec": round((t_chat1 - t0), 3),
                "secondary_sec": 0.0,
                "web_search_sec": round((t_web_end - t0), 3) if enable_web_final else 0.0,
            }
            if tokens:
                ret["profile"]["tokens"] = tokens

        emit("done", message="AI 解读完成", model=primary_model, provider=provider)
        return ret

    prompt_primary = build_prompt(result, show_thoughts=True, user_message=user_message, history=history)
    if provider == "gemini":
        primary_timeout = _env_int("GEMINI_TIMEOUT_SECONDS", _env_int("OLLAMA_TIMEOUT_SECONDS", 60))
        t_p0 = time.time()
        primary = _call_gemini_generate(primary_model, prompt_primary, timeout_seconds=primary_timeout)
        t_p1 = time.time()
        tokens = _usage_from_raw(primary.get("raw", {}))
        thinking_log.append({
            "model": primary_model,
            "provider": provider,
            "status": primary.get("status", "ok"),
            "thoughts": primary.get("thoughts", []),
            "answer": primary.get("answer", ""),
            "raw": primary.get("raw", ""),
        })
        final_answer = primary.get("answer", "")
        final_answer = _prepend_web_digest(final_answer, web_results)
        compact = _format_compact_answer(final_answer, context=result)
        ret: dict[str, Any] = {
            "answer": compact,
            "thinking": thinking_log if show_thoughts else [],
            "models": [
                {"name": primary_model, "status": primary.get("status", "ok"), "provider": provider},
            ],
        }
        ret["selected_model"] = primary_model
        ret["web_used"] = bool(enable_web_final)
        if web_note:
            ret["web_note"] = web_note
        if web_results:
            ret["web_results"] = web_results
        if profile_final:
            ret["profile"] = {
                "web_search_sec": round((t_web_end - t0), 3) if 't_web_end' in locals() else 0.0,
                "primary_sec": round((t_p1 - t_p0), 3),
                "secondary_sec": 0.0,
                "total_sec": round((t_p1 - t0), 3),
            }
            if tokens:
                ret["profile"]["tokens"] = tokens
        emit("done", message="AI 解读完成", model=primary_model, provider=provider)
        return ret

    primary_timeout = _env_int("OLLAMA_TIMEOUT_SECONDS", 60)
    primary_retries = _env_int("OLLAMA_RETRIES", 2)
    t_p0 = time.time()
    primary = _call_ollama(endpoint, primary_model, prompt_primary, primary_options, timeout_seconds=primary_timeout, retries=primary_retries)
    t_p1 = time.time()
    thinking_log.append({
        "model": primary_model,
        "status": primary.get("status", "ok"),
        "thoughts": primary.get("thoughts", []),
        "answer": primary.get("answer", ""),
        "raw": primary.get("raw", ""),
    })

    followup_prompt = build_followup_prompt(result, primary, user_message=user_message, history=history)
    secondary_timeout = _env_int("OLLAMA_SECONDARY_TIMEOUT_SECONDS", _env_int("OLLAMA_TIMEOUT_SECONDS", 75))
    secondary_retries = _env_int("OLLAMA_SECONDARY_RETRIES", 1)
    t_s0 = time.time()
    secondary = _call_ollama(secondary_endpoint, secondary_model, followup_prompt, secondary_options, timeout_seconds=secondary_timeout, retries=secondary_retries)
    t_s1 = time.time()
    thinking_log.append({
        "model": secondary_model,
        "status": secondary.get("status", "ok"),
        "thoughts": secondary.get("thoughts", []),
        "answer": secondary.get("answer", ""),
        "raw": secondary.get("raw", ""),
    })

    final_answer = (
        (secondary.get("answer") or "").strip()
        or (primary.get("answer") or "").strip()
        or (secondary.get("error") or primary.get("error") or "AI 未返回有效内容，请稍后重试。")
    )
    final_answer = _prepend_web_digest(final_answer, web_results)

    if is_question:
        compact = _format_qa_answer(final_answer, user_question=(user_message or ""))
    else:
        compact = _format_compact_answer(final_answer, context=result)

    ret: dict[str, Any] = {
        "answer": compact,
        "thinking": thinking_log if show_thoughts else [],
        "models": [
            {"name": primary_model, "status": primary.get("status", "ok")},
            {"name": secondary_model, "status": secondary.get("status", "ok")},
        ],
    }
    ret["selected_model"] = primary_model
    ret["web_used"] = bool(enable_web_final)
    if web_note:
        ret["web_note"] = web_note
    if web_results:
        ret["web_results"] = web_results
    if profile_final:
        ret["profile"] = {
            "web_search_sec": round((t_web_end - t0), 3) if 't_web_end' in locals() else 0.0,
            "primary_sec": round((t_p1 - t_p0), 3) if 't_p1' in locals() and 't_p0' in locals() else 0.0,
            "secondary_sec": round((t_s1 - t_s0), 3) if 't_s1' in locals() and 't_s0' in locals() else 0.0,
            "total_sec": round(((t_s1 if 't_s1' in locals() else time.time()) - t0), 3),
        }
    emit("done", message="AI 解读完成", model=primary_model)
    return ret


def _call_ollama(endpoint: str, model: str, prompt: str, options: dict[str, Any], *, timeout_seconds: int, retries: int) -> dict[str, Any]:
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}

    # Simple retry with exponential backoff
    last_exc: Exception | None = None
    tries = max(1, retries)
    for attempt in range(tries):
        try:
            response = http_client.post(
                endpoint,
                json=payload,
                timeout=timeout_seconds,
                retries=0,
            )
            try:
                data = response.json()
            except ValueError as exc:  # invalid json
                last_exc = exc
                time.sleep(1.5 * (2 ** attempt))
                continue
            text = (data or {}).get("response", "").strip()
            thoughts, answer = _split_thoughts_and_answer(text)
            return {
                "model": model,
                "status": "ok",
                "thoughts": thoughts,
                "answer": answer,
                "raw": text,
                "error": "",
            }
        except HttpClientError as exc:
            last_exc = exc
        time.sleep(1.5 * (2 ** attempt))

    # Failed after retries
    return {
        "model": model,
        "status": f"error: {last_exc}" if last_exc else "error",
        "thoughts": [],
        "answer": "",
        "error": str(last_exc) if last_exc else "unknown error",
    }


def _call_ollama_stream(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any] | None,
    *,
    timeout_seconds: int,
    keep_alive: str | None = None,
) -> tuple[str, list[str]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if options:
        payload["options"] = options
    if keep_alive:
        payload["keep_alive"] = keep_alive
    try:
        resp = http_client.post(
            endpoint,
            json=payload,
            timeout=timeout_seconds,
            retries=0,
            stream=True,
        )
    except HttpClientError as exc:
        raise LLMIntegrationError(f"Ollama 流式请求失败：{exc}")
    buffer: list[str] = []
    thoughts: list[str] = []
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        msg = data.get("message") or data.get("delta") or {}
        content = msg.get("content") or ""
        if content:
            buffer.append(content)
        if data.get("done"):
            break
    answer = "".join(buffer).strip()
    if not answer:
        raise LLMIntegrationError("Ollama 流式未返回有效内容，请稍后重试。")
    return answer, thoughts


def _split_thoughts_and_answer(text: str) -> tuple[list[str], str]:
    """Robustly split thoughts vs. conclusion across common LLM formats."""
    if not text:
        return [], ""

    # 1) <think> ... </think>
    think_block = re.findall(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE)
    if think_block:
        thoughts = [line.strip("-• 。") for line in think_block[0].splitlines() if line.strip()]
        answer = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
        return thoughts, answer

    # 2) Fenced code blocks ```thinking ...```
    fence = re.findall(r"```(?:thinking|think|reasoning)?\n([\s\S]*?)\n```", text, flags=re.IGNORECASE)
    if fence:
        thoughts = [line.strip("-• 。") for line in fence[0].splitlines() if line.strip()]
        answer = re.sub(r"```(?:thinking|think|reasoning)?[\s\S]*?```", "", text, flags=re.IGNORECASE).strip()
        return thoughts, answer

    # 3) 中文分段：思考过程 / 结论
    thought_pattern = re.compile(r"思考过程[:：]\s*", re.IGNORECASE)
    conclusion_pattern = re.compile(r"结论[:：]\s*", re.IGNORECASE)
    if "思考" in text and "结论" in text:
        parts = conclusion_pattern.split(text, maxsplit=1)
        if len(parts) == 2:
            thought_section = thought_pattern.split(parts[0], maxsplit=1)[-1]
            conclusion_section = parts[1]
            thoughts = [line.strip("-• 。") for line in thought_section.splitlines() if line.strip()]
            answer = "结论:\n" + conclusion_section.strip()
            return thoughts, answer.strip()

    # Fallback: no explicit separation
    return [], text.strip()


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 100)].rstrip() + "\n...（内容过长，已截断）"


def build_prompt(
    result: dict[str, Any],
    show_thoughts: bool = True,
    user_message: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    metrics_lines = "\n".join(
        f"- {item['label']}: {item['value']}" for item in result.get("metrics", [])
    )
    benchmark_ticker = result.get("benchmark_ticker")
    benchmark_metrics = result.get("benchmark_metrics") or []
    benchmark_lines = "\n".join(f"- {item['label']}: {item['value']}" for item in benchmark_metrics)
    risk_profile = result.get("risk_profile", "未指定")
    capital = result.get("capital")
    capital_line = f"可支配资金: {capital:,.0f}" if capital else "可支配资金: 未提供"
    recommendations = result.get("recommendations") or []
    plans_lines = "\n".join(
        f"- {plan['title']}: {plan['actions']}"
        for plan in recommendations[:3]
    ) or "- 暂无方案"

    quick_summary_lines = "\n".join(f"- {item}" for item in result.get("quick_summary", [])) or "- 无"
    action_plan = result.get("action_plan", [])
    action_plan_lines = "\n".join(
        f"- {step['title']}（{step.get('priority', '中')}优先）: {step['detail']}"
        for step in action_plan
    ) or "- 无"
    risk_alerts = result.get("risk_alerts", [])
    risk_alerts_lines = "\n".join(f"- {alert}" for alert in risk_alerts) or "- 无"
    education_tips = result.get("education_tips", [])
    education_lines = "\n".join(f"- {tip}" for tip in education_tips) or "- 无"
    horizon_label = result.get("investment_horizon_label", "")
    experience_label = result.get("experience_label", "")
    goal_label = result.get("primary_goal_label", "")
    confidence_label = result.get("confidence_label", "")

    timeframe_hint = result.get("web_timeframe_hint")
    market_context = result.get("market_context") or {}
    market_items = market_context.get("news", []) if isinstance(market_context, dict) else []
    now_label = _now_utc_label()
    market_lines = "\n".join(
        f"- [{item.get('retrieved_at') or market_context.get('retrieved_at') or now_label}] "
        f"{item.get('title', '资讯')}（{item.get('host') or urlsplit(item.get('url') or '').netloc or '未知来源'}）: "
        f"{item.get('snippet', '')} <{item.get('url', '无链接')}>"
        for item in market_items
    ) or market_context.get("message", "暂无外部资讯")
    related_tickers = ", ".join(market_context.get("tickers", []) or [])
    interest_terms = market_context.get("interest_terms", []) or []
    interest_line = ", ".join(interest_terms) if interest_terms else "未填写"

    recent_rows = result.get("recent_rows", [])[-5:]
    signal_lines = "\n".join(
        f"- {row['date']}: 持仓={row['position']}, 收盘={row['adj_close']}, 短期均线={row['sma_short']}, 长期均线={row['sma_long']}, RSI={row['rsi']}"
        for row in recent_rows
    ) or "- 无历史数据"

    ticker = result.get("ticker", "Unknown")
    start_date = result.get("start_date", "")
    end_date = result.get("end_date", "")

    executive_cards = result.get("executive_briefing") or []
    executive_lines = "\n".join(
        f"- {card.get('title')}: {card.get('status', '—')} | {card.get('body', '')}"
        for card in executive_cards
    ) or "- 暂无执行摘要"
    playbook_sections = (result.get("advisor_playbook") or {}).get("sections", [])
    playbook_lines = "\n".join(
        f"- {section.get('title')}: " + "；".join(section.get("points", [])[:3])
        for section in playbook_sections
    ) or "- 暂无投顾提要"

    user_questions = result.get("user_questions") or []
    user_q_lines = "\n".join(
        f"- {qa.get('question')}: {qa.get('answer')}"
        for qa in user_questions
    ) or "- 已为用户生成四大问题的标准解答"

    model_weights = result.get("model_weights") or {}
    if model_weights.get("available"):
        weights_lines = "; ".join(
            f"{item['name']} {item['weight']:.2f}"
            for item in model_weights.get("allocations", [])[:4]
        )
    else:
        weights_lines = "暂无权重建议"

    risk_dashboard = result.get("risk_dashboard") or {}
    risk_level = risk_dashboard.get("risk_level", "未知")
    risk_insight = risk_dashboard.get("insight", "")

    macro_bundle = result.get("macro_bundle") or {}

    def _format_macro_line(entry: dict[str, Any]) -> str:
        label = entry.get("label") or ""
        short = entry.get("short") or ""
        if label and short:
            name = f"{label} ({short})"
        else:
            name = label or short or "宏观指标"
        if not entry.get("available"):
            detail = entry.get("message") or "—"
        else:
            latest = entry.get("latest", "—")
            change = entry.get("change_21d", "—")
            detail = f"最新 {latest} | 21日 {change}%"
        return f"- {name}: {detail}"

    macro_lines = "\n".join(
        _format_macro_line(entry)
        for entry in macro_bundle.values()
        if isinstance(entry, dict)
    ) or "- 无宏观数据"

    scenario_bundle = result.get("scenario_simulation") or {}
    if scenario_bundle.get("available"):
        scenario_lines = "\n".join(
            f"- {item.get('label')}: {item.get('return')} → {item.get('description')}"
            for item in scenario_bundle.get("scenarios", [])[:3]
        )
    else:
        scenario_lines = "- 暂无情景模拟"

    opportunity_bundle = result.get("opportunity_radar") or {}
    if opportunity_bundle.get("available"):
        leaders = opportunity_bundle.get("leaders", [])
        laggards = opportunity_bundle.get("laggards", [])
        summary_text = opportunity_bundle.get("summary", "")
        opportunity_lines = (
            f"- 领先: {', '.join(item['ticker'] for item in leaders[:3]) or '暂无'}\n"
            f"- 回调: {', '.join(item['ticker'] for item in laggards[:3]) or '暂无'}\n"
            f"- 结论: {summary_text[:120]}"
        )
    else:
        opportunity_lines = "- 暂无机会推荐"

    event_bundle = result.get("event_bundle") or []
    event_lines = "\n".join(
        f"- {', '.join(event.get('types', []))}: {event.get('title')}"
        for event in event_bundle[:5]
    ) or "- 最近暂无重大事件"

    factor_effectiveness = result.get("factor_effectiveness") or {}
    factor_ic = factor_effectiveness.get("composite")
    factor_top = ", ".join(
        item["name"] for item in factor_effectiveness.get("top_factors", [])[:3]
    ) if factor_effectiveness.get("available") else ""
    factor_line = (
        f"平均 IC={factor_ic:.2f}，重点因子：{factor_top}"
        if factor_effectiveness.get("available")
        else "因子信号无显著优势或样本不足"
    )

    deep_bundle = result.get("deep_signal_bundle") or {}
    deep_line = (
        f"{deep_bundle.get('best_model', '深度模型')} 准确率 {deep_bundle.get('accuracy', 0.0)*100:.1f}%"
        if deep_bundle else "深度信号暂不可用"
    )

    history = history or []
    history_lines = "\n".join(
        f"- {entry['role']}: {entry['content']}" for entry in history[-6:]
    )

    return (
        "你是资深量化投顾，需要用通俗中文向非专业用户解释这次回测，并给出可执行的投资方案。\n"
        "请结合机器学习、深度模型、因子分析、宏观与知识图谱等模块的结果，重点回答以下问题：能否赚钱？怎么投？风险点？下一步做什么？\n"
        "要求：\n"
        "- 文字务必接地气，少用专业术语，必要时给出简短解释；\n"
        "- 每个段落只写 3 条要点，以“- ”开头；\n"
        "- 要把“能赚多少”“怎么调仓”“风险怎么控”“下一步”讲清楚，并给出时间或阈值；\n"
        "- 不要罗列大段指标原文，优先整合并解释它们的含义；\n"
        "- 若依据不足要直说“证据不足”；\n"
        "- 若上方“实时资讯引用”存在内容，务必引用至少两条并标注来源/日期，特别关注用户指定的时间范围；如资讯为空则明确处于离线模式。\n"
        "- 除指定标题外不得添加其他标题、表格或代码块。\n"
        "输出格式（固定 4 段，每段 3 行）：\n"
        "### 盈利空间\n- ...\n- ...\n- ...\n"
        "### 操作路线\n- ...\n- ...\n- ...\n"
        "### 风险防守\n- ...\n- ...\n- ...\n"
        "### 下一步\n- ...\n- ...\n- ...\n"
        f"当前真实日期（UTC）: {now_label}\n"
        f"标的: {ticker}\n时间窗口: {start_date} 至 {end_date}\n{capital_line}\n"
        f"风险偏好: {risk_profile}\n"
        f"用户画像: 经验水平={experience_label or '未填写'}, 投资期限={horizon_label or '未填写'}, 核心目标={goal_label or '未填写'}, 信心等级={confidence_label or '未知'}\n"
        f"策略关键指标:\n{metrics_lines}\n"
        f"基准指标:\n{benchmark_lines if benchmark_lines else '无'}\n"
        f"执行摘要:\n{executive_lines}\n"
        f"投顾提要:\n{playbook_lines}\n"
        f"用户核心问题与标准回答:\n{user_q_lines}\n"
        f"系统自动概要:\n{quick_summary_lines}\n"
        f"执行清单:\n{action_plan_lines}\n"
        f"风险提醒:\n{risk_alerts_lines}\n"
        f"教育提示:\n{education_lines}\n"
        f"潜在关联资产/行业: {related_tickers or '暂无'} | 用户关注关键词: {interest_line}\n"
        f"模型权重建议: {weights_lines}\n"
        f"风险雷达: 等级={risk_level} | 解读={risk_insight}\n"
        f"深度信号: {deep_line}\n"
        f"因子有效性: {factor_line}\n"
        f"宏观重点: {result.get('macro_highlight', '暂无宏观提示')}\n"
        f"近期信号摘要:\n{signal_lines}\n"
        f"系统生成的策略方案:\n{plans_lines}\n"
        f"市场资讯与备选标的:\n{market_lines}\n"
        f"联网摘要:\n{result.get('web_news_digest', '—')}\n"
        f"时间范围提示: {('需覆盖近 ' + str(timeframe_hint) + ' 个月') if timeframe_hint else '未指定'}\n"
        f"宏观脉搏:\n{macro_lines}\n"
        f"情景模拟:\n{scenario_lines}\n"
        f"机会雷达:\n{opportunity_lines}\n"
        f"事件雷达:\n{event_lines}\n"
        f"投资者追加问题: {user_message if user_message else '无'}\n"
        f"近期对话历史:\n{history_lines if history_lines else '- 无历史对话'}\n"
        "请在结论部分给出分步骤的执行指引（买入/减仓/对冲/风控阈值）。"
    )


def build_followup_prompt(
    result: dict[str, Any],
    primary: dict[str, Any],
    user_message: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    max_chars = _env_int("OLLAMA_FOLLOWUP_MAX_CHARS", 2400)
    max_thought_lines = _env_int("OLLAMA_FOLLOWUP_MAX_THOUGHT_LINES", 16)
    base_summary = _truncate_text(primary.get("answer", "") or "", max_chars)
    thoughts = primary.get("thoughts", []) or []
    primary_thoughts = "\n".join(thoughts[:max_thought_lines])

    model_weights = result.get("model_weights") or {}
    if model_weights.get("available"):
        weights_summary = "; ".join(
            f"{item['name']} {item['weight']:.2f}"
            for item in model_weights.get("allocations", [])[:4]
        )
    else:
        weights_summary = "暂无"

    risk_dashboard = result.get("risk_dashboard") or {}
    risk_line = f"风险等级={risk_dashboard.get('risk_level', '未知')} | {risk_dashboard.get('insight', '')}"

    executive_cards = result.get("executive_briefing") or []
    executive_lines = "\n".join(
        f"- {card.get('title')}: {card.get('status', '—')} | {card.get('body', '')}"
        for card in executive_cards[:4]
    ) or "- 无执行摘要"
    scenario_bundle = result.get("scenario_simulation") or {}
    scenario_line = "基线暂无" if not scenario_bundle.get("available") else (
        f"基线 {scenario_bundle['scenarios'][1]['return']} / 防守 {scenario_bundle['scenarios'][-1]['return']}"
        if len(scenario_bundle.get("scenarios", [])) >= 2
        else "基线 {scenario_bundle.get('scenarios', [{'return': '—'}])[0]['return']}"
    )
    opportunity_bundle = result.get("opportunity_radar") or {}
    opportunity_line = (
        ", ".join(item["ticker"] for item in opportunity_bundle.get("leaders", [])[:3])
        if opportunity_bundle.get("available")
        else "暂无"
    )
    playbook_sections = (result.get("advisor_playbook") or {}).get("sections", [])
    playbook_summary = "; ".join(
        f"{section.get('title')}: " + "、".join(section.get("points", [])[:2])
        for section in playbook_sections[:3]
    ) or "暂无"
    return (
        "你是第二位投顾，需要在第一位分析师结论基础上进行复核与强化，确保建议易懂且可执行。\n"
        "请重点围绕“能赚多少、怎么投、风险怎么控、下一步”四个问题，结合模型权重与风险雷达给出明确操作。\n"
        "输出格式仍为四段（盈利空间/操作路线/风险防守/下一步），每段 3 条。若发现前一结论不足，请直接指出并修正。\n"
        f"第一位思考要点:\n{primary_thoughts or '无'}\n"
        f"第一位结论摘要:\n{base_summary or '无'}\n"
        f"执行摘要卡片:\n{executive_lines}\n"
        f"投顾提要: {playbook_summary}\n"
        f"当前模型权重: {weights_summary}\n"
        f"风险雷达: {risk_line}\n"
        f"宏观提示: {result.get('macro_highlight', '—')}\n"
        f"情景基线: {scenario_line}\n"
        f"热门机会: {opportunity_line}\n"
        f"请输出改进后的最终方案。"
    )


def build_qa_prompt(
    result: dict[str, Any], user_message: str, history: list[dict[str, str]] | None = None
) -> str:
    history = history or []
    ticker = result.get("ticker", "Unknown")
    start_date = result.get("start_date", "")
    end_date = result.get("end_date", "")
    quick_summary = "\n".join(f"- {s}" for s in (result.get("quick_summary") or [])[:3]) or "- 无"
    risk_line = f"风险等级={result.get('risk_dashboard', {}).get('risk_level', '未知')}"
    weights_summary = "; ".join(
        f"{item['name']} {item['weight']:.2f}"
        for item in (result.get('model_weights', {}).get('allocations') or [])[:3]
    ) or "暂无"
    scenario_bundle = result.get("scenario_simulation") or {}
    scenario_line = (
        scenario_bundle.get("scenarios", [{}])[1].get("return", "—")
        if scenario_bundle.get("available") and len(scenario_bundle.get("scenarios", [])) >= 2
        else "—"
    )
    opportunity_bundle = result.get("opportunity_radar") or {}
    opportunity_line = (
        ", ".join(item["ticker"] for item in opportunity_bundle.get("leaders", [])[:2])
        if opportunity_bundle.get("available")
        else "暂无"
    )
    playbook_sections = (result.get("advisor_playbook") or {}).get("sections", [])
    playbook_line = (
        " | ".join(
            f"{section.get('title')}:" + "、".join(section.get("points", [])[:2])
            for section in playbook_sections[:3]
        )
        if playbook_sections
        else "暂无投顾提要"
    )
    market_context = result.get("market_context") or {}
    market_items = market_context.get("news") or []
    now_label = _now_utc_label()
    if market_items:
        market_lines = "\n".join(
            f"- [{item.get('retrieved_at') or market_context.get('retrieved_at') or now_label}] "
            f"{item.get('title', '资讯')}（{item.get('host') or urlsplit(item.get('url') or '').netloc or '未知来源'}）: "
            f"{item.get('snippet', '')} <{item.get('url', '无链接')}>"
            for item in market_items[:5]
        )
    else:
        market_lines = "- 当前没有实时资讯，请明确告知用户需要先开启联网按钮或稍后重试。"
    timeframe_hint = result.get("web_timeframe_hint")
    news_digest = result.get("web_news_digest") or "- 无联网摘要"
    return (
        "你是一位资深金融顾问，在“答疑模式”下继续与客户对话。除系统自动生成的首段概览外，后续回复需以顾问视角给出精炼、可执行的建议，避免重复摘要或堆砌指标。\n"
        "请用 2-4 句自然语言说明收益预期、关键依据与执行步骤，保持专业而直接的语气，不要使用标题、项目符号或模板化寒暄。\n"
        "务必遵守：若下方提供了“实时资讯”列表，你只能引用列表中的事实，且需按“来源（URL）”格式说明，并覆盖指定时间范围；若列表为空时才可以说明未取到新闻。\n"
        f"标的: {ticker} 时间: {start_date}→{end_date}\n"
        f"当前真实日期（UTC）: {now_label}\n"
        f"系统概要:\n{quick_summary}\n"
        f"模型权重: {weights_summary} | {risk_line}\n"
        f"情景基线: {scenario_line} | 机会雷达: {opportunity_line}\n"
        f"投顾提要: {playbook_line}\n"
        f"实时资讯（仅可引用以下条目）:\n{market_lines}\n"
        f"联网摘要（便于交叉引用）:\n{news_digest}\n"
        f"时间范围要求：{('需覆盖近 ' + str(timeframe_hint) + ' 个月') if timeframe_hint else '未额外指定'}\n"
        f"用户问题: {user_message}\n"
        "禁止复制整段指标或原始数据，答案必须落地、易懂。"
    )


def build_qa_followup_prompt(
    result: dict[str, Any],
    primary: dict[str, Any],
    user_message: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Secondary model refines the primary Q&A answer."""
    history = history or []
    p_answer = primary.get("answer", "")
    p_thoughts = "\n".join(primary.get("thoughts", [])[:8])
    return (
        "你是第二位资深金融顾问，请审阅第一位的“答疑模式”回复，修正不准确处并补充关键遗漏。\n"
        "请继续以顾问口吻自然回应，用 2-4 句覆盖结论、关键依据与行动步骤，避免重复系统报告、指标罗列或与前文相同的句子。\n"
        f"第一位要点:\n{p_thoughts or '无'}\n"
        f"第一位结论:\n{p_answer or '无'}\n"
        f"用户问题: {user_message}\n"
        "请直接给出修订后的建议，确保语句顺畅。"
    )


def _format_compact_answer(text: str, *, context: dict[str, Any] | None = None) -> str:
    """Compact and normalize the final answer into 4 sections with 3 bullets each.

    - Keep only allowed sections in fixed order
    - Convert any free text into bullets
    - Trim bullet length and deduplicate
    - Enforce total length bound
    """
    if not text:
        return ""

    max_total = _env_int("AI_MAX_TOTAL_CHARS", 2600)
    max_bullets = _env_int("AI_MAX_BULLETS_PER_SECTION", 3)
    max_bullet_chars = _env_int("AI_MAX_BULLET_CHARS", 120)

    allowed = ["盈利空间", "操作路线", "风险防守", "下一步"]
    aliases = {
        "盈利机会": "盈利空间",
        "操作计划": "操作路线",
        "风险盲区": "风险防守",
        "额外关注": "下一步",
    }
    sections: dict[str, list[str]] = {name: [] for name in allowed}
    current: str | None = None

    def push_bullet(sec: str, content: str) -> None:
        content = re.sub(r"\s+", " ", content).strip("- •。;；:： ")
        if not content:
            return
        content = content[:max_bullet_chars]
        if content not in sections[sec]:
            if len(sections[sec]) < max_bullets:
                sections[sec].append(content)

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("### "):
            title = line[4:].strip()
            canonical = aliases.get(title, title)
            current = canonical if canonical in sections else None
            continue
        if current is None:
            continue
        if line.startswith("- "):
            push_bullet(current, line[2:])
        elif re.match(r"^\d+\.", line):
            push_bullet(current, re.sub(r"^\d+\.\s*", "", line))
        else:
            push_bullet(current, line)

    # If any section lacks bullets, fill from structured context as fallback
    if context:
        try:
            if not sections["盈利空间"]:
                src = context.get("key_takeaways") or context.get("quick_summary") or []
                for s in src:
                    push_bullet("盈利空间", str(s))
            if not sections["操作路线"]:
                for step in context.get("action_plan", []) or []:
                    push_bullet("操作路线", str(step.get("detail") or step.get("title") or ""))
            if not sections["风险防守"]:
                for s in context.get("risk_alerts", []) or []:
                    push_bullet("风险防守", str(s))
            if not sections["下一步"]:
                tips = context.get("education_tips", []) or []
                if tips:
                    for s in tips:
                        push_bullet("下一步", str(s))
                else:
                    mc = context.get("market_context") or {}
                    if isinstance(mc, dict):
                        if mc.get("analysis"):
                            push_bullet("下一步", str(mc.get("analysis")))
                        for item in (mc.get("news") or [])[:3]:
                            t = item.get("title") or "相关新闻"
                            push_bullet("下一步", str(t))
        except Exception:
            pass

    # Final guard: still missing → placeholder
    for name in allowed:
        if not sections[name]:
            sections[name] = ["证据不足，建议继续跟踪。"]

    lines: list[str] = []
    for name in allowed:
        lines.append(f"### {name}")
        for item in sections[name][:max_bullets]:
            lines.append(f"- {item}")

    compact = "\n".join(lines)
    return _truncate_text(compact, max_total)


def _format_qa_answer(text: str, *, user_question: str = "") -> str:
    """Compact Q&A answer into three short sections."""
    if not text:
        return ""
    max_total = _env_int("AI_QA_MAX_TOTAL_CHARS", 2000)
    max_bullets = min(3, _env_int("AI_QA_MAX_BULLETS", 3))
    max_bullet_chars = _env_int("AI_QA_MAX_BULLET_CHARS", 120)

    sections = {"快速答复": [], "依据/条件": [], "行动建议": []}

    def push(sec: str, s: str) -> None:
        s = re.sub(r"\s+", " ", s).strip("- •。;；:： ")
        if not s:
            return
        s = s[:max_bullet_chars]
        if len(sections[sec]) < max_bullets and s not in sections[sec]:
            sections[sec].append(s)

    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("### "):
            title = line[4:].strip()
            current = title if title in sections else None
            continue
        if current is None:
            # Treat free text as part of 快速答复
            current = "快速答复"
        if line.startswith("- "):
            push(current, line[2:])
        elif re.match(r"^\d+\.", line):
            push(current, re.sub(r"^\d+\.\s*", "", line))
        else:
            push(current, line)

    summary_parts: list[str] = []

    def join_section(label: str, prefix: str) -> None:
        if sections[label]:
            joined = "；".join(sections[label][:max_bullets])
            summary_parts.append(f"{prefix}{joined}")

    join_section("快速答复", "结论：")
    join_section("依据/条件", "依据：")
    join_section("行动建议", "建议：")

    if summary_parts:
        return _truncate_text(" ".join(summary_parts), max_total)

    # Fallback: strip markdown markers but keep natural text
    plain_segments: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("### "):
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        plain_segments.append(line)

    if not plain_segments and user_question:
        plain_segments.append(f"目前缺乏更多数据支撑，请补充信息后我再评估。")

    return _truncate_text(" ".join(plain_segments) or text.strip(), max_total)
