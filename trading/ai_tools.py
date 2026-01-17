from __future__ import annotations

import json
from typing import Any

from .alpaca_data import fetch_stock_bars, fetch_stock_snapshots
from .ai_rag import query as rag_query
from .web_search import search_web


DEFAULT_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "实时联网搜索并返回摘要结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 12},
                    "mode": {"type": "string", "enum": ["news", "text"]},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "在用户知识库中检索相关内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alpaca_snapshot",
            "description": "获取 Alpaca 行情快照。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["symbols"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "alpaca_bars",
            "description": "获取 Alpaca K 线数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}},
                    "timeframe": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["symbols"],
            },
        },
    },
]


def get_tool_definitions(enabled: list[str] | None = None) -> list[dict[str, Any]]:
    if not enabled:
        return list(DEFAULT_TOOL_DEFINITIONS)
    filtered: list[dict[str, Any]] = []
    allowed = {name.strip().lower() for name in enabled if name}
    for tool in DEFAULT_TOOL_DEFINITIONS:
        name = tool.get("function", {}).get("name", "").lower()
        if name in allowed:
            filtered.append(tool)
    return filtered


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def execute_tool_call(name: str, arguments: Any, *, user_id: str | None = None) -> dict[str, Any]:
    args = _parse_args(arguments)
    tool = (name or "").strip().lower()
    if tool == "web_search":
        query = args.get("query") or ""
        max_results = int(args.get("max_results") or 6)
        mode = args.get("mode") or "news"
        return {"results": search_web(str(query), max_results=max_results, mode=str(mode))}
    if tool == "rag_search":
        query = args.get("query") or ""
        top_k = int(args.get("top_k") or 5)
        return {"results": rag_query(str(query), user_id=user_id, top_k=top_k)}
    if tool == "alpaca_snapshot":
        symbols = args.get("symbols") or []
        return {"snapshots": fetch_stock_snapshots(symbols, user_id=user_id)}
    if tool == "alpaca_bars":
        symbols = args.get("symbols") or []
        timeframe = args.get("timeframe") or "1Day"
        start = args.get("start")
        end = args.get("end")
        limit = args.get("limit")
        return {
            "bars": fetch_stock_bars(
                symbols,
                timeframe=str(timeframe),
                start=start,
                end=end,
                limit=limit,
                feed="sip",
                adjustment="split",
                user_id=user_id,
            )
        }
    return {"error": f"Unknown tool: {name}"}
