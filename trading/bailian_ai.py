from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from .http_client import http_client, HttpClientError
from .profile import load_api_credentials, resolve_api_credential


CHAT_PATH = os.environ.get("BAILIAN_CHAT_PATH", "/api/v1/services/aigc/text-generation/generation")
EMBEDDING_PATH = os.environ.get("BAILIAN_EMBEDDING_PATH", "/api/v1/services/embeddings/text-embedding/embedding")
VL_PATH = os.environ.get("BAILIAN_VL_PATH", "/api/v1/services/aigc/multimodal-generation/generation")


def _resolve_key(user_id: str | None = None, user: Any | None = None) -> str | None:
    if user is not None:
        return resolve_api_credential(user, "bailian_api_key")
    if user_id:
        try:
            creds = load_api_credentials(str(user_id))
        except Exception:
            creds = {}
        if isinstance(creds, dict):
            key = creds.get("bailian_api_key")
            if key:
                return key
        return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("BAILIAN_API_KEY")
    return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("BAILIAN_API_KEY")


def _base_url() -> str:
    return os.environ.get("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com").rstrip("/")


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _extract_answer(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    output = data.get("output") or {}
    choices = output.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            parts = []
            for item in content:
                text = item.get("text") if isinstance(item, dict) else str(item)
                if text:
                    parts.append(str(text))
            content = "\n".join(parts)
        return str(content).strip(), message
    text = output.get("text") or output.get("output_text") or ""
    return str(text).strip(), {}


def chat(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout_seconds: int,
    api_key: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    api_key = api_key or _resolve_key(user_id=user_id)
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY 环境变量，无法调用阿里云百炼。")
    url = f"{_base_url()}{CHAT_PATH}"
    parameters: dict[str, Any] = {
        "result_format": "message",
        "temperature": float(os.environ.get("BAILIAN_TEMPERATURE", os.environ.get("DASHSCOPE_TEMPERATURE", "0.35"))),
    }
    max_tokens = int(os.environ.get("BAILIAN_MAX_TOKENS", os.environ.get("DASHSCOPE_MAX_TOKENS", "900")))
    if max_tokens > 0:
        parameters["max_tokens"] = max_tokens
    if tool_choice is not None:
        parameters["tool_choice"] = tool_choice
    if response_format is not None:
        parameters["response_format"] = response_format
    if extra_params:
        parameters.update(extra_params)
    payload: dict[str, Any] = {
        "model": model,
        "input": {"messages": messages},
        "parameters": parameters,
    }
    if tools:
        payload["input"]["tools"] = tools
    try:
        response = http_client.post(url, json=payload, headers=_headers(api_key), timeout=timeout_seconds, retries=0)
        data = response.json()
    except HttpClientError as exc:
        raise RuntimeError(f"百炼请求失败：{exc}")
    except ValueError:
        raise RuntimeError("百炼返回了无法解析的响应")
    if isinstance(data, dict) and data.get("code"):
        raise RuntimeError(f"百炼错误：{data.get('message') or data.get('code')}")
    answer, message = _extract_answer(data if isinstance(data, dict) else {})
    return {"status": "ok", "answer": answer, "raw": data, "message": message}


def embed_texts(
    texts: list[str],
    *,
    model: str | None = None,
    user_id: str | None = None,
    api_key: str | None = None,
    timeout_seconds: int | None = None,
) -> list[list[float]]:
    api_key = api_key or _resolve_key(user_id=user_id)
    if not api_key:
        return []
    if not model and user_id:
        try:
            creds = load_api_credentials(str(user_id))
        except Exception:
            creds = {}
        if isinstance(creds, dict):
            preferred = str(creds.get("ai_embedding_model") or "").strip()
            if preferred:
                lowered = preferred.lower()
                if lowered.startswith(("bailian:", "dashscope:", "aliyun:")):
                    preferred = preferred.split(":", 1)[1].strip()
                model = preferred
    payload = {
        "model": model or os.environ.get("BAILIAN_EMBEDDING_MODEL", os.environ.get("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v2")),
        "input": {"texts": [str(text) for text in texts]},
    }
    url = f"{_base_url()}{EMBEDDING_PATH}"
    try:
        response = http_client.post(
            url,
            json=payload,
            headers=_headers(api_key),
            timeout=timeout_seconds or 30,
            retries=0,
        )
        data = response.json()
    except (HttpClientError, ValueError):
        return []
    output = data.get("output") if isinstance(data, dict) else None
    embeddings = output.get("embeddings") if isinstance(output, dict) else None
    if isinstance(embeddings, list):
        vectors: list[list[float]] = []
        for item in embeddings:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if isinstance(vector, list):
                vectors.append([float(x) for x in vector])
        return vectors
    return []


def _encode_image(path: str | Path) -> str | None:
    try:
        binary = Path(path).read_bytes()
    except Exception:
        return None
    encoded = base64.b64encode(binary).decode("ascii")
    suffix = Path(path).suffix.lower().lstrip(".") or "png"
    return f"data:image/{suffix};base64,{encoded}"


def multimodal_chat(
    model: str,
    messages: list[dict[str, Any]],
    *,
    images: list[str] | None = None,
    timeout_seconds: int | None = None,
    api_key: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    api_key = api_key or _resolve_key(user_id=user_id)
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY 环境变量，无法调用阿里云百炼。")
    url = f"{_base_url()}{VL_PATH}"
    image_parts: list[dict[str, str]] = []
    for item in images or []:
        if not item:
            continue
        if item.startswith("data:image/"):
            image_parts.append({"image": item})
        elif item.startswith("http://") or item.startswith("https://"):
            image_parts.append({"image": item})
        else:
            encoded = _encode_image(item)
            if encoded:
                image_parts.append({"image": encoded})
    if image_parts:
        if messages:
            messages = list(messages)
            latest = messages[-1]
            content = latest.get("content")
            parts = []
            if isinstance(content, list):
                parts.extend(content)
            elif isinstance(content, str) and content.strip():
                parts.append({"text": content})
            parts.extend(image_parts)
            latest = dict(latest)
            latest["content"] = parts
            messages[-1] = latest
    payload: dict[str, Any] = {
        "model": model,
        "input": {"messages": messages},
        "parameters": {"result_format": "message"},
    }
    try:
        response = http_client.post(url, json=payload, headers=_headers(api_key), timeout=timeout_seconds or 60, retries=0)
        data = response.json()
    except HttpClientError as exc:
        raise RuntimeError(f"百炼多模态请求失败：{exc}")
    except ValueError:
        raise RuntimeError("百炼多模态返回了无法解析的响应")
    if isinstance(data, dict) and data.get("code"):
        raise RuntimeError(f"百炼错误：{data.get('message') or data.get('code')}")
    answer, message = _extract_answer(data if isinstance(data, dict) else {})
    return {"status": "ok", "answer": answer, "raw": data, "message": message}
