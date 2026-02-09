from __future__ import annotations

import os


def get_ai_model_choices() -> list[str]:
    env_choices = os.environ.get("OLLAMA_MODEL_CHOICES", "")
    choices: list[str] = []
    if env_choices:
        choices = [item.strip() for item in env_choices.split(",") if item.strip()]
    default_model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
    fallbacks = [
        default_model,
        os.environ.get("OLLAMA_SECONDARY_MODEL"),
        "deepseek-r1:8b",
        "llama3.2",
        "llama3.2:3b",
        "qwen2:7b",
    ]
    bailian_env_choices = os.environ.get("BAILIAN_MODEL_CHOICES") or os.environ.get("DASHSCOPE_MODEL_CHOICES") or ""
    bailian_default = os.environ.get("BAILIAN_MODEL") or os.environ.get("DASHSCOPE_MODEL") or "qwen-max"
    bailian_candidates_raw = [bailian_default, "qwen-max", "qwen-plus", "qwen-turbo"]
    bailian_candidates: list[str] = []
    bailian_env_list = bailian_env_choices.split(",") if bailian_env_choices else []
    for item in bailian_env_list:
        normalized = item.strip()
        if normalized:
            if ":" not in normalized:
                normalized = f"bailian:{normalized}"
            if normalized not in bailian_candidates:
                bailian_candidates.append(normalized)
    for candidate in bailian_candidates_raw:
        if not candidate:
            continue
        normalized = str(candidate).strip()
        if not normalized:
            continue
        if ":" not in normalized:
            normalized = f"bailian:{normalized}"
        if normalized not in bailian_candidates:
            bailian_candidates.append(normalized)
    gemini_candidates = [
        os.environ.get("GEMINI_MODEL"),
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-latest",
        "gemini-3.0-pro",
    ]
    ai_provider = (os.environ.get("AI_PROVIDER") or "").strip().lower()
    if ai_provider == "gemini":
        gemini_candidates = [os.environ.get("GEMINI_MODEL", "gemini-3.0-pro")] + gemini_candidates
    if ai_provider == "bailian":
        preferred = bailian_default.strip() if isinstance(bailian_default, str) else str(bailian_default)
        if ":" not in preferred:
            preferred = f"bailian:{preferred}"
        bailian_candidates = [preferred] + [item for item in bailian_candidates if item != preferred]
    for candidate in fallbacks:
        if candidate and candidate.strip() and candidate not in choices:
            choices.append(candidate.strip())
    for candidate in gemini_candidates:
        if candidate and candidate.strip() and candidate not in choices:
            choices.append(candidate.strip())
    for candidate in bailian_candidates:
        if candidate and candidate.strip() and candidate not in choices:
            choices.append(candidate.strip())
    return choices


def get_embedding_model_choices() -> list[str]:
    env_choices = os.environ.get("BAILIAN_EMBEDDING_MODEL_CHOICES") or os.environ.get("DASHSCOPE_EMBEDDING_MODEL_CHOICES") or ""
    choices: list[str] = []
    if env_choices:
        choices = [item.strip() for item in env_choices.split(",") if item.strip()]
    default_model = (
        os.environ.get("BAILIAN_EMBEDDING_MODEL")
        or os.environ.get("DASHSCOPE_EMBEDDING_MODEL")
        or "text-embedding-v2"
    )
    fallbacks = [
        default_model,
        "text-embedding-v2",
    ]
    for candidate in fallbacks:
        if candidate and candidate.strip() and candidate not in choices:
            choices.append(candidate.strip())
    return choices
