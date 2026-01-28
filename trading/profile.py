from __future__ import annotations

import os
from typing import Any

from django.contrib.auth import get_user_model

from .models import UserProfile

DEFAULT_PROFILE = {
    "display_name": "",
    "cover_color": "#116e5f",
    "bio": "",
    "avatar_path": "",
    "feature_image_path": "",
    "gallery_paths": [],
}

API_CREDENTIAL_FIELDS = {
    "alpaca_trading_mode": {
        "label": "Alpaca Trading Mode",
        "env": "ALPACA_TRADING_MODE",
        "help": "选择 live 或 paper 作为默认交易环境。",
        "mask": False,
    },
    "alpaca_paper_api_key_id": {
        "label": "Alpaca Paper API Key ID",
        "env": "ALPACA_PAPER_API_KEY_ID",
        "help": "用于 Alpaca Paper 交易环境的 Key ID。",
    },
    "alpaca_paper_api_secret_key": {
        "label": "Alpaca Paper API Secret",
        "env": "ALPACA_PAPER_API_SECRET_KEY",
        "help": "用于 Alpaca Paper 交易环境的 Secret Key。",
    },
    "alpaca_live_api_key_id": {
        "label": "Alpaca Live API Key ID",
        "env": "ALPACA_LIVE_API_KEY_ID",
        "help": "用于 Alpaca Live 交易环境的 Key ID。",
    },
    "alpaca_live_api_secret_key": {
        "label": "Alpaca Live API Secret",
        "env": "ALPACA_LIVE_API_SECRET_KEY",
        "help": "用于 Alpaca Live 交易环境的 Secret Key。",
    },
    "alpaca_api_key_id": {
        "label": "Alpaca API Key ID",
        "env": "ALPACA_API_KEY_ID",
        "help": "用于 Alpaca 行情/交易 API 的 Key ID。",
    },
    "alpaca_api_secret_key": {
        "label": "Alpaca API Secret",
        "env": "ALPACA_API_SECRET_KEY",
        "help": "用于 Alpaca 行情/交易 API 的 Secret Key。",
    },
    "bailian_api_key": {
        "label": "BaiLian (DashScope) API Key",
        "env": "DASHSCOPE_API_KEY",
        "help": "用于阿里云百炼（通义千问）API Key。",
    },
    "aliyun_access_key_id": {
        "label": "Aliyun AccessKey ID",
        "env": "ALIYUN_ACCESS_KEY_ID",
        "help": "用于阿里云服务的 AccessKey ID。",
    },
    "aliyun_access_key_secret": {
        "label": "Aliyun AccessKey Secret",
        "env": "ALIYUN_ACCESS_KEY_SECRET",
        "help": "用于阿里云服务的 AccessKey Secret。",
    },
    "gemini_api_key": {
        "label": "Gemini API Key",
        "env": "GEMINI_API_KEY",
        "help": "用于 Google Gemini 模型调用。",
    },
    "ollama_api_key": {
        "label": "Ollama API Key",
        "env": "OLLAMA_API_KEY",
        "help": "用于 Ollama Web Search 或云端接口（可选）。",
    },
    "strategy_update_auth_token": {
        "label": "Strategy Update Token",
        "env": "STRATEGY_UPDATE_AUTH_TOKEN",
        "help": "用于拉取远程策略覆写（可选）。",
    },
}


def _get_or_create_profile(user_id: str) -> UserProfile:
    UserModel = get_user_model()
    user = UserModel.objects.get(pk=user_id)
    profile, _ = UserProfile.objects.get_or_create(user=user)
    return profile


def _normalize_gallery(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    return []


def _normalize_api_credentials(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, str] = {}
    for key in API_CREDENTIAL_FIELDS:
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            normalized[key] = text
    return normalized


def load_profile(user_id: str) -> dict[str, Any]:
    profile_obj = _get_or_create_profile(user_id)
    payload = {
        "display_name": profile_obj.display_name or "",
        "cover_color": profile_obj.cover_color or DEFAULT_PROFILE["cover_color"],
        "bio": profile_obj.bio or "",
        "avatar_path": profile_obj.avatar_path or "",
        "feature_image_path": profile_obj.feature_image_path or "",
        "gallery_paths": _normalize_gallery(profile_obj.gallery_paths),
        "slug": str(profile_obj.slug),
    }
    return payload


def save_profile(user_id: str, profile: dict[str, Any]) -> None:
    profile_obj = _get_or_create_profile(user_id)
    for key in ("display_name", "cover_color", "bio", "avatar_path", "feature_image_path"):
        value = profile.get(key)
        if value is not None:
            setattr(profile_obj, key, value)
    if "gallery_paths" in profile:
        profile_obj.gallery_paths = _normalize_gallery(profile.get("gallery_paths"))
    profile_obj.save()


def load_api_credentials(user_id: str) -> dict[str, str]:
    profile_obj = _get_or_create_profile(user_id)
    return _normalize_api_credentials(profile_obj.api_credentials)


def save_api_credentials(user_id: str, updates: dict[str, Any], *, replace: bool = False) -> dict[str, str]:
    profile_obj = _get_or_create_profile(user_id)
    credentials = {} if replace else _normalize_api_credentials(profile_obj.api_credentials)
    for key, value in updates.items():
        if key not in API_CREDENTIAL_FIELDS:
            continue
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        credentials[key] = text
    profile_obj.api_credentials = credentials
    profile_obj.save(update_fields=["api_credentials", "updated_at"])
    return credentials


def clear_api_credentials(user_id: str) -> None:
    profile_obj = _get_or_create_profile(user_id)
    profile_obj.api_credentials = {}
    profile_obj.save(update_fields=["api_credentials", "updated_at"])


def mask_credential(value: str | None) -> str:
    if not value:
        return ""
    text = str(value)
    tail = text[-4:] if len(text) > 4 else text
    return f"•••• {tail}"


def resolve_api_credential(user: Any, key: str) -> str | None:
    if user and getattr(user, "is_authenticated", False):
        creds = load_api_credentials(str(user.id))
        value = creds.get(key)
        if value:
            return value
    env = API_CREDENTIAL_FIELDS.get(key, {}).get("env")
    if env:
        return os.environ.get(env)
    return None
