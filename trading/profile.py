from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings
from django.contrib.auth import get_user_model

from .models import UserProfile
from .observability import record_metric

LOGGER = logging.getLogger(__name__)

DEFAULT_PROFILE = {
    "display_name": "",
    "cover_color": "#116e5f",
    "bio": "",
    "avatar_path": "",
    "feature_image_path": "",
    "gallery_paths": [],
}

API_CREDENTIAL_FIELDS = {
    "market_data_provider": {
        "label": "Market Data Provider",
        "env": "MARKET_DATA_PROVIDER",
        "help": "选择行情数据源：alpaca 或 massive。",
        "mask": False,
    },
    "market_news_provider": {
        "label": "Market News Provider",
        "env": "MARKET_NEWS_PROVIDER",
        "help": "选择新闻源：follow_data、alpaca 或 massive。",
        "mask": False,
    },
    "massive_api_key": {
        "label": "Massive API Key",
        "env": "MASSIVE_API_KEY",
        "help": "用于 Massive Stocks Advanced 的 API Key。",
    },
    "massive_s3_access_key_id": {
        "label": "Massive S3 Access Key ID",
        "env": "MASSIVE_S3_ACCESS_KEY_ID",
        "help": "用于 Massive Flat Files (S3) 访问的 Access Key ID（可选）。",
    },
    "massive_s3_secret_access_key": {
        "label": "Massive S3 Secret Access Key",
        "env": "MASSIVE_S3_SECRET_ACCESS_KEY",
        "help": "用于 Massive Flat Files (S3) 访问的 Secret Access Key（可选）。",
    },
    "massive_rest_url": {
        "label": "Massive REST URL",
        "env": "MASSIVE_REST_URL",
        "help": "Massive REST 根地址（可选，默认官方地址）。",
        "mask": False,
    },
    "massive_ws_url": {
        "label": "Massive WebSocket URL",
        "env": "MASSIVE_WS_URL",
        "help": "Massive WebSocket 地址（可选，默认官方地址）。",
        "mask": False,
    },
    "massive_plan": {
        "label": "Massive Plan",
        "env": "MASSIVE_PLAN",
        "help": "当前套餐标识，默认 stocks_advanced。",
        "mask": False,
    },
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
    "ai_model": {
        "label": "AI Model",
        "env": "AI_MODEL",
        "help": "默认 AI 模型（示例：bailian:qwen-max）。",
        "mask": False,
    },
    "ai_embedding_model": {
        "label": "AI Embedding Model",
        "env": "BAILIAN_EMBEDDING_MODEL",
        "help": "默认向量化模型（示例：text-embedding-v2）。",
        "mask": False,
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

_CREDENTIALS_FIELD_PREFIX = "fernet:"


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


def _resolve_credentials_encryption_key() -> bytes | None:
    configured = (os.environ.get("DJANGO_CREDENTIALS_ENCRYPTION_KEY") or "").strip()
    if configured:
        try:
            decoded = base64.urlsafe_b64decode(configured.encode("utf-8"))
            if len(decoded) == 32:
                return configured.encode("utf-8")
        except Exception:
            pass
        # Support passphrase-style keys by deriving a Fernet-compatible key.
        return base64.urlsafe_b64encode(hashlib.sha256(configured.encode("utf-8")).digest())

    secret_key = str(getattr(settings, "SECRET_KEY", "") or "").strip()
    if not secret_key:
        return None
    return base64.urlsafe_b64encode(hashlib.sha256(secret_key.encode("utf-8")).digest())


def _get_credentials_cipher() -> Fernet | None:
    key = _resolve_credentials_encryption_key()
    if not key:
        return None
    try:
        return Fernet(key)
    except Exception as exc:
        LOGGER.warning("Failed to initialize API credential cipher: %s", exc)
    return None


def _encrypt_api_credentials(credentials: dict[str, str]) -> str | None:
    cipher = _get_credentials_cipher()
    if cipher is None:
        return None
    payload = json.dumps(credentials, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    token = cipher.encrypt(payload).decode("utf-8")
    return f"{_CREDENTIALS_FIELD_PREFIX}{token}"


def _decrypt_api_credentials(encrypted: str | None) -> dict[str, str] | None:
    text = (encrypted or "").strip()
    if not text:
        return None
    cipher = _get_credentials_cipher()
    if cipher is None:
        return None
    token = text[len(_CREDENTIALS_FIELD_PREFIX) :] if text.startswith(_CREDENTIALS_FIELD_PREFIX) else text
    try:
        decrypted = cipher.decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return None
    except Exception:
        return None
    try:
        payload = json.loads(decrypted)
    except (TypeError, ValueError):
        return None
    return _normalize_api_credentials(payload)


def _persist_encrypted_api_credentials(profile_obj: UserProfile, credentials: dict[str, str], *, clear_legacy: bool) -> bool:
    encrypted = _encrypt_api_credentials(credentials)
    if not encrypted:
        return False
    profile_obj.api_credentials_encrypted = encrypted
    update_fields = ["api_credentials_encrypted", "updated_at"]
    if clear_legacy:
        profile_obj.api_credentials = {}
        update_fields.insert(1, "api_credentials")
    profile_obj.save(update_fields=update_fields)
    return True


def _load_api_credentials_from_profile(profile_obj: UserProfile) -> dict[str, str]:
    encrypted_value = profile_obj.api_credentials_encrypted or ""
    decrypted = _decrypt_api_credentials(encrypted_value)
    if decrypted is not None:
        if isinstance(profile_obj.api_credentials, dict) and profile_obj.api_credentials:
            profile_obj.api_credentials = {}
            profile_obj.save(update_fields=["api_credentials", "updated_at"])
            record_metric(
                "credential.migration.success",
                user_id=str(profile_obj.user_id),
                reason="clear_legacy_after_decrypt",
            )
        return decrypted

    if encrypted_value:
        record_metric(
            "credential.migration.failure",
            user_id=str(profile_obj.user_id),
            reason="decrypt_failed",
        )
        LOGGER.warning("Failed to decrypt api_credentials_encrypted for user_id=%s", profile_obj.user_id)

    legacy = _normalize_api_credentials(profile_obj.api_credentials)
    if legacy and _persist_encrypted_api_credentials(profile_obj, legacy, clear_legacy=True):
        record_metric(
            "credential.migration.success",
            user_id=str(profile_obj.user_id),
            reason="read_rewrite",
        )
        return legacy
    if legacy:
        record_metric(
            "credential.migration.failure",
            user_id=str(profile_obj.user_id),
            reason="encryption_unavailable",
        )
    return legacy


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
    return _load_api_credentials_from_profile(profile_obj)


def save_api_credentials(user_id: str, updates: dict[str, Any], *, replace: bool = False) -> dict[str, str]:
    profile_obj = _get_or_create_profile(user_id)
    credentials = {} if replace else _load_api_credentials_from_profile(profile_obj)
    clearable_fields = {
        "ai_model",
        "ai_embedding_model",
        "market_data_provider",
        "market_news_provider",
        "massive_api_key",
        "massive_s3_access_key_id",
        "massive_s3_secret_access_key",
        "massive_rest_url",
        "massive_ws_url",
        "massive_plan",
    }
    for key, value in updates.items():
        if key not in API_CREDENTIAL_FIELDS:
            continue
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            if key in clearable_fields:
                credentials.pop(key, None)
            continue
        credentials[key] = text
    if not _persist_encrypted_api_credentials(profile_obj, credentials, clear_legacy=True):
        raise RuntimeError("Credential encryption unavailable. Configure DJANGO_CREDENTIALS_ENCRYPTION_KEY.")
    record_metric(
        "credential.migration.success",
        user_id=str(profile_obj.user_id),
        reason="write_encrypted",
    )
    return credentials


def clear_api_credentials(user_id: str) -> None:
    profile_obj = _get_or_create_profile(user_id)
    profile_obj.api_credentials = {}
    profile_obj.api_credentials_encrypted = ""
    profile_obj.save(update_fields=["api_credentials", "api_credentials_encrypted", "updated_at"])


def migrate_legacy_api_credentials(profile_obj: UserProfile, *, dry_run: bool = False) -> tuple[bool, str]:
    legacy = _normalize_api_credentials(profile_obj.api_credentials)
    if not legacy:
        return False, "empty"
    if dry_run:
        return True, "dry_run"
    if _persist_encrypted_api_credentials(profile_obj, legacy, clear_legacy=True):
        record_metric(
            "credential.migration.success",
            user_id=str(profile_obj.user_id),
            reason="bulk_migrate",
        )
        return True, "migrated"
    record_metric(
        "credential.migration.failure",
        user_id=str(profile_obj.user_id),
        reason="encryption_unavailable",
    )
    return False, "encryption_unavailable"


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
