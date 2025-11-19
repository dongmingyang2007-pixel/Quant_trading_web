from __future__ import annotations

import logging
import os
import base64
import binascii
import re
from io import BytesIO
from pathlib import Path
from typing import Optional
import uuid

from PIL import Image, UnidentifiedImageError
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

MAX_IMAGE_PIXELS = int(os.environ.get("IMAGE_MAX_PIXELS", 12_000_000))
MAX_IMAGE_BYTES = int(os.environ.get("IMAGE_MAX_BYTES", 5 * 1024 * 1024))  # 5MB 默认
MAX_IMAGE_SIDE = int(os.environ.get("IMAGE_MAX_SIDE", 5000))

IMAGE_ERROR_MESSAGES = {
    "invalid_image": "图片无法解析，请使用标准 JPG/PNG 文件。",
    "image_too_large": f"图片体积或尺寸过大，请控制在 {MAX_IMAGE_SIDE}px 内且小于 {MAX_IMAGE_BYTES // (1024 * 1024)}MB。",
}
DATA_URL_PATTERN = re.compile(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$")
ALLOWED_DATAURL_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
}

LOGGER = logging.getLogger(__name__)


def media_root() -> Path:
    return Path(settings.MEDIA_ROOT)


def _normalize_storage_path(path: str | os.PathLike[str] | None) -> str:
    if not path:
        return ""
    text = str(path).strip()
    if not text:
        return ""
    normalized = text.replace("\\", "/").lstrip("/")
    normalized = re.sub(r"/{2,}", "/", normalized)
    return normalized.strip("/")


def _default_media_url(normalized: str) -> str:
    base = (settings.MEDIA_URL or "").rstrip("/")
    if not base:
        return "/" + normalized
    return f"{base}/{normalized}"


def _storage_url(path: str) -> str:
    try:
        return default_storage.url(path)
    except (NotImplementedError, AttributeError, ValueError):
        return _default_media_url(path)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to build media url for %s: %s", path, exc)
        return _default_media_url(path)


def _prepare_image(upload) -> BytesIO:
    """
    使用 Pillow 校验上传的图片内容，去除 EXIF 并重新编码为 JPEG。
    若图片无效或超过阈值，抛出 ValueError。
    """
    try:
        upload.seek(0)
    except AttributeError:
        pass
    try:
        image = Image.open(upload)
        image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("invalid_image") from exc
    try:
        upload.seek(0)
    except AttributeError:
        pass
    image = Image.open(upload)
    image = image.convert("RGB")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("image_too_large")
    if image.width > MAX_IMAGE_SIDE or image.height > MAX_IMAGE_SIDE:
        raise ValueError("image_too_large")

    buffer = BytesIO()
    image.save(buffer, format="JPEG", optimize=True, quality=88)
    size = buffer.tell()
    if size > MAX_IMAGE_BYTES:
        raise ValueError("image_too_large")
    buffer.seek(0)
    return buffer


def save_uploaded_file(upload, *, subdir: str, filename_prefix: Optional[str] = None) -> str:
    sanitized = _prepare_image(upload)
    extra = uuid.uuid4().hex[:8]
    if filename_prefix:
        prefix = f"{filename_prefix}-{extra}"
    else:
        prefix = extra + uuid.uuid4().hex[:8]
    filename = f"{prefix}.jpg"
    try:
        sanitized.seek(0)
    except Exception:
        pass
    relative_dir = _normalize_storage_path(subdir)
    storage_path = "/".join(part for part in (relative_dir, filename) if part)
    content = ContentFile(sanitized.read(), name=filename)
    saved_path = default_storage.save(storage_path, content)
    return _normalize_storage_path(saved_path)


def describe_image_error(error: Exception | str | None) -> str:
    """
    将 Pillow 校验阶段的错误转换为友好的提示信息。
    """
    code = ""
    if isinstance(error, Exception) and error.args:
        code = str(error.args[0])
    elif isinstance(error, str):
        code = error
    if code in IMAGE_ERROR_MESSAGES:
        return IMAGE_ERROR_MESSAGES[code]
    return "图片上传失败，请更换文件或稍后再试。"


def delete_media_file(path: str | None) -> None:
    if not path:
        return
    try:
        text = str(path).strip()
    except Exception:
        return
    if text.startswith(("http://", "https://")):
        return
    normalized = _normalize_storage_path(text)
    if not normalized:
        return
    try:
        default_storage.delete(normalized)
    except NotImplementedError:
        target = media_root() / normalized
        try:
            if target.exists() and target.is_file():
                target.unlink()
        except OSError:
            pass
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to delete media file %s: %s", normalized, exc)


def decode_data_url_image(data_url: str | None, *, filename_prefix: str = "upload") -> ContentFile | None:
    if not data_url:
        return None
    match = DATA_URL_PATTERN.match(data_url)
    if not match:
        return None
    mime, b64_data = match.groups()
    ext = ALLOWED_DATAURL_MIME.get(mime.lower())
    if not ext:
        return None
    try:
        binary = base64.b64decode(b64_data, validate=True)
    except (ValueError, binascii.Error):
        return None
    if len(binary) > MAX_IMAGE_BYTES * 2:
        return None
    filename = f"{filename_prefix}-{uuid.uuid4().hex[:8]}{ext}"
    return ContentFile(binary, name=filename)


def resolve_media_url(path: str | os.PathLike[str] | None) -> str:
    if not path:
        return ""
    text = str(path).strip()
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        return text
    normalized = _normalize_storage_path(text)
    if not normalized:
        return ""
    try:
        return _storage_url(normalized)
    except Exception as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Falling back to default media URL for %s: %s", normalized, exc)
        fallback = _default_media_url(normalized)
        return fallback
