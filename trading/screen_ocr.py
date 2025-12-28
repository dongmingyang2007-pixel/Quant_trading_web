from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageFilter, ImageOps

try:  # optional dependency
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional
    pytesseract = None  # type: ignore


_TIMEFRAME_PATTERN = re.compile(r"\b(\d{1,3})([smhdwSMHDW])\b")
_TIMEFRAME_ALIAS = re.compile(r"\b(\d{1,3})(min|m|h|d|w|wk|mo|M)\b", re.IGNORECASE)
_SYMBOL_PATTERN = re.compile(r"\b[A-Z]{1,6}(?:\.[A-Z]{1,3})?\b")


@dataclass(slots=True)
class OcrResult:
    symbol: Optional[str]
    timeframe: Optional[str]
    raw_text: str
    available: bool
    error: Optional[str] = None


def extract_header_metadata(image: Image.Image) -> OcrResult:
    if pytesseract is None or not shutil.which("tesseract"):
        return OcrResult(symbol=None, timeframe=None, raw_text="", available=False, error="ocr_unavailable")
    if image is None:
        return OcrResult(symbol=None, timeframe=None, raw_text="", available=False, error="no_image")

    prepared = _prepare_ocr_image(image)
    try:
        text = pytesseract.image_to_string(prepared, config="--psm 6")
    except Exception:
        return OcrResult(symbol=None, timeframe=None, raw_text="", available=False, error="ocr_failed")

    normalized = " ".join(text.split())
    symbol = _extract_symbol(normalized)
    timeframe = _extract_timeframe(normalized)
    return OcrResult(symbol=symbol, timeframe=timeframe, raw_text=normalized, available=True)


def _prepare_ocr_image(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = gray.resize((gray.width * 2, gray.height * 2))
    gray = gray.point(lambda value: 255 if value > 160 else 0)
    return gray


def _extract_symbol(text: str) -> Optional[str]:
    matches = _SYMBOL_PATTERN.findall(text or "")
    if not matches:
        return None
    blacklist = {"UTC", "NYSE", "NASDAQ", "AMEX", "BATS"}
    for candidate in matches:
        if candidate in blacklist:
            continue
        return candidate
    return matches[0]


def _extract_timeframe(text: str) -> Optional[str]:
    if not text:
        return None
    match = _TIMEFRAME_PATTERN.search(text)
    if match:
        return f"{match.group(1)}{match.group(2).lower()}"
    match = _TIMEFRAME_ALIAS.search(text)
    if match:
        unit = match.group(2).lower()
        return f"{match.group(1)}{unit}"
    return None


__all__ = ["extract_header_metadata", "OcrResult"]
