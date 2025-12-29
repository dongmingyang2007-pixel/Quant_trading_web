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


_TIMEFRAME_TOKEN = re.compile(r"\b(\d{1,3})\s*([a-zA-Z]{1,6})\b")
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
    return canonicalize_timeframe(text)


def canonicalize_timeframe(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    unit_map = {
        "s": "s",
        "sec": "s",
        "secs": "s",
        "second": "s",
        "seconds": "s",
        "m": "m",
        "min": "m",
        "mins": "m",
        "minute": "m",
        "minutes": "m",
        "h": "h",
        "hr": "h",
        "hrs": "h",
        "hour": "h",
        "hours": "h",
        "d": "d",
        "day": "d",
        "days": "d",
        "w": "w",
        "wk": "w",
        "wks": "w",
        "week": "w",
        "weeks": "w",
        "mo": "m",
        "mon": "m",
        "month": "m",
        "months": "m",
    }
    for match in _TIMEFRAME_TOKEN.finditer(text):
        number = match.group(1)
        unit_raw = match.group(2).lower()
        unit = unit_map.get(unit_raw)
        if unit:
            return f"{int(number)}{unit}"
    return None


__all__ = ["extract_header_metadata", "canonicalize_timeframe", "OcrResult"]
