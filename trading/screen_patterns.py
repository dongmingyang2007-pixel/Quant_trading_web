from __future__ import annotations

import base64
import colorsys
import io
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

try:
    from django.conf import settings
except Exception:  # pragma: no cover - optional during isolated use
    settings = None  # type: ignore


_MAX_BYTES = int(
    getattr(settings, "SCREEN_ANALYZER_MAX_BYTES", os.environ.get("SCREEN_ANALYZER_MAX_BYTES", "900000"))
    if settings
    else os.environ.get("SCREEN_ANALYZER_MAX_BYTES", "900000")
)
_MAX_WIDTH = int(
    getattr(settings, "SCREEN_ANALYZER_MAX_WIDTH", os.environ.get("SCREEN_ANALYZER_MAX_WIDTH", "1280"))
    if settings
    else os.environ.get("SCREEN_ANALYZER_MAX_WIDTH", "1280")
)
_MAX_HEIGHT = int(
    getattr(settings, "SCREEN_ANALYZER_MAX_HEIGHT", os.environ.get("SCREEN_ANALYZER_MAX_HEIGHT", "720"))
    if settings
    else os.environ.get("SCREEN_ANALYZER_MAX_HEIGHT", "720")
)
_MIN_SERIES_POINTS = int(os.environ.get("SCREEN_ANALYZER_MIN_POINTS", "120"))

_HUE_TOLERANCE = int(os.environ.get("SCREEN_ANALYZER_HUE_TOLERANCE", "8"))
_SAT_THRESHOLD = int(os.environ.get("SCREEN_ANALYZER_SAT_THRESHOLD", "70"))
_VAL_THRESHOLD = int(os.environ.get("SCREEN_ANALYZER_VAL_THRESHOLD", "60"))
_HSV_SAT_TOLERANCE = int(os.environ.get("SCREEN_ANALYZER_HSV_SAT_TOLERANCE", "70"))
_HSV_VAL_TOLERANCE = int(os.environ.get("SCREEN_ANALYZER_HSV_VAL_TOLERANCE", "70"))
_LINE_RANGE_RATIO = float(os.environ.get("SCREEN_ANALYZER_LINE_RANGE_RATIO", "0.08"))
_LINE_COVERAGE_MIN = float(os.environ.get("SCREEN_ANALYZER_LINE_COVERAGE_MIN", "0.22"))

_FLAT_SLOPE = float(os.environ.get("SCREEN_ANALYZER_FLAT_SLOPE", "0.02"))
_SLOPE_PARALLEL = float(os.environ.get("SCREEN_ANALYZER_SLOPE_PARALLEL", "0.018"))
_CONVERGENCE_MIN = float(os.environ.get("SCREEN_ANALYZER_CONVERGENCE_MIN", "0.12"))

PATTERN_KEYS = [
    "rising_wedge",
    "falling_wedge",
    "ascending_triangle",
    "descending_triangle",
    "sym_triangle",
    "channel_up",
    "channel_down",
    "range",
    "trend_up",
    "trend_down",
    "trend_flat",
    "converging",
    "rising_converging",
    "falling_converging",
]

FEATURE_NAMES = [
    "slope",
    "r2",
    "volatility",
    "mean_return",
    "max_drawdown",
    "amplitude",
    "upper_slope",
    "lower_slope",
    "convergence",
    "maxima_ratio",
    "minima_ratio",
]


@dataclass(slots=True)
class SeriesResult:
    series: np.ndarray
    diagnostics: Dict[str, Any]
    mode: str


@dataclass(slots=True)
class LineFit:
    slope: float
    intercept: float
    r2: float


def analyze_screen_frame(
    data_url: str,
    *,
    mode: str = "auto",
    calibration: Optional[Dict[str, Any]] = None,
    header_image: Optional[str] = None,
    enable_ocr: bool = True,
    include_features: bool = False,
) -> Dict[str, Any]:
    image = _decode_data_url(data_url)
    if image is None:
        return {"error": "invalid_image"}
    image = _resize_for_analysis(image)
    mode = (mode or "auto").lower()
    if mode not in {"auto", "line", "candlestick"}:
        mode = "auto"
    calibration = _normalize_calibration(calibration)

    series_result = _extract_series(image, mode, calibration)
    if series_result is None:
        return {"error": "series_not_found"}

    pattern = _classify_pattern(series_result.series)
    model_result = _maybe_apply_model(series_result.series, pattern)
    overlay = _render_overlay(image, pattern.get("upper_fit"), pattern.get("lower_fit"))

    symbol = None
    timeframe = None
    ocr_payload: Dict[str, Any] = {}
    if enable_ocr and header_image:
        ocr_payload = _extract_ocr_payload(header_image)
        symbol = ocr_payload.get("symbol")
        timeframe = ocr_payload.get("timeframe")

    response = {
        "pattern_key": pattern["pattern_key"],
        "confidence": pattern["confidence"],
        "bias": pattern["bias"],
        "probabilities": pattern["probabilities"],
        "overlay_image": overlay,
        "analysis_mode": series_result.mode,
        "symbol": symbol,
        "timeframe": timeframe,
        "diagnostics": {
            **series_result.diagnostics,
            **pattern["diagnostics"],
            **ocr_payload.get("diagnostics", {}),
            **model_result.get("diagnostics", {}),
        },
    }
    if include_features:
        response["features"] = build_feature_vector(series_result.series)
    return response


def build_feature_vector(series: np.ndarray) -> list[float]:
    if series.size == 0:
        return [0.0 for _ in FEATURE_NAMES]
    length = len(series)
    x = np.linspace(0.0, 1.0, length)
    fit = _fit_line(x, series)
    returns = np.diff(series)
    volatility = float(np.std(returns)) if returns.size else 0.0
    mean_return = float(np.mean(returns)) if returns.size else 0.0
    max_drawdown = _max_drawdown(series)
    amplitude = float(np.max(series) - np.min(series)) if length else 0.0
    maxima, minima = _find_extrema(series, order=5)
    maxima_ratio = len(maxima) / max(1, length)
    minima_ratio = len(minima) / max(1, length)
    convergence = _estimate_convergence(series, maxima, minima)
    upper_slope = convergence.get("upper_slope", 0.0)
    lower_slope = convergence.get("lower_slope", 0.0)
    converging = convergence.get("convergence", 0.0)
    return [
        round(float(fit.slope), 6),
        round(float(fit.r2), 6),
        round(float(volatility), 6),
        round(float(mean_return), 6),
        round(float(max_drawdown), 6),
        round(float(amplitude), 6),
        round(float(upper_slope), 6),
        round(float(lower_slope), 6),
        round(float(converging), 6),
        round(float(maxima_ratio), 6),
        round(float(minima_ratio), 6),
    ]


def _normalize_calibration(calibration: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    if not isinstance(calibration, dict):
        return result
    for key in ("line", "candle_up", "candle_down"):
        payload = calibration.get(key)
        if not isinstance(payload, dict):
            continue
        rgb = _normalize_rgb(payload)
        if rgb is None:
            continue
        h, s, v = _rgb_to_hsv(rgb)
        result[key] = {
            "r": rgb[0],
            "g": rgb[1],
            "b": rgb[2],
            "h": h,
            "s": s,
            "v": v,
            "tol_h": float(payload.get("tol_h", _HUE_TOLERANCE)),
            "tol_s": float(payload.get("tol_s", _HSV_SAT_TOLERANCE)),
            "tol_v": float(payload.get("tol_v", _HSV_VAL_TOLERANCE)),
        }
    return result


def _normalize_rgb(payload: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    if not isinstance(payload, dict):
        return None
    for key in ("r", "g", "b"):
        if key not in payload:
            return None
    try:
        r = int(payload.get("r", 0))
        g = int(payload.get("g", 0))
        b = int(payload.get("b", 0))
    except (TypeError, ValueError):
        return None
    return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))


def _rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = [value / 255.0 for value in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 255.0, s * 255.0, v * 255.0


def _extract_series(image: Image.Image, mode: str, calibration: Dict[str, Any]) -> SeriesResult | None:
    if mode == "line":
        return _extract_line_series(image, calibration)
    if mode == "candlestick":
        return _extract_candlestick_series(image, calibration)
    result = _extract_line_series(image, calibration)
    if result is not None:
        return result
    return _extract_candlestick_series(image, calibration)


def _extract_ocr_payload(header_image: str) -> Dict[str, Any]:
    from .screen_ocr import extract_header_metadata

    header = _decode_data_url(header_image)
    if header is None:
        return {"diagnostics": {"ocr_error": "invalid_header"}}
    header = _resize_for_analysis(header)
    ocr = extract_header_metadata(header)
    diagnostics = {"ocr_available": ocr.available}
    if ocr.error:
        diagnostics["ocr_error"] = ocr.error
    return {
        "symbol": ocr.symbol,
        "timeframe": ocr.timeframe,
        "diagnostics": diagnostics,
    }


def _maybe_apply_model(series: np.ndarray, pattern: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from .screen_training import load_trained_model
    except Exception:
        return {"diagnostics": {"model_used": False}}

    model_bundle = load_trained_model()
    if model_bundle is None:
        return {"diagnostics": {"model_used": False}}
    model, labels = model_bundle
    features = build_feature_vector(series)
    try:
        probs = model.predict_proba([features])[0]
    except Exception:
        return {"diagnostics": {"model_used": False}}
    best_index = int(np.argmax(probs))
    best_label = labels[best_index] if best_index < len(labels) else None
    confidence = float(probs[best_index]) if probs is not None else 0.0
    diagnostics = {"model_used": True, "model_confidence": round(confidence, 3)}
    if best_label and confidence >= 0.55:
        pattern["pattern_key"] = best_label
        pattern["bias"] = _pattern_bias(best_label)
        pattern["confidence"] = round(max(pattern["confidence"], confidence), 3)
        pattern["probabilities"] = _bias_probabilities(pattern["bias"], pattern["confidence"])
        diagnostics["model_override"] = True
    else:
        diagnostics["model_override"] = False
    return {"diagnostics": diagnostics}


def _decode_data_url(data_url: str) -> Image.Image | None:
    if not data_url or not isinstance(data_url, str):
        return None
    if not data_url.startswith("data:image/"):
        return None
    try:
        _, encoded = data_url.split(",", 1)
    except ValueError:
        return None
    estimated_bytes = (len(encoded) * 3) // 4
    if estimated_bytes > _MAX_BYTES:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception:
        return None
    if len(raw) > _MAX_BYTES:
        return None
    try:
        image = Image.open(io.BytesIO(raw))
        return image.convert("RGB")
    except Exception:
        return None


def _resize_for_analysis(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width <= _MAX_WIDTH and height <= _MAX_HEIGHT:
        return image
    scale = min(_MAX_WIDTH / max(width, 1), _MAX_HEIGHT / max(height, 1), 1.0)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def _extract_line_series(image: Image.Image, calibration: Dict[str, Any]) -> SeriesResult | None:
    hsv = image.convert("HSV")
    arr = np.array(hsv)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    hue = arr[:, :, 0].astype(np.int16)
    sat = arr[:, :, 1].astype(np.int16)
    val = arr[:, :, 2].astype(np.int16)

    line_mask = None
    dominant = None
    calibrated = False
    target = calibration.get("line") if isinstance(calibration, dict) else None
    if isinstance(target, dict):
        calibrated = True
        line_mask = _mask_from_target(hue, sat, val, target)
    else:
        base_mask = (sat >= _SAT_THRESHOLD) & (val >= _VAL_THRESHOLD)
        if int(base_mask.sum()) < max(200, image.size[0]):
            return None
        hist = np.bincount(hue[base_mask].ravel(), minlength=256)
        dominant = int(hist.argmax())
        diff = np.abs(((hue - dominant + 128) % 256) - 128)
        line_mask = base_mask & (diff <= _HUE_TOLERANCE)

    coverage, median_range = _measure_line_shape(line_mask)
    if coverage < _LINE_COVERAGE_MIN or median_range > image.size[1] * _LINE_RANGE_RATIO:
        return None

    width, height = image.size
    x_indices = np.arange(width)
    y_values = np.full(width, np.nan)
    for x in range(width):
        ys = np.where(line_mask[:, x])[0]
        if ys.size:
            y_values[x] = float(np.median(ys))
    valid = ~np.isnan(y_values)
    if int(valid.sum()) < max(_MIN_SERIES_POINTS, width * 0.2):
        return None
    y_values = np.interp(x_indices, x_indices[valid], y_values[valid])
    series = 1.0 - (y_values / max(1.0, float(height - 1)))
    diagnostics = {
        "dominant_hue": dominant,
        "coverage": round(float(coverage), 3),
        "median_range": round(float(median_range), 2),
        "points": int(valid.sum()),
        "calibrated": calibrated,
    }
    return SeriesResult(series=series, diagnostics=diagnostics, mode="line")


def _measure_line_shape(mask: np.ndarray) -> Tuple[float, float]:
    height, width = mask.shape
    ranges: list[int] = []
    columns = 0
    for x in range(width):
        ys = np.where(mask[:, x])[0]
        if ys.size:
            columns += 1
            ranges.append(int(ys[-1] - ys[0]))
    if not ranges or width == 0:
        return 0.0, float(height)
    coverage = columns / width
    median_range = float(np.median(ranges))
    return coverage, median_range


def _mask_from_target(hue: np.ndarray, sat: np.ndarray, val: np.ndarray, target: Dict[str, Any]) -> np.ndarray:
    try:
        target_h = float(target.get("h", 0.0))
        target_s = float(target.get("s", 0.0))
        target_v = float(target.get("v", 0.0))
        tol_h = float(target.get("tol_h", _HUE_TOLERANCE))
        tol_s = float(target.get("tol_s", _HSV_SAT_TOLERANCE))
        tol_v = float(target.get("tol_v", _HSV_VAL_TOLERANCE))
    except (TypeError, ValueError):
        target_h = 0.0
        target_s = 0.0
        target_v = 0.0
        tol_h = _HUE_TOLERANCE
        tol_s = _HSV_SAT_TOLERANCE
        tol_v = _HSV_VAL_TOLERANCE
    diff = np.abs(((hue - target_h + 128) % 256) - 128)
    mask = diff <= tol_h
    mask &= np.abs(sat - target_s) <= tol_s
    mask &= np.abs(val - target_v) <= tol_v
    return mask


def _extract_candlestick_series(image: Image.Image, calibration: Dict[str, Any]) -> SeriesResult | None:
    hsv = image.convert("HSV")
    arr = np.array(hsv)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    hue = arr[:, :, 0].astype(np.int16)
    sat = arr[:, :, 1].astype(np.int16)
    val = arr[:, :, 2].astype(np.int16)
    up_target = calibration.get("candle_up") if isinstance(calibration, dict) else None
    down_target = calibration.get("candle_down") if isinstance(calibration, dict) else None
    if not isinstance(up_target, dict) and not isinstance(down_target, dict):
        return None

    masks = []
    if isinstance(up_target, dict):
        masks.append(_mask_from_target(hue, sat, val, up_target))
    if isinstance(down_target, dict):
        masks.append(_mask_from_target(hue, sat, val, down_target))
    if not masks:
        return None
    combined = np.logical_or.reduce(masks)
    coverage, median_range = _measure_line_shape(combined)
    if coverage < _LINE_COVERAGE_MIN * 0.6:
        return None

    width, height = image.size
    x_indices = np.arange(width)
    y_values = np.full(width, np.nan)
    for x in range(width):
        ys = np.where(combined[:, x])[0]
        if ys.size:
            y_values[x] = float(np.median(ys))
    valid = ~np.isnan(y_values)
    if int(valid.sum()) < max(_MIN_SERIES_POINTS, width * 0.15):
        return None
    y_values = np.interp(x_indices, x_indices[valid], y_values[valid])
    series = 1.0 - (y_values / max(1.0, float(height - 1)))
    diagnostics = {
        "coverage": round(float(coverage), 3),
        "median_range": round(float(median_range), 2),
        "points": int(valid.sum()),
        "calibrated": True,
    }
    return SeriesResult(series=series, diagnostics=diagnostics, mode="candlestick")


def _classify_pattern(series: np.ndarray) -> Dict[str, Any]:
    length = len(series)
    x = np.linspace(0.0, 1.0, length)
    maxima, minima = _find_extrema(series, order=5)
    diagnostics: Dict[str, Any] = {
        "series_length": length,
        "maxima": int(len(maxima)),
        "minima": int(len(minima)),
    }

    if len(maxima) < 4 or len(minima) < 4:
        fit = _fit_line(x, series)
        diagnostics.update(
            {
                "upper_slope": fit.slope,
                "lower_slope": fit.slope,
                "convergence": 0.0,
            }
        )
        return _trend_fallback(fit, diagnostics)

    upper_fit = _fit_line(x[maxima], series[maxima])
    lower_fit = _fit_line(x[minima], series[minima])
    gap_start = upper_fit.intercept - lower_fit.intercept
    gap_end = (upper_fit.slope + upper_fit.intercept) - (lower_fit.slope + lower_fit.intercept)
    convergence = 0.0
    if gap_start > 0:
        convergence = max(0.0, min(1.0, (gap_start - gap_end) / gap_start))

    diagnostics.update(
        {
            "upper_slope": upper_fit.slope,
            "lower_slope": lower_fit.slope,
            "upper_r2": upper_fit.r2,
            "lower_r2": lower_fit.r2,
            "convergence": convergence,
        }
    )

    pattern_key = _resolve_pattern_key(upper_fit.slope, lower_fit.slope, convergence, gap_start, gap_end)
    bias = _pattern_bias(pattern_key)
    confidence = _pattern_confidence(
        pattern_key,
        upper_fit,
        lower_fit,
        convergence,
        len(maxima),
        len(minima),
    )
    probabilities = _bias_probabilities(bias, confidence)

    return {
        "pattern_key": pattern_key,
        "bias": bias,
        "confidence": confidence,
        "probabilities": probabilities,
        "upper_fit": upper_fit,
        "lower_fit": lower_fit,
        "diagnostics": diagnostics,
    }


def _trend_fallback(fit: LineFit, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    slope = fit.slope
    if slope > _FLAT_SLOPE:
        pattern_key = "trend_up"
    elif slope < -_FLAT_SLOPE:
        pattern_key = "trend_down"
    else:
        pattern_key = "trend_flat"
    bias = _pattern_bias(pattern_key)
    confidence = max(0.2, min(0.85, 0.25 + 0.6 * max(0.0, fit.r2)))
    probabilities = _bias_probabilities(bias, confidence)
    return {
        "pattern_key": pattern_key,
        "bias": bias,
        "confidence": round(float(confidence), 3),
        "probabilities": probabilities,
        "upper_fit": None,
        "lower_fit": None,
        "diagnostics": diagnostics,
    }


def _resolve_pattern_key(
    upper_slope: float,
    lower_slope: float,
    convergence: float,
    gap_start: float,
    gap_end: float,
) -> str:
    upper_flat = abs(upper_slope) <= _FLAT_SLOPE
    lower_flat = abs(lower_slope) <= _FLAT_SLOPE
    converging = convergence >= _CONVERGENCE_MIN and gap_end >= 0.0

    if converging:
        if upper_slope > _FLAT_SLOPE and lower_slope > _FLAT_SLOPE:
            return "rising_wedge" if lower_slope > upper_slope else "rising_converging"
        if upper_slope < -_FLAT_SLOPE and lower_slope < -_FLAT_SLOPE:
            return "falling_wedge" if lower_slope > upper_slope else "falling_converging"
        if upper_flat and lower_slope > _FLAT_SLOPE:
            return "ascending_triangle"
        if lower_flat and upper_slope < -_FLAT_SLOPE:
            return "descending_triangle"
        if upper_slope < -_FLAT_SLOPE and lower_slope > _FLAT_SLOPE:
            return "sym_triangle"
        return "converging"

    if abs(upper_slope - lower_slope) <= _SLOPE_PARALLEL and gap_start > 0:
        if upper_slope > _FLAT_SLOPE:
            return "channel_up"
        if upper_slope < -_FLAT_SLOPE:
            return "channel_down"
        return "range"

    if upper_slope > _FLAT_SLOPE and lower_slope > _FLAT_SLOPE:
        return "trend_up"
    if upper_slope < -_FLAT_SLOPE and lower_slope < -_FLAT_SLOPE:
        return "trend_down"
    return "range"


def _pattern_bias(pattern_key: str) -> str:
    bearish = {
        "rising_wedge",
        "descending_triangle",
        "channel_down",
        "trend_down",
        "falling_converging",
    }
    bullish = {
        "falling_wedge",
        "ascending_triangle",
        "channel_up",
        "trend_up",
        "rising_converging",
    }
    neutral = {
        "sym_triangle",
        "range",
        "trend_flat",
        "converging",
    }
    if pattern_key in bearish:
        return "bearish"
    if pattern_key in bullish:
        return "bullish"
    if pattern_key in neutral:
        return "neutral"
    return "neutral"


def _pattern_confidence(
    pattern_key: str,
    upper_fit: LineFit,
    lower_fit: LineFit,
    convergence: float,
    max_count: int,
    min_count: int,
) -> float:
    base_r2 = max(0.0, min(1.0, min(upper_fit.r2, lower_fit.r2)))
    extrema_score = min(1.0, (max_count + min_count) / 18.0)
    convergence_score = min(1.0, max(0.0, convergence))
    weight = 0.2 if pattern_key in {"trend_up", "trend_down", "trend_flat"} else 0.35
    confidence = 0.2 + 0.45 * base_r2 + 0.2 * extrema_score + weight * convergence_score
    return round(float(max(0.1, min(0.95, confidence))), 3)


def _bias_probabilities(bias: str, confidence: float) -> Dict[str, float]:
    conf = max(0.1, min(0.95, confidence))
    if bias == "neutral":
        neutral = 0.55 + 0.25 * conf
        remain = max(0.0, 1.0 - neutral)
        up = remain / 2
        down = remain / 2
    else:
        primary = 0.45 + 0.45 * conf
        secondary = 0.1 + 0.2 * (1.0 - conf)
        neutral = max(0.0, 1.0 - primary - secondary)
        if bias == "bullish":
            up, down = primary, secondary
        else:
            up, down = secondary, primary
    return {
        "up": round(float(up), 3),
        "down": round(float(down), 3),
        "neutral": round(float(neutral), 3),
    }


def _find_extrema(series: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    maxima: list[int] = []
    minima: list[int] = []
    length = len(series)
    for idx in range(order, length - order):
        window = series[idx - order : idx + order + 1]
        value = series[idx]
        if value >= window.max():
            if not maxima or idx - maxima[-1] >= order:
                maxima.append(idx)
        if value <= window.min():
            if not minima or idx - minima[-1] >= order:
                minima.append(idx)
    return np.array(maxima, dtype=int), np.array(minima, dtype=int)


def _estimate_convergence(series: np.ndarray, maxima: np.ndarray, minima: np.ndarray) -> Dict[str, float]:
    if series.size == 0 or len(maxima) < 3 or len(minima) < 3:
        return {"upper_slope": 0.0, "lower_slope": 0.0, "convergence": 0.0}
    length = len(series)
    x = np.linspace(0.0, 1.0, length)
    upper_fit = _fit_line(x[maxima], series[maxima])
    lower_fit = _fit_line(x[minima], series[minima])
    gap_start = upper_fit.intercept - lower_fit.intercept
    gap_end = (upper_fit.slope + upper_fit.intercept) - (lower_fit.slope + lower_fit.intercept)
    convergence = 0.0
    if gap_start > 0:
        convergence = max(0.0, min(1.0, (gap_start - gap_end) / gap_start))
    return {
        "upper_slope": float(upper_fit.slope),
        "lower_slope": float(lower_fit.slope),
        "convergence": float(convergence),
    }


def _max_drawdown(series: np.ndarray) -> float:
    if series.size == 0:
        return 0.0
    peak = float(series[0])
    max_dd = 0.0
    for value in series:
        value = float(value)
        if value > peak:
            peak = value
        if peak > 0:
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
    return float(max_dd)


def _fit_line(x: np.ndarray, y: np.ndarray) -> LineFit:
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0
    if denom > 0:
        r2 = 1.0 - float(np.sum((y - y_pred) ** 2)) / denom
    return LineFit(slope=float(slope), intercept=float(intercept), r2=float(r2))


def _render_overlay(image: Image.Image, upper_fit: LineFit | None, lower_fit: LineFit | None) -> str:
    if upper_fit is None or lower_fit is None:
        return ""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size

    upper_line = _line_pixels(upper_fit, width, height)
    lower_line = _line_pixels(lower_fit, width, height)
    draw.line(upper_line, fill=(220, 38, 38), width=2)
    draw.line(lower_line, fill=(16, 185, 129), width=2)

    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _line_pixels(fit: LineFit, width: int, height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    def _series_to_px(value: float) -> int:
        return int(round((1.0 - value) * max(1, height - 1)))

    y0 = _series_to_px(fit.intercept)
    y1 = _series_to_px(fit.slope + fit.intercept)
    return (0, y0), (max(1, width - 1), y1)


__all__ = [
    "analyze_screen_frame",
    "build_feature_vector",
    "PATTERN_KEYS",
    "FEATURE_NAMES",
]
