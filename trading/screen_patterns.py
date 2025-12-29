from __future__ import annotations

import base64
import colorsys
import io
import math
import os
import threading
import time
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
_QUALITY_MIN = float(
    getattr(
        settings,
        "SCREEN_ANALYZER_QUALITY_MIN",
        os.environ.get("SCREEN_ANALYZER_QUALITY_MIN", "0.35"),
    )
    if settings
    else os.environ.get("SCREEN_ANALYZER_QUALITY_MIN", "0.35")
)
_QUALITY_NEUTRAL = float(
    getattr(
        settings,
        "SCREEN_ANALYZER_QUALITY_NEUTRAL",
        os.environ.get("SCREEN_ANALYZER_QUALITY_NEUTRAL", "0.45"),
    )
    if settings
    else os.environ.get("SCREEN_ANALYZER_QUALITY_NEUTRAL", "0.45")
)
_MODEL_MIN_CONF = float(
    getattr(
        settings,
        "SCREEN_ANALYZER_MODEL_MIN_CONF",
        os.environ.get("SCREEN_ANALYZER_MODEL_MIN_CONF", "0.55"),
    )
    if settings
    else os.environ.get("SCREEN_ANALYZER_MODEL_MIN_CONF", "0.55")
)
_FUSION_ALIGN_BONUS = float(
    getattr(
        settings,
        "SCREEN_FUSION_ALIGN_BONUS",
        os.environ.get("SCREEN_FUSION_ALIGN_BONUS", "0.05"),
    )
    if settings
    else os.environ.get("SCREEN_FUSION_ALIGN_BONUS", "0.05")
)
_FUSION_CONFLICT_PENALTY = float(
    getattr(
        settings,
        "SCREEN_FUSION_CONFLICT_PENALTY",
        os.environ.get("SCREEN_FUSION_CONFLICT_PENALTY", "0.07"),
    )
    if settings
    else os.environ.get("SCREEN_FUSION_CONFLICT_PENALTY", "0.07")
)
_FUSION_PIVOT_SHIFT_THRESHOLD = float(
    getattr(
        settings,
        "SCREEN_FUSION_PIVOT_SHIFT_THRESHOLD",
        os.environ.get("SCREEN_FUSION_PIVOT_SHIFT_THRESHOLD", "0.02"),
    )
    if settings
    else os.environ.get("SCREEN_FUSION_PIVOT_SHIFT_THRESHOLD", "0.02")
)
_OVERLAY_LAYERS_RAW = (
    getattr(settings, "SCREEN_ANALYZER_OVERLAY_LAYERS", os.environ.get("SCREEN_ANALYZER_OVERLAY_LAYERS"))
    if settings
    else os.environ.get("SCREEN_ANALYZER_OVERLAY_LAYERS")
)

_MULTI_SCALE_ENABLED = (
    getattr(
        settings,
        "SCREEN_ANALYZER_MULTI_SCALE_ENABLED",
        os.environ.get("SCREEN_ANALYZER_MULTI_SCALE_ENABLED", "1") in {"1", "true", "True"},
    )
    if settings
    else os.environ.get("SCREEN_ANALYZER_MULTI_SCALE_ENABLED", "1") in {"1", "true", "True"}
)
_MULTI_SCALE_FACTORS_RAW = (
    getattr(settings, "SCREEN_ANALYZER_MULTI_SCALE_FACTORS", os.environ.get("SCREEN_ANALYZER_MULTI_SCALE_FACTORS"))
    if settings
    else os.environ.get("SCREEN_ANALYZER_MULTI_SCALE_FACTORS")
)
_SERIES_SMOOTH_ENABLED = (
    getattr(
        settings,
        "SCREEN_ANALYZER_SMOOTH_ENABLED",
        os.environ.get("SCREEN_ANALYZER_SMOOTH_ENABLED", "1") in {"1", "true", "True"},
    )
    if settings
    else os.environ.get("SCREEN_ANALYZER_SMOOTH_ENABLED", "1") in {"1", "true", "True"}
)
_SERIES_SMOOTH_WINDOW = int(
    getattr(
        settings,
        "SCREEN_ANALYZER_SMOOTH_WINDOW",
        os.environ.get("SCREEN_ANALYZER_SMOOTH_WINDOW", "5"),
    )
    if settings
    else os.environ.get("SCREEN_ANALYZER_SMOOTH_WINDOW", "5")
)
_ADAPTIVE_INTERVAL_ENABLED = (
    getattr(
        settings,
        "SCREEN_ANALYZER_ADAPTIVE_ENABLED",
        os.environ.get("SCREEN_ANALYZER_ADAPTIVE_ENABLED", "1") in {"1", "true", "True"},
    )
    if settings
    else os.environ.get("SCREEN_ANALYZER_ADAPTIVE_ENABLED", "1") in {"1", "true", "True"}
)
_ADAPTIVE_INTERVAL_MIN = int(
    getattr(
        settings,
        "SCREEN_ANALYZER_ADAPTIVE_MIN_INTERVAL_MS",
        os.environ.get("SCREEN_ANALYZER_ADAPTIVE_MIN_INTERVAL_MS", "700"),
    )
    if settings
    else os.environ.get("SCREEN_ANALYZER_ADAPTIVE_MIN_INTERVAL_MS", "700")
)
_ADAPTIVE_INTERVAL_MAX = int(
    getattr(
        settings,
        "SCREEN_ANALYZER_ADAPTIVE_MAX_INTERVAL_MS",
        os.environ.get("SCREEN_ANALYZER_ADAPTIVE_MAX_INTERVAL_MS", "2000"),
    )
    if settings
    else os.environ.get("SCREEN_ANALYZER_ADAPTIVE_MAX_INTERVAL_MS", "2000")
)
_ADAPTIVE_VOL_LOW = float(
    getattr(
        settings,
        "SCREEN_ANALYZER_ADAPTIVE_VOL_LOW",
        os.environ.get("SCREEN_ANALYZER_ADAPTIVE_VOL_LOW", "0.004"),
    )
    if settings
    else os.environ.get("SCREEN_ANALYZER_ADAPTIVE_VOL_LOW", "0.004")
)
_ADAPTIVE_VOL_HIGH = float(
    getattr(
        settings,
        "SCREEN_ANALYZER_ADAPTIVE_VOL_HIGH",
        os.environ.get("SCREEN_ANALYZER_ADAPTIVE_VOL_HIGH", "0.02"),
    )
    if settings
    else os.environ.get("SCREEN_ANALYZER_ADAPTIVE_VOL_HIGH", "0.02")
)
_MOMENTUM_WINDOW = int(
    getattr(
        settings,
        "SCREEN_SIGNAL_MOMENTUM_WINDOW",
        os.environ.get("SCREEN_SIGNAL_MOMENTUM_WINDOW", "32"),
    )
    if settings
    else os.environ.get("SCREEN_SIGNAL_MOMENTUM_WINDOW", "32")
)
_MOMENTUM_MIN_DELTA = float(
    getattr(
        settings,
        "SCREEN_SIGNAL_MOMENTUM_MIN_DELTA",
        os.environ.get("SCREEN_SIGNAL_MOMENTUM_MIN_DELTA", "0.02"),
    )
    if settings
    else os.environ.get("SCREEN_SIGNAL_MOMENTUM_MIN_DELTA", "0.02")
)
_MOMENTUM_DELTA_SCALE = float(
    getattr(
        settings,
        "SCREEN_SIGNAL_MOMENTUM_DELTA_SCALE",
        os.environ.get("SCREEN_SIGNAL_MOMENTUM_DELTA_SCALE", "0.08"),
    )
    if settings
    else os.environ.get("SCREEN_SIGNAL_MOMENTUM_DELTA_SCALE", "0.08")
)
_MOMENTUM_MIN_CONF = float(
    getattr(
        settings,
        "SCREEN_SIGNAL_MOMENTUM_MIN_CONF",
        os.environ.get("SCREEN_SIGNAL_MOMENTUM_MIN_CONF", "0.35"),
    )
    if settings
    else os.environ.get("SCREEN_SIGNAL_MOMENTUM_MIN_CONF", "0.35")
)
_MOMENTUM_WEIGHT = float(
    getattr(
        settings,
        "SCREEN_SIGNAL_MOMENTUM_WEIGHT",
        os.environ.get("SCREEN_SIGNAL_MOMENTUM_WEIGHT", "0.15"),
    )
    if settings
    else os.environ.get("SCREEN_SIGNAL_MOMENTUM_WEIGHT", "0.15")
)
_SIGNAL_SMOOTH_ALPHA = float(
    getattr(
        settings,
        "SCREEN_SIGNAL_SMOOTH_ALPHA",
        os.environ.get("SCREEN_SIGNAL_SMOOTH_ALPHA", "0.45"),
    )
    if settings
    else os.environ.get("SCREEN_SIGNAL_SMOOTH_ALPHA", "0.45")
)
_SIGNAL_STATE_TTL = int(
    getattr(
        settings,
        "SCREEN_SIGNAL_STATE_TTL",
        os.environ.get("SCREEN_SIGNAL_STATE_TTL", "30"),
    )
    if settings
    else os.environ.get("SCREEN_SIGNAL_STATE_TTL", "30")
)
_SIGNAL_STATE_MAX = int(
    getattr(
        settings,
        "SCREEN_SIGNAL_STATE_MAX",
        os.environ.get("SCREEN_SIGNAL_STATE_MAX", "200"),
    )
    if settings
    else os.environ.get("SCREEN_SIGNAL_STATE_MAX", "200")
)
_SIGNAL_COOLDOWN_SECONDS = int(
    getattr(
        settings,
        "SCREEN_SIGNAL_COOLDOWN_SECONDS",
        os.environ.get("SCREEN_SIGNAL_COOLDOWN_SECONDS", "6"),
    )
    if settings
    else os.environ.get("SCREEN_SIGNAL_COOLDOWN_SECONDS", "6")
)
_SIGNAL_COOLDOWN_PULLBACK = float(
    getattr(
        settings,
        "SCREEN_SIGNAL_COOLDOWN_PULLBACK",
        os.environ.get("SCREEN_SIGNAL_COOLDOWN_PULLBACK", "0.1"),
    )
    if settings
    else os.environ.get("SCREEN_SIGNAL_COOLDOWN_PULLBACK", "0.1")
)
_SIGNAL_CALIBRATION_TEMP = float(
    getattr(
        settings,
        "SCREEN_SIGNAL_CALIBRATION_TEMP",
        os.environ.get("SCREEN_SIGNAL_CALIBRATION_TEMP", "1.15"),
    )
    if settings
    else os.environ.get("SCREEN_SIGNAL_CALIBRATION_TEMP", "1.15")
)
_SIGNAL_DIRECTION_THRESHOLD = float(
    getattr(
        settings,
        "SCREEN_SIGNAL_DIRECTION_THRESHOLD",
        os.environ.get("SCREEN_SIGNAL_DIRECTION_THRESHOLD", "0.55"),
    )
    if settings
    else os.environ.get("SCREEN_SIGNAL_DIRECTION_THRESHOLD", "0.55")
)

_WAVE_ENABLED = (
    getattr(settings, "SCREEN_WAVE_ENABLED", os.environ.get("SCREEN_WAVE_ENABLED", "1") in {"1", "true", "True"})
    if settings
    else os.environ.get("SCREEN_WAVE_ENABLED", "1") in {"1", "true", "True"}
)
_WAVE_FUSION_PATTERN_WEIGHT = float(
    getattr(
        settings,
        "SCREEN_WAVE_FUSION_PATTERN_WEIGHT",
        os.environ.get("SCREEN_WAVE_FUSION_PATTERN_WEIGHT", "0.6"),
    )
    if settings
    else os.environ.get("SCREEN_WAVE_FUSION_PATTERN_WEIGHT", "0.6")
)
_WAVE_FUSION_WAVE_WEIGHT = float(
    getattr(
        settings,
        "SCREEN_WAVE_FUSION_WAVE_WEIGHT",
        os.environ.get("SCREEN_WAVE_FUSION_WAVE_WEIGHT", "0.4"),
    )
    if settings
    else os.environ.get("SCREEN_WAVE_FUSION_WAVE_WEIGHT", "0.4")
)
_WAVE_FUSION_MIN_CONF = float(
    getattr(
        settings,
        "SCREEN_WAVE_FUSION_MIN_CONF",
        os.environ.get("SCREEN_WAVE_FUSION_MIN_CONF", "0.35"),
    )
    if settings
    else os.environ.get("SCREEN_WAVE_FUSION_MIN_CONF", "0.35")
)
_WAVE_FUSION_CONFLICT_THRESHOLD = float(
    getattr(
        settings,
        "SCREEN_WAVE_FUSION_CONFLICT_THRESHOLD",
        os.environ.get("SCREEN_WAVE_FUSION_CONFLICT_THRESHOLD", "0.55"),
    )
    if settings
    else os.environ.get("SCREEN_WAVE_FUSION_CONFLICT_THRESHOLD", "0.55")
)
_WAVE_FUSION_CONFLICT_PULLBACK = float(
    getattr(
        settings,
        "SCREEN_WAVE_FUSION_CONFLICT_PULLBACK",
        os.environ.get("SCREEN_WAVE_FUSION_CONFLICT_PULLBACK", "0.12"),
    )
    if settings
    else os.environ.get("SCREEN_WAVE_FUSION_CONFLICT_PULLBACK", "0.12")
)

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


@dataclass(slots=True)
class SignalState:
    probabilities: Dict[str, float]
    direction: str
    ts: float


_SIGNAL_STATE: Dict[str, SignalState] = {}
_SIGNAL_LOCK = threading.Lock()


def _parse_float_list(raw: Optional[str], default: list[float]) -> list[float]:
    if not raw:
        return default
    values = []
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = float(chunk)
        except ValueError:
            continue
        if value <= 0:
            continue
        values.append(value)
    return values or default


def _parse_str_list(raw: Optional[str], default: list[str]) -> list[str]:
    if not raw:
        return default
    values: list[str] = []
    for chunk in str(raw).split(","):
        item = chunk.strip().lower()
        if item:
            values.append(item)
    return values or default


_MULTI_SCALE_FACTORS = _parse_float_list(_MULTI_SCALE_FACTORS_RAW, [1.0, 0.8, 0.6])
_OVERLAY_LAYERS_DEFAULT = _parse_str_list(_OVERLAY_LAYERS_RAW, ["trendlines"])

def analyze_screen_frame(
    data_url: str,
    *,
    mode: str = "auto",
    calibration: Optional[Dict[str, Any]] = None,
    header_image: Optional[str] = None,
    enable_ocr: bool = True,
    include_features: bool = False,
    include_waves: bool = True,
    include_fusion: bool = True,
    include_timings: bool = False,
    overlay_layers: Optional[list[str]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    stage_start = time.perf_counter()
    image, decode_diag, decode_error, decode_action = _decode_data_url_with_diagnostics(data_url)
    timings["decode"] = _elapsed_ms(stage_start)
    if image is None:
        return _error_response(
            decode_error or "invalid_image",
            next_action=decode_action,
            diagnostics=decode_diag,
            timings=timings if include_timings else None,
        )
    stage_start = time.perf_counter()
    image = _resize_for_analysis(image)
    timings["resize"] = _elapsed_ms(stage_start)
    mode = (mode or "auto").lower()
    if mode not in {"auto", "line", "candlestick"}:
        mode = "auto"
    calibration = _normalize_calibration(calibration)
    if isinstance(overlay_layers, list):
        overlay_layers = [str(layer).lower() for layer in overlay_layers if layer]

    stage_start = time.perf_counter()
    series_result = _extract_series(image, mode, calibration)
    timings["extract_series"] = _elapsed_ms(stage_start)
    if series_result is None:
        return _error_response(
            "series_not_found",
            next_action="Adjust the crop area or recalibrate line/candle colors.",
            diagnostics={"mode": mode, "image_width": image.size[0], "image_height": image.size[1]},
            timings=timings if include_timings else None,
        )

    suggested_interval = None
    analysis_volatility = None
    if _ADAPTIVE_INTERVAL_ENABLED:
        suggestion = _suggest_interval_ms(series_result.series)
        if suggestion:
            suggested_interval, analysis_volatility = suggestion

    stage_start = time.perf_counter()
    pattern = _classify_pattern(series_result.series)
    timings["classify_pattern"] = _elapsed_ms(stage_start)
    stage_start = time.perf_counter()
    model_result = _maybe_apply_model(series_result.series, pattern)
    timings["model"] = _elapsed_ms(stage_start)
    quality_factor, quality_diagnostics = _apply_quality_gating(pattern, series_result.diagnostics)
    series_result.diagnostics.update(quality_diagnostics)

    symbol = None
    timeframe = None
    ocr_payload: Dict[str, Any] = {}
    if enable_ocr and header_image:
        stage_start = time.perf_counter()
        ocr_payload = _extract_ocr_payload(header_image)
        timings["ocr"] = _elapsed_ms(stage_start)
        symbol = ocr_payload.get("symbol")
        timeframe = ocr_payload.get("timeframe")

    stage_start = time.perf_counter()
    momentum_payload = _momentum_signal(series_result.series)
    timings["momentum"] = _elapsed_ms(stage_start)
    wave_payload: Optional[Dict[str, Any]] = None
    fused_probabilities: Optional[Dict[str, float]] = None
    fusion_diagnostics: Dict[str, Any] = {}
    include_waves = bool(include_waves) and _WAVE_ENABLED
    include_fusion = bool(include_fusion)
    if include_waves:
        try:
            from .screen_waves import analyze_waves

            stage_start = time.perf_counter()
            wave_result = analyze_waves(
                series_result.series,
                symbol=symbol,
                timeframe=timeframe,
                analysis_mode=series_result.mode,
                session_id=session_id,
            )
            timings["waves"] = _elapsed_ms(stage_start)
            wave_payload = wave_result.to_dict()
        except Exception as exc:  # pragma: no cover - defensive
            fusion_diagnostics = {"wave_error": str(exc)}

    signals = [
        {
            "name": "pattern",
            "probabilities": pattern.get("probabilities", {}),
            "confidence": pattern.get("confidence", 0.0),
            "weight": _WAVE_FUSION_PATTERN_WEIGHT,
            "quality": quality_factor,
        }
    ]
    if wave_payload and isinstance(wave_payload, dict):
        wave_quality = _wave_quality_factor(wave_payload)
        diagnostics = wave_payload.get("diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics["quality_factor"] = round(wave_quality, 3)
        signals.append(
            {
                "name": "wave",
                "probabilities": wave_payload.get("probabilities", {}),
                "confidence": wave_payload.get("confidence", 0.0),
                "weight": _WAVE_FUSION_WAVE_WEIGHT,
                "wave_key": wave_payload.get("wave_key"),
                "quality": wave_quality,
            }
        )
    if momentum_payload:
        signals.append(
            {
                "name": "momentum",
                "probabilities": momentum_payload.get("probabilities", {}),
                "confidence": momentum_payload.get("confidence", 0.0),
                "weight": _MOMENTUM_WEIGHT,
                "direction": momentum_payload.get("direction"),
                "quality": quality_factor,
            }
        )
    fusion_payload: Optional[Dict[str, Any]] = None
    if include_fusion:
        stage_start = time.perf_counter()
        fused_probabilities, fusion_info = _fuse_probabilities(
            signals,
            symbol=symbol,
            timeframe=timeframe,
            analysis_mode=series_result.mode,
            session_id=session_id,
        )
        timings["fusion"] = _elapsed_ms(stage_start)
        fusion_diagnostics = {**fusion_diagnostics, **fusion_info}
        if fused_probabilities:
            fused_direction = _dominant_direction(fused_probabilities, _SIGNAL_DIRECTION_THRESHOLD)
            if fused_direction == "up":
                fusion_bias = "bullish"
                fusion_confidence = max(fused_probabilities.get("up", 0.0), fused_probabilities.get("down", 0.0))
            elif fused_direction == "down":
                fusion_bias = "bearish"
                fusion_confidence = max(fused_probabilities.get("up", 0.0), fused_probabilities.get("down", 0.0))
            else:
                fusion_bias = "neutral"
                fusion_confidence = fused_probabilities.get("neutral", 0.0)
            alignment = None
            pivot_shift = None
            bonus = 0.0
            penalty = 0.0
            if wave_payload and isinstance(wave_payload, dict):
                wave_direction = wave_payload.get("direction")
                pivot_shift = None
                diagnostics = wave_payload.get("diagnostics")
                if isinstance(diagnostics, dict):
                    pivot_shift = diagnostics.get("pivot_shift")
                bias_dir = "neutral"
                if pattern.get("bias") == "bullish":
                    bias_dir = "up"
                elif pattern.get("bias") == "bearish":
                    bias_dir = "down"
                if wave_direction in {"up", "down"} and bias_dir in {"up", "down"}:
                    if wave_direction == bias_dir:
                        alignment = "aligned"
                        bonus = _FUSION_ALIGN_BONUS
                    elif pivot_shift is not None and float(pivot_shift) < _FUSION_PIVOT_SHIFT_THRESHOLD:
                        alignment = "conflict"
                        penalty = _FUSION_CONFLICT_PENALTY
            fusion_confidence = max(0.1, min(0.95, fusion_confidence + bonus - penalty))
            fusion_payload = {
                "probabilities": fused_probabilities,
                "bias": fusion_bias,
                "confidence": round(float(fusion_confidence), 3),
                "method": "weighted_log_smooth_v1",
                "diagnostics": {
                    **fusion_diagnostics,
                    "alignment": alignment,
                    "pivot_shift": round(float(pivot_shift), 4) if pivot_shift is not None else None,
                    "confidence_bonus": round(float(bonus), 3) if bonus else 0.0,
                    "confidence_penalty": round(float(penalty), 3) if penalty else 0.0,
                },
            }

    stage_start = time.perf_counter()
    overlay = ""
    try:
        overlay = _render_overlay(
            image,
            pattern.get("upper_fit"),
            pattern.get("lower_fit"),
            wave_payload=wave_payload,
            overlay_layers=overlay_layers,
            series_length=len(series_result.series),
        )
    except Exception as exc:  # pragma: no cover - defensive
        response_error = f"{exc.__class__.__name__}"
        response = {"overlay_error": response_error}
        series_result.diagnostics.update(response)
    timings["overlay"] = _elapsed_ms(stage_start)

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
            **fusion_diagnostics,
        },
    }
    if suggested_interval is not None:
        response["suggested_interval_ms"] = suggested_interval
        if analysis_volatility is not None:
            response["diagnostics"]["analysis_volatility"] = round(float(analysis_volatility), 6)
    if include_waves and wave_payload is not None:
        response["wave"] = wave_payload
    if include_fusion and fused_probabilities is not None:
        response["fused_probabilities"] = fused_probabilities
    if include_fusion and fusion_payload is not None:
        response["fusion"] = fusion_payload
    if momentum_payload:
        response["diagnostics"].update(
            {
                "momentum_direction": momentum_payload.get("direction"),
                "momentum_confidence": momentum_payload.get("confidence"),
                "momentum_window": momentum_payload.get("window"),
            }
        )
    if include_timings:
        response["timings_ms"] = _finalize_timings(timings)
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
    candidates = _extract_series_candidates(image, mode, calibration)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0]["result"]


def _extract_series_candidates(
    image: Image.Image, mode: str, calibration: Dict[str, Any]
) -> list[Dict[str, Any]]:
    scales = _MULTI_SCALE_FACTORS if _MULTI_SCALE_ENABLED else [1.0]
    candidates: list[Dict[str, Any]] = []
    for scale in scales:
        if scale <= 0:
            continue
        if scale == 1.0:
            scaled = image
        else:
            width, height = image.size
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            scaled = image.resize(new_size, Image.LANCZOS)
        candidate = _extract_series_single(scaled, mode, calibration, scale=scale)
        if candidate:
            candidates.append(candidate)
    return candidates


def _extract_series_single(
    image: Image.Image, mode: str, calibration: Dict[str, Any], *, scale: float
) -> Optional[Dict[str, Any]]:
    if mode == "line":
        result = _extract_line_series(image, calibration)
    elif mode == "candlestick":
        result = _extract_candlestick_series(image, calibration)
    else:
        result = _extract_line_series(image, calibration) or _extract_candlestick_series(image, calibration)
    if result is None:
        return None
    series = result.series
    smoothed = False
    noise = 0.0
    if _SERIES_SMOOTH_ENABLED and _SERIES_SMOOTH_WINDOW > 1:
        series, smoothed, noise = _smooth_series(series, _SERIES_SMOOTH_WINDOW)
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "scale": round(float(scale), 3),
            "smoothed": smoothed,
            "smooth_window": _SERIES_SMOOTH_WINDOW if smoothed else 0,
            "noise": round(float(noise), 5),
        }
    )
    score = _series_quality_score(series, diagnostics)
    diagnostics["quality_score"] = round(float(score), 4)
    return {
        "result": SeriesResult(series=series, diagnostics=diagnostics, mode=result.mode),
        "score": score,
    }


def _smooth_series(series: np.ndarray, window: int) -> Tuple[np.ndarray, bool, float]:
    if series.size == 0 or window <= 1:
        return series, False, 0.0
    window = max(3, int(window))
    if window % 2 == 0:
        window -= 1
    if series.size < window:
        return series, False, 0.0
    pad = window // 2
    kernel = np.ones(window) / window
    padded = np.pad(series, pad, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    noise = float(np.mean(np.abs(series - smoothed))) if series.size else 0.0
    return smoothed, True, noise


def _series_quality_score(series: np.ndarray, diagnostics: Dict[str, Any]) -> float:
    coverage = float(diagnostics.get("coverage", 0.0))
    points = float(diagnostics.get("points", 0.0))
    width = float(diagnostics.get("width", max(1, series.size)))
    height = float(diagnostics.get("height", 1.0))
    median_range = float(diagnostics.get("median_range", height))
    points_ratio = min(1.0, points / max(1.0, width))
    range_ratio = 1.0 - min(1.0, median_range / max(1.0, height))
    score = coverage * (0.6 + 0.4 * points_ratio) * (0.55 + 0.45 * range_ratio)
    if diagnostics.get("calibrated"):
        score *= 1.05
    noise = float(diagnostics.get("noise", 0.0))
    if noise > 0:
        score *= max(0.6, 1.0 - min(0.4, noise * 4.5))
    return float(max(0.0, min(1.0, score)))


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
    min_conf = _MODEL_MIN_CONF
    threshold_source = "default"
    try:
        from .screen_training import load_model_meta

        meta = load_model_meta()
        if isinstance(meta, dict):
            override_threshold = meta.get("override_threshold")
            if override_threshold is not None:
                min_conf = float(override_threshold)
                source = meta.get("override_source")
                threshold_source = source if isinstance(source, str) else "meta"
    except Exception:
        pass
    min_conf = max(0.0, min(1.0, float(min_conf)))
    diagnostics = {
        "model_used": True,
        "model_confidence": round(confidence, 3),
        "model_threshold": round(min_conf, 3),
        "model_threshold_source": threshold_source,
    }
    if best_label and confidence >= min_conf:
        pattern["pattern_key"] = best_label
        pattern["bias"] = _pattern_bias(best_label)
        pattern["confidence"] = round(max(pattern["confidence"], confidence), 3)
        pattern["probabilities"] = _bias_probabilities(pattern["bias"], pattern["confidence"])
        diagnostics["model_override"] = True
    else:
        diagnostics["model_override"] = False
    return {"diagnostics": diagnostics}


def _decode_data_url_with_diagnostics(
    data_url: str,
) -> Tuple[Optional[Image.Image], Dict[str, Any], Optional[str], Optional[str]]:
    diagnostics: Dict[str, Any] = {}
    if not data_url or not isinstance(data_url, str):
        return None, diagnostics, "invalid_image", "Provide a valid image data URL."
    if not data_url.startswith("data:image/"):
        return None, diagnostics, "invalid_image", "Use a data:image/... URL."
    try:
        _, encoded = data_url.split(",", 1)
    except ValueError:
        return None, diagnostics, "invalid_image", "Use a valid data URL."
    estimated_bytes = (len(encoded) * 3) // 4
    diagnostics["estimated_bytes"] = int(estimated_bytes)
    if estimated_bytes > _MAX_BYTES:
        return None, diagnostics, "invalid_image", "Reduce capture size or quality."
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception:
        return None, diagnostics, "invalid_image", "Re-capture the chart area."
    diagnostics["decoded_bytes"] = int(len(raw))
    if len(raw) > _MAX_BYTES:
        return None, diagnostics, "invalid_image", "Reduce capture size or quality."
    try:
        image = Image.open(io.BytesIO(raw))
        image = image.convert("RGB")
        diagnostics["image_width"] = image.size[0]
        diagnostics["image_height"] = image.size[1]
        return image, diagnostics, None, None
    except Exception:
        return None, diagnostics, "invalid_image", "Use a clear chart screenshot."


def _decode_data_url(data_url: str) -> Image.Image | None:
    image, _, _, _ = _decode_data_url_with_diagnostics(data_url)
    return image


def _resize_for_analysis(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width <= _MAX_WIDTH and height <= _MAX_HEIGHT:
        return image
    scale = min(_MAX_WIDTH / max(width, 1), _MAX_HEIGHT / max(height, 1), 1.0)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


def _finalize_timings(timings: Dict[str, float]) -> Dict[str, float]:
    total = round(sum(timings.values()), 2)
    return {**timings, "total": total}


def _error_response(
    error: str,
    *,
    next_action: Optional[str] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    response: Dict[str, Any] = {"error": error}
    if next_action:
        response["next_action"] = next_action
    if diagnostics:
        response["diagnostics"] = diagnostics
    if timings:
        response["timings_ms"] = _finalize_timings(timings)
    return response


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
    mask_coverage = float(line_mask.sum()) / max(1.0, float(line_mask.size))
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
        "mask_coverage": round(float(mask_coverage), 4),
        "median_range": round(float(median_range), 2),
        "points": int(valid.sum()),
        "calibrated": calibrated,
        "width": int(width),
        "height": int(height),
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
    masks = []
    calibrated = False
    if isinstance(up_target, dict):
        masks.append(_mask_from_target(hue, sat, val, up_target))
        calibrated = True
    if isinstance(down_target, dict):
        masks.append(_mask_from_target(hue, sat, val, down_target))
        calibrated = True
    auto_hues: list[int] = []
    if not masks:
        base_mask = (sat >= _SAT_THRESHOLD) & (val >= _VAL_THRESHOLD)
        if int(base_mask.sum()) < max(200, image.size[0]):
            return None
        hist = np.bincount(hue[base_mask].ravel(), minlength=256)
        top1 = int(hist.argmax())
        hist2 = hist.copy()
        span = max(8, int(_HUE_TOLERANCE * 2))
        hist2[max(0, top1 - span) : min(256, top1 + span + 1)] = 0
        top2 = int(hist2.argmax())
        auto_hues.append(top1)
        if hist2[top2] >= max(30, hist[top1] * 0.08):
            auto_hues.append(top2)
        for hue_center in auto_hues:
            diff = np.abs(((hue - hue_center + 128) % 256) - 128)
            masks.append(base_mask & (diff <= _HUE_TOLERANCE))
        if not masks:
            return None
    combined = np.logical_or.reduce(masks)
    coverage, median_range = _measure_line_shape(combined)
    mask_coverage = float(combined.sum()) / max(1.0, float(combined.size))
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
        "mask_coverage": round(float(mask_coverage), 4),
        "median_range": round(float(median_range), 2),
        "points": int(valid.sum()),
        "calibrated": calibrated,
        "width": int(width),
        "height": int(height),
    }
    if auto_hues:
        diagnostics["auto_hues"] = auto_hues
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


def _direction_probabilities(direction: str, confidence: float) -> Dict[str, float]:
    conf = max(0.1, min(0.95, confidence))
    if direction == "neutral":
        neutral = 0.55 + 0.25 * conf
        remain = max(0.0, 1.0 - neutral)
        up = remain / 2
        down = remain / 2
    else:
        primary = 0.45 + 0.45 * conf
        secondary = 0.1 + 0.2 * (1.0 - conf)
        neutral = max(0.0, 1.0 - primary - secondary)
        if direction == "up":
            up, down = primary, secondary
        else:
            up, down = secondary, primary
    return {
        "up": round(float(up), 3),
        "down": round(float(down), 3),
        "neutral": round(float(neutral), 3),
    }


def _quality_factor_from_diagnostics(diagnostics: Dict[str, Any]) -> Tuple[float, list[str]]:
    reasons: list[str] = []
    quality_score = diagnostics.get("quality_score")
    if isinstance(quality_score, (int, float)):
        quality = float(quality_score)
    else:
        coverage = float(diagnostics.get("coverage", 0.0))
        points = float(diagnostics.get("points", 0.0))
        width = float(diagnostics.get("width", max(1.0, points)))
        points_ratio = min(1.0, points / max(1.0, width))
        noise = float(diagnostics.get("noise", 0.0))
        noise_factor = max(0.0, 1.0 - min(1.0, noise * 5.0))
        quality = coverage * (0.6 + 0.4 * points_ratio) * (0.55 + 0.45 * noise_factor)
    mask_coverage = float(diagnostics.get("mask_coverage", 0.0))
    if quality < _QUALITY_NEUTRAL:
        reasons.append("low_quality")
    if float(diagnostics.get("coverage", 0.0)) < _LINE_COVERAGE_MIN * 0.8:
        reasons.append("coverage_low")
    if mask_coverage and mask_coverage < 0.004:
        reasons.append("mask_sparse")
    if float(diagnostics.get("points", 0.0)) < max(20.0, _MIN_SERIES_POINTS * 0.4):
        reasons.append("points_low")
    if float(diagnostics.get("noise", 0.0)) > 0.08:
        reasons.append("noise_high")
    quality = max(0.0, min(1.0, quality))
    return quality, reasons


def _apply_quality_gating(pattern: Dict[str, Any], diagnostics: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    quality, reasons = _quality_factor_from_diagnostics(diagnostics)
    gated = False
    if quality < _QUALITY_NEUTRAL:
        pattern["bias"] = "neutral"
        gated = True
    confidence = float(pattern.get("confidence", 0.0))
    adjusted_confidence = confidence * (0.6 + 0.4 * quality)
    if quality < _QUALITY_MIN:
        adjusted_confidence = min(adjusted_confidence, 0.35)
    pattern["confidence"] = round(max(0.1, min(0.95, adjusted_confidence)), 3)
    pattern["probabilities"] = _bias_probabilities(pattern.get("bias", "neutral"), pattern["confidence"])
    quality_diagnostics = {
        "quality_factor": round(float(quality), 3),
        "quality_gated": gated,
    }
    if reasons:
        quality_diagnostics["quality_reasons"] = reasons
    return quality, quality_diagnostics


def _wave_quality_factor(wave_payload: Dict[str, Any]) -> float:
    pivots = wave_payload.get("pivots")
    pivot_count = len(pivots) if isinstance(pivots, list) else 0
    pivot_factor = min(1.0, pivot_count / 6.0)
    confidence = float(wave_payload.get("confidence", 0.0))
    quality = 0.5 * pivot_factor + 0.5 * min(1.0, max(0.0, confidence))
    return max(0.0, min(1.0, quality))


def _momentum_signal(series: np.ndarray) -> Optional[Dict[str, Any]]:
    if series.size < max(10, _MOMENTUM_WINDOW // 2):
        return None
    window = min(_MOMENTUM_WINDOW, series.size)
    segment = series[-window:]
    x = np.linspace(0.0, 1.0, window)
    fit = _fit_line(x, segment)
    delta = float(segment[-1] - segment[0])
    direction = "neutral"
    if delta >= _MOMENTUM_MIN_DELTA:
        direction = "up"
    elif delta <= -_MOMENTUM_MIN_DELTA:
        direction = "down"
    strength = min(1.0, abs(delta) / max(1e-6, _MOMENTUM_DELTA_SCALE))
    confidence = 0.25 + 0.5 * strength + 0.25 * max(0.0, min(1.0, fit.r2))
    confidence = max(0.1, min(0.9, confidence))
    return {
        "direction": direction,
        "confidence": round(float(confidence), 3),
        "probabilities": _direction_probabilities(direction, confidence),
        "window": int(window),
        "fit_r2": round(float(fit.r2), 3),
    }


def _suggest_interval_ms(series: np.ndarray) -> Optional[Tuple[int, float]]:
    if series.size < 4:
        return None
    returns = np.diff(series)
    if returns.size == 0:
        return None
    volatility = float(np.std(returns))
    low = min(_ADAPTIVE_VOL_LOW, _ADAPTIVE_VOL_HIGH)
    high = max(_ADAPTIVE_VOL_LOW, _ADAPTIVE_VOL_HIGH)
    if high <= low:
        return int(_ADAPTIVE_INTERVAL_MAX), volatility
    ratio = (volatility - low) / (high - low)
    ratio = max(0.0, min(1.0, ratio))
    interval = _ADAPTIVE_INTERVAL_MAX - int(round((_ADAPTIVE_INTERVAL_MAX - _ADAPTIVE_INTERVAL_MIN) * ratio))
    interval = max(min(interval, _ADAPTIVE_INTERVAL_MAX), _ADAPTIVE_INTERVAL_MIN)
    return int(interval), volatility


def _normalize_probabilities(probabilities: Dict[str, float]) -> Dict[str, float]:
    values = {}
    total = 0.0
    for key in ("up", "down", "neutral"):
        raw = probabilities.get(key, 0.0)
        try:
            value = max(0.0, float(raw))
        except (TypeError, ValueError):
            value = 0.0
        values[key] = value
        total += value
    if total <= 0:
        return {"up": 0.333, "down": 0.333, "neutral": 0.334}
    return {key: value / total for key, value in values.items()}


def _dominant_direction(probabilities: Dict[str, float], threshold: float) -> str:
    best_key = max(probabilities, key=probabilities.get)
    if best_key in {"up", "down"} and probabilities.get(best_key, 0.0) >= threshold:
        return best_key
    return "neutral"


def _apply_conflict_pullback(probabilities: Dict[str, float], amount: Optional[float] = None) -> Dict[str, float]:
    pullback = _WAVE_FUSION_CONFLICT_PULLBACK if amount is None else float(amount)
    neutral = min(0.9, probabilities.get("neutral", 0.0) + pullback)
    remaining = max(0.0, 1.0 - neutral)
    ud_sum = probabilities.get("up", 0.0) + probabilities.get("down", 0.0)
    if ud_sum <= 0:
        return {"up": remaining / 2, "down": remaining / 2, "neutral": neutral}
    up = remaining * probabilities.get("up", 0.0) / ud_sum
    down = remaining * probabilities.get("down", 0.0) / ud_sum
    return {"up": up, "down": down, "neutral": neutral}


def _temperature_calibrate(probabilities: Dict[str, float], temperature: float) -> Dict[str, float]:
    if temperature <= 0.0 or abs(temperature - 1.0) < 1e-3:
        return probabilities
    epsilon = 1e-6
    adjusted = {
        key: math.exp(math.log(max(epsilon, probabilities.get(key, 0.0))) / temperature)
        for key in ("up", "down", "neutral")
    }
    return _normalize_probabilities(adjusted)


def _signal_state_key(
    symbol: Optional[str],
    timeframe: Optional[str],
    analysis_mode: Optional[str],
    *,
    session_id: Optional[str] = None,
) -> Optional[str]:
    if symbol and timeframe:
        return f"{symbol}:{timeframe}:{analysis_mode or 'auto'}"
    if session_id:
        return f"session:{session_id}:{analysis_mode or 'auto'}"
    return None


def _get_signal_state(key: str) -> Optional[SignalState]:
    now = time.time()
    with _SIGNAL_LOCK:
        _prune_signal_state(now)
        return _SIGNAL_STATE.get(key)


def _set_signal_state(key: str, probabilities: Dict[str, float], direction: str) -> None:
    now = time.time()
    with _SIGNAL_LOCK:
        _prune_signal_state(now)
        if len(_SIGNAL_STATE) >= _SIGNAL_STATE_MAX:
            oldest = sorted(_SIGNAL_STATE.items(), key=lambda item: item[1].ts)[: max(1, _SIGNAL_STATE_MAX // 4)]
            for item_key, _ in oldest:
                _SIGNAL_STATE.pop(item_key, None)
        _SIGNAL_STATE[key] = SignalState(probabilities=probabilities, direction=direction, ts=now)


def _prune_signal_state(now: float) -> None:
    expired = [key for key, state in _SIGNAL_STATE.items() if now - state.ts > _SIGNAL_STATE_TTL]
    for key in expired:
        _SIGNAL_STATE.pop(key, None)


def _apply_signal_smoothing(
    probabilities: Dict[str, float],
    *,
    symbol: Optional[str],
    timeframe: Optional[str],
    analysis_mode: Optional[str],
    session_id: Optional[str],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"signal_smoothed": False}
    key = _signal_state_key(symbol, timeframe, analysis_mode, session_id=session_id)
    if not key:
        return probabilities, diagnostics
    prev = _get_signal_state(key)
    if prev is None:
        direction = _dominant_direction(probabilities, _SIGNAL_DIRECTION_THRESHOLD)
        _set_signal_state(key, probabilities, direction)
        diagnostics["signal_state_new"] = True
        return probabilities, diagnostics
    alpha = max(0.0, min(1.0, _SIGNAL_SMOOTH_ALPHA))
    smoothed = {}
    for prob_key in ("up", "down", "neutral"):
        current = float(probabilities.get(prob_key, 0.0))
        previous = float(prev.probabilities.get(prob_key, 0.0))
        smoothed[prob_key] = alpha * current + (1.0 - alpha) * previous
    smoothed = _normalize_probabilities(smoothed)
    direction = _dominant_direction(smoothed, _SIGNAL_DIRECTION_THRESHOLD)
    cooldown_applied = False
    if prev.direction != direction and (time.time() - prev.ts) < _SIGNAL_COOLDOWN_SECONDS:
        smoothed = _normalize_probabilities(_apply_conflict_pullback(smoothed, _SIGNAL_COOLDOWN_PULLBACK))
        cooldown_applied = True
        direction = _dominant_direction(smoothed, _SIGNAL_DIRECTION_THRESHOLD)
    _set_signal_state(key, smoothed, direction)
    diagnostics.update(
        {
            "signal_smoothed": True,
            "signal_alpha": round(alpha, 3),
            "signal_direction": direction,
            "signal_prev_direction": prev.direction,
            "signal_cooldown": cooldown_applied,
        }
    )
    return smoothed, diagnostics


def _fuse_probabilities(
    signals: list[Dict[str, Any]],
    *,
    symbol: Optional[str],
    timeframe: Optional[str],
    analysis_mode: Optional[str],
    session_id: Optional[str],
) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {"fusion_used": False}
    if not signals:
        diagnostics["fusion_reason"] = "no_signals"
        return None, diagnostics

    epsilon = 1e-6
    weighted: list[Tuple[str, Dict[str, float], float, float]] = []
    signal_details: Dict[str, Any] = {}
    for signal in signals:
        name = signal.get("name") or "signal"
        probs = signal.get("probabilities")
        if not isinstance(probs, dict):
            continue
        wave_key = signal.get("wave_key")
        if name == "wave" and wave_key == "unknown":
            signal_details[name] = {"skipped": "wave_unknown"}
            continue
        conf = max(0.0, min(1.0, float(signal.get("confidence") or 0.0)))
        if name == "wave" and conf < _WAVE_FUSION_MIN_CONF:
            signal_details[name] = {"skipped": "low_conf", "confidence": round(conf, 3)}
            continue
        if name == "momentum" and conf < _MOMENTUM_MIN_CONF:
            signal_details[name] = {"skipped": "low_conf", "confidence": round(conf, 3)}
            continue
        weight = max(0.0, float(signal.get("weight") or 0.0))
        if weight <= 0:
            continue
        quality = max(0.0, min(1.0, float(signal.get("quality") or 1.0)))
        norm_probs = _normalize_probabilities(probs)
        adj_weight = weight * (0.6 + 0.4 * conf) * (0.6 + 0.4 * quality)
        weighted.append((name, norm_probs, adj_weight, conf))
        signal_details[name] = {
            "weight": round(adj_weight, 3),
            "confidence": round(conf, 3),
            "quality": round(quality, 3),
            "direction": _dominant_direction(norm_probs, _SIGNAL_DIRECTION_THRESHOLD),
            "wave_key": wave_key,
        }

    if len(weighted) < 2:
        diagnostics["fusion_reason"] = "insufficient_signals"
        diagnostics["signals"] = signal_details
        return None, diagnostics

    total_weight = sum(item[2] for item in weighted)
    if total_weight <= 0:
        diagnostics["fusion_reason"] = "zero_weight"
        diagnostics["signals"] = signal_details
        return None, diagnostics

    fused_raw = {}
    for key in ("up", "down", "neutral"):
        log_sum = 0.0
        for _, probs, weight, _ in weighted:
            log_sum += (weight / total_weight) * math.log(max(epsilon, probs.get(key, 0.0)))
        fused_raw[key] = math.exp(log_sum)
    fused = _normalize_probabilities(fused_raw)

    conflict = False
    directions = {item[0]: _dominant_direction(item[1], _WAVE_FUSION_CONFLICT_THRESHOLD) for item in weighted}
    if "up" in directions.values() and "down" in directions.values():
        conflict = True
        fused = _normalize_probabilities(_apply_conflict_pullback(fused))

    fused = _temperature_calibrate(fused, _SIGNAL_CALIBRATION_TEMP)
    fused, smoothing_diagnostics = _apply_signal_smoothing(
        fused,
        symbol=symbol,
        timeframe=timeframe,
        analysis_mode=analysis_mode,
        session_id=session_id,
    )
    diagnostics.update(smoothing_diagnostics)
    diagnostics.update(
        {
            "fusion_used": True,
            "fusion_conflict": conflict,
            "signals": signal_details,
        }
    )
    return {key: round(float(value), 3) for key, value in fused.items()}, diagnostics


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


def _render_overlay(
    image: Image.Image,
    upper_fit: LineFit | None,
    lower_fit: LineFit | None,
    *,
    wave_payload: Optional[Dict[str, Any]] = None,
    overlay_layers: Optional[list[str]] = None,
    series_length: Optional[int] = None,
) -> str:
    layers = {layer for layer in (overlay_layers or _OVERLAY_LAYERS_DEFAULT) if layer}
    if not layers:
        return ""
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    drawn = False

    if "trendlines" in layers and upper_fit is not None and lower_fit is not None:
        upper_line = _line_pixels(upper_fit, width, height)
        lower_line = _line_pixels(lower_fit, width, height)
        draw.line(upper_line, fill=(220, 38, 38), width=2)
        draw.line(lower_line, fill=(16, 185, 129), width=2)
        drawn = True

    if wave_payload and series_length and series_length > 1:
        pivots = wave_payload.get("pivots")
        if isinstance(pivots, list) and "pivots" in layers:
            for pivot in pivots:
                idx = pivot.get("idx")
                value = pivot.get("value")
                kind = pivot.get("kind")
                if idx is None or value is None:
                    continue
                x = int(round((float(idx) / (series_length - 1)) * (width - 1)))
                y = int(round((1.0 - float(value)) * (height - 1)))
                radius = 4
                color = (239, 68, 68) if kind == "H" else (16, 185, 129)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(15, 23, 42))
                drawn = True

        counts = wave_payload.get("count")
        if isinstance(counts, list) and isinstance(pivots, list) and "wave_count" in layers:
            for item in counts:
                pivot_index = item.get("pivot_index")
                label = item.get("label")
                if pivot_index is None or label is None:
                    continue
                if not isinstance(pivot_index, int) or pivot_index >= len(pivots):
                    continue
                pivot = pivots[pivot_index]
                idx = pivot.get("idx")
                value = pivot.get("value")
                if idx is None or value is None:
                    continue
                x = int(round((float(idx) / (series_length - 1)) * (width - 1)))
                y = int(round((1.0 - float(value)) * (height - 1)))
                draw.text((x + 6, y - 12), str(label), fill=(15, 23, 42))
                drawn = True

    if not drawn:
        return ""
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
