from __future__ import annotations

import math
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from django.conf import settings
except Exception:  # pragma: no cover - optional
    settings = None  # type: ignore


def _get_setting(name: str, default):
    if settings is None:
        return default
    return getattr(settings, name, default)


MAX_POINTS = int(_get_setting("SCREEN_WAVE_MAX_POINTS", 360))
SMOOTH_WINDOW = int(_get_setting("SCREEN_WAVE_SMOOTH_WINDOW", 3))
PIVOT_ORDER = int(_get_setting("SCREEN_WAVE_PIVOT_ORDER", 5))
ZIGZAG_THRESHOLD = float(_get_setting("SCREEN_WAVE_ZIGZAG", 0.03))
STATE_TTL_SECONDS = int(_get_setting("SCREEN_WAVE_STATE_TTL", 20))
STATE_MAX_SIZE = int(_get_setting("SCREEN_WAVE_STATE_MAX", 200))
STABILITY_BONUS = float(_get_setting("SCREEN_WAVE_STABILITY_BONUS", 0.05))
STABILITY_PENALTY = float(_get_setting("SCREEN_WAVE_STABILITY_PENALTY", 0.1))
STABILITY_THRESHOLD = float(_get_setting("SCREEN_WAVE_STABILITY_THRESHOLD", 0.02))


@dataclass(slots=True)
class WaveResult:
    wave_key: str
    stage: str
    direction: str
    confidence: float
    probabilities: Dict[str, float]
    pivots: List[Dict[str, Any]]
    count: List[Dict[str, Any]]
    fib: Dict[str, Any]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wave_key": self.wave_key,
            "stage": self.stage,
            "direction": self.direction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "pivots": self.pivots,
            "count": self.count,
            "fib": self.fib,
            "diagnostics": self.diagnostics,
        }


@dataclass(slots=True)
class WaveState:
    wave_key: str
    pivots: List[Dict[str, Any]]
    ts: float


_STATE: Dict[str, WaveState] = {}
_STATE_LOCK = threading.Lock()


def analyze_waves(
    series: np.ndarray,
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    analysis_mode: Optional[str] = None,
    session_id: Optional[str] = None,
) -> WaveResult:
    prepared = _prepare_series(series)
    pivots = _pivots_from_series(prepared)
    diagnostics: Dict[str, Any] = {
        "pivot_count": len(pivots),
        "series_length": int(prepared.size),
    }

    if len(pivots) < 4:
        return _unknown_result(pivots, diagnostics)

    candidates = []
    impulse = _score_impulse(prepared, pivots)
    if impulse:
        candidates.append(impulse)
    abc = _score_abc(prepared, pivots)
    if abc:
        candidates.append(abc)

    if not candidates:
        return _unknown_result(pivots, diagnostics)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    chosen = candidates[0]
    diagnostics.update(
        {
            "candidate_count": len(candidates),
            "chosen_rank": 1,
        }
    )

    key = _state_key(symbol, timeframe, analysis_mode, session_id=session_id)
    stability = _apply_stability(chosen, key)
    diagnostics.update(stability)
    chosen["confidence"] = _clamp(chosen["confidence"])

    result = WaveResult(
        wave_key=chosen["wave_key"],
        stage=chosen["stage"],
        direction=chosen["direction"],
        confidence=chosen["confidence"],
        probabilities=_probabilities_from_direction(chosen["direction"], chosen["confidence"]),
        pivots=chosen["pivots"],
        count=chosen["count"],
        fib=chosen["fib"],
        diagnostics={**diagnostics, **chosen.get("diagnostics", {})},
    )
    return result


def _unknown_result(pivots: List[Dict[str, Any]], diagnostics: Dict[str, Any]) -> WaveResult:
    return WaveResult(
        wave_key="unknown",
        stage="unknown",
        direction="neutral",
        confidence=0.2,
        probabilities=_probabilities_from_direction("neutral", 0.2),
        pivots=pivots,
        count=[],
        fib={},
        diagnostics=diagnostics,
    )


def _prepare_series(series: np.ndarray) -> np.ndarray:
    if series.size == 0:
        return series
    trimmed = series[-MAX_POINTS:]
    if SMOOTH_WINDOW <= 1 or trimmed.size < SMOOTH_WINDOW + 2:
        return _normalize(trimmed)
    window = min(SMOOTH_WINDOW, trimmed.size)
    kernel = np.ones(window) / window
    smoothed = np.convolve(trimmed, kernel, mode="same")
    return _normalize(smoothed)


def _normalize(series: np.ndarray) -> np.ndarray:
    if series.size == 0:
        return series
    minimum = float(np.min(series))
    maximum = float(np.max(series))
    if maximum == minimum:
        return series
    return (series - minimum) / (maximum - minimum)


def _pivots_from_series(series: np.ndarray) -> List[Dict[str, Any]]:
    order = max(3, min(PIVOT_ORDER, max(3, int(series.size / 40))))
    maxima, minima = _find_extrema(series, order=order)
    candidates = sorted(
        [(idx, "H") for idx in maxima] + [(idx, "L") for idx in minima],
        key=lambda item: item[0],
    )
    pivots: List[Tuple[int, float, str]] = []
    last_idx = None
    last_val = None
    last_kind = None
    for idx, kind in candidates:
        val = float(series[idx])
        if last_val is None:
            pivots.append((idx, val, kind))
            last_idx, last_val, last_kind = idx, val, kind
            continue
        if kind == last_kind:
            replace = (kind == "H" and val >= last_val) or (kind == "L" and val <= last_val)
            if replace:
                pivots[-1] = (idx, val, kind)
                last_idx, last_val = idx, val
            continue
        if abs(val - last_val) < ZIGZAG_THRESHOLD:
            continue
        pivots.append((idx, val, kind))
        last_idx, last_val, last_kind = idx, val, kind

    trimmed = pivots[-12:]
    return [{"idx": idx, "value": round(val, 6), "kind": kind} for idx, val, kind in trimmed]


def _score_impulse(series: np.ndarray, pivots: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(pivots) < 6:
        return None
    segment = pivots[-6:]
    kinds = [pivot["kind"] for pivot in segment]
    if kinds == ["L", "H", "L", "H", "L", "H"]:
        direction = "up"
    elif kinds == ["H", "L", "H", "L", "H", "L"]:
        direction = "down"
    else:
        return None
    values = [float(pivot["value"]) for pivot in segment]
    lengths = [abs(values[i + 1] - values[i]) for i in range(5)]
    w1, w2, w3, w4, w5 = lengths
    penalties = {}

    if direction == "up":
        penalties["wave2_breach"] = 1.0 if values[2] <= values[0] else 0.0
        penalties["wave4_overlap"] = 1.0 if values[4] <= values[1] else 0.0
    else:
        penalties["wave2_breach"] = 1.0 if values[2] >= values[0] else 0.0
        penalties["wave4_overlap"] = 1.0 if values[4] >= values[1] else 0.0

    penalties["wave3_shortest"] = 1.0 if w3 <= min(w1, w5) else 0.0

    ratios = {
        "w2_ratio": _safe_ratio(w2, w1),
        "w3_ratio": _safe_ratio(w3, w1),
        "w5_ratio": _safe_ratio(w5, w1),
    }
    fib_penalty = (
        _ratio_penalty(ratios["w2_ratio"], [0.382, 0.5, 0.618])
        + _ratio_penalty(ratios["w3_ratio"], [1.0, 1.618, 2.618])
        + _ratio_penalty(ratios["w5_ratio"], [0.618, 1.0, 1.618])
    ) / 3.0

    rule_penalty = (
        penalties["wave2_breach"] * 0.5 + penalties["wave4_overlap"] * 0.3 + penalties["wave3_shortest"] * 0.4
    )
    score = 1.0 - min(1.0, rule_penalty + fib_penalty * 0.6)
    fit_r2 = _pivot_fit_r2(series, segment)
    score *= 0.5 + 0.5 * fit_r2

    wave_key = "impulse_up" if direction == "up" else "impulse_down"
    count_labels = ["0", "1", "2", "3", "4", "5"]
    count = [{"pivot_index": idx, "label": label} for idx, label in enumerate(count_labels)]
    return {
        "wave_key": wave_key,
        "stage": "impulse",
        "direction": direction,
        "confidence": round(score, 3),
        "score": score,
        "pivots": segment,
        "count": count,
        "fib": ratios,
        "diagnostics": {"rule_penalties": penalties, "fit_r2": round(fit_r2, 3)},
    }


def _score_abc(series: np.ndarray, pivots: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(pivots) < 4:
        return None
    segment = pivots[-4:]
    kinds = [pivot["kind"] for pivot in segment]
    if kinds == ["L", "H", "L", "H"]:
        direction = "up"
    elif kinds == ["H", "L", "H", "L"]:
        direction = "down"
    else:
        return None
    values = [float(pivot["value"]) for pivot in segment]
    a = abs(values[1] - values[0])
    b = abs(values[2] - values[1])
    c = abs(values[3] - values[2])
    ratios = {
        "b_ratio": _safe_ratio(b, a),
        "c_ratio": _safe_ratio(c, a),
    }
    penalties = {
        "b_ratio": _ratio_penalty(ratios["b_ratio"], [0.382, 0.5, 0.618, 0.786]),
        "c_ratio": _ratio_penalty(ratios["c_ratio"], [1.0, 1.272, 1.618]),
    }
    score = 1.0 - min(1.0, (penalties["b_ratio"] + penalties["c_ratio"]) / 2.0)
    fit_r2 = _pivot_fit_r2(series, segment)
    score *= 0.5 + 0.5 * fit_r2

    wave_key = "abc_up" if direction == "up" else "abc_down"
    count_labels = ["A", "B", "C"]
    count = [{"pivot_index": idx, "label": label} for idx, label in enumerate(count_labels)]
    return {
        "wave_key": wave_key,
        "stage": "correction",
        "direction": direction,
        "confidence": round(score, 3),
        "score": score,
        "pivots": segment,
        "count": count,
        "fib": ratios,
        "diagnostics": {"ratio_penalties": penalties, "fit_r2": round(fit_r2, 3)},
    }


def _pivot_fit_r2(series: np.ndarray, pivots: List[Dict[str, Any]]) -> float:
    if series.size == 0 or not pivots:
        return 0.0
    indices = [int(p["idx"]) for p in pivots]
    values = [float(p["value"]) for p in pivots]
    x = np.arange(series.size)
    y_pred = np.interp(x, indices, values)
    y = series
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom == 0:
        return 0.0
    return max(0.0, 1.0 - float(np.sum((y - y_pred) ** 2)) / denom)


def _ratio_penalty(ratio: float, targets: List[float]) -> float:
    if ratio <= 0:
        return 1.0
    diffs = [abs(ratio - target) / target for target in targets]
    return min(1.0, min(diffs))


def _safe_ratio(value: float, base: float) -> float:
    if base <= 0:
        return 0.0
    return value / base


def _probabilities_from_direction(direction: str, confidence: float) -> Dict[str, float]:
    conf = _clamp(confidence, 0.1, 0.95)
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


def _find_extrema(series: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    maxima: List[int] = []
    minima: List[int] = []
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


def _state_key(
    symbol: Optional[str],
    timeframe: Optional[str],
    analysis_mode: Optional[str],
    *,
    session_id: Optional[str] = None,
) -> Optional[str]:
    mode = analysis_mode or "auto"
    if symbol and timeframe:
        return f"{symbol}:{timeframe}:{mode}"
    if session_id:
        return f"session:{session_id}:{mode}"
    return None


def _apply_stability(candidate: Dict[str, Any], key: Optional[str]) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {"stability_applied": False}
    if not key:
        return diagnostics
    prev = _get_state(key)
    if prev is None:
        _set_state(key, candidate)
        return diagnostics
    diagnostics["stability_applied"] = True
    shift = _pivot_shift(prev.pivots, candidate["pivots"])
    diagnostics["pivot_shift"] = round(shift, 4) if shift is not None else None
    if prev.wave_key == candidate["wave_key"] and shift is not None and shift < STABILITY_THRESHOLD:
        candidate["confidence"] = _clamp(candidate["confidence"] + STABILITY_BONUS)
        diagnostics["stability_bonus"] = True
    elif prev.wave_key != candidate["wave_key"] and shift is not None and shift < STABILITY_THRESHOLD:
        candidate["confidence"] = _clamp(candidate["confidence"] - STABILITY_PENALTY)
        diagnostics["stability_penalty"] = True
    _set_state(key, candidate)
    return diagnostics


def _pivot_shift(prev: List[Dict[str, Any]], curr: List[Dict[str, Any]]) -> Optional[float]:
    if not prev or not curr:
        return None
    count = min(len(prev), len(curr), 4)
    if count <= 0:
        return None
    diffs = []
    for idx in range(1, count + 1):
        diffs.append(abs(float(prev[-idx]["value"]) - float(curr[-idx]["value"])))
    return float(sum(diffs) / count)


def _get_state(key: str) -> Optional[WaveState]:
    now = time.time()
    with _STATE_LOCK:
        _prune_state(now)
        return _STATE.get(key)


def _set_state(key: str, candidate: Dict[str, Any]) -> None:
    now = time.time()
    with _STATE_LOCK:
        _prune_state(now)
        if len(_STATE) >= STATE_MAX_SIZE:
            oldest = sorted(_STATE.items(), key=lambda item: item[1].ts)[: max(1, STATE_MAX_SIZE // 4)]
            for item_key, _ in oldest:
                _STATE.pop(item_key, None)
        _STATE[key] = WaveState(
            wave_key=candidate["wave_key"],
            pivots=candidate["pivots"],
            ts=now,
        )


def _prune_state(now: float) -> None:
    expired = [key for key, state in _STATE.items() if now - state.ts > STATE_TTL_SECONDS]
    for key in expired:
        _STATE.pop(key, None)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


__all__ = ["analyze_waves", "WaveResult"]
