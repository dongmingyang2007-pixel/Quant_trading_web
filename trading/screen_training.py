from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from django.conf import settings
except Exception:  # pragma: no cover - optional
    settings = None  # type: ignore

try:  # optional dependency
    import joblib  # type: ignore
except Exception:  # pragma: no cover - fallback
    joblib = None  # type: ignore

try:  # optional dependency
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import accuracy_score  # type: ignore
except Exception:  # pragma: no cover - optional
    RandomForestClassifier = None  # type: ignore
    train_test_split = None  # type: ignore
    accuracy_score = None  # type: ignore

from .screen_patterns import FEATURE_NAMES, PATTERN_KEYS

_MODEL_MIN_CONF_DEFAULT = float(
    getattr(settings, "SCREEN_ANALYZER_MODEL_MIN_CONF", os.environ.get("SCREEN_ANALYZER_MODEL_MIN_CONF", "0.55"))
    if settings
    else os.environ.get("SCREEN_ANALYZER_MODEL_MIN_CONF", "0.55")
)
_MODEL_THRESHOLD_CANDIDATES = [
    float(item)
    for item in os.environ.get(
        "SCREEN_ANALYZER_MODEL_THRESHOLD_CANDIDATES",
        "0.45,0.5,0.55,0.6,0.65,0.7,0.75",
    ).split(",")
    if item.strip()
]
_DEFAULT_NAMESPACE = "screen_analyzer"
_NAMESPACE_SANITIZER = re.compile(r"[^a-z0-9_\-]+")


@dataclass(slots=True)
class TrainingMetrics:
    total_samples: int
    classes: Dict[str, int]
    accuracy: Optional[float]
    test_size: int
    override_threshold: Optional[float] = None
    override_accuracy: Optional[float] = None
    override_coverage: Optional[float] = None
    override_samples: Optional[int] = None
    override_source: Optional[str] = None


def _resolve_namespace(namespace: str | None) -> str:
    raw = (namespace or _DEFAULT_NAMESPACE).strip().lower()
    if not raw:
        raw = _DEFAULT_NAMESPACE
    sanitized = _NAMESPACE_SANITIZER.sub("_", raw).strip("._-")
    if not sanitized:
        sanitized = _DEFAULT_NAMESPACE
    return sanitized[:64]


def _data_dir(namespace: str | None = None) -> Path:
    base = Path(getattr(settings, "DATA_CACHE_DIR", Path.cwd()))
    path = base / _resolve_namespace(namespace)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _samples_path(namespace: str | None = None) -> Path:
    return _data_dir(namespace) / "pattern_samples.jsonl"


def _model_path(namespace: str | None = None) -> Path:
    return _data_dir(namespace) / "pattern_model.pkl"


def _model_meta_path(namespace: str | None = None) -> Path:
    return _data_dir(namespace) / "pattern_model_meta.json"


def save_sample(
    features: List[float],
    label: str,
    meta: Optional[Dict[str, Any]] = None,
    *,
    namespace: str | None = None,
) -> None:
    if label not in PATTERN_KEYS:
        raise ValueError("invalid_label")
    entry = {
        "label": label,
        "features": features,
        "meta": meta or {},
        "ts": time.time(),
    }
    path = _samples_path(namespace)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_samples(*, namespace: str | None = None) -> List[Dict[str, Any]]:
    path = _samples_path(namespace)
    if not path.exists():
        return []
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            samples.append(entry)
    return samples


def train_model(min_samples: int = 18, *, namespace: str | None = None) -> TrainingMetrics:
    if RandomForestClassifier is None or train_test_split is None or accuracy_score is None:
        raise RuntimeError("sklearn_unavailable")
    samples = load_samples(namespace=namespace)
    valid: List[Dict[str, Any]] = [
        sample
        for sample in samples
        if isinstance(sample.get("features"), list)
        and len(sample.get("features", [])) == len(FEATURE_NAMES)
        and isinstance(sample.get("label"), str)
        and sample["label"] in PATTERN_KEYS
    ]
    if len(valid) < min_samples:
        raise RuntimeError("insufficient_samples")
    x = [sample["features"] for sample in valid]
    y = [sample["label"] for sample in valid]
    class_counts: Dict[str, int] = {}
    for label in y:
        class_counts[label] = class_counts.get(label, 0) + 1
    if len(class_counts) < 2:
        raise RuntimeError("insufficient_classes")

    test_size = 0.2 if len(valid) >= 25 else 0.0
    if test_size > 0:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=13)
    else:
        x_train, x_test, y_train, y_test = x, [], y, []

    model = RandomForestClassifier(
        n_estimators=180,
        random_state=13,
        class_weight="balanced",
        max_depth=8,
    )
    model.fit(x_train, y_train)

    labels = list(getattr(model, "classes_", sorted(set(y))))
    accuracy = None
    override_summary: Optional[Dict[str, Any]] = None
    if x_test:
        preds = model.predict(x_test)
        accuracy = float(accuracy_score(y_test, preds))
        try:
            probs = model.predict_proba(x_test)
            override_summary = _select_override_threshold(
                probs,
                y_test,
                labels=labels,
                thresholds=_MODEL_THRESHOLD_CANDIDATES,
            )
        except Exception:
            override_summary = None

    _save_model(model, labels=labels, namespace=namespace)
    override_threshold = _MODEL_MIN_CONF_DEFAULT
    override_source = "default"
    override_accuracy = None
    override_coverage = None
    override_samples = None
    if override_summary:
        override_threshold = float(override_summary["threshold"])
        override_source = "validation"
        override_accuracy = float(override_summary["accuracy"])
        override_coverage = float(override_summary["coverage"])
        override_samples = int(override_summary["samples"])
    meta = TrainingMetrics(
        total_samples=len(valid),
        classes=class_counts,
        accuracy=accuracy,
        test_size=len(x_test),
        override_threshold=override_threshold,
        override_accuracy=override_accuracy,
        override_coverage=override_coverage,
        override_samples=override_samples,
        override_source=override_source,
    )
    _write_meta(meta, namespace=namespace)
    return meta


def _save_model(model: Any, *, labels: List[str], namespace: str | None = None) -> None:
    path = _model_path(namespace)
    payload = {"model": model, "labels": labels, "features": FEATURE_NAMES}
    if joblib is not None:
        joblib.dump(payload, path)
        return
    import pickle

    with path.open("wb") as fh:
        pickle.dump(payload, fh)


def _write_meta(metrics: TrainingMetrics, *, namespace: str | None = None) -> None:
    path = _model_meta_path(namespace)
    meta = {
        "trained_at": time.time(),
        "total_samples": metrics.total_samples,
        "classes": metrics.classes,
        "accuracy": metrics.accuracy,
        "test_size": metrics.test_size,
        "features": FEATURE_NAMES,
        "override_threshold": metrics.override_threshold,
        "override_accuracy": metrics.override_accuracy,
        "override_coverage": metrics.override_coverage,
        "override_samples": metrics.override_samples,
        "override_source": metrics.override_source,
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_trained_model(*, namespace: str | None = None) -> Optional[Tuple[Any, List[str]]]:
    path = _model_path(namespace)
    if not path.exists():
        return None
    try:
        if joblib is not None:
            payload = joblib.load(path)
        else:
            import pickle

            with path.open("rb") as fh:
                payload = pickle.load(fh)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    labels = payload.get("labels") or []
    if model is None or not isinstance(labels, list):
        return None
    return model, labels


def load_model_meta(*, namespace: str | None = None) -> Optional[Dict[str, Any]]:
    path = _model_meta_path(namespace)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _select_override_threshold(
    probabilities: Any,
    labels_true: List[str],
    *,
    labels: List[str],
    thresholds: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    if not labels_true or not labels:
        return None
    rows = list(probabilities) if probabilities is not None else []
    if not rows:
        return None
    total = min(len(rows), len(labels_true))
    if total <= 0:
        return None
    thresholds = thresholds or _MODEL_THRESHOLD_CANDIDATES
    best: Optional[Dict[str, Any]] = None
    for threshold in thresholds:
        correct = 0
        covered = 0
        for row, truth in zip(rows, labels_true):
            idx, conf = _max_class_confidence(row)
            if idx is None or conf < threshold:
                continue
            covered += 1
            if idx < len(labels) and labels[idx] == truth:
                correct += 1
        if covered <= 0:
            accuracy = 0.0
        else:
            accuracy = correct / covered
        coverage = covered / total
        score = accuracy * (0.6 + 0.4 * coverage)
        if best is None or score > best["score"]:
            best = {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "coverage": float(coverage),
                "samples": int(covered),
                "score": float(score),
            }
    return best


def _max_class_confidence(row: Any) -> Tuple[Optional[int], float]:
    best_idx = None
    best_val = None
    try:
        iterator = enumerate(row)
    except TypeError:
        return None, 0.0
    for idx, value in iterator:
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if best_val is None or score > best_val:
            best_val = score
            best_idx = int(idx)
    if best_idx is None or best_val is None:
        return None, 0.0
    return best_idx, float(best_val)


__all__ = [
    "save_sample",
    "train_model",
    "load_samples",
    "load_trained_model",
    "load_model_meta",
    "TrainingMetrics",
]
