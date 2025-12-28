from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(slots=True)
class TrainingMetrics:
    total_samples: int
    classes: Dict[str, int]
    accuracy: Optional[float]
    test_size: int


def _data_dir() -> Path:
    base = Path(getattr(settings, "DATA_CACHE_DIR", Path.cwd()))
    path = base / "screen_analyzer"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _samples_path() -> Path:
    return _data_dir() / "pattern_samples.jsonl"


def _model_path() -> Path:
    return _data_dir() / "pattern_model.pkl"


def _model_meta_path() -> Path:
    return _data_dir() / "pattern_model_meta.json"


def save_sample(features: List[float], label: str, meta: Optional[Dict[str, Any]] = None) -> None:
    if label not in PATTERN_KEYS:
        raise ValueError("invalid_label")
    entry = {
        "label": label,
        "features": features,
        "meta": meta or {},
        "ts": time.time(),
    }
    path = _samples_path()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_samples() -> List[Dict[str, Any]]:
    path = _samples_path()
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


def train_model(min_samples: int = 18) -> TrainingMetrics:
    if RandomForestClassifier is None or train_test_split is None or accuracy_score is None:
        raise RuntimeError("sklearn_unavailable")
    samples = load_samples()
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

    accuracy = None
    if x_test:
        preds = model.predict(x_test)
        accuracy = float(accuracy_score(y_test, preds))

    labels = list(getattr(model, "classes_", sorted(set(y))))
    _save_model(model, labels=labels)
    meta = TrainingMetrics(
        total_samples=len(valid),
        classes=class_counts,
        accuracy=accuracy,
        test_size=len(x_test),
    )
    _write_meta(meta)
    return meta


def _save_model(model: Any, *, labels: List[str]) -> None:
    path = _model_path()
    payload = {"model": model, "labels": labels, "features": FEATURE_NAMES}
    if joblib is not None:
        joblib.dump(payload, path)
        return
    import pickle

    with path.open("wb") as fh:
        pickle.dump(payload, fh)


def _write_meta(metrics: TrainingMetrics) -> None:
    path = _model_meta_path()
    meta = {
        "total_samples": metrics.total_samples,
        "classes": metrics.classes,
        "accuracy": metrics.accuracy,
        "test_size": metrics.test_size,
        "features": FEATURE_NAMES,
    }
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_trained_model() -> Optional[Tuple[Any, List[str]]]:
    path = _model_path()
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


__all__ = ["save_sample", "train_model", "load_samples", "load_trained_model", "TrainingMetrics"]
