from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from django.conf import settings

from .bailian_ai import embed_texts


DEFAULT_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "420") or 420)
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "80") or 80)
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "5") or 5)


@dataclass(slots=True)
class RagChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def _kb_root(user_id: str | None) -> Path:
    base = Path(getattr(settings, "DATA_CACHE_DIR", Path(settings.DATA_ROOT) / "data_cache"))
    owner = str(user_id or "global")
    root = base / "knowledge_base" / owner
    root.mkdir(parents=True, exist_ok=True)
    return root


def _manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def _vector_path(root: Path) -> Path:
    return root / "vectors.npy"


def _split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - max(0, overlap))
    for idx in range(0, len(cleaned), step):
        chunk = cleaned[idx : idx + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _normalize_vectors(vectors: Iterable[Iterable[float]]) -> np.ndarray:
    array = np.array(list(vectors), dtype=np.float32)
    if array.ndim != 2 or not array.size:
        return np.empty((0, 0), dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def _load_manifest(root: Path) -> list[dict[str, Any]]:
    path = _manifest_path(root)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def _save_manifest(root: Path, manifest: list[dict[str, Any]]) -> None:
    path = _manifest_path(root)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_vectors(root: Path) -> np.ndarray:
    path = _vector_path(root)
    if not path.exists():
        return np.empty((0, 0), dtype=np.float32)
    try:
        return np.load(path)
    except Exception:
        return np.empty((0, 0), dtype=np.float32)


def _save_vectors(root: Path, vectors: np.ndarray) -> None:
    path = _vector_path(root)
    np.save(path, vectors)


def ingest_texts(
    texts: Iterable[str],
    *,
    user_id: str | None,
    metadata: dict[str, Any] | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> dict[str, Any]:
    root = _kb_root(user_id)
    manifest = _load_manifest(root)
    vectors = _load_vectors(root)
    chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    overlap = overlap if overlap is not None else DEFAULT_CHUNK_OVERLAP

    chunks: list[RagChunk] = []
    base_meta = metadata or {}
    for text in texts:
        for chunk in _split_text(str(text), chunk_size=chunk_size, overlap=overlap):
            chunk_hash = sha256(chunk.encode("utf-8")).hexdigest()[:16]
            chunks.append(RagChunk(chunk_id=chunk_hash, text=chunk, metadata=dict(base_meta)))
    if not chunks:
        return {"chunks": 0, "message": "empty"}

    embeddings = embed_texts([item.text for item in chunks], user_id=user_id)
    if not embeddings:
        return {"chunks": 0, "message": "embed_failed"}
    new_vectors = _normalize_vectors(embeddings)
    if vectors.size == 0:
        vectors = new_vectors
    else:
        vectors = np.vstack([vectors, new_vectors])

    for item in chunks:
        manifest.append({"id": item.chunk_id, "text": item.text, "metadata": item.metadata})

    _save_manifest(root, manifest)
    _save_vectors(root, vectors)
    return {"chunks": len(chunks), "total": len(manifest)}


def query(
    text: str,
    *,
    user_id: str | None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    root = _kb_root(user_id)
    manifest = _load_manifest(root)
    vectors = _load_vectors(root)
    if not manifest or vectors.size == 0:
        return []
    embeddings = embed_texts([text], user_id=user_id)
    if not embeddings:
        return []
    query_vec = _normalize_vectors(embeddings)
    if query_vec.size == 0:
        return []
    scores = vectors @ query_vec[0]
    top_k = max(1, min(top_k or DEFAULT_TOP_K, len(manifest)))
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    results: list[dict[str, Any]] = []
    for idx in ranked_idx:
        item = manifest[int(idx)]
        results.append(
            {
                "id": item.get("id"),
                "text": item.get("text"),
                "metadata": item.get("metadata", {}),
                "score": float(scores[int(idx)]),
            }
        )
    return results
