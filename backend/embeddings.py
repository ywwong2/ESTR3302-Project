from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

MODEL_SIGLIP = "google/siglip-base-patch16-224"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
_LOCAL_SIGLIP_DIR = MODELS_DIR / "siglip-base-patch16-224"

QUERY_LABELS = ["cat", "dog", "plane", "clothes", "car"]

# Lazy-loaded singletons
_SIGLIP_MODEL = None
_SIGLIP_PROCESSOR = None


def _log(msg: str) -> None:
    try:
        from backend.debug_log import log as _dl
        _dl(msg)
    except Exception:
        print(msg)


def _resolve_siglip_source() -> str:
    if _LOCAL_SIGLIP_DIR.exists():
        return str(_LOCAL_SIGLIP_DIR)
    return MODEL_SIGLIP


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── SigLIP loader (lazy singleton) ───────────────────────────

def _load_siglip():
    global _SIGLIP_MODEL, _SIGLIP_PROCESSOR
    if _SIGLIP_MODEL is None:
        import torch
        from transformers import SiglipProcessor, SiglipModel

        source = _resolve_siglip_source()
        _log(f"[SigLIP] Loading from: {source}")
        t0 = time.time()
        _SIGLIP_PROCESSOR = SiglipProcessor.from_pretrained(source)
        _SIGLIP_MODEL = SiglipModel.from_pretrained(source)
        _SIGLIP_MODEL.eval()
        _log(f"[SigLIP] Loaded in {time.time()-t0:.1f}s")
    return _SIGLIP_MODEL, _SIGLIP_PROCESSOR


# ── Image embedding ───────────────────────────────────────────

def embed_image(file_path: Path) -> list[float]:
    """Embed an image file using SigLIP vision encoder. Returns L2-normalised vector."""
    import torch
    from PIL import Image

    _log(f"[SigLIP] Embedding image: {file_path.name}")
    model, processor = _load_siglip()
    t0 = time.time()
    with Image.open(file_path) as img:
        inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        vec = out.pooler_output[0].tolist()
    _log(f"[SigLIP] Image embedded in {time.time()-t0:.1f}s, dim={len(vec)}")
    return _l2_normalize(vec)


# ── Text embedding (with JSON cache) ─────────────────────────

def embed_query_text(query: str, cache_path: Path | None = None) -> list[float]:
    """Embed a text query. Reads/writes cache_path (query_cache.json) if provided."""
    import torch

    key = query.strip().lower()

    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if key in cache:
            _log(f"[SigLIP] Query cache hit: '{key[:60]}'")
            return cache[key]
    else:
        cache = {}

    _log(f"[SigLIP] Embedding text query: '{key[:80]}'")
    t0 = time.time()
    model, processor = _load_siglip()
    inputs = processor(text=[query], return_tensors="pt", padding="max_length", truncation=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        vec = out.pooler_output[0].tolist() if hasattr(out, "pooler_output") else out[0].tolist()
    vec = _l2_normalize(vec)
    _log(f"[SigLIP] Text embedded in {time.time()-t0:.1f}s")

    if cache_path:
        cache[key] = vec
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    return vec


# ── Cosine-by-query computation ───────────────────────────────

def compute_cosines_for_image(
    image_vec: list[float],
    cache_path: Path | None = None,
) -> dict[str, float]:
    """Compute cosine similarity between an image vector and each of the 5 query labels.
    Uses cached query embeddings from cache_path if available."""
    cosines: dict[str, float] = {}
    for label in QUERY_LABELS:
        q_vec = embed_query_text(label, cache_path=cache_path)
        cosines[label] = round(cosine_similarity(image_vec, q_vec), 6)
    return cosines


def detect_category(cosines: dict[str, float]) -> str:
    """Return the query label with highest cosine — the auto-detected category."""
    return max(cosines, key=lambda k: cosines[k])


# ── Backward-compat helpers (used by old main.py callers) ─────

def generate_embedding(
    media_type: str,
    file_path: Path,
    preprocess_result: dict[str, Any],
) -> dict[str, Any]:
    """Thin wrapper kept for compatibility. Returns vector dict."""
    if media_type != "image":
        raise ValueError(f"Unsupported media type: {media_type}")
    vec = embed_image(file_path)
    return {
        "modality": "image",
        "model_name": MODEL_SIGLIP,
        "vector_dim": len(vec),
        "vector": vec,
    }
