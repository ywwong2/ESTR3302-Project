from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from backend.debug_log import log

MODEL_IMAGE_VIDEO = "google/siglip-base-patch16-224"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_MODEL_DIRS = {
    MODEL_IMAGE_VIDEO: MODELS_DIR / "siglip-base-patch16-224",
}

VECTOR_DIMENSIONS = {
    "image": 768,
}

# Lazy-loaded model singletons
_SIGLIP_MODEL = None
_SIGLIP_PROCESSOR = None


def _resolve_model_source(model_name: str) -> str:
    """Use local model dir if available, else fall back to HF hub name."""
    local_dir = LOCAL_MODEL_DIRS.get(model_name)
    if local_dir and local_dir.exists():
        return str(local_dir)
    return model_name


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


# ── Model loaders (lazy, singleton) ──────────────────────────


def _load_siglip():
    global _SIGLIP_MODEL, _SIGLIP_PROCESSOR
    if _SIGLIP_MODEL is None:
        import torch
        from transformers import SiglipProcessor, SiglipModel

        source = _resolve_model_source(MODEL_IMAGE_VIDEO)
        log(f"[DEBUG] Loading SigLIP model+processor from: {source}")
        t0 = time.time()
        _SIGLIP_PROCESSOR = SiglipProcessor.from_pretrained(source)
        log(f"[DEBUG] SigLIP processor (text+image) loaded in {time.time()-t0:.1f}s")
        t1 = time.time()
        _SIGLIP_MODEL = SiglipModel.from_pretrained(source)
        _SIGLIP_MODEL.eval()
        log(f"[DEBUG] SigLIP model loaded in {time.time()-t1:.1f}s")
    else:
        log("[DEBUG] SigLIP model already cached")
    return _SIGLIP_MODEL, _SIGLIP_PROCESSOR


# ── Per-modality embedding functions ─────────────────────────


def _embed_image(file_path: Path, preprocess_result: dict[str, Any]) -> list[float]:
    import traceback
    import torch
    from PIL import Image

    log(f"[DEBUG] _embed_image called for: {file_path}")
    model, processor = _load_siglip()
    image_path = Path(preprocess_result.get("processed_path", file_path))
    log(f"[DEBUG] Opening image: {image_path}")
    try:
        with Image.open(image_path) as img:
            log(f"[DEBUG] Image opened: size={img.size}, mode={img.mode}")
            rgb = img.convert("RGB")
            log(f"[DEBUG] Converted to RGB, running processor...")
            inputs = processor(images=rgb, return_tensors="pt")
        log(f"[DEBUG] Processor done, running model inference...")
        with torch.no_grad():
            out = model.get_image_features(**inputs)
            features = out.pooler_output[0].tolist()
        log(f"[DEBUG] Image embedding done, dim={len(features)}")
        return _l2_normalize(features)
    except Exception as e:
        log(f"[ERROR] _embed_image failed: {e}")
        log(traceback.format_exc())
        raise


# ── Pooling ──────────────────────────────────────────────────


def mean_pool(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


def max_pool(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    return [max(v[i] for v in vectors) for i in range(dim)]


# ── Main entry point ─────────────────────────────────────────


def _model_name_for(media_type: str) -> str:
    if media_type == "image":
        return MODEL_IMAGE_VIDEO
    raise ValueError(f"Unsupported media type: {media_type}")


def generate_embedding(
    media_type: str,
    file_path: Path,
    preprocess_result: dict[str, Any],
    pooling_type: str = "mean",
) -> dict[str, Any]:
    model_name = _model_name_for(media_type)
    vector_dim = VECTOR_DIMENSIONS[media_type]
    log(f"\n{'='*60}")
    log(f"[DEBUG] generate_embedding: type={media_type}, model={model_name}, file={file_path}")
    log(f"{'='*60}")
    t_start = time.time()

    if media_type == "image":
        segment_vectors = [_embed_image(file_path, preprocess_result)]
    else:
        raise ValueError(f"Unsupported media type: {media_type}")

    log(f"[DEBUG] Got {len(segment_vectors)} segment(s), pooling with {pooling_type}...")
    pool_fn = mean_pool if pooling_type == "mean" else max_pool
    final_vector = pool_fn(segment_vectors)
    log(f"[DEBUG] generate_embedding DONE in {time.time()-t_start:.1f}s, final dim={len(final_vector)}")

    vector_hash = hashlib.sha256(
        json.dumps(final_vector, separators=(",", ":")).encode()
    ).hexdigest()

    return {
        "modality": media_type,
        "model_name": model_name,
        "vector_dim": vector_dim,
        "pooling_type": pooling_type,
        "num_segments": len(segment_vectors),
        "vector": final_vector,
        "vector_id": f"vec_{vector_hash[:20]}",
    }


def save_vector_file(vector_id: str, vector: list[float], vector_dir: Path) -> Path:
    vector_dir.mkdir(parents=True, exist_ok=True)
    out_path = vector_dir / f"{vector_id}.json"
    out_path.write_text(json.dumps({"id": vector_id, "vector": vector}))
    return out_path


def load_vector_file(vector_id: str, vector_dir: Path) -> list[float] | None:
    """Load a stored vector by its ID. Returns None if file missing."""
    fpath = vector_dir / f"{vector_id}.json"
    if not fpath.exists():
        return None
    data = json.loads(fpath.read_text())
    return data.get("vector")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_query_text(query: str) -> list[float]:
    """Embed a search query using SigLIP text encoder (shared space with images)."""
    import torch

    log(f"[SEARCH] Embedding query with SigLIP text encoder: '{query[:80]}'")
    t0 = time.time()
    model, processor = _load_siglip()
    inputs = processor(text=[query], return_tensors="pt", padding="max_length", truncation=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        if hasattr(out, "pooler_output"):
            vec = out.pooler_output[0].tolist()
        else:
            vec = out[0].tolist()
    log(f"[SEARCH] SigLIP query embedded in {time.time()-t0:.1f}s, dim={len(vec)}")
    return _l2_normalize(vec)
