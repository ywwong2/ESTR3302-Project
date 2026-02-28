from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from backend.debug_log import log

MODEL_IMAGE_VIDEO = "google/siglip-base-patch16-224"
MODEL_AUDIO = "laion/clap-htsat-unfused"
MODEL_TEXT = "google/embeddinggemma-300m"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOCAL_MODEL_DIRS = {
    MODEL_IMAGE_VIDEO: MODELS_DIR / "siglip-base-patch16-224",
    MODEL_AUDIO: MODELS_DIR / "clap-htsat-unfused",
    MODEL_TEXT: MODELS_DIR / "embeddinggemma-300m",
}

VECTOR_DIMENSIONS = {
    "image": 768,
    "video": 768,
    "audio": 512,
    "text": 768,
}

# Lazy-loaded model singletons
_SIGLIP_MODEL = None
_SIGLIP_PROCESSOR = None
_CLAP_MODEL = None
_CLAP_PROCESSOR = None
_TEXT_MODEL = None


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


def _load_clap():
    global _CLAP_MODEL, _CLAP_PROCESSOR
    if _CLAP_MODEL is None:
        import torch
        from transformers import AutoProcessor, ClapModel

        source = _resolve_model_source(MODEL_AUDIO)
        log(f"[DEBUG] Loading CLAP model from: {source}")
        t0 = time.time()
        _CLAP_PROCESSOR = AutoProcessor.from_pretrained(source)
        log(f"[DEBUG] CLAP processor loaded in {time.time()-t0:.1f}s")
        t1 = time.time()
        _CLAP_MODEL = ClapModel.from_pretrained(source)
        _CLAP_MODEL.eval()
        log(f"[DEBUG] CLAP model loaded in {time.time()-t1:.1f}s")
    else:
        log("[DEBUG] CLAP model already cached")
    return _CLAP_MODEL, _CLAP_PROCESSOR


def _load_text_model():
    global _TEXT_MODEL
    if _TEXT_MODEL is None:
        from sentence_transformers import SentenceTransformer

        source = _resolve_model_source(MODEL_TEXT)
        log(f"[DEBUG] Loading EmbeddingGemma model from: {source}")
        t0 = time.time()
        _TEXT_MODEL = SentenceTransformer(source, device="cpu")
        log(f"[DEBUG] EmbeddingGemma model loaded in {time.time()-t0:.1f}s")
    else:
        log("[DEBUG] EmbeddingGemma model already cached")
    return _TEXT_MODEL


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


def _embed_audio(file_path: Path, preprocess_result: dict[str, Any]) -> list[list[float]]:
    import torch
    import torchaudio

    log(f"[DEBUG] _embed_audio called for: {file_path}")
    model, processor = _load_clap()
    log(f"[DEBUG] Loading audio waveform...")
    waveform, sr = torchaudio.load(str(file_path))
    log(f"[DEBUG] Waveform loaded: shape={waveform.shape}, sr={sr}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 48000:
        waveform = torchaudio.functional.resample(waveform, sr, 48000)
        sr = 48000
    audio = waveform.squeeze(0).tolist()

    chunks = preprocess_result.get("chunks", [{"start": 0.0, "end": len(audio) / sr}])
    vectors = []
    for chunk in chunks:
        start = int(chunk["start"] * sr)
        end = int(chunk["end"] * sr)
        audio_chunk = audio[start:end] or [0.0] * sr
        inputs = processor(audios=audio_chunk, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            features = model.get_audio_features(**inputs)[0].tolist()
        vectors.append(_l2_normalize(features))
    return vectors


def _embed_video(file_path: Path, preprocess_result: dict[str, Any]) -> list[list[float]]:
    timestamps = preprocess_result.get("timestamps_sec", [0.0])
    # Frame extraction not yet implemented; replicate image embedding per timestamp
    image_vec = _embed_image(file_path, preprocess_result)
    return [image_vec[:] for _ in timestamps]


def _embed_text(file_path: Path) -> list[float]:
    """Embed text content using SigLIP's text encoder (shared space with images)."""
    import torch

    log(f"[DEBUG] _embed_text (SigLIP) called for: {file_path}")
    model, processor = _load_siglip()
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip() or " "
    log(f"[DEBUG] Text length: {len(text)} chars, encoding with SigLIP text encoder...")
    t0 = time.time()
    inputs = processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        if hasattr(out, "pooler_output"):
            vec = out.pooler_output[0].tolist()
        else:
            vec = out[0].tolist()
    log(f"[DEBUG] SigLIP text encoding done in {time.time()-t0:.1f}s, dim={len(vec)}")
    return _l2_normalize(vec)


# ── Pooling ──────────────────────────────────────────────────


def mean_pool(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


def max_pool(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    return [max(v[i] for v in vectors) for i in range(dim)]


# ── Main entry point ─────────────────────────────────────────


def _model_name_for(media_type: str) -> str:
    if media_type in {"image", "video", "text"}:
        return MODEL_IMAGE_VIDEO
    if media_type == "audio":
        return MODEL_AUDIO
    return MODEL_IMAGE_VIDEO


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
    elif media_type == "video":
        segment_vectors = _embed_video(file_path, preprocess_result)
    elif media_type == "audio":
        segment_vectors = _embed_audio(file_path, preprocess_result)
    elif media_type == "text":
        segment_vectors = [_embed_text(file_path)]
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
    """Embed a search query using SigLIP's text encoder (shared space with images & text)."""
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


def embed_query_audio(query: str) -> list[float]:
    """Embed a search query using CLAP's text encoder (shared space with audio)."""
    import torch

    log(f"[SEARCH] Embedding query with CLAP text encoder: '{query[:80]}'")
    t0 = time.time()
    model, processor = _load_clap()
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        vec = text_features[0].tolist()
    log(f"[SEARCH] CLAP query embedded in {time.time()-t0:.1f}s, dim={len(vec)}")
    return _l2_normalize(vec)
