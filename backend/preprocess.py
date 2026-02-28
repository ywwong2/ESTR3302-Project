from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_TARGET_SIZE = (224, 224)
VIDEO_TARGET_FRAMES = 10
AUDIO_TARGET_CHUNKS = 10
TEXT_MAX_CHARS = 4000


def detect_media_type(content_type: str) -> str:
    ct = (content_type or "").lower()
    if ct.startswith("image/"):
        return "image"
    if ct.startswith("video/"):
        return "video"
    if ct.startswith("audio/"):
        return "audio"
    if ct.startswith("text/"):
        return "text"
    raise ValueError("Unsupported content type. Allowed: image/*, video/*, audio/*, text/*")


def uniform_timestamps(duration_sec: float, num_samples: int) -> list[float]:
    step = duration_sec / num_samples
    return [round((i + 0.5) * step, 3) for i in range(num_samples)]


def uniform_audio_chunks(duration_sec: float, chunks: int) -> list[dict[str, float]]:
    boundaries = [round(i * duration_sec / chunks, 3) for i in range(chunks + 1)]
    return [
        {"start": boundaries[i], "end": boundaries[i + 1]}
        for i in range(chunks)
    ]


def preprocess_image(file_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{file_path.stem}_224.jpg"

    with Image.open(file_path) as img:
        original_w, original_h = img.size
        processed = img.convert("RGB").resize(IMAGE_TARGET_SIZE)
        processed.save(out_file, format="JPEG", quality=90)

    return {
        "modality": "image",
        "target_size": list(IMAGE_TARGET_SIZE),
        "original_size": [original_w, original_h],
        "processed_path": str(out_file),
    }


def preprocess_text(file_path: Path) -> dict[str, Any]:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    trimmed = raw[:TEXT_MAX_CHARS]
    return {
        "modality": "text",
        "original_length": len(raw),
        "final_length": len(trimmed),
        "preview": trimmed[:200],
    }


def preprocess_video(file_path: Path) -> dict[str, Any]:
    assumed_duration_sec = 20.0
    return {
        "modality": "video",
        "target_frames": VIDEO_TARGET_FRAMES,
        "timestamps_sec": uniform_timestamps(assumed_duration_sec, VIDEO_TARGET_FRAMES),
    }


def preprocess_audio(file_path: Path) -> dict[str, Any]:
    duration_sec = 30.0
    try:
        with wave.open(str(file_path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate > 0 and frames > 0:
                duration_sec = frames / float(rate)
    except Exception:
        pass

    return {
        "modality": "audio",
        "duration_sec": round(duration_sec, 3),
        "chunks": uniform_audio_chunks(duration_sec, AUDIO_TARGET_CHUNKS),
    }


def run_preprocessing(media_type: str, file_path: Path, out_dir: Path) -> dict[str, Any]:
    if media_type == "image":
        result = preprocess_image(file_path, out_dir)
    elif media_type == "video":
        result = preprocess_video(file_path)
    elif media_type == "audio":
        result = preprocess_audio(file_path)
    elif media_type == "text":
        result = preprocess_text(file_path)
    else:
        raise ValueError(f"Unsupported media type: {media_type}")

    # Save preprocessing manifest for debugging
    manifest_path = out_dir / f"{file_path.stem}_preprocess.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["manifest_path"] = str(manifest_path)
    return result
