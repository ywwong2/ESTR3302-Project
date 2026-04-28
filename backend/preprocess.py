from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_TARGET_SIZE = (224, 224)


def detect_media_type(content_type: str) -> str:
    ct = (content_type or "").lower()
    if ct.startswith("image/"):
        return "image"
    raise ValueError("Unsupported content type. Allowed: image/*")


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


def run_preprocessing(media_type: str, file_path: Path, out_dir: Path) -> dict[str, Any]:
    if media_type == "image":
        return preprocess_image(file_path, out_dir)
    raise ValueError(f"Unsupported media type: {media_type}")
