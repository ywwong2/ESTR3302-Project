from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image

CATEGORY_DIR_TO_QUERY = {
    "cats": "cat",
    "dogs": "dog",
    "planes": "plane",
    "clothes": "clothes",
    "cars": "car",
}
QUERY_ORDER = ["cat", "dog", "plane", "clothes", "car"]
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".tif", ".tiff"}


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_vector(output) -> list[float]:
    if hasattr(output, "pooler_output"):
        return output.pooler_output[0].tolist()
    return output[0].tolist()


def list_dataset_images(dataset_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    for folder, query in CATEGORY_DIR_TO_QUERY.items():
        folder_path = dataset_root / folder
        if not folder_path.exists():
            raise FileNotFoundError(f"Missing category folder: {folder_path}")

        files = [
            p for p in sorted(folder_path.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
        if not files:
            raise ValueError(f"No supported image files found under {folder_path}")

        for p in files:
            records.append(
                {
                    "relative_path": str(p.relative_to(dataset_root)),
                    "absolute_path": str(p),
                    "category": query,
                }
            )

    return records


def load_siglip_components(project_root: Path):
    import torch
    from transformers import SiglipModel, SiglipProcessor

    local_model_dir = project_root / "models" / "siglip-base-patch16-224"
    source = str(local_model_dir) if local_model_dir.exists() else "google/siglip-base-patch16-224"

    processor = SiglipProcessor.from_pretrained(source)
    model = SiglipModel.from_pretrained(source)
    model.eval()

    return model, processor, torch


def build_cache(dataset_root: Path, output_path: Path) -> dict:
    project_root = Path(__file__).resolve().parent.parent
    model, processor, torch = load_siglip_components(project_root)

    images = list_dataset_images(dataset_root)

    query_vectors: dict[str, list[float]] = {}
    for query in QUERY_ORDER:
        inputs = processor(text=[query], return_tensors="pt", padding="max_length", truncation=True)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
        query_vectors[query] = l2_normalize(extract_vector(out))

    image_rows: list[dict] = []
    pairs: list[dict] = []

    for idx, item in enumerate(images):
        abs_path = Path(item["absolute_path"])
        with Image.open(abs_path) as img:
            rgb = img.convert("RGB")
            inputs = processor(images=rgb, return_tensors="pt")

        with torch.no_grad():
            out = model.get_image_features(**inputs)
        img_vec = l2_normalize(extract_vector(out))

        cosines = {}
        for query in QUERY_ORDER:
            score = cosine_similarity(query_vectors[query], img_vec)
            cosines[query] = round(float(score), 8)
            pairs.append(
                {
                    "query": query,
                    "relative_path": item["relative_path"],
                    "category": item["category"],
                    "cosine": round(float(score), 8),
                }
            )

        image_rows.append(
            {
                "image_id": idx,
                "relative_path": item["relative_path"],
                "category": item["category"],
                "cosine_by_query": cosines,
            }
        )

    result = {
        "model_name": "google/siglip-base-patch16-224",
        "queries": QUERY_ORDER,
        "dataset_root": str(dataset_root),
        "image_count": len(image_rows),
        "pair_count": len(pairs),
        "images": image_rows,
        "pairs": pairs,
    }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SigLIP query-image cosine cache for dataset folders.")
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--output", type=str, default="dataset/cosine_cache_siglip.json")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)

    result = build_cache(dataset_root=dataset_root, output_path=output_path)
    print(f"Cache written: {output_path}")
    print(f"Images: {result['image_count']}")
    print(f"Pairs: {result['pair_count']}")


if __name__ == "__main__":
    main()
