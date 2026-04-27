from __future__ import annotations

import json
import csv
from pathlib import Path
from collections import defaultdict

def load_cache(cache_path: Path) -> tuple[list[dict], list[str]]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    images_raw = data.get("images", [])
    query_labels = ["cat", "dog", "plane", "clothes", "car"]
    return images_raw, query_labels

def load_final_weights(csv_path: Path) -> dict[str, float]:
    """Get the last row's weights from simulation CSV."""
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        last = rows[-1]
    weights = {}
    for k, v in last.items():
        if k.startswith("w") and ("cos" in k or "fresh" in k or "lr" in k or "ctr" in k or "awt" in k or "social" in k):
            weights[k] = float(v)
    return weights

def compute_score(image: dict, query_label: str, weights: dict[str, float], normalize: bool = True) -> float:
    """Compute ranking score for an image given a query."""
    cos = float(image.get("cosine_by_query", {}).get(query_label, 0.0))
    freshness = image.get("freshness", 0.5)
    lr = image.get("lr", 0.05)
    ctr = image.get("ctr", 0.1)
    awt = image.get("awt", 0.1)
    social = image.get("social_proof", 0.0)
    
    # Feature vector: [cos, fresh, lr, ctr, awt, social]
    x = [cos, freshness, lr, ctr, awt, social]
    
    # Normalize to [0,1] if requested
    if normalize:
        mins = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        maxs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        x = [(x[i] - mins[i]) / (maxs[i] - mins[i] + 1e-8) for i in range(6)]
    
    # Weights from CSV
    w = [
        weights.get("w0_cos", 0.166),
        weights.get("w1_fresh", 0.166),
        weights.get("w2_lr", 0.166),
        weights.get("w3_ctr", 0.166),
        weights.get("w4_awt", 0.166),
        weights.get("w5_social", 0.166),
    ]
    
    score = sum(wi * xi for wi, xi in zip(w, x))
    return score

def analyze_ranking(csv_path: Path, cache_path: Path, out_path: Path, normalize: bool = True):
    images_raw, query_labels = load_cache(cache_path)
    weights = load_final_weights(csv_path)
    
    label_to_idx = {label: i for i, label in enumerate(query_labels)}
    
    results = {}
    for query_label in query_labels:
        # Score all images for this query
        scored = []
        for img in images_raw:
            score = compute_score(img, query_label, weights, normalize=normalize)
            category = img.get("category", "unknown")
            scored.append((score, category, img.get("image_id", -1)))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Compute semantic relevance: % of top-10 that match the query
        target_idx = label_to_idx[query_label]
        top10 = scored[:10]
        matches = sum(1 for _, cat, _ in top10 if cat == query_label)
        relevance = matches / 10.0
        
        results[query_label] = {
            "top10": [(s, cat, iid) for s, cat, iid in top10],
            "relevance": relevance,
            "weights": weights,
        }
    
    # Write report
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Ranking Examples Analysis\n\n")
        f.write(f"Final weights: {weights}\n\n")
        
        for query_label, res in results.items():
            f.write(f"## Query: {query_label}\n")
            f.write(f"Top-10 semantic relevance: {res['relevance']*100:.1f}%\n\n")
            f.write("| Rank | Score | Category | Image ID |\n")
            f.write("|------|-------|----------|----------|\n")
            for i, (score, cat, iid) in enumerate(res["top10"], 1):
                match = "✓" if cat == query_label else " "
                f.write(f"| {i} | {score:.4f} | {cat} {match} | {iid} |\n")
            f.write("\n")
    
    print(f"Ranking analysis written to {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python analyze_ranking_examples.py <csv_path> <cache_path> <out_path> [--no-normalize]")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    cache_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    normalize = "--no-normalize" not in sys.argv
    
    analyze_ranking(csv_path, cache_path, out_path, normalize=normalize)
