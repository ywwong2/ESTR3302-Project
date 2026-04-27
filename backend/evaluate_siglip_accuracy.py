from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def softmax(values: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = {k: v * temperature for k, v in values.items()}
    max_v = max(scaled.values())
    exps = {k: math.exp(v - max_v) for k, v in scaled.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def rank_of_label(scores: dict[str, float], label: str) -> int:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for idx, (k, _) in enumerate(ordered, start=1):
        if k == label:
            return idx
    return len(ordered)


def evaluate(cache_path: Path, temperature: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    queries = payload["queries"]
    images = payload["images"]

    per_class_total: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    per_class_correct_cosine: dict[str, list[float]] = defaultdict(list)
    per_class_correct_prob: dict[str, list[float]] = defaultdict(list)

    confusion: dict[str, dict[str, int]] = {
        q: {p: 0 for p in queries} for q in queries
    }

    rows: list[dict[str, Any]] = []

    top1_correct = 0
    correct_cos_values: list[float] = []
    best_cos_values: list[float] = []
    margin_values: list[float] = []
    correct_prob_values: list[float] = []
    top1_prob_values: list[float] = []
    mrr_sum = 0.0

    for item in images:
        true_label = item["category"]
        scores = {k: float(v) for k, v in item["cosine_by_query"].items()}
        probs = softmax(scores, temperature=temperature)

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        pred_label, pred_score = ordered[0]
        true_score = scores[true_label]
        best_wrong = max(v for k, v in scores.items() if k != true_label)
        margin = true_score - best_wrong
        rank_true = rank_of_label(scores, true_label)

        correct = int(pred_label == true_label)

        per_class_total[true_label] += 1
        per_class_correct[true_label] += correct
        per_class_correct_cosine[true_label].append(true_score)
        per_class_correct_prob[true_label].append(probs[true_label])
        confusion[true_label][pred_label] += 1

        top1_correct += correct
        correct_cos_values.append(true_score)
        best_cos_values.append(pred_score)
        margin_values.append(margin)
        correct_prob_values.append(probs[true_label])
        top1_prob_values.append(max(probs.values()))
        mrr_sum += 1.0 / rank_true

        rows.append(
            {
                "image_id": item.get("image_id", ""),
                "relative_path": item["relative_path"],
                "true_category": true_label,
                "predicted_category": pred_label,
                "top1_correct": correct,
                "true_cosine": round(true_score, 8),
                "pred_cosine": round(pred_score, 8),
                "margin_true_minus_best_wrong": round(margin, 8),
                "rank_of_true": rank_true,
                "true_softmax_confidence": round(probs[true_label], 8),
                "top1_softmax_confidence": round(max(probs.values()), 8),
            }
        )

    total = len(images)
    summary: dict[str, Any] = {
        "cache_path": str(cache_path),
        "model_name": payload.get("model_name", "unknown"),
        "queries": queries,
        "temperature": temperature,
        "num_images": total,
        "top1_correct": top1_correct,
        "top1_accuracy": top1_correct / total if total else 0.0,
        "mean_true_cosine": sum(correct_cos_values) / total if total else 0.0,
        "mean_top1_cosine": sum(best_cos_values) / total if total else 0.0,
        "mean_margin_true_minus_best_wrong": sum(margin_values) / total if total else 0.0,
        "mean_true_softmax_confidence": sum(correct_prob_values) / total if total else 0.0,
        "mean_top1_softmax_confidence": sum(top1_prob_values) / total if total else 0.0,
        "mean_reciprocal_rank": mrr_sum / total if total else 0.0,
        "confusion_matrix": confusion,
        "per_category": {},
    }

    for label in queries:
        n = per_class_total[label]
        if n == 0:
            summary["per_category"][label] = {
                "count": 0,
                "top1_correct": 0,
                "top1_accuracy": 0.0,
                "mean_true_cosine": 0.0,
                "mean_true_softmax_confidence": 0.0,
            }
            continue

        summary["per_category"][label] = {
            "count": n,
            "top1_correct": per_class_correct[label],
            "top1_accuracy": per_class_correct[label] / n,
            "mean_true_cosine": sum(per_class_correct_cosine[label]) / n,
            "mean_true_softmax_confidence": sum(per_class_correct_prob[label]) / n,
        }

    return summary, rows


def write_details_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SigLIP category accuracy from query-image cosine cache. "
            "Reports top-1 accuracy and confidence metrics."
        )
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="dataset/cosine_cache_siglip.json",
        help="Path to cosine cache json",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=100.0,
        help="Softmax temperature to map cosine scores into confidence values",
    )
    parser.add_argument(
        "--details-csv",
        type=str,
        default="dataset/siglip_accuracy_details.csv",
        help="Where to write per-image prediction details csv",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="dataset/siglip_accuracy_summary.json",
        help="Where to write summary json",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    cache_path = Path(args.cache)
    details_csv = Path(args.details_csv)
    summary_json = Path(args.summary_json)

    summary, rows = evaluate(cache_path=cache_path, temperature=args.temperature)
    write_details_csv(rows, details_csv)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("SigLIP Accuracy Evaluation")
    print(f"cache: {cache_path}")
    print(f"num_images: {summary['num_images']}")
    print(f"top1_correct: {summary['top1_correct']}")
    print(f"top1_accuracy: {summary['top1_accuracy']:.4f}")
    print(f"mean_true_cosine: {summary['mean_true_cosine']:.6f}")
    print(f"mean_margin_true_minus_best_wrong: {summary['mean_margin_true_minus_best_wrong']:.6f}")
    print(f"mean_true_softmax_confidence: {summary['mean_true_softmax_confidence']:.6f}")
    print(f"mean_top1_softmax_confidence: {summary['mean_top1_softmax_confidence']:.6f}")
    print(f"mrr: {summary['mean_reciprocal_rank']:.6f}")
    print(f"details_csv: {details_csv}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
