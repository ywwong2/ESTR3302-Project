from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict

def load_item_stats(path: Path) -> dict[int, list[dict]]:
    """Load per-image stats keyed by image_id."""
    data: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = int(row["image_id"])
            data[iid].append({
                "round": int(row["round"]),
                "ctr": float(row["ctr"]),
                "lr": float(row["lr"]),
                "awt": float(row["awt"]),
                "social": float(row["social_proof"]),
                "score": float(row["own_query_score"]),
            })
    return data


def load_query_ranks(path: Path, query_label: str) -> dict[int, list[dict]]:
    """Load rank trajectories for a specific query, keyed by image_id."""
    data: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["query_label"] == query_label:
                iid = int(row["image_id"])
                data[iid].append({
                    "round": int(row["round"]),
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                })
    return data


def ma(series: list[float], window: int) -> list[float]:
    """Simple moving average."""
    result = []
    for i in range(len(series)):
        start = max(0, i - window + 1)
        result.append(sum(series[start:i+1]) / (i - start + 1))
    return result


def plot_figure1(data: dict[int, list[dict]], image_id: int, out_path: Path) -> None:
    """Plot smoothed CTR, LR, AWT, social proof, and ranking score for one image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable")
        return

    rows = data[image_id]
    rounds = [r["round"] for r in rows]
    ctr = [r["ctr"] for r in rows]
    lr = [r["lr"] for r in rows]
    awt = [r["awt"] for r in rows]
    social = [r["social"] for r in rows]
    score = [r["score"] for r in rows]

    # Apply light smoothing for readability
    window = 20
    ctr_s = ma(ctr, window)
    lr_s = ma(lr, window)
    awt_s = ma(awt, window)
    social_s = ma(social, window)
    score_s = ma(score, window)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Panel 1: CTR and LR (rates)
    ax = axes[0]
    ax.plot(rounds, ctr_s, label="CTR (smoothed)", color="#2563eb", linewidth=1.5)
    ax.plot(rounds, lr_s, label="Like Rate (smoothed)", color="#dc2626", linewidth=1.5)
    ax.set_ylabel("Rate")
    ax.set_title(f"Image {image_id} — Estimated Engagement Rates")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, max(max(ctr_s), max(lr_s)) * 1.15)

    # Panel 2: AWT and Social Proof
    ax = axes[1]
    ax.plot(rounds, awt_s, label="Avg Watch Time (smoothed)", color="#059669", linewidth=1.5)
    ax.plot(rounds, social_s, label="Social Proof (smoothed)", color="#7c3aed", linewidth=1.5)
    ax.set_ylabel("Score")
    ax.set_title(f"Image {image_id} — Depth Engagement & Social Proof")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    # Panel 3: Ranking Score
    ax = axes[2]
    ax.plot(rounds, score, label="Raw score", color="#9ca3af", alpha=0.4, linewidth=0.8)
    ax.plot(rounds, score_s, label="Score (MA20)", color="#0f766e", linewidth=1.8)
    ax.set_xlabel("Round")
    ax.set_ylabel("Ranking Score")
    ax.set_title(f"Image {image_id} — Ranking Score (Own-Category Query)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Figure 1 saved: {out_path}")


def plot_figure2(data: dict[int, list[dict]], image_ids: list[int], out_path: Path) -> None:
    """Plot rank positions over time for 2-3 images under a fixed query."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#2563eb", "#dc2626", "#059669"]
    for idx, iid in enumerate(image_ids):
        rows = data[iid]
        rounds = [r["round"] for r in rows]
        ranks = [r["rank"] for r in rows]
        # Invert y-axis so rank 1 is at top
        ax.plot(rounds, ranks, label=f"Image {iid}", color=colors[idx], linewidth=1.5, alpha=0.8)

    ax.invert_yaxis()
    ax.set_xlabel("Round")
    ax.set_ylabel("Rank Position (lower = higher rank)")
    ax.set_title("Rank Trajectories — Cat Query")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Figure 2 saved: {out_path}")


if __name__ == "__main__":
    import sys
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("backend/data/final_config5")
    item_path = base / "item_stats.csv"
    rank_path = base / "query_ranks.csv"
    out_dir = base

    item_data = load_item_stats(item_path)
    cat_ranks = load_query_ranks(rank_path, "cat")

    # Figure 1: Image 5 (cat) — clear winner with rich trajectory
    plot_figure1(item_data, 5, out_dir / "figure1_image5_trajectory.png")

    # Figure 2: Images 5 (winner), 0 (mid-tier decliner), 2 (never-ranked) under cat query
    plot_figure2(cat_ranks, [5, 0, 2], out_dir / "figure2_cat_ranks.png")
