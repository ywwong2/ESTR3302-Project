from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict


def load_query_ranks(path: Path, query_label: str) -> dict[int, list[dict]]:
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


def moving_average(series: list[float], window: int) -> list[float]:
    result = []
    for i in range(len(series)):
        start = max(0, i - window + 1)
        result.append(sum(series[start:i+1]) / (i - start + 1))
    return result


def load_quality_map(base: Path) -> dict[int, float]:
    state_path = base / "final_state.json"
    if state_path.exists():
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return {int(img["image_id"]): float(img["quality"]) for img in data.get("images", [])}

    item_path = base / "item_stats.csv"
    if item_path.exists():
        with item_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            qualities: dict[int, float] = {}
            for row in reader:
                if int(row["round"]) == 1:
                    qualities[int(row["image_id"])] = float(row["quality"])
            return qualities
    return {}


def load_categories(base: Path) -> dict[int, str]:
    CAT_NAMES = ["cat", "dog", "plane", "clothes", "car"]
    state_path = base / "final_state.json"
    if state_path.exists():
        data = json.loads(state_path.read_text(encoding="utf-8"))
        categories = {}
        for img in data.get("images", []):
            cat_val = img.get("category", "")
            try:
                cat_idx = int(cat_val)
                cat_val = CAT_NAMES[cat_idx] if cat_idx < len(CAT_NAMES) else str(cat_idx)
            except (ValueError, TypeError):
                pass
            categories[int(img["image_id"])] = str(cat_val)
        return categories
    item_path = base / "item_stats.csv"
    if item_path.exists():
        categories: dict[int, str] = {}
        with item_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["round"]) != 1:
                    continue
                iid = int(row["image_id"])
                cat_idx = int(row["category"])
                categories[iid] = CAT_NAMES[cat_idx] if cat_idx < len(CAT_NAMES) else str(cat_idx)
        return categories
    return {}


def select_diff_quality(image_ids: list[int], qualities: dict[int, float]) -> list[int]:
    """Select images at 1st/2nd (high), 5th/6th (mid), 9th/10th (low) quality positions."""
    sorted_ids = sorted(image_ids, key=lambda i: qualities.get(i, 0.0), reverse=True)  # highest first
    n = len(sorted_ids)
    result = []
    for pos in [0, 1, 4, 5, 8, 9]:  # 1st, 2nd, 5th, 6th, 9th, 10th (0-indexed)
        if pos < n:
            result.append(sorted_ids[pos])
    return result


def select_tight_quality_window(image_ids: list[int], qualities: dict[int, float], window: int = 6) -> list[int]:
    if len(image_ids) <= window:
        return image_ids
    sorted_ids = sorted(image_ids, key=lambda i: qualities.get(i, 0.0))
    best = sorted_ids[:window]
    best_range = qualities.get(best[-1], 0.0) - qualities.get(best[0], 0.0)
    for i in range(len(sorted_ids) - window + 1):
        chunk = sorted_ids[i:i + window]
        rng = qualities.get(chunk[-1], 0.0) - qualities.get(chunk[0], 0.0)
        if rng < best_range:
            best = chunk
            best_range = rng
    return best


def plot_quality_tiers(
    rank_path: Path,
    query_label: str,
    qualities: dict[int, float],
    categories: dict[int, str],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable")
        return

    image_ids = [iid for iid, label in categories.items() if label == query_label]
    if len(image_ids) < 3:
        print(f"Not enough images to plot quality tiers for {query_label}.")
        return

    sorted_ids = sorted(image_ids, key=lambda i: qualities.get(i, 0.0))
    n = len(sorted_ids)
    low_ids = set(sorted_ids[: n // 3])
    mid_ids = set(sorted_ids[n // 3 : (2 * n) // 3])
    high_ids = set(sorted_ids[(2 * n) // 3 :])

    tier_ranks: dict[str, dict[int, list[int]]] = {
        "low": defaultdict(list),
        "mid": defaultdict(list),
        "high": defaultdict(list),
    }

    with rank_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["query_label"] != query_label:
                continue
            iid = int(row["image_id"])
            round_id = int(row["round"])
            rank = int(row["rank"])
            if iid in low_ids:
                tier_ranks["low"][round_id].append(rank)
            elif iid in mid_ids:
                tier_ranks["mid"][round_id].append(rank)
            elif iid in high_ids:
                tier_ranks["high"][round_id].append(rank)

    rounds = sorted({r for tier in tier_ranks.values() for r in tier.keys()})
    if not rounds:
        print(f"No rank data for {query_label}.")
        return

    def mean_rank(tier: str) -> list[float]:
        values = []
        for r in rounds:
            vals = tier_ranks[tier].get(r, [])
            values.append(sum(vals) / len(vals) if vals else float("nan"))
        return values

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for tier, color in [("low", "#b91c1c"), ("mid", "#7c3aed"), ("high", "#047857")]:
        series = mean_rank(tier)
        series = [x for x in series]
        ax.plot(rounds, series, label=f"{tier} quality", color=color, linewidth=2.0)

    ax.invert_yaxis()
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Average Rank", fontsize=11)
    ax.set_title(f"Quality Tier Rank Separation ({query_label})", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_figure(
    rank_data: dict[int, list[dict]],
    image_ids: list[int],
    qualities: list[float],
    labels: list[str],
    colors: list[str],
    title: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for idx, iid in enumerate(image_ids):
        rows = rank_data[iid]
        rounds = [r["round"] for r in rows]
        ranks = [r["rank"] for r in rows]
        ma = moving_average(ranks, window=10)

        ax.plot(rounds, ranks, color=colors[idx], alpha=0.12, linewidth=0.7)
        ax.plot(rounds, ma, color=colors[idx], linewidth=2.2,
                label=f"Image {iid} ({labels[idx]}, q={qualities[idx]:.3f})")

    ax.invert_yaxis()
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Rank Position (lower = higher rank)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.25)
    # Set y-axis to show ranks 1 to 13
    ax.set_ylim(13.5, 0.5)
    ax.set_yticks(range(1, 14))

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_figure1_combined(
    rank_data: dict[int, list[dict]],
    image_ids: list[int],
    qualities_list: list[float],
    labels: list[str],
    colors: list[str],
    title: str,
    out_path: Path,
    quality_map: dict[int, float],
    category_map: dict[int, str],
) -> None:
    """Figure 1: left=rank dynamics, right=quality distribution per category."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception:
        print("matplotlib unavailable")
        return

    category_labels = ["cat", "dog", "plane", "clothes", "car"]
    cat_colors_dist = ["#047857", "#2563eb", "#d97706", "#7c3aed", "#b91c1c"]

    # Gather quality values per category (only original images, id < 50)
    cat_quality_vals: dict[str, list[float]] = {c: [] for c in category_labels}
    for iid, q in quality_map.items():
        if iid >= 50:
            continue
        cat_name = category_map.get(iid, "")
        # category_map may store int string or label; handle both
        if isinstance(cat_name, int):
            if cat_name < len(category_labels):
                cat_name = category_labels[cat_name]
        if cat_name in cat_quality_vals:
            cat_quality_vals[cat_name].append(q)

    fig, (ax_rank, ax_dist) = plt.subplots(1, 2, figsize=(16, 5.5),
                                            gridspec_kw={"width_ratios": [3, 1.2]})

    # --- Left: rank dynamics ---
    for idx, iid in enumerate(image_ids):
        rows = rank_data[iid]
        rounds = [r["round"] for r in rows]
        ranks = [r["rank"] for r in rows]
        ma = moving_average(ranks, window=10)
        ax_rank.plot(rounds, ranks, color=colors[idx], alpha=0.12, linewidth=0.7)
        ax_rank.plot(rounds, ma, color=colors[idx], linewidth=2.2,
                     label=f"Image {iid} ({labels[idx]}, q={qualities_list[idx]:.3f})")
    ax_rank.invert_yaxis()
    ax_rank.set_ylim(13.5, 0.5)
    ax_rank.set_yticks(range(1, 14))
    ax_rank.set_xlabel("Round", fontsize=11)
    ax_rank.set_ylabel("Rank Position (lower = higher rank)", fontsize=11)
    ax_rank.set_title(title, fontsize=12)
    ax_rank.legend(loc="upper right", fontsize=8.5)
    ax_rank.grid(alpha=0.25)

    # --- Right: quality distribution per category ---
    positions = list(range(1, len(category_labels) + 1))
    bplot = ax_dist.boxplot(
        [cat_quality_vals[c] for c in category_labels],
        positions=positions,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bplot["boxes"], cat_colors_dist):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for flier, color in zip(bplot["fliers"], cat_colors_dist):
        flier.set(markerfacecolor=color, markeredgecolor=color)
    ax_dist.set_xticks(positions)
    ax_dist.set_xticklabels(category_labels, fontsize=9, rotation=15)
    ax_dist.set_ylabel("Quality Value", fontsize=10)
    ax_dist.set_title("Quality Distribution\nper Category", fontsize=11)
    ax_dist.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_quality_distribution(
    quality_map: dict[int, float],
    category_map: dict[int, str],
    out_path: Path,
) -> None:
    """Dot/strip plot of quality values per category (original images only)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import random as _rng
    except Exception:
        print("matplotlib unavailable")
        return

    category_labels = ["cat", "dog", "plane", "clothes", "car"]
    cat_colors_dist = ["#047857", "#2563eb", "#d97706", "#7c3aed", "#b91c1c"]

    cat_quality_vals: dict[str, list[float]] = {c: [] for c in category_labels}
    for iid, q in quality_map.items():
        if iid >= 50:
            continue
        cat_name = category_map.get(iid, "")
        if cat_name in cat_quality_vals:
            cat_quality_vals[cat_name].append(q)

    fig, ax = plt.subplots(figsize=(8, 5))
    _r = _rng.Random(0)

    for xi, (cat, color) in enumerate(zip(category_labels, cat_colors_dist), start=1):
        vals = sorted(cat_quality_vals[cat])
        jitter = [xi + _r.uniform(-0.18, 0.18) for _ in vals]
        ax.scatter(jitter, vals, color=color, alpha=0.75, s=45, edgecolors="white", linewidths=0.4, label=cat)
        # Median line
        if vals:
            med = sorted(vals)[len(vals) // 2]
            ax.plot([xi - 0.25, xi + 0.25], [med, med], color=color, linewidth=2.0, solid_capstyle="round")

    ax.set_xticks(range(1, len(category_labels) + 1))
    ax.set_xticklabels(category_labels, fontsize=10)
    ax.set_ylabel("Quality Value", fontsize=11)
    ax.set_title("Quality Score Distribution per Category\n(each dot = one image, line = median)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(0.4, len(category_labels) + 0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import sys
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("backend/data/final_search_first_v2")
    rank_path = base / "query_ranks.csv"
    out_dir = base

    quality_map = load_quality_map(base)
    category_map = load_categories(base)
    if not quality_map:
        quality_map = {2: 0.039, 9: 0.086, 8: 0.309, 1: 0.338, 4: 0.369, 5: 0.401,
                       27: 0.057, 29: 0.241, 23: 0.259, 21: 0.264, 20: 0.308, 28: 0.347}

    # Exclude injected images (id >= 50) from selection
    INJECT_THRESHOLD = 50
    cat_ids_orig = [iid for iid, label in category_map.items() if label == "cat" and iid < INJECT_THRESHOLD]
    if not cat_ids_orig:
        cat_ids_orig = [2, 9, 8, 1, 4, 5]
    cat_ids = select_diff_quality(cat_ids_orig, quality_map)
    cat_qualities = [quality_map.get(i, 0.0) for i in cat_ids]
    cat_labels = ["high", "high", "mid", "mid", "low", "low"][: len(cat_ids)]
    cat_colors = ["#047857", "#6ee7b7", "#7c3aed", "#c4b5fd", "#b91c1c", "#fca5a5"][: len(cat_ids)]

    # Figure 1: Rank dynamics only (6 lines)
    cat_data = load_query_ranks(rank_path, "cat")
    plot_figure(
        cat_data,
        image_ids=cat_ids,
        qualities=cat_qualities,
        labels=cat_labels,
        colors=cat_colors,
        title="Figure 1: Rank Dynamics — Cat Images (1st/2nd, 5th/6th, 9th/10th Quality)",
        out_path=out_dir / "figure1_diff_quality_6lines.png",
    )

    # Figure 4: Quality distribution per category (dot/strip plot)
    plot_quality_distribution(
        quality_map=quality_map,
        category_map=category_map,
        out_path=out_dir / "figure4_quality_distribution.png",
    )

    # Figure 2: Similar-quality plane images (6 lines, tightest 6-tuple)
    plane_ids = [iid for iid, label in category_map.items() if label == "plane"]
    if not plane_ids:
        plane_ids = [27, 29, 23, 21, 20, 28]
    plane_ids = select_tight_quality_window(plane_ids, quality_map, window=6)
    plane_qualities = [quality_map.get(i, 0.0) for i in plane_ids]
    plane_labels = [f"q≈{q:.3f}" for q in plane_qualities]
    plane_colors = ["#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0d9488"][: len(plane_ids)]
    plane_data = load_query_ranks(rank_path, "plane")
    plot_figure(
        plane_data,
        image_ids=plane_ids,
        qualities=plane_qualities,
        labels=plane_labels,
        colors=plane_colors,
        title="Figure 2: Rank Dynamics — Similar-Quality Plane Images (tightest 6-tuple)",
        out_path=out_dir / "figure2_similar_quality_planes_6lines.png",
    )

    # Figure 3: Injected images (rounds 500-1000 only)
    injected_ids = [50, 51, 52]  # IDs of injected images
    injected_qualities = [0.4, 0.25, 0.01]
    injected_labels = ["high-qual", "mid-qual", "low-qual"]
    injected_colors = ["#047857", "#7c3aed", "#b91c1c"]
    injected_data = load_query_ranks(rank_path, "cat")
    
    # Filter data to only include rounds >= 500
    for iid in injected_ids:
        if iid in injected_data:
            injected_data[iid] = [row for row in injected_data[iid] if row["round"] >= 500]
    
    if any(injected_data.get(iid, []) for iid in injected_ids):
        plot_figure(
            injected_data,
            image_ids=injected_ids,
            qualities=injected_qualities,
            labels=injected_labels,
            colors=injected_colors,
            title="Figure 3: Injected Images Rank Dynamics (rounds 500-1000)",
            out_path=out_dir / "figure3_injected_images.png",
        )

