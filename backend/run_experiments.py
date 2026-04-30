#!/usr/bin/env python3
"""
Run three verification experiments and generate comparison plots.

Usage:
    python backend/run_experiments.py          # run all 3
    python backend/run_experiments.py exp1     # run only experiment 1
    python backend/run_experiments.py exp2
    python backend/run_experiments.py exp3
    python backend/run_experiments.py plots    # regenerate plots only (no simulation)
"""
from __future__ import annotations

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
SIM = "backend/simulate_ranking.py"
CACHE = "dataset/cosine_cache_siglip.json"

BAYESIAN_DIR = ROOT / "backend" / "data" / "final_search_first_v3_twostage_injection_v5_fixed"
BAYESIAN_DEF_DIR = ROOT / "backend" / "data" / "exp_bayesian_defended"
EXP1_DIR = ROOT / "backend" / "data" / "exp_bayesian_vs_freq"
EXP1_FREQ10_DIR = ROOT / "backend" / "data" / "exp_bayesian_vs_freq_w10"
EXP1_FREQ30_DIR = ROOT / "backend" / "data" / "exp_bayesian_vs_freq_w30"
EXP1_DEFENDED_DIR = ROOT / "backend" / "data" / "exp_bayesian_trust_ramp"
EXP2_DIR = ROOT / "backend" / "data" / "exp_drift"
EXP3_DIR = ROOT / "backend" / "data" / "exp_synthetic"

# ── Shared base params (canonical run) ─────────────────────────
BASE_PARAMS = [
    "--seed", "42",
    "--dataset-cache", CACHE,
    "--w-init", "0.25", "0.15", "0.15", "0.15", "0.15", "0.15",
    "--tau", "0.40", "--beta", "0.005", "--eta", "0.004",
    "--batch-size", "64", "--rounds", "1000", "--top-k", "10",
    "--rho", "0.20", "--gamma-pos", "0.90",
    "--lambda-v", "1.2", "--lambda-t", "2.2", "--lambda-l", "1.2",
    "--alpha", "-1.40", "1.40", "3.40", "0.05", "0.10",
    "--alpha-q", "0.30",
    "--gamma", "-0.90", "0.25", "0.90", "0.08", "0.05",
    "--gamma-q", "4.50",
    "--delta", "-2.60", "0.05", "0.25", "0.03", "0.05",
    "--delta-t", "3.00", "--delta-q", "4.00",
    "--kappa", "12",
    "--omega-l", "1.5", "--m-social", "6.2166",  # ln(1+500)
    "--two-stage-top-k", "13",
    "--quality-weight-cos", "1.0", "--quality-weight-match", "2.0",
    "--quality-weight-fresh", "0.5", "--quality-weight-ctr", "0.5",
    "--quality-weight-lr", "5.0", "--quality-weight-awt", "2.0",
    "--normalize-features",
    "--min-fresh-weight", "0.05",
    "--freshness-decay-scale", "5.0",
    "--recovery-boost-interval", "50", "--recovery-boost-strength", "0.5",
    "--item-diagnostics",
]


def run_sim(extra_args: list[str], label: str) -> None:
    cmd = [PYTHON, SIM] + BASE_PARAMS + extra_args
    print(f"\n{'='*60}\n  Running: {label}\n{'='*60}")
    print(f"  cmd: {' '.join(cmd[:6])} ... ({len(cmd)} args)")
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    print(f"  ✅ {label} done\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EXPERIMENT 1: Bayesian vs Frequency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_exp1() -> None:
    # Bayesian smoothing (no defense — isolate update rule)
    run_sim([
        "--inject-images", "--inject-round", "501", "--inject-category", "cat",
        "--out-dir", str(BAYESIAN_DEF_DIR),
    ], "Exp 1: Bayesian smoothing")
    # Frequency baseline W=50 (no defense — isolate update rule)
    run_sim([
        "--inject-images", "--inject-round", "501", "--inject-category", "cat",
        "--frequency-update", "--frequency-window", "50",
        "--out-dir", str(EXP1_DIR),
    ], "Exp 1: Frequency baseline W=50")
    # Frequency baseline W=30
    run_sim([
        "--inject-images", "--inject-round", "501", "--inject-category", "cat",
        "--frequency-update", "--frequency-window", "30",
        "--out-dir", str(EXP1_FREQ30_DIR),
    ], "Exp 1: Frequency baseline W=30")
    # Frequency baseline W=10
    run_sim([
        "--inject-images", "--inject-round", "501", "--inject-category", "cat",
        "--frequency-update", "--frequency-window", "10",
        "--out-dir", str(EXP1_FREQ10_DIR),
    ], "Exp 1: Frequency baseline W=10")


def run_exp1_defended() -> None:
    """Standalone: Bayesian smoothing + trust ramp (production config, with injection)."""
    run_sim([
        "--inject-images", "--inject-round", "501", "--inject-category", "cat",
        "--trust-ramp-period", "200",
        "--out-dir", str(EXP1_DEFENDED_DIR),
    ], "Exp 1b: Bayesian + Trust Ramp")


def plot_exp1_defended() -> None:
    """Single-panel rank-trajectory plot for Bayesian + trust ramp."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load_query_ranks(base: Path, query: str) -> dict[int, list[dict]]:
        data: dict[int, list[dict]] = defaultdict(list)
        with (base / "query_ranks.csv").open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["query_label"] == query:
                    data[int(row["image_id"])].append(
                        {"round": int(row["round"]), "rank": int(row["rank"])}
                    )
        return data

    def moving_avg(vals: list[float], w: int = 10) -> list[float]:
        out = []
        for i in range(len(vals)):
            s = max(0, i - w + 1)
            out.append(sum(vals[s:i+1]) / (i - s + 1))
        return out

    injected = [50, 51, 52]
    labels_q = ["high (q=0.4)", "mid (q=0.25)", "low (q=0.01)"]
    colors = ["#047857", "#7c3aed", "#b91c1c"]

    ranks = load_query_ranks(EXP1_DEFENDED_DIR, "cat")

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, iid in enumerate(injected):
        rows = [r for r in ranks.get(iid, []) if r["round"] >= 500]
        if not rows:
            continue
        rr = [r["round"] for r in rows]
        rk = [r["rank"] for r in rows]
        ax.plot(rr, rk, color=colors[idx], alpha=0.12, linewidth=0.7)
        ax.plot(rr, moving_avg(rk), color=colors[idx], linewidth=2.2,
                label=f"Image {iid} ({labels_q[idx]})")
    ax.invert_yaxis()
    ax.set_ylim(13.5, 0.5)
    ax.set_yticks(range(1, 14))
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Rank Position", fontsize=11)
    ax.set_title("Cold-Start Rank Dynamics: Bayesian Smoothing + Trust Ramp", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = EXP1_DEFENDED_DIR / "fig_bayesian_trust_ramp.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_exp1() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load_item_stats(base: Path) -> dict[int, list[dict]]:
        data: dict[int, list[dict]] = defaultdict(list)
        path = base / "item_stats.csv"
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[int(row["image_id"])].append({
                    "round": int(row["round"]),
                    "ctr": float(row["ctr"]),
                    "lr": float(row["lr"]),
                    "awt": float(row["awt"]),
                })
        return data

    def load_query_ranks(base: Path, query: str) -> dict[int, list[dict]]:
        data: dict[int, list[dict]] = defaultdict(list)
        path = base / "query_ranks.csv"
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["query_label"] == query:
                    data[int(row["image_id"])].append({
                        "round": int(row["round"]),
                        "rank": int(row["rank"]),
                    })
        return data

    def moving_avg(vals: list[float], w: int = 10) -> list[float]:
        out = []
        for i in range(len(vals)):
            s = max(0, i - w + 1)
            out.append(sum(vals[s:i+1]) / (i - s + 1))
        return out

    injected = [50, 51, 52]
    labels_q = ["high (q=0.4)", "mid (q=0.25)", "low (q=0.01)"]
    colors = ["#047857", "#7c3aed", "#b91c1c"]

    # ── Plot A: Side-by-side rank dynamics ──
    bay_ranks = load_query_ranks(BAYESIAN_DEF_DIR, "cat")
    freq30_ranks = load_query_ranks(EXP1_FREQ30_DIR, "cat")
    freq10_ranks = load_query_ranks(EXP1_FREQ10_DIR, "cat")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # LHS: Bayesian — solid lines
    for idx, iid in enumerate(injected):
        rows = [r for r in bay_ranks.get(iid, []) if r["round"] >= 500]
        if not rows:
            continue
        rounds = [r["round"] for r in rows]
        ma = moving_avg([r["rank"] for r in rows])
        ax1.plot(rounds, ma, color=colors[idx], linewidth=2.2,
                 label=f"Image {iid} ({labels_q[idx]})")
    ax1.set_title("Bayesian Smoothing", fontsize=12)

    # RHS: W=30 solid, W=10 dotted; same color per image
    variants = [
        (freq30_ranks, "-",  2.2, "W=30"),
        (freq10_ranks, ":",  2.0, "W=10"),
    ]
    for freq_ranks, ls, lw, wlabel in variants:
        for idx, iid in enumerate(injected):
            rows = [r for r in freq_ranks.get(iid, []) if r["round"] >= 500]
            if not rows:
                continue
            rounds = [r["round"] for r in rows]
            ma = moving_avg([r["rank"] for r in rows])
            if ls == "-":
                label = f"Image {iid} ({labels_q[idx]}) W=30"
            elif idx == 0:
                label = "W=10 (dotted)"
            else:
                label = None
            ax2.plot(rounds, ma, color=colors[idx], linestyle=ls, linewidth=lw, label=label)
    ax2.set_title("Frequency (solid=W=30, dotted=W=10)", fontsize=12)

    for ax in (ax1, ax2):
        ax.invert_yaxis()
        ax.set_ylim(13.5, 0.5)
        ax.set_yticks(range(1, 14))
        ax.set_xlabel("Round", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
    ax1.set_ylabel("Rank Position", fontsize=11)
    fig.suptitle("Cold-Start Rank Dynamics: Bayesian vs Frequency", fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = EXP1_DIR / "fig_rank_comparison.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ── Plot B: CTR/LR over time for injected images ──
    bay_stats = load_item_stats(BAYESIAN_DEF_DIR)
    freq_stats = load_item_stats(EXP1_DIR)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for metric_idx, (metric, ax) in enumerate(zip(["ctr", "lr"], axes)):
        for idx, iid in enumerate(injected):
            for src, src_label, ls in [(bay_stats, "Bayesian", "-"), (freq_stats, "Freq", "--")]:
                rows = src.get(iid, [])
                rows = [r for r in rows if r["round"] >= 500]
                if not rows:
                    continue
                rounds = [r["round"] for r in rows]
                vals = [r[metric] for r in rows]
                lbl = f"Img {iid} ({src_label})" if metric_idx == 0 else None
                ax.plot(rounds, vals, color=colors[idx], linestyle=ls, linewidth=1.5, label=lbl)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=7.5, ncol=2)
    axes[0].set_title("Cold-Start Metric Stability: Bayesian (solid) vs Frequency (dashed)", fontsize=12)
    axes[1].set_xlabel("Round", fontsize=11)
    fig.tight_layout()
    out_path = EXP1_DIR / "fig_cold_start_metrics.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EXPERIMENT 2: Temporal Drift
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_exp2() -> None:
    run_sim([
        "--drift-round", "500",
        "--drift-alpha-q", "0.05",
        "--drift-gamma-q", "1.00",
        "--drift-delta-q", "1.00",
        "--drift-alpha", "-1.40", "1.40", "3.40", "0.40", "0.80",
        "--out-dir", str(EXP2_DIR),
    ], "Exp 2: Temporal drift at round 500")


def plot_exp2() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load simulation_rounds.csv
    csv_path = EXP2_DIR / "simulation_rounds.csv"
    rounds, rewards = [], []
    w_hist: dict[str, list[float]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["round"])
            rounds.append(t)
            rewards.append(float(row["avg_reward"]))
            for key in ["w0_cos", "w1_fresh", "w2_lr", "w3_ctr", "w4_awt", "w5_social"]:
                w_hist[key].append(float(row[key]))

    drift_round = 500

    # ── Plot A: Reward curve ──
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(rounds, rewards, color="#0f766e", linewidth=1.4)
    ax.axvline(drift_round, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Drift at round {drift_round}")
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Avg Reward", fontsize=11)
    ax.set_title("Reward Under Temporal Drift", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = EXP2_DIR / "fig_drift_reward.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Plot B: Weight trajectory ──
    labels_w = ["cos", "fresh", "lr", "ctr", "awt", "social"]
    colors_w = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed", "#64748b"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, lbl) in enumerate(zip(sorted(w_hist.keys()), labels_w)):
        ax.plot(rounds, w_hist[key], color=colors_w[i], linewidth=1.5, label=lbl)
    ax.axvline(drift_round, color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Weight", fontsize=11)
    ax.set_title("Weight Trajectory Under Temporal Drift", fontsize=12)
    ax.legend(ncol=3, fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = EXP2_DIR / "fig_drift_weights.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Plot C: Rank dynamics for cat images (high-q vs low-q) ──
    item_path = EXP2_DIR / "item_stats.csv"
    qualities: dict[int, float] = {}
    categories: dict[int, int] = {}
    with item_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["round"]) == 1:
                iid = int(row["image_id"])
                qualities[iid] = float(row["quality"])
                categories[iid] = int(row["category"])

    cat_ids = [iid for iid, cat in categories.items() if cat == 0]
    cat_ids_sorted = sorted(cat_ids, key=lambda i: qualities.get(i, 0), reverse=True)
    pick_high = cat_ids_sorted[:2]
    pick_low = cat_ids_sorted[-2:]
    picked = pick_high + pick_low

    rank_path = EXP2_DIR / "query_ranks.csv"
    rank_data: dict[int, list[dict]] = defaultdict(list)
    with rank_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["query_label"] == "cat":
                iid = int(row["image_id"])
                if iid in picked:
                    rank_data[iid].append({"round": int(row["round"]), "rank": int(row["rank"])})

    def ma(vals, w=10):
        out = []
        for i in range(len(vals)):
            s = max(0, i - w + 1)
            out.append(sum(vals[s:i+1]) / (i - s + 1))
        return out

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_r = ["#047857", "#6ee7b7", "#b91c1c", "#fca5a5"]
    labels_r = [f"Img {iid} (q={qualities[iid]:.3f})" for iid in picked]
    for idx, iid in enumerate(picked):
        rows = rank_data.get(iid, [])
        if not rows:
            continue
        rr = [r["round"] for r in rows]
        rk = [r["rank"] for r in rows]
        ax.plot(rr, rk, color=colors_r[idx], alpha=0.12, linewidth=0.7)
        ax.plot(rr, ma(rk), color=colors_r[idx], linewidth=2.2, label=labels_r[idx])
    ax.axvline(drift_round, color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.6, label="Drift")
    ax.invert_yaxis()
    ax.set_ylim(13.5, 0.5)
    ax.set_yticks(range(1, 14))
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Rank", fontsize=11)
    ax.set_title("Rank Dynamics Under Temporal Drift (Cat Query)", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = EXP2_DIR / "fig_drift_ranks.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved: {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EXPERIMENT 3: Resilience to Synthetic Interaction Attacks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ATTACK_INTENSITIES = [0, 5, 10, 20, 50]  # 0 = natural baseline
DEFENSE_CLAMP = 2.0  # velocity clamp factor for defended runs
DEFENSE_TRUST_RAMP = 200  # trust ramp period (rounds) for defended runs

def run_exp3() -> None:
    # Undefended runs
    for n_views in ATTACK_INTENSITIES:
        label = f"natural" if n_views == 0 else f"{n_views}v/round"
        out = EXP3_DIR / f"intensity_{n_views}"
        run_sim([
            "--synthetic-image-round", "200",
            "--synthetic-image-quality", "0.01",
            "--synthetic-start-round", str(220 if n_views > 0 else 0),
            "--synthetic-views-per-round", str(n_views),
            "--synthetic-like-ratio", "0.8",
            "--out-dir", str(out),
        ], f"Exp 3: Synthetic attack ({label})")
    # Defended runs (velocity clamp + trust ramp)
    for n_views in ATTACK_INTENSITIES:
        label = f"natural+def" if n_views == 0 else f"{n_views}v/round+def"
        out = EXP3_DIR / f"defended_{n_views}"
        run_sim([
            "--synthetic-image-round", "200",
            "--synthetic-image-quality", "0.01",
            "--synthetic-start-round", str(220 if n_views > 0 else 0),
            "--synthetic-views-per-round", str(n_views),
            "--synthetic-like-ratio", "0.8",
            "--velocity-clamp", str(DEFENSE_CLAMP),
            "--trust-ramp-period", str(DEFENSE_TRUST_RAMP),
            "--out-dir", str(out),
        ], f"Exp 3: Defended ({label})")


def run_exp3d() -> None:
    """Run only the defended variants (for re-running defense only)."""
    for n_views in ATTACK_INTENSITIES:
        label = f"natural+def" if n_views == 0 else f"{n_views}v/round+def"
        out = EXP3_DIR / f"defended_{n_views}"
        run_sim([
            "--synthetic-image-round", "200",
            "--synthetic-image-quality", "0.01",
            "--synthetic-start-round", str(220 if n_views > 0 else 0),
            "--synthetic-views-per-round", str(n_views),
            "--synthetic-like-ratio", "0.8",
            "--velocity-clamp", str(DEFENSE_CLAMP),
            "--trust-ramp-period", str(DEFENSE_TRUST_RAMP),
            "--out-dir", str(out),
        ], f"Exp 3: Defended ({label})")


def plot_exp3() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors_atk = ["#64748b", "#2563eb", "#d97706", "#dc2626", "#7c3aed"]

    # ── Plot A: Rank trajectory at each intensity ──
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rounds_to_top3: dict[int, int | None] = {}
    natural_avg_engagement = 64 * 0.3 * 0.2  # ~3.8 views/image/round

    for idx, n_views in enumerate(ATTACK_INTENSITIES):
        out = EXP3_DIR / f"intensity_{n_views}"
        rank_path = out / "query_ranks.csv"
        if not rank_path.exists():
            print(f"  ⚠ Missing: {rank_path}")
            continue

        # Find the synthetic image id (last image added, should be id=50)
        item_path = out / "item_stats.csv"
        syn_id = None
        with item_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iid = int(row["image_id"])
                if iid >= 50:
                    syn_id = iid
                    break
        if syn_id is None:
            print(f"  ⚠ Synthetic image not found in {item_path}")
            continue

        # Load rank trajectory
        rank_data: list[dict] = []
        with rank_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["query_label"] == "cat" and int(row["image_id"]) == syn_id:
                    rank_data.append({"round": int(row["round"]), "rank": int(row["rank"])})

        if not rank_data:
            continue
        rank_data = [r for r in rank_data if r["round"] >= 200]
        rr = [r["round"] for r in rank_data]
        rk = [r["rank"] for r in rank_data]

        # Moving average
        def ma(vals, w=10):
            out = []
            for i in range(len(vals)):
                s = max(0, i - w + 1)
                out.append(sum(vals[s:i+1]) / (i - s + 1))
            return out

        label = "Natural (no attack)" if n_views == 0 else f"{n_views} fake views/round"
        ax.plot(rr, rk, color=colors_atk[idx], alpha=0.10, linewidth=0.7)
        ax.plot(rr, ma(rk), color=colors_atk[idx], linewidth=2.2, label=label)

        # Find first round where rank ≤ 3
        first_top3 = None
        for r in rank_data:
            if r["rank"] <= 3 and r["round"] >= 220:
                first_top3 = r["round"]
                break
        rounds_to_top3[n_views] = first_top3

    ax.axvline(220, color="#64748b", linestyle=":", linewidth=1, alpha=0.6, label="Attack starts")
    ax.invert_yaxis()
    ax.set_ylim(13.5, 0.5)
    ax.set_yticks(range(1, 14))
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Rank", fontsize=11)
    ax.set_title("Synthetic Attack: Low-Quality Image Rank Trajectory", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = EXP3_DIR / "fig_attack_rank_trajectory.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ── Plot B: Effort bar chart ──
    intensities = [n for n in ATTACK_INTENSITIES if n > 0]
    reached = []
    total_fake = []
    for n in intensities:
        rt3 = rounds_to_top3.get(n)
        if rt3 is not None:
            rounds_needed = rt3 - 220
            reached.append(rounds_needed)
            total_fake.append(rounds_needed * n)
        else:
            reached.append(float("nan"))
            total_fake.append(float("nan"))

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(intensities))
    bars = ax.bar(x_pos, reached, color=colors_atk[1:len(intensities)+1], width=0.6, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{n} views/round" for n in intensities], fontsize=10)
    ax.set_ylabel("Rounds to Reach Top-3", fontsize=11)
    ax.set_title("Effort Required for Synthetic Attack", fontsize=12)

    for i, (bar, tf) in enumerate(zip(bars, total_fake)):
        if not (tf != tf):  # not NaN
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{int(tf)} total\n({intensities[i]/natural_avg_engagement:.1f}× pop.)",
                    ha="center", va="bottom", fontsize=8.5)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 5,
                    "Never\nreached\ntop-3", ha="center", va="bottom", fontsize=8.5, color="#dc2626")
            bar.set_height(0)

    ax.set_ylim(0, max([r for r in reached if r == r], default=100) * 1.4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = EXP3_DIR / "fig_attack_effort.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print summary
    print("\n── Exp 3 Summary (Undefended) ──")
    print(f"Natural avg engagement: ~{natural_avg_engagement:.1f} views/image/round")
    for n in intensities:
        rt3 = rounds_to_top3.get(n)
        if rt3 is not None:
            print(f"  {n} fake views/round: top-3 at round {rt3} ({rt3-220} rounds, {(rt3-220)*n} total fake, {n/natural_avg_engagement:.1f}× population rate)")
        else:
            print(f"  {n} fake views/round: never reached top-3")

    # ── Plot C: Side-by-side undefended vs defended ──
    _plot_exp3_defended(plt, natural_avg_engagement)


def _plot_exp3_defended(plt, natural_avg_engagement: float) -> None:
    """Side-by-side: undefended vs defended rank trajectories."""
    colors_atk = ["#64748b", "#2563eb", "#d97706", "#dc2626", "#7c3aed"]

    def _load_syn_ranks(base: Path):
        item_path = base / "item_stats.csv"
        syn_id = None
        with item_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["image_id"]) >= 50:
                    syn_id = int(row["image_id"])
                    break
        if syn_id is None:
            return [], None
        rank_data = []
        with (base / "query_ranks.csv").open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["query_label"] == "cat" and int(row["image_id"]) == syn_id:
                    rank_data.append({"round": int(row["round"]), "rank": int(row["rank"])})
        return [r for r in rank_data if r["round"] >= 200], syn_id

    def ma(vals, w=10):
        out = []
        for i in range(len(vals)):
            s = max(0, i - w + 1)
            out.append(sum(vals[s:i+1]) / (i - s + 1))
        return out

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for panel_idx, (prefix, title, ax) in enumerate([
        ("intensity", "No Defense", ax1),
        ("defended", f"Velocity Clamp ({DEFENSE_CLAMP:.0f}x) + Trust Ramp ({DEFENSE_TRUST_RAMP}r)", ax2),
    ]):
        for idx, n_views in enumerate(ATTACK_INTENSITIES):
            base = EXP3_DIR / f"{prefix}_{n_views}"
            if not (base / "query_ranks.csv").exists():
                continue
            rank_data, _ = _load_syn_ranks(base)
            if not rank_data:
                continue
            rr = [r["round"] for r in rank_data]
            rk = [r["rank"] for r in rank_data]
            lbl = "Natural" if n_views == 0 else f"{n_views} views/rnd"
            ax.plot(rr, rk, color=colors_atk[idx], alpha=0.10, linewidth=0.7)
            ax.plot(rr, ma(rk), color=colors_atk[idx], linewidth=2.2, label=lbl)
        ax.axvline(220, color="#64748b", linestyle=":", linewidth=1, alpha=0.6)
        ax.invert_yaxis()
        ax.set_ylim(13.5, 0.5)
        ax.set_yticks(range(1, 14))
        ax.set_xlabel("Round", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
    ax1.set_ylabel("Rank", fontsize=11)
    fig.suptitle("Synthetic Attack: Undefended vs Velocity-Clamped", fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = EXP3_DIR / "fig_attack_defended.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print defended summary
    print("\n── Exp 3 Summary (Defended, clamp={:.0f}×) ──".format(DEFENSE_CLAMP))
    for n_views in [n for n in ATTACK_INTENSITIES if n > 0]:
        base = EXP3_DIR / f"defended_{n_views}"
        rank_data, _ = _load_syn_ranks(base)
        first_top3 = None
        for r in rank_data:
            if r["rank"] <= 3 and r["round"] >= 220:
                first_top3 = r["round"]
                break
        if first_top3 is not None:
            rounds_needed = first_top3 - 220
            print(f"  {n_views} fake views/round: top-3 at round {first_top3} ({rounds_needed} rounds, {rounds_needed*n_views} total fake)")
        else:
            print(f"  {n_views} fake views/round: NEVER reached top-3")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target in ("all", "exp1"):
        run_exp1()
    if target in ("all", "exp2"):
        run_exp2()
    if target in ("all", "exp3"):
        run_exp3()
    if target == "exp3d":
        run_exp3d()

    # Generate plots
    if target in ("all", "exp1", "plots"):
        print("\n── Generating Exp 1 plots ──")
        plot_exp1()
    if target in ("all", "exp2", "plots"):
        print("\n── Generating Exp 2 plots ──")
        plot_exp2()
    if target in ("all", "exp3", "exp3d", "plots"):
        print("\n── Generating Exp 3 plots ──")
        plot_exp3()

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
