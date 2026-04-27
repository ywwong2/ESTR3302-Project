from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import csv
import json
import math

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

def run_single(seed: int, out_dir: str, extra_args: list[str], rounds: int = 1000) -> dict:
    cmd = [
        sys.executable, "backend/simulate_ranking.py",
        "--seed", str(seed),
        "--rounds", str(rounds),
        "--out-dir", out_dir,
        "--dataset-cache", "dataset/cosine_cache_siglip.json",
        "--no-plots",
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def parse_csv(csv_path: Path) -> dict[str, list[float]]:
    data = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return data

root = Path("backend/data/ablations_extra")
root.mkdir(parents=True, exist_ok=True)
seeds = [42, 43, 44]
rounds = 1000

configs = [
    ("balanced_reward", ["--tau", "1.0", "--beta", "0.001", "--normalize-features", "--lambda-v", "2.0", "--lambda-t", "2.0", "--lambda-l", "3.0"]),
    ("less_deep_reward", ["--tau", "1.0", "--beta", "0.001", "--normalize-features", "--lambda-v", "1.0", "--lambda-t", "1.0", "--lambda-l", "2.0"]),
    ("balanced_reward_no_norm", ["--tau", "1.0", "--beta", "0.001", "--lambda-v", "2.0", "--lambda-t", "2.0", "--lambda-l", "3.0"]),
    ("tau_0.5_norm", ["--tau", "0.5", "--normalize-features"]),
    ("tau_0.5_beta_1e-3_norm", ["--tau", "0.5", "--beta", "0.001", "--normalize-features"]),
]

results = {}
for name, extra in configs:
    print(f"\n=== Config: {name} ===")
    for seed in seeds:
        run_dir = root / f"{name}_seed{seed}"
        ok = run_single(seed, str(run_dir), extra, rounds=rounds)
        if ok:
            data = parse_csv(run_dir / "simulation_rounds.csv")
            results.setdefault(name, {})[seed] = data
            print(f"  seed {seed} done")
        else:
            print(f"  seed {seed} FAILED")

# Summarize
summary = {}
for name, runs in results.items():
    metrics = {k: [] for k in ["last100_reward", "w_cos", "w_fresh", "w_lr", "w_ctr", "w_awt", "w_social"]}
    for data in runs.values():
        metrics["last100_reward"].append(sum(data["avg_reward"][-100:]) / 100)
        metrics["w_cos"].append(sum(data["w0_cos"][-100:]) / 100)
        metrics["w_fresh"].append(sum(data["w1_fresh"][-100:]) / 100)
        metrics["w_lr"].append(sum(data["w2_lr"][-100:]) / 100)
        metrics["w_ctr"].append(sum(data["w3_ctr"][-100:]) / 100)
        metrics["w_awt"].append(sum(data["w4_awt"][-100:]) / 100)
        metrics["w_social"].append(sum(data.get("w5_social", [0]*1000)[-100:]) / 100)
    summary[name] = {k: (sum(v)/len(v), math.sqrt(sum((x-sum(v)/len(v))**2 for x in v)/len(v)) if len(v)>1 else 0.0) for k, v in metrics.items()}

print("\n" + "=" * 100)
print(f"{'Config':<30} {'Reward':>10} {'w_cos':>8} {'w_fresh':>8} {'w_lr':>8} {'w_ctr':>8} {'w_awt':>8} {'w_social':>8}")
print("-" * 100)
for name, s in summary.items():
    r = s["last100_reward"]
    print(f"{name:<30} {r[0]:>6.2f}±{r[1]:<4.2f} {s['w_cos'][0]:>6.3f}±{s['w_cos'][1]:<4.3f} {s['w_fresh'][0]:>6.3f}±{s['w_fresh'][1]:<4.3f} {s['w_lr'][0]:>6.3f}±{s['w_lr'][1]:<4.3f} {s['w_ctr'][0]:>6.3f}±{s['w_ctr'][1]:<4.3f} {s['w_awt'][0]:>6.3f}±{s['w_awt'][1]:<4.3f} {s['w_social'][0]:>6.3f}±{s['w_social'][1]:<4.3f}")

# Save summary
with (root / "summary.json").open("w") as f:
    json.dump({k: {m: {"mean": mv[0], "std": mv[1]} for m, mv in v.items()} for k, v in summary.items()}, f, indent=2)

if HAS_MPL:
    for name, runs in results.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        data = runs[sorted(runs.keys())[0]]
        rounds = list(range(1, len(data["avg_reward"]) + 1))
        for wk, wlabel, color in [
            ("w0_cos", "cos", "#1f77b4"),
            ("w1_fresh", "fresh", "#ff7f0e"),
            ("w2_lr", "lr", "#2ca02c"),
            ("w3_ctr", "ctr", "#d62728"),
            ("w4_awt", "awt", "#9467bd"),
            ("w5_social", "social", "#8c564b"),
        ]:
            if wk in data:
                ax.plot(rounds, data[wk], label=wlabel, color=color, linewidth=1.2)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Round")
        ax.set_ylabel("Weight")
        ax.set_title(f"Weight Trajectory: {name}")
        ax.legend(ncol=3)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(root / f"weights_{name}.png", dpi=140)
        plt.close(fig)
    print(f"Plots saved to {root}")
