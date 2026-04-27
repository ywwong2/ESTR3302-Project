import json
import math
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("backend/data/ablations")
summary_path = root / "ablation_summary.json"
with summary_path.open() as f:
    summary = json.load(f)

# Collect results from individual CSVs
results = {}
for subdir in root.iterdir():
    if not subdir.is_dir():
        continue
    parts = subdir.name.rsplit("_seed", 1)
    if len(parts) != 2:
        continue
    name, seed = parts
    seed = int(seed)
    csv_path = subdir / "simulation_rounds.csv"
    if not csv_path.exists():
        continue
    data = {k: [] for k in ["avg_reward", "w0_cos", "w1_fresh", "w2_lr", "w3_ctr", "w4_awt", "w5_social"]}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in data:
                data[k].append(float(row[k]))
    results.setdefault(name, {})[seed] = data

# Reward curves
fig, ax = plt.subplots(figsize=(14, 8))
for name in sorted(results.keys()):
    seeds = sorted(results[name].keys())
    arr = [results[name][s]["avg_reward"] for s in seeds]
    min_len = min(len(x) for x in arr)
    arr = [x[:min_len] for x in arr]
    mean_r = [sum(x[t] for x in arr) / len(arr) for t in range(min_len)]
    std_r = [math.sqrt(sum((x[t] - mean_r[t]) ** 2 for x in arr) / len(arr)) for t in range(min_len)]
    xs = list(range(1, min_len + 1))
    ax.plot(xs, mean_r, label=name, linewidth=1.2)
    ax.fill_between(xs, [m - s for m, s in zip(mean_r, std_r)], [m + s for m, s in zip(mean_r, std_r)], alpha=0.12)
ax.set_xlabel("Round")
ax.set_ylabel("Avg Reward")
ax.set_title("Reward Curves (mean ± std over seeds)")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(root / "reward_curves_all.png", dpi=180)
plt.close(fig)
print(f"Saved {root / 'reward_curves_all.png'}")

# Weight trajectories for selected configs
selected = ["baseline", "tau_0.5", "tau_1.0", "tau_1.0_beta_1e-3", "tau_1.0_beta_1e-3_norm", "no_social", "normalize_features"]
for name in selected:
    if name not in results:
        continue
    fig, ax = plt.subplots(figsize=(10, 5))
    # Just plot first seed for clarity
    data = results[name][sorted(results[name].keys())[0]]
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
    ax.set_title(f"Weight Trajectory: {name} (seed 42)")
    ax.legend(ncol=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / f"weights_{name}.png", dpi=140)
    plt.close(fig)
print(f"Saved weight trajectory plots")
