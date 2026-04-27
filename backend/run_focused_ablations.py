from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import csv
import json
import math
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

def run_single(seed: int, out_dir: str, extra_args: list[str], rounds: int = 1000) -> bool:
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

def moving_average(values: list[float], window: int) -> list[float]:
    if window >= len(values):
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result

def compute_diagnostics(data: dict[str, list[float]], last_n: int = 100) -> dict:
    last_reward = data["avg_reward"][-last_n:]
    mean_reward = sum(last_reward) / len(last_reward)
    std_reward = math.sqrt(sum((x - mean_reward)**2 for x in last_reward) / len(last_reward))
    cv_reward = std_reward / mean_reward if mean_reward > 0 else 0.0
    
    # Weight stats
    weight_keys = [k for k in data.keys() if k.startswith("w")]
    weight_stats = {}
    for wk in weight_keys:
        w_vals = data[wk][-last_n:]
        weight_stats[wk] = {
            "mean": sum(w_vals) / len(w_vals),
            "std": math.sqrt(sum((x - sum(w_vals)/len(w_vals))**2 for x in w_vals) / len(w_vals))
        }
    
    # Check for collapse/dominance
    collapsed = [wk for wk, ws in weight_stats.items() if ws["mean"] < 0.01]
    dominated = [wk for wk, ws in weight_stats.items() if ws["mean"] > 0.6]
    
    # Global metrics
    global_ctr = sum(data["global_ctr"][-last_n:]) / last_n
    global_lr = sum(data["global_lr"][-last_n:]) / last_n
    global_awt = sum(data["global_awt"][-last_n:]) / last_n
    
    return {
        "reward_mean": mean_reward,
        "reward_std": std_reward,
        "reward_cv": cv_reward,
        "weight_stats": weight_stats,
        "collapsed_features": collapsed,
        "dominated_features": dominated,
        "global_ctr": global_ctr,
        "global_lr": global_lr,
        "global_awt": global_awt,
    }

root = Path("backend/data/focused_ablations")
root.mkdir(parents=True, exist_ok=True)
seeds = [42, 43, 44]
rounds = 1000

# 5 prioritized configs from user
configs = [
    ("config1_cos0.20_tau0.5_beta1e-3_eta0.005_bs64", 
     ["--tau", "0.5", "--beta", "0.001", "--normalize-features", "--freeze-cos", "0.20", "--eta", "0.005", "--batch-size", "64"]),
    ("config2_cos0.20_tau0.3_beta1e-3_eta0.005_bs64",
     ["--tau", "0.3", "--beta", "0.001", "--normalize-features", "--freeze-cos", "0.20", "--eta", "0.005", "--batch-size", "64"]),
    ("config3_cos0.25_tau0.5_beta1e-3_eta0.005_bs64",
     ["--tau", "0.5", "--beta", "0.001", "--normalize-features", "--freeze-cos", "0.25", "--eta", "0.005", "--batch-size", "64"]),
    ("config4_cos0.20_tau0.5_beta5e-3_eta0.005_bs128",
     ["--tau", "0.5", "--beta", "0.005", "--normalize-features", "--freeze-cos", "0.20", "--eta", "0.005", "--batch-size", "128"]),
    ("config5_cos0.20_tau0.5_beta1e-3_eta0.005_bs64_lambda323",
     ["--tau", "0.5", "--beta", "0.001", "--normalize-features", "--freeze-cos", "0.20", "--eta", "0.005", "--batch-size", "64", "--lambda-v", "3.0", "--lambda-t", "2.0", "--lambda-l", "3.0"]),
]

results = {}
for name, extra in configs:
    print(f"\n=== Config: {name} ===")
    for seed in seeds:
        run_dir = root / f"{name}_seed{seed}"
        ok = run_single(seed, str(run_dir), extra, rounds=rounds)
        if ok:
            data = parse_csv(run_dir / "simulation_rounds.csv")
            diag = compute_diagnostics(data, last_n=100)
            results.setdefault(name, {})[seed] = {"data": data, "diag": diag}
            print(f"  seed {seed} done (reward={diag['reward_mean']:.2f}±{diag['reward_std']:.2f}, cv={diag['reward_cv']:.3f})")
        else:
            print(f"  seed {seed} FAILED")

# Aggregate results
summary = {}
for name, runs in results.items():
    agg = {
        "reward_mean": [],
        "reward_std": [],
        "reward_cv": [],
        "w_cos": [], "w_fresh": [], "w_lr": [], "w_ctr": [], "w_awt": [], "w_social": [],
        "global_ctr": [], "global_lr": [], "global_awt": [],
    }
    for r in runs.values():
        d = r["diag"]
        agg["reward_mean"].append(d["reward_mean"])
        agg["reward_std"].append(d["reward_std"])
        agg["reward_cv"].append(d["reward_cv"])
        agg["global_ctr"].append(d["global_ctr"])
        agg["global_lr"].append(d["global_lr"])
        agg["global_awt"].append(d["global_awt"])
        ws = d["weight_stats"]
        agg["w_cos"].append(ws.get("w0_cos", {"mean": 0})["mean"])
        agg["w_fresh"].append(ws.get("w1_fresh", {"mean": 0})["mean"])
        agg["w_lr"].append(ws.get("w2_lr", {"mean": 0})["mean"])
        agg["w_ctr"].append(ws.get("w3_ctr", {"mean": 0})["mean"])
        agg["w_awt"].append(ws.get("w4_awt", {"mean": 0})["mean"])
        agg["w_social"].append(ws.get("w5_social", {"mean": 0})["mean"])
    
    summary[name] = {k: (sum(v)/len(v), math.sqrt(sum((x-sum(v)/len(v))**2 for x in v)/len(v)) if len(v)>1 else 0.0) for k, v in agg.items()}

print("\n" + "=" * 130)
print(f"{'Config':<45} {'Reward':>10} {'Std':>8} {'CV':>8} {'w_cos':>8} {'w_fresh':>8} {'w_lr':>8} {'w_ctr':>8} {'w_awt':>8} {'w_social':>8}")
print("-" * 130)
for name, s in summary.items():
    r = s["reward_mean"]
    std = s["reward_std"]
    cv = s["reward_cv"]
    print(f"{name:<45} {r[0]:>6.2f}±{r[1]:<4.2f} {std[0]:>6.2f}±{std[1]:<4.2f} {cv[0]:>6.3f}±{cv[1]:<4.3f} {s['w_cos'][0]:>6.3f}±{s['w_cos'][1]:<4.3f} {s['w_fresh'][0]:>6.3f}±{s['w_fresh'][1]:<4.3f} {s['w_lr'][0]:>6.3f}±{s['w_lr'][1]:<4.3f} {s['w_ctr'][0]:>6.3f}±{s['w_ctr'][1]:<4.3f} {s['w_awt'][0]:>6.3f}±{s['w_awt'][1]:<4.3f} {s['w_social'][0]:>6.3f}±{s['w_social'][1]:<4.3f}")

# Save summary
with (root / "summary.json").open("w") as f:
    json.dump({k: {m: {"mean": mv[0], "std": mv[1]} for m, mv in v.items()} for k, v in summary.items()}, f, indent=2)

# Plot reward curves with moving averages
if HAS_MPL:
    for name, runs in results.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        data = runs[sorted(runs.keys())[0]]["data"]
        rounds_idx = list(range(1, len(data["avg_reward"]) + 1))
        
        # Raw reward
        ax1.plot(rounds_idx, data["avg_reward"], alpha=0.6, linewidth=1, label="Raw")
        ma20 = moving_average(data["avg_reward"], 20)
        ma50 = moving_average(data["avg_reward"], 50)
        ax1.plot(rounds_idx, ma20, linewidth=1.5, label="MA(20)")
        ax1.plot(rounds_idx, ma50, linewidth=1.5, label="MA(50)")
        ax1.set_ylabel("Reward")
        ax1.set_title(f"Reward Curve: {name}")
        ax1.legend()
        ax1.grid(alpha=0.25)
        
        # Weight trajectories
        for wk, wlabel, color in [
            ("w0_cos", "cos", "#1f77b4"),
            ("w1_fresh", "fresh", "#ff7f0e"),
            ("w2_lr", "lr", "#2ca02c"),
            ("w3_ctr", "ctr", "#d62728"),
            ("w4_awt", "awt", "#9467bd"),
            ("w5_social", "social", "#8c564b"),
        ]:
            if wk in data:
                ax2.plot(rounds_idx, data[wk], label=wlabel, color=color, linewidth=1.2)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Weight")
        ax2.set_title("Weight Trajectory")
        ax2.legend(ncol=3)
        ax2.grid(alpha=0.25)
        
        fig.tight_layout()
        fig.savefig(root / f"curves_{name}.png", dpi=140)
        plt.close(fig)
    
    print(f"Plots saved to {root}")
