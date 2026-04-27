from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

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
    if result.returncode != 0:
        print(f"ERROR running seed {seed}: {result.stderr}")
        return {"error": True, "stderr": result.stderr}
    return {"error": False}


def parse_csv(csv_path: Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data[k].append(float(v))
    return dict(data)


def last_n_mean_std(values: list[float], n: int = 100) -> tuple[float, float]:
    arr = values[-n:]
    if not arr:
        return 0.0, 0.0
    m = sum(arr) / len(arr)
    s = math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr)) if len(arr) > 1 else 0.0
    return m, s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out-root", type=str, default="backend/data/ablations")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = list(range(42, 42 + args.seeds))

    configs = [
        ("baseline", []),
        ("no_social", ["--drop-social"]),
        ("tau_0.5", ["--tau", "0.5"]),
        ("tau_1.0", ["--tau", "1.0"]),
        ("tau_2.0", ["--tau", "2.0"]),
        ("beta_1e-4", ["--beta", "0.0001"]),
        ("beta_1e-3", ["--beta", "0.001"]),
        ("beta_1e-2", ["--beta", "0.01"]),
        ("omega_1.0", ["--omega-l", "1.0"]),
        ("omega_2.0", ["--omega-l", "2.0"]),
        ("omega_3.0", ["--omega-l", "3.0"]),
        ("lambda_l_2.0", ["--lambda-l", "2.0"]),
        ("lambda_l_3.0", ["--lambda-l", "3.0"]),
        ("normalize_features", ["--normalize-features"]),
        ("tau_1.0_beta_1e-3", ["--tau", "1.0", "--beta", "0.001"]),
        ("tau_1.0_beta_1e-3_norm", ["--tau", "1.0", "--beta", "0.001", "--normalize-features"]),
        ("tau_1.0_omega_2.0", ["--tau", "1.0", "--omega-l", "2.0"]),
        ("tau_1.0_omega_2.0_beta_1e-3", ["--tau", "1.0", "--omega-l", "2.0", "--beta", "0.001"]),
        ("freeze_social_0.1", ["--freeze-social", "0.1"]),
        ("freeze_social_0.2", ["--freeze-social", "0.2"]),
        ("lagged_social", ["--lagged-social"]),
    ]

    results: dict[str, list[dict]] = defaultdict(list)

    for name, extra in configs:
        print(f"\n=== Config: {name} ===")
        for seed in seeds:
            run_dir = out_root / f"{name}_seed{seed}"
            res = run_single(seed, str(run_dir), extra, rounds=args.rounds)
            if res.get("error"):
                continue
            csv_path = run_dir / "simulation_rounds.csv"
            if not csv_path.exists():
                print(f"Missing CSV for {name} seed {seed}")
                continue
            data = parse_csv(csv_path)
            results[name].append(data)
            print(f"  seed {seed} done")

    # Aggregate
    summary: dict[str, dict] = {}
    for name, runs in results.items():
        if not runs:
            continue
        metrics = {
            "final_reward": [],
            "last100_reward_mean": [],
            "last100_reward_std": [],
            "final_w_cos": [],
            "final_w_fresh": [],
            "final_w_lr": [],
            "final_w_ctr": [],
            "final_w_awt": [],
            "final_w_social": [],
            "last100_w_cos_mean": [],
            "last100_w_fresh_mean": [],
            "last100_w_lr_mean": [],
            "last100_w_ctr_mean": [],
            "last100_w_awt_mean": [],
            "last100_w_social_mean": [],
            "last100_w_cos_std": [],
            "last100_w_fresh_std": [],
            "last100_w_lr_std": [],
            "last100_w_ctr_std": [],
            "last100_w_awt_std": [],
            "last100_w_social_std": [],
            "final_ctr": [],
            "final_lr": [],
            "final_awt": [],
        }
        for data in runs:
            def get(k):
                return data.get(k, [])
            metrics["final_reward"].append(get("avg_reward")[-1])
            m, s = last_n_mean_std(get("avg_reward"), 100)
            metrics["last100_reward_mean"].append(m)
            metrics["last100_reward_std"].append(s)
            metrics["final_w_cos"].append(get("w0_cos")[-1])
            metrics["final_w_fresh"].append(get("w1_fresh")[-1])
            metrics["final_w_lr"].append(get("w2_lr")[-1])
            metrics["final_w_ctr"].append(get("w3_ctr")[-1])
            metrics["final_w_awt"].append(get("w4_awt")[-1])
            metrics["final_w_social"].append(get("w5_social")[-1] if get("w5_social") else 0.0)
            for wk, wname in [("w0_cos", "cos"), ("w1_fresh", "fresh"), ("w2_lr", "lr"), ("w3_ctr", "ctr"), ("w4_awt", "awt"), ("w5_social", "social")]:
                m, s = last_n_mean_std(get(wk), 100)
                metrics[f"last100_w_{wname}_mean"].append(m)
                metrics[f"last100_w_{wname}_std"].append(s)
            metrics["final_ctr"].append(get("global_ctr")[-1])
            metrics["final_lr"].append(get("global_lr")[-1])
            metrics["final_awt"].append(get("global_awt")[-1])

        def mean_std(arr):
            if not arr:
                return 0.0, 0.0
            m = sum(arr) / len(arr)
            s = math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr)) if len(arr) > 1 else 0.0
            return m, s

        summary[name] = {k: mean_std(v) for k, v in metrics.items()}

    # Write summary JSON
    summary_path = out_root / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({k: {m: {"mean": mv[0], "std": mv[1]} for m, mv in v.items()} for k, v in summary.items()}, f, indent=2)
    print(f"\nSummary JSON: {summary_path}")

    # Print table
    print("\n" + "=" * 140)
    header = f"{'Config':<30} {'Rwd(last100)':>14} {'w_cos':>8} {'w_fresh':>8} {'w_lr':>8} {'w_ctr':>8} {'w_awt':>8} {'w_soc':>8} {'CTR':>7} {'LR':>7} {'AWT':>7}"
    print(header)
    print("-" * 140)
    for name in sorted(summary.keys()):
        s = summary[name]
        rwd = s["last100_reward_mean"]
        print(f"{name:<30} {rwd[0]:>6.2f}±{rwd[1]:<5.2f} {s['last100_w_cos_mean'][0]:>6.3f}±{s['last100_w_cos_std'][0]:<4.3f} {s['last100_w_fresh_mean'][0]:>6.3f}±{s['last100_w_fresh_std'][0]:<4.3f} {s['last100_w_lr_mean'][0]:>6.3f}±{s['last100_w_lr_std'][0]:<4.3f} {s['last100_w_ctr_mean'][0]:>6.3f}±{s['last100_w_ctr_std'][0]:<4.3f} {s['last100_w_awt_mean'][0]:>6.3f}±{s['last100_w_awt_std'][0]:<4.3f} {s['last100_w_social_mean'][0]:>6.3f}±{s['last100_w_social_std'][0]:<4.3f} {s['final_ctr'][0]:>6.3f} {s['final_lr'][0]:>6.3f} {s['final_awt'][0]:>6.3f}")

    if HAS_MPL:
        # Plot reward curves
        fig, ax = plt.subplots(figsize=(12, 7))
        for name, runs in results.items():
            if not runs:
                continue
            arr = []
            for data in runs:
                arr.append(data["avg_reward"])
            min_len = min(len(x) for x in arr)
            arr = [x[:min_len] for x in arr]
            mean_r = [sum(x[t] for x in arr) / len(arr) for t in range(min_len)]
            std_r = [math.sqrt(sum((x[t] - mean_r[t]) ** 2 for x in arr) / len(arr)) for t in range(min_len)]
            xs = list(range(1, min_len + 1))
            ax.plot(xs, mean_r, label=name, linewidth=1.2)
            ax.fill_between(xs, [m - s for m, s in zip(mean_r, std_r)], [m + s for m, s in zip(mean_r, std_r)], alpha=0.15)
        ax.set_xlabel("Round")
        ax.set_ylabel("Avg Reward")
        ax.set_title("Reward Curves (mean ± std over seeds)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_root / "reward_curves_all.png", dpi=180)
        plt.close(fig)
        print(f"Plot: {out_root / 'reward_curves_all.png'}")

        # Plot weight trajectories for selected configs
        selected = ["baseline", "tau_1.0", "tau_1.0_beta_1e-3", "tau_1.0_beta_1e-3_norm", "no_social", "normalize_features"]
        for name in selected:
            if name not in results or not results[name]:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            data = results[name][0]
            rounds = list(range(1, len(data["avg_reward"]) + 1))
            for wk, wlabel in [("w0_cos", "cos"), ("w1_fresh", "fresh"), ("w2_lr", "lr"), ("w3_ctr", "ctr"), ("w4_awt", "awt"), ("w5_social", "social")]:
                if wk in data:
                    ax.plot(rounds, data[wk], label=wlabel, linewidth=1.2)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Round")
            ax.set_ylabel("Weight")
            ax.set_title(f"Weight Trajectory: {name}")
            ax.legend(ncol=3)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_root / f"weights_{name}.png", dpi=140)
            plt.close(fig)
        print(f"Weight plots saved to {out_root}")


if __name__ == "__main__":
    main()
