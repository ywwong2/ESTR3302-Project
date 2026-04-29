"""
Generate:
  1. Precision@10 table for all 5 categories
  2. Rank-dynamics figures (like figure1) for dog / plane / clothes / car
     Saved to: exp_bayesian_defended/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────
SRC  = Path(__file__).parent / "final_search_first_v3_twostage_injection_v5_fixed"
OUT  = Path(__file__).parent / "exp_bayesian_defended"
OUT.mkdir(parents=True, exist_ok=True)

qr  = pd.read_csv(SRC / "query_ranks.csv")
its = pd.read_csv(SRC / "item_stats.csv")

CATS  = ['cat','dog','plane','clothes','car']
QCAT  = {c: i for i, c in enumerate(CATS)}
ICAT  = {i: c for i, c in enumerate(CATS)}

# ── quality lookup (use round-1 so it's the initial intrinsic quality) ──
q_snap = (its[its['round']==1][['image_id','quality','category']]
          .drop_duplicates('image_id').set_index('image_id'))

# ════════════════════════════════════════════════════════════════
# 1. Precision@10
# ════════════════════════════════════════════════════════════════
def prec10_series(ql):
    sub = qr[qr['query_label']==ql].copy()
    top10 = sub[sub['rank']<=10]
    hits  = top10[top10['category']==QCAT[ql]].groupby('round').size()
    total = top10.groupby('round').size()
    return (hits / total).fillna(0)

print("\n=== Precision@10 at selected rounds ===")
checkpoints = [1, 50, 100, 200, 300, 500, 750, 1000]
avail = sorted(qr['round'].unique())
snaps = [min(avail, key=lambda x: abs(x-c)) for c in checkpoints]
snaps = sorted(set(snaps))

header = f"{'Query':>8}" + "".join(f"  R{r:>4}" for r in snaps)
print(header)
print("-"*len(header))
for ql in CATS:
    ps = prec10_series(ql)
    row = f"{ql:>8}"
    for r in snaps:
        nr = min(avail, key=lambda x: abs(x-r))
        v  = ps.get(nr, float('nan'))
        row += f"  {v:.3f}"
    print(row)

# ════════════════════════════════════════════════════════════════
# 2. Rank-dynamics figures for dog / plane / clothes / car
# ════════════════════════════════════════════════════════════════
QUALITY_TIERS = {
    'high': ('#1a7a4a', '#5eca8c'),
    'mid':  ('#3d3db4', '#9090d8'),
    'low':  ('#b03030', '#e89090'),
}

def pick_6(cat_idx, ql):
    """Return 6 image_ids: top-2, mid-2, bottom-2 by quality in this category."""
    imgs = q_snap[q_snap['category']==cat_idx].sort_values('quality', ascending=False)
    n = len(imgs)
    if n < 6:
        return None
    ids = imgs.index.tolist()
    # 1st/2nd, middle pair, 9th/10th (or n-2/n-1)
    mid_i = max(0, n//2 - 1)
    bot_i = max(0, n - 2)
    chosen = [ids[0], ids[1], ids[mid_i], ids[mid_i+1], ids[bot_i], ids[bot_i+1]]
    labels_raw = ['high','high','mid','mid','low','low']
    return list(zip(chosen, labels_raw))

def make_figure(ql):
    cat_idx = QCAT[ql]
    pairs   = pick_6(cat_idx, ql)
    if pairs is None:
        print(f"  Not enough images for {ql}, skipping.")
        return

    sub = qr[(qr['query_label']==ql)].copy()
    rounds = np.array(sorted(sub['round'].unique()))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    tier_colors = {'high': QUALITY_TIERS['high'],
                   'mid':  QUALITY_TIERS['mid'],
                   'low':  QUALITY_TIERS['low']}
    tier_count  = {'high':0,'mid':0,'low':0}
    tier_labels = {'high':['1st quality','2nd quality'],
                   'mid': ['5th quality','6th quality'],
                   'low': ['9th quality','10th quality']}

    for img_id, tier in pairs:
        tidx = tier_count[tier]
        color = tier_colors[tier][tidx]
        tier_count[tier] += 1
        q_val = q_snap.loc[img_id, 'quality'] if img_id in q_snap.index else 0.0
        tier_str = tier_labels[tier][tidx]

        rank_by_round = sub[sub['image_id']==img_id].set_index('round')['rank']
        ys_raw = np.array([rank_by_round.get(r, np.nan) for r in rounds], dtype=float)

        # rolling mean (window=20)
        s = pd.Series(ys_raw)
        ys_smooth = s.rolling(20, min_periods=1, center=True).mean().values

        ax.plot(rounds, ys_raw, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(rounds, ys_smooth, color=color, linewidth=2.2,
                label=f"Image {img_id} ({tier}, q={q_val:.3f})")

    ax.invert_yaxis()
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Rank Position (lower = higher rank)", fontsize=12)
    title_cat = ql.capitalize()
    ax.set_title(
        f"Rank Dynamics — {title_cat} Images (1st/2nd, 5th/6th, 9th/10th Quality)",
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(rounds[0], rounds[-1])

    fname = OUT / f"figure_{ql}_rank_dynamics.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

print("\n=== Generating rank-dynamics figures ===")
for ql in ['dog','plane','clothes','car']:
    make_figure(ql)

print("\nDone.")
