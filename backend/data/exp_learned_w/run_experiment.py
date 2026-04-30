"""
Experiment: Two-stage ranking where Stage-2 is driven by the LEARNED weight
vector w (instead of fixed quality_weight_* constants).

Stage 1 : filter top-K1 = 13 by cosine similarity  (unchanged)
Stage 2 : re-rank by  score_i = dot(w, x_i)
          where x_i = [cos_ui, freshness_i, lr_i, ctr_i, awt_i, social_i]
          w  is updated each round by policy gradient and is projected onto
          the unit simplex with a floor on freshness.

Three variants are run back-to-back:
  A) w init = converged values from existing experiment
  B) w init = uniform  (1/6 each)
  C) w init = converged values, higher learning rate (eta * 5)

Output saved to:   backend/data/exp_learned_w/
"""

import csv, json, math, random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
CACHE      = HERE.parent.parent.parent / "dataset" / "cosine_cache_siglip.json"
ITEM_STATS = HERE.parent / "final_search_first_v3_twostage_injection_v5_fixed" / "item_stats.csv"
OUT        = HERE / "k1_10"
OUT.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────
def dot(a, b):
    return sum(x*y for x,y in zip(a,b))

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))

def softmax(logits):
    m = max(logits)
    e = [math.exp(v - m) for v in logits]
    s = sum(e)
    return [v/s for v in e]

def project_simplex(v):
    n = len(v)
    u = sorted(v, reverse=True)
    cssv = 0.0
    rho = 0
    for i, ui in enumerate(u):
        cssv += ui
        if ui - (cssv - 1.0) / (i + 1) > 0:
            rho = i
    theta = (sum(u[:rho+1]) - 1.0) / (rho + 1)
    return [max(vi - theta, 0.0) for vi in v]

# ── load cosine cache ─────────────────────────────────────────────
print("Loading cosine cache …")
with open(CACHE) as f:
    raw = json.load(f)

QUERY_LABELS = ["cat", "dog", "plane", "clothes", "car"]
QCAT = {q: i for i, q in enumerate(QUERY_LABELS)}
NUM_CATS = len(QUERY_LABELS)

# Load quality from item_stats round-1 snapshot
import csv as _csv
quality_map = {}
with open(ITEM_STATS) as f:
    for row in _csv.DictReader(f):
        if int(row["round"]) == 1:
            quality_map[int(row["image_id"])] = float(row["quality"])

# Build image list from cache
images_master = []
for info in raw["images"]:
    cat_label = info.get("category", "")
    if cat_label not in QCAT:
        continue
    img_id  = int(info["image_id"])
    cosines = {ql: float(info["cosine_by_query"].get(ql, 0.0)) for ql in QUERY_LABELS}
    images_master.append({
        "id":      img_id,
        "cat_idx": QCAT[cat_label],
        "quality": quality_map.get(img_id, 0.3),
        "cosines": cosines,
    })

N = len(images_master)
print(f"  {N} images loaded, categories: {[sum(1 for im in images_master if im['cat_idx']==i) for i in range(5)]}")

# ── simulation params ─────────────────────────────────────────────
SIM = dict(
    top_k         = 10,
    two_stage_k   = 10,   # K1=K2=10: Stage 1 perfectly captures all correct images
    batch_size    = 30,
    num_rounds    = 500,
    eta           = 0.01,
    beta          = 0.01,      # L2 regularisation
    tau           = 1.0,
    min_fresh_w   = 0.05,
    rho           = 0.1,       # EWMA smoothing
    c_ctr = 2.0, m_ctr = 0.1,
    c_lr  = 2.0, m_lr  = 0.05,
    gamma_pos     = 0.5,
    # user behaviour (alpha, gamma, delta  dim=5: [bias, match, cos, fresh, social])
    alpha   = [-1.5, 0.3, 2.5, 0.2, 0.1],
    alpha_q = 0.3,
    gamma   = [-0.5, 0.2, 1.0, 0.1, 0.2],
    gamma_q = 4.5,
    kappa   = 12,
    delta   = [-2.6, 0.05, 0.25, 0.03, 0.05],
    delta_t = 3.0,
    delta_q = 4.0,
    lambda_v = 1.2, lambda_t = 2.2, lambda_l = 1.2,
    freshness_decay = 5.0,
    recovery_boost_interval = 50,
    recovery_boost_strength = 0.5,
)

W_CONVERGED = [0.2556, 0.050, 0.1954, 0.0581, 0.2264, 0.2148]
W_UNIFORM   = [1/6]*6

VARIANTS = [
    ("A_converged_w",    W_CONVERGED[:], SIM["eta"]),
    ("B_uniform_w",      W_UNIFORM[:],   SIM["eta"]),
    ("C_converged_highLR", W_CONVERGED[:], SIM["eta"] * 5),
]

# ── per-run simulation ────────────────────────────────────────────
def run(variant_name, w_init, eta, seed=42):
    rng  = random.Random(seed)
    p    = SIM
    w    = w_init[:]

    # image state
    imgs = []
    for im in images_master:
        imgs.append({**im,
            "tilde_e": 0.0, "tilde_v": 0.0, "tilde_l": 0.0,
            "ctr": p["m_ctr"], "lr": p["m_lr"], "awt": 0.0,
            "social_proof": 0.0,
            "freshness": rng.uniform(0.2, 1.0),
            "arrival_time": 0,
            "last_shown": 0,
        })

    rows_round  = []   # round-level log
    rows_prec   = []   # precision@10 per round per query

    for t in range(1, p["num_rounds"] + 1):
        # freshness decay
        for im in imgs:
            age = t - im["arrival_time"]
            im["freshness"] = 2.0 ** (-age / p["freshness_decay"])

        # recovery boost every 50 rounds
        if p["recovery_boost_interval"] > 0 and t % p["recovery_boost_interval"] == 0:
            for im in imgs:
                if t - im["last_shown"] >= p["recovery_boost_interval"]:
                    im["freshness"] = min(1.0, im["freshness"] + p["recovery_boost_strength"])

        # ── per-query precision logging ──────────────────────────
        for qi, ql in enumerate(QUERY_LABELS):
            cos_vals = [im["cosines"][ql] for im in imgs]
            stage1   = sorted(range(N), key=lambda i: cos_vals[i], reverse=True)[:p["two_stage_k"]]
            x_s1     = [[cos_vals[idx],
                         imgs[idx]["freshness"],
                         imgs[idx]["lr"],
                         imgs[idx]["ctr"],
                         imgs[idx]["awt"],
                         imgs[idx]["social_proof"]] for idx in stage1]
            s2_scores = [dot(w, x) for x in x_s1]
            s2_ranked = sorted(range(len(stage1)), key=lambda i: s2_scores[i], reverse=True)
            displayed_ids = [stage1[i] for i in s2_ranked[:p["top_k"]]]
            hits = sum(1 for idx in displayed_ids if imgs[idx]["cat_idx"] == qi)
            rows_prec.append({"round": t, "query": ql, "prec10": hits / p["top_k"]})

        # ── simulate batch_size users ────────────────────────────
        grad = [0.0] * 6
        total_reward = 0.0

        # aggregate engagement
        agg = {im["id"]: {"e":0.0,"v":0.0,"l":0.0,"ws":0.0,"wc":0.0} for im in imgs}

        for _ in range(p["batch_size"]):
            q_cat = rng.randrange(NUM_CATS)
            ql    = QUERY_LABELS[q_cat]

            cos_vals = [im["cosines"][ql] for im in imgs]

            # Stage 1
            stage1 = sorted(range(N), key=lambda i: cos_vals[i], reverse=True)[:p["two_stage_k"]]

            # Stage 2: rank by dot(w, x)
            x_vecs   = []
            s2scores = []
            for idx in stage1:
                im = imgs[idx]
                x  = [cos_vals[idx], im["freshness"], im["lr"], im["ctr"], im["awt"], im["social_proof"]]
                x_vecs.append(x)
                s2scores.append(dot(w, x))

            s2_ranked  = sorted(range(len(stage1)), key=lambda i: s2scores[i], reverse=True)
            disp_local = s2_ranked[:p["top_k"]]
            displayed  = [stage1[i] for i in disp_local]

            # policy distribution over stage1
            pol_scores = [0.0] * N
            for li, gi in enumerate(stage1):
                pol_scores[gi] = s2scores[li]
            logits = [p["tau"] * s for s in pol_scores]
            a      = softmax(logits)

            rewards = [0.0] * N

            for rank, idx in enumerate(displayed):
                im = imgs[idx]
                im["last_shown"] = t

                match = 1.0 if im["cat_idx"] == q_cat else 0.0
                h     = [1.0, match, cos_vals[idx], im["freshness"], im["social_proof"]]
                q_val = im["quality"]

                eta_r    = (1.0 + rank) ** (-p["gamma_pos"])
                examined = 1.0 if rng.random() < eta_r else 0.0
                if not examined:
                    continue

                view = 1.0 if rng.random() < sigmoid(dot(p["alpha"], h) + p["alpha_q"]*q_val) else 0.0
                watch = like = 0.0
                if view:
                    mu  = sigmoid(dot(p["gamma"], h) + p["gamma_q"]*q_val)
                    ab  = max(p["kappa"]*mu, 1e-6)
                    bb  = max(p["kappa"]*(1-mu), 1e-6)
                    watch = rng.betavariate(ab, bb)
                    like  = 1.0 if rng.random() < sigmoid(dot(p["delta"], h) + p["delta_t"]*watch + p["delta_q"]*q_val) else 0.0

                r_ui = p["lambda_v"]*view + p["lambda_t"]*watch + p["lambda_l"]*like
                rewards[idx] = r_ui
                total_reward += r_ui

                ag = agg[im["id"]]
                ag["e"] += 1.0; ag["v"] += view; ag["l"] += like
                if view: ag["ws"] += watch; ag["wc"] += 1.0

            exp_r = sum(a[i]*rewards[i] for i in range(N))
            for idx in displayed:
                x = [cos_vals[idx], imgs[idx]["freshness"],
                     imgs[idx]["lr"], imgs[idx]["ctr"],
                     imgs[idx]["awt"], imgs[idx]["social_proof"]]
                coeff = -p["tau"] * a[idx] * (rewards[idx] - exp_r)
                for j in range(6):
                    grad[j] += coeff * x[j]

        # gradient step
        bs   = max(p["batch_size"], 1)
        grad = [g/bs + 2.0*p["beta"]*wj for g,wj in zip(grad,w)]
        w    = project_simplex([wj - eta*gj for wj,gj in zip(w,grad)])
        w    = [max(wj, 0.0) for wj in w]
        w[1] = max(w[1], p["min_fresh_w"])   # freshness floor

        # EWMA update
        rho = p["rho"]
        for im in imgs:
            ag = agg[im["id"]]
            te,tv,tl = ag["e"],ag["v"],ag["l"]
            im["tilde_e"] = (1-rho)*im["tilde_e"] + rho*te
            im["tilde_v"] = (1-rho)*im["tilde_v"] + rho*tv
            im["tilde_l"] = (1-rho)*im["tilde_l"] + rho*tl
            im["ctr"] = (im["tilde_v"] + p["c_ctr"]*p["m_ctr"]) / (im["tilde_e"] + p["c_ctr"])
            im["lr"]  = (im["tilde_l"] + p["c_lr"]*p["m_lr"])   / (im["tilde_v"] + p["c_lr"])
            mw = ag["ws"]/ag["wc"] if ag["wc"] > 0 else 0.0
            im["awt"] = (1-rho)*im["awt"] + rho*mw

        rows_round.append({"round": t, "reward": total_reward/bs, **{f"w{j}": w[j] for j in range(6)}})

        if t % 100 == 0:
            mean_prec = sum(r["prec10"] for r in rows_prec if r["round"]==t) / NUM_CATS
            print(f"  [{variant_name}] Round {t:4d}  reward={total_reward/bs:.3f}  mean P@10={mean_prec:.3f}  w={[round(x,3) for x in w]}")

    return rows_round, rows_prec


# ── run all variants ──────────────────────────────────────────────
all_prec = {}
all_w    = {}

for vname, w_init, eta in VARIANTS:
    print(f"\n{'='*60}\nVariant: {vname}  eta={eta}\n{'='*60}")
    rnd, prec = run(vname, w_init, eta)
    all_prec[vname] = prec
    all_w[vname]    = rnd

    # save CSVs
    with open(OUT / f"{vname}_rounds.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rnd[0].keys()); wr.writeheader(); wr.writerows(rnd)
    with open(OUT / f"{vname}_prec.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=prec[0].keys()); wr.writeheader(); wr.writerows(prec)

# ── plot 1: mean Precision@10 over time for all 3 variants ───────
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
COLORS = {"cat":"#1a7a4a","dog":"#3d3db4","plane":"#b03030","clothes":"#b07000","car":"#555"}

for ax, (vname, _, _) in zip(axes, VARIANTS):
    prec = all_prec[vname]
    rounds = sorted(set(r["round"] for r in prec))
    for ql in QUERY_LABELS:
        ys = [r["prec10"] for r in prec if r["query"]==ql]
        import pandas as pd
        ys_s = pd.Series(ys).rolling(20, min_periods=1).mean().values
        ax.plot(rounds, ys_s, label=ql, color=COLORS[ql], linewidth=1.6)
    mean_ys = [sum(r["prec10"] for r in prec if r["round"]==rr)/NUM_CATS for rr in rounds]
    mean_s  = pd.Series(mean_ys).rolling(20, min_periods=1).mean().values
    ax.plot(rounds, mean_s, "k--", linewidth=2, label="mean")
    ax.set_title(vname.replace("_"," "), fontsize=10)
    ax.set_xlabel("Round"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    if ax == axes[0]: ax.set_ylabel("Precision@10")
    ax.legend(fontsize=7)

fig.suptitle("Learned-w Two-Stage (K1=K2=10): Precision@10 per Query", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_prec10.png", dpi=150)
plt.close(fig)
print(f"\nSaved fig_prec10.png")

# ── plot 2: weight trajectories for all 3 variants ───────────────
WLABELS = ["cos","fresh","lr","ctr","awt","social"]
WCOLS   = ["#2563eb","#f59e0b","#16a34a","#dc2626","#7c3aed","#0891b2"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, (vname, _, _) in zip(axes, VARIANTS):
    rnd = all_w[vname]
    rounds = [r["round"] for r in rnd]
    for j, (lbl, col) in enumerate(zip(WLABELS, WCOLS)):
        ys = [r[f"w{j}"] for r in rnd]
        ax.plot(rounds, ys, label=lbl, color=col, linewidth=1.8)
    ax.set_title(vname.replace("_"," "), fontsize=10)
    ax.set_xlabel("Round"); ax.set_ylim(0, 0.7); ax.grid(alpha=0.3)
    if ax == axes[0]: ax.set_ylabel("Weight value")
    ax.legend(fontsize=7)

fig.suptitle("Learned-w Two-Stage (K1=K2=10): Weight Trajectories", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_weights.png", dpi=150)
plt.close(fig)
print(f"Saved fig_weights.png")
print("\nAll done.")
