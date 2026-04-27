# Ranking Simulation Debug Report

## 1. Root Cause Diagnosis

The social-proof weight collapse is caused by **three interacting problems**:

1. **Feature scale mismatch**: `cos` has mean ~0.003–0.005 with std ~0.047, while `freshness` is ~0.65, `AWT` ~0.12, and `social` grows toward 0.09+. Because the gradient is `grad[j] += coeff * x[j]`, features with larger magnitudes receive much larger gradient signals. `cos` gets gradient contributions ~10–100× smaller than `AWT` and `social`, so its weight is driven to 0 within the first ~50 rounds.

2. **Sharp softmax (tau = 5.0)**: With tau=5, the policy is extremely sharp. Small score differences are amplified, making the ranker hypersensitive to whatever features have the largest scale. Combined with the scale mismatch, this creates a runaway feedback loop where social proof and AWT dominate the ranking, get more exposure, accumulate more reward, and therefore receive even stronger gradients.

3. **Positive feedback loop**: Social proof is defined as `min(1, log(1 + tilde_v + omega_l * tilde_l) / M)`. With `omega_l = 5.0`, likes heavily inflate the social score. Popular items get ranked higher → more views → more likes → higher social score → ranked even higher. Because `tau=5` is so sharp, this creates an almost deterministic "rich get richer" dynamic.

4. **Reward structure favors engagement over relevance**: The reward weights `lambda_v=1, lambda_t=2, lambda_l=5` heavily weight watch time and likes, which are driven by latent `quality`, not by `cos`. The ranker rationally learns that `AWT` and `social` are better proxies for quality than `cos`, causing `cos` to collapse even when its scale is fixed.

**Freshness** collapses to 0 because it is initialized once (`rng.uniform(0.2, 1.0)`) and never updated during the simulation, so it carries no predictive signal for reward.

---

## 2. Code Changes Made

### `backend/simulate_ranking.py`

Four new CLI flags were added to `get_parser()`:

- `--diagnostics` — writes an extra `diagnostics.csv` with per-round feature means/stds, gradient norms, and pre/post projection weights.
- `--normalize-features` — min-max normalizes each of the 6 ranking features to `[0,1]` per query before computing `w^T x`.
- `--drop-social` — removes `social_proof` from the ranking feature vector (reduces to 5 dims).
- `--freeze-social <float>` — fixes `w_social` to the given value and projects the remaining 5 weights onto the simplex of size `1 - frozen`.
- `--freeze-cos <float>` — fixes `w_cos` to the given value and projects the remaining 5 weights onto the simplex of size `1 - frozen`.
- `--lagged-social` — uses the previous round's `social_proof` instead of the current round's.

The `run_simulation` signature was relaxed to accept `**kwargs` overrides so the ablation script can call it programmatically without shell parsing.

The weight projection block (around line 514) was updated to handle `freeze_social` and `freeze_cos`:

```python
if freeze_social and use_social:
    frozen = args.freeze_social
    w_free = project_to_simplex([w_pre[j] / (1.0 - frozen + 1e-12) for j in range(5)])
    w = [w_free[j] * (1.0 - frozen) for j in range(5)] + [frozen]
elif freeze_cos:
    frozen = args.freeze_cos
    w_free = project_to_simplex([w_pre[j] / (1.0 - frozen + 1e-12) for j in range(1, dim)])
    w = [frozen] + [w_free[j - 1] * (1.0 - frozen) for j in range(1, dim)]
else:
    w = project_to_simplex(w_pre)
```

Feature normalization is applied per-user, per-query inside the inner loop before ranking:

```python
if args.normalize_features:
    num_feats = dim
    mins = [min(x_vectors[i][j] for i in range(len(images))) for j in range(num_feats)]
    maxs = [max(x_vectors[i][j] for i in range(len(images))) for j in range(num_feats)]
    for i in range(len(images)):
        for j in range(num_feats):
            den = maxs[j] - mins[j]
            if den > 1e-8:
                x_vectors[i][j] = (x_vectors[i][j] - mins[j]) / den
            else:
                x_vectors[i][j] = 0.0
    scores = [dot(w, x_vectors[i]) for i in range(len(images))]
```

### New scripts

- `backend/run_ablations.py` — orchestrates 21 configurations × 5 seeds, aggregates CSVs, prints a comparison table, and plots reward/weight curves.
- `backend/run_extra_ablations.py` — tests rebalanced reward weights (`lambda_v=2, lambda_t=2, lambda_l=3`) and lower tau + normalization combos.
- `backend/run_freeze_cos_ablations.py` — tests `freeze_cos ∈ {0.10, 0.15, 0.20}` combined with `tau=0.5, beta=1e-3, normalize-features`.
- `backend/plot_ablations.py` — regenerates overlay plots from existing CSVs after `matplotlib` was installed.

---

## 3. Ablation Results Table

Results are averaged over the **last 100 rounds** across 5 random seeds (except extra ablations which used 3 seeds).

| Configuration | Reward (last 100) | w_cos | w_fresh | w_lr | w_ctr | w_awt | w_social |
|-------------|-------------------|-------|---------|------|-------|-------|----------|
| **baseline** | 9.09 ± 0.49 | 0.000 | 0.000 | 0.000 | 0.000 | 0.026 | **0.974** |
| tau = 0.5 | 9.17 ± 0.52 | 0.000 | 0.124 | 0.083 | 0.083 | 0.385 | 0.325 |
| tau = 1.0 | 9.24 ± 0.44 | 0.000 | 0.005 | 0.000 | 0.000 | 0.566 | 0.429 |
| tau = 2.0 | 9.40 ± 0.53 | 0.000 | 0.000 | 0.000 | 0.000 | 0.629 | 0.371 |
| beta = 1e-3 | 9.06 ± 0.56 | 0.000 | 0.000 | 0.000 | 0.000 | 0.029 | 0.971 |
| normalize features only | 9.88 ± 0.70 | 0.000 | 0.000 | **0.996** | 0.004 | 0.000 | 0.000 |
| tau=1.0 + beta=1e-3 + norm | 9.96 ± 0.64 | 0.000 | 0.000 | 0.142 | 0.198 | 0.380 | 0.280 |
| no social | 9.46 ± 0.63 | 0.000 | 0.000 | 0.018 | 0.000 | **0.982** | — |
| freeze social 0.1 | 9.42 ± 0.18 | 0.000 | 0.000 | 0.000 | 0.000 | 0.900 | 0.100 |
| tau=0.5 + beta=1e-3 + norm | **10.08 ± 0.73** | 0.000 | 0.000 | 0.190 | 0.215 | 0.331 | 0.264 |
| **freeze cos 0.10** + above | 9.78 ± 0.52 | **0.100** | 0.000 | 0.166 | 0.197 | 0.294 | 0.243 |
| **freeze cos 0.15** + above | 9.95 ± 0.73 | **0.150** | 0.000 | 0.147 | 0.182 | 0.291 | 0.230 |
| **freeze cos 0.20** + above | **10.11 ± 0.40** | **0.200** | 0.000 | 0.137 | 0.170 | 0.276 | 0.218 |

**Global metrics** (all configs are similar):
- CTR converges to ~0.138–0.142
- LR converges to ~0.080–0.092
- AWT converges to ~0.113–0.122

---

## 4. Key Findings

1. **Lowering `tau` from 5.0 to 0.5 is the single most effective fix** for preventing single-feature dominance. With `tau=0.5`, no feature exceeds ~40% weight, even without normalization.

2. **Feature normalization (`normalize-features`) helps** but is dangerous with `tau=5.0`. When combined with sharp softmax, normalization causes LR to completely dominate (~99.6%). Normalization must be paired with a lower `tau`.

3. **L2 regularization (`beta`) has almost no effect** on weight distribution when `tau=5.0`. The gradients on AWT/social are so large that `beta=1e-3` or even `1e-2` cannot prevent collapse.

4. **Dropping social just shifts dominance to AWT** (~98%). The problem is not specifically social proof; it is any dynamic feedback signal that correlates with reward.

5. **Rebalancing reward weights** (e.g., `lambda_v=2, lambda_t=2, lambda_l=3`) does not restore `cos` weight. The simulation is structured so that watch time and likes are quality-driven, and `cos` is a poor proxy for quality.

6. **`freeze_cos` is the only mechanism that guarantees `cos` retains meaningful weight** in this reward structure. With `freeze_cos=0.15`, `tau=0.5`, `beta=1e-3`, and `normalize-features`, the final weights are well-balanced and the reward is higher than baseline.

---

## 5. Recommended Final Hyperparameters

| Hyperparameter | Baseline | **Recommended** |
|----------------|----------|-----------------|
| `tau` | 5.0 | **0.5** |
| `beta` | 0.0 | **0.001** |
| `normalize-features` | False | **True** |
| `freeze-cos` | — | **0.15** |
| `omega_l` | 5.0 | 5.0 *(unchanged)* |
| `rho` | 0.1 | 0.1 *(unchanged)* |
| `lambda_v` | 1.0 | 1.0 *(unchanged)* |
| `lambda_t` | 2.0 | 2.0 *(unchanged)* |
| `lambda_l` | 5.0 | 5.0 *(unchanged)* |

**Command to run:**

```bash
python backend/simulate_ranking.py \
  --tau 0.5 \
  --beta 0.001 \
  --normalize-features \
  --freeze-cos 0.15 \
  --dataset-cache dataset/cosine_cache_siglip.json \
  --rounds 1000
```

**Why this works:**

- **`tau=0.5`** softens the softmax policy. Instead of the top 2–3 items receiving 90%+ of the policy mass, the mass is spread across ~5–8 items. This dramatically reduces the winner-take-all dynamic that drives popularity collapse.
- **`normalize-features`** removes the 100× scale mismatch between `cos` (~0.003) and `freshness` (~0.65). After normalization, all features are in `[0,1]`, so gradient magnitudes are comparable across dimensions.
- **`beta=1e-3`** adds a small L2 penalty that prevents weights from drifting to the simplex boundary too aggressively. It works synergistically with the lower `tau`.
- **`freeze-cos=0.15`** guarantees that semantic relevance always contributes at least 15% of the ranking score. In this simulation, `cos` is rationally underweighted because watch time and likes are quality-driven, not relevance-driven. Freezing it prevents the ranker from abandoning relevance entirely.

**Expected behavior:**
- Final weights approximately: `cos=0.15`, `lr≈0.15`, `ctr≈0.18`, `awt≈0.29`, `social≈0.23`.
- No single feature dominates.
- Average reward ~10.0 (higher than baseline ~9.1).
- Stable across seeds (std < 0.75).
- Reward curve is smooth and non-oscillatory after ~100 rounds.

---

## 6. Plots

See the generated plots in:
- `backend/data/ablations/reward_curves_all.png` — reward curves across all ablations
- `backend/data/ablations/weights_baseline.png` — baseline collapse
- `backend/data/ablations/weights_tau_1.0_beta_1e-3_norm.png` — best learned-only config
- `backend/data/ablations_freeze_cos/weights_freeze_cos_0.15.png` — recommended config

---

## 7. Optional Further Improvements

If you want to make `cos` *learned* rather than frozen, the simulation's reward structure needs to be adjusted so that relevance directly predicts watch time or likes. Two minimal ways to do this without changing the dataset:

1. **Increase `lambda_v`** relative to `lambda_t` and `lambda_l` (e.g., `lambda_v=5, lambda_t=2, lambda_l=3`). In quick tests, this raised `cos` to ~0.13 by round 50, but it still declined to 0 by round 300 as AWT/social accumulated quality signal.

2. **Add a relevance penalty** to the loss: penalize the ranker when high-ranked items have very low `cos`. This is a form of fairness/regularization and can be implemented by adding `-lambda_rel * cos` to the reward for non-matching items.

Both options change the optimization objective rather than the user simulation, so they are compatible with your constraints.
