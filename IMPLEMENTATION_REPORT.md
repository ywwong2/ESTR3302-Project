# Image Ranking Simulation — Implementation Report

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Dataset & Cosine Cache](#3-dataset--cosine-cache)
4. [Image State & Feature Representation](#4-image-state--feature-representation)
5. [User Behaviour Oracle](#5-user-behaviour-oracle)
6. [Ranking Algorithm](#6-ranking-algorithm)
   - 6.1 [Single-Stage Scoring (initial approach)](#61-single-stage-scoring-initial-approach)
   - 6.2 [Two-Stage Ranking (final design)](#62-two-stage-ranking-final-design)
7. [Policy Gradient Weight Learning](#7-policy-gradient-weight-learning)
8. [Engagement Signal Smoothing](#8-engagement-signal-smoothing)
9. [Freshness Mechanism](#9-freshness-mechanism)
10. [Exploration Injection](#10-exploration-injection)
11. [Non-Click Penalty](#11-non-click-penalty)
12. [Aggressive Forgetting (rho)](#12-aggressive-forgetting-rho)
13. [Image Injection Experiment](#13-image-injection-experiment)
    - 13.1 [Injection Design](#131-injection-design)
    - 13.2 [Cold-Start Fix: Seeded Engagement](#132-cold-start-fix-seeded-engagement)
    - 13.3 [Recovery Boost for Old Images](#133-recovery-boost-for-old-images)
14. [Debugging Journey & Root Cause Fixes](#14-debugging-journey--root-cause-fixes)
15. [Output Files & Plots](#15-output-files--plots)
16. [How to Run](#16-how-to-run)
17. [Final Configuration Parameters](#17-final-configuration-parameters)

---

## 1. Project Overview

This project implements a simulation of an **image search ranking system** that learns to rank images by observing simulated user behaviour signals (click-through rate, like rate, average watch time). The simulation models a real-world recommendation feedback loop: the ranking policy determines which images users see, user interactions update the quality signals of those images, and the updated signals in turn reshape future rankings.

The key research question is: **does the system learn to surface high-quality, semantically relevant images over time, and how does it respond when new images are injected mid-simulation?**

---

## 2. System Architecture

```
dataset/cosine_cache_siglip.json   ← pre-computed SigLIP cosine similarities
        │
        ▼
backend/simulate_ranking.py        ← main simulation (all ranking + learning logic)
        │
        ├─ setup_images_from_cache()  ← loads 50 images, 5 categories
        ├─ setup_users()              ← 1000 users with random-effect offsets
        ├─ run_simulation()           ← main loop (1000 rounds)
        │      ├─ freshness decay
        │      ├─ image injection (round 501)
        │      ├─ recovery boost
        │      ├─ two-stage ranking
        │      ├─ user behaviour simulation
        │      ├─ policy gradient update
        │      └─ EWMA engagement smoothing
        └─ outputs → data/final_search_first_v3_twostage_injection_v5_fixed/
                       ├─ simulation_rounds.csv
                       ├─ item_stats.csv
                       ├─ query_ranks.csv
                       ├─ reward_curve.png
                       └─ weight_trajectory.png

backend/plot_rank_dynamics_v3.py   ← post-simulation plotting
        └─ figure1_diff_quality_6lines.png
        └─ figure2_similar_quality_planes_6lines.png
        └─ figure3_injected_images.png
        └─ figure4_quality_distribution.png
```

---

## 3. Dataset & Cosine Cache

**File:** `dataset/cosine_cache_siglip.json`

Rather than using random synthetic embeddings, the simulation loads pre-computed **SigLIP** cosine similarities between 50 real images and 5 category query prototypes (`cat`, `dog`, `plane`, `clothes`, `car`). Each image has 10 images per category (50 total).

**Why:** SigLIP is a vision-language model that embeds images into a semantically meaningful space. Using real cosine values means the simulation faithfully reflects how a real retrieval system would score images — the cosine values are not artificially equal or orderly. This surfaces realistic edge cases (e.g., a dog image might have a moderate cosine for the `cat` query due to visual similarity).

Each image entry in the cache contains:
```json
{
  "category": "cat",
  "cosine_by_query": {
    "cat":    0.0832,
    "dog":    0.0612,
    "plane":  0.0211,
    "clothes":0.0198,
    "car":    0.0243
  }
}
```

---

## 4. Image State & Feature Representation

**Class:** `ImageState` (dataclass)

Each image carries the following state fields updated each round:

| Field | Description |
|---|---|
| `image_id` | Integer index |
| `category` | Integer category index (0–4) |
| `quality` | Oracle quality score (Beta(2,5) distributed, hidden from ranker) |
| `freshness` | Exponential decay score based on image age |
| `ctr` | Bayesian-smoothed click-through rate |
| `lr` | Bayesian-smoothed like rate |
| `awt` | EWMA average watch time |
| `social_proof` | Normalised engagement popularity signal |
| `tilde_e/v/l` | EWMA accumulators for examinations, views, likes |
| `arrival_time` | Round at which image was added (0 for originals) |
| `last_shown_round` | Most recent round the image was displayed (for recovery boost) |

The **ranking feature vector** fed to the learned policy is:
```
x = [cosine, freshness, lr, ctr, awt, social_proof]
```

Feature vector dimension is 6. The learned weight vector `w` lives on the unit simplex (all weights ≥ 0, sum to 1).

---

## 5. User Behaviour Oracle

The oracle simulates probabilistic user actions for each image displayed at rank `r`:

### 5.1 Examination (position bias)
```
P(examined | rank r) = (1 + r)^(-γ_pos)     γ_pos = 0.90
```
Users examine images near the top with higher probability. This models real-world position bias where lower-ranked items are seen less.

### 5.2 View / Click
```
P(view | examined) = σ(α·h + α_q·quality + ξ_view)
```
Where `h = [1, match, cosine, freshness, social_proof]` is the context vector, `α` are learned oracle coefficients, and `ξ_view ~ N(0, 0.1)` is a user-specific random effect.

**Parameters used:**
- `α = [-1.40, 1.40, 3.40, 0.05, 0.10]`
- `α_q = 0.30`

The `match` term (1 if category matches query, else 0) captures semantic relevance. The high coefficient on cosine (3.40) reflects that visually similar images are more likely to be clicked.

### 5.3 Watch Time
```
μ = σ(γ·h + γ_q·quality + ξ_watch)
watch ~ Beta(κ·μ, κ·(1−μ))     κ = 12
```
Watch time is sampled from a Beta distribution parameterised by a quality-and-context-dependent mean. Higher quality and matching images generate longer watch times.

**Parameters:** `γ = [-0.90, 0.25, 0.90, 0.08, 0.05]`, `γ_q = 4.50`

### 5.4 Like
```
P(like | view) = σ(δ·h + δ_t·watch + δ_q·quality + ξ_like)
```
Likes depend on watch time (users only like what they watched) and quality.

**Parameters:** `δ = [-2.60, 0.05, 0.25, 0.03, 0.05]`, `δ_t = 3.00`, `δ_q = 4.00`

### 5.5 Reward Signal
```
r_ui = λ_v · view + λ_t · watch + λ_l · like
     = 1.2·view + 2.2·watch + 1.2·like
```
The reward combines all three engagement signals. Watch time is weighted highest to encourage the system to surface videos users genuinely engage with, not just click.

---

## 6. Ranking Algorithm

### 6.1 Single-Stage Scoring (initial approach)

The initial design used a single learned weight vector `w` to score every image and display the top-10:
```
score(image) = w · x = w_cos·cos + w_fresh·fresh + w_lr·lr + w_ctr·ctr + w_awt·awt + w_social·social
```

**Problem discovered:** The cosine weight was dominant (w_cos ≈ 0.80) to ensure semantic relevance, which left very little room for quality-driven features to influence ranking. High-cosine but low-quality images would consistently outrank low-cosine but high-quality images. The feedback loop was too weak: quality only influenced ranking indirectly through slowly-building engagement metrics.

### 6.2 Two-Stage Ranking (final design)

**Why added:** To decouple semantic filtering from quality-based re-ranking, enabling both concerns to operate at full strength simultaneously.

**Stage 1 — Cosine Filter:**
```
stage1_top_k = top-13 images by cosine(image, query)
```
The top-13 (10 original category images + 3 injected images after round 501) are selected purely on semantic similarity. This guarantees the displayed images are at least semantically relevant.

**Stage 2 — Quality Re-ranking:**
```
quality_score = w_cos·cos + w_match·match + w_fresh·fresh
              + w_ctr·ctr + w_lr·lr + w_awt·awt
```
The top-13 candidates are re-ranked by `quality_score`. Each weight is a fixed hyperparameter (not learned — they represent domain knowledge about what matters for quality).

**Stage 2 Weights used:**
| Signal | Weight | Rationale |
|---|---|---|
| cosine | 1.0 | Semantic relevance still matters in final rank |
| match | 2.0 | Strong bonus for exact category match |
| freshness | 0.5 | Reward recency; penalises old unseen images |
| CTR | 0.5 | Click rate as engagement proxy |
| LR | 5.0 | Like rate is a strong quality signal (rare, deliberate action) |
| AWT | 2.0 | Watch time reflects genuine engagement |

**Result:** The two-stage approach improved semantic accuracy from ~50% to ~60% correct-category images in top-10, and the highest-quality cat image (#3, q=0.72) now consistently ranks #1.

---

## 7. Policy Gradient Weight Learning

The learned weight vector `w` governs the Stage 1 scoring. The system uses **REINFORCE (policy gradient)** to optimise `w` to maximise expected reward.

### Softmax Policy
```
a_i = softmax(τ · score_i)_i     τ = 0.40 (temperature)
```
The probability of displaying image `i` follows a softmax over scores. Temperature `τ` controls exploration vs. exploitation — a lower value makes the policy more stochastic.

### Gradient Update
```
∇L ≈ -τ · Σ_i  a_i · (r_i − E[r]) · x_i  +  2β·w
```
The gradient is the REINFORCE baseline-subtracted policy gradient plus L2 regularisation.

### Simplex Projection
After each gradient step, `w` is projected onto the unit probability simplex `{w | wᵢ ≥ 0, Σwᵢ = 1}` using the O(n log n) algorithm. This ensures weights remain non-negative and interpretable as relative importance scores.

**Why projection:** Without it, weights can drift negative or grow unbounded, losing the interpretability of relative feature importance.

### Minimum Freshness Weight
```
w[1] = max(w[1], 0.05)   # freshness weight floor
```
**Why added:** The policy gradient tended to drive the freshness weight to near-zero because freshness does not directly generate reward (users do not "reward" recency explicitly). Without this floor, freshness would be eliminated from the learned policy, making it impossible for new or recently-boosted images to compete. A minimum of 0.05 ensures the system always retains some sensitivity to recency.

**Key parameters:**
- Learning rate η = 0.004
- L2 regularisation β = 0.005
- Batch size = 64 users per round
- Temperature τ = 0.40

---

## 8. Engagement Signal Smoothing

Raw per-round engagement counts are too noisy for stable ranking. Two smoothing methods are combined:

### EWMA (Exponential Weighted Moving Average)
```
ẽ_t = (1 − ρ)·ẽ_{t−1} + ρ·e_t
ṽ_t = (1 − ρ)·ṽ_{t−1} + ρ·v_t
l̃_t = (1 − ρ)·l̃_{t−1} + ρ·l_t
```
`ρ = 0.20` (aggressive forgetting — see Section 12).

### Bayesian Smoothing for CTR and LR
```
CTR = (ṽ + c_ctr · m_ctr) / (ẽ + c_ctr)
LR  = (l̃ + c_lr  · m_lr)  / (ṽ + c_lr)
```
With priors `c_ctr = 10, m_ctr = 0.10` and `c_lr = 10, m_lr = 0.05`. For an image with few observations, the estimate is pulled toward the prior mean. As observations accumulate, the estimate converges to the true empirical rate.

**Why Bayesian smoothing:** A newly injected image with zero observations would have CTR = 0/0 = undefined. Cold-start images need a reasonable prior to be comparable to well-observed images. The Bayesian estimate also prevents high-CTR flukes from a single lucky impression from dominating.

### Social Proof
```
social_proof = min(1.0, log(1 + ṽ + ω_l · l̃) / m_social)
```
`ω_l = 1.5` (likes weighted more than views), `m_social = log(1 + 500)`.

---

## 9. Freshness Mechanism

**Why added:** In a real search/recommendation system, new content needs a boost to compete against well-established images that have accumulated engagement data over hundreds of rounds. Without freshness, the system would settle into a fixed ranking and never surface new images.

### Freshness Decay Formula
```
freshness(t) = 2^(−age / scale)     age = t − t_arrival,   scale = 5 rounds
```
At age 0 (injection): freshness = 1.0. At age 5: freshness = 0.5. At age 10: freshness = 0.25. The fast decay ensures the freshness advantage is temporary — new images must build genuine engagement to maintain their rank.

**Original images** (arrival_time = 0) have age = t from round 1, so their freshness is essentially 0 by the time injection happens at round 501. This is intentional: old images rely on their accumulated CTR/LR/AWT rather than freshness.

### Freshness in Stage 2 Quality Score
**Why this matters:** Early in the debugging process, freshness was tracked but **not included in the Stage 2 quality_score formula**. This meant the freshness decay, minimum freshness weight, and recovery boost all had zero effect on actual ranking — new images were ranked purely on their (zero) engagement signals. Adding `w_fresh · freshness` to Stage 2 was a critical fix.

---

## 10. Exploration Injection

**Why added:** The ranking system suffers from a **filter bubble**: only images already in the top-10 get shown to users, accumulate engagement, and thus stay in the top-10. Images ranked 11+ never get exposure and therefore never improve. This is the classic explore-exploit dilemma.

**Mechanism:** With 5% probability each round, the 10th-ranked (bottom) display slot is replaced with a random image from the Stage 1 pool that is not already in the displayed top-10:
```python
if rng.random() < 0.05:
    random_idx = rng.choice(non_displayed_pool)
    displayed[-1] = random_idx
```

This gives borderline-ranked images occasional exposure without significantly degrading the overall display quality (the replaced slot is the lowest-ranked anyway).

---

## 11. Non-Click Penalty

**Why added:** Without this, the system only received positive feedback — images that got clicked were rewarded, but images that were shown and *not* clicked were treated as if they were never shown at all (no negative signal). This caused the CTR estimate to be overly optimistic.

**Mechanism:** When an image is examined (`examined = 1`) but not clicked (`view = 0`), an immediate CTR update is applied:
```python
if view == 0.0 and examined:
    current_ctr = (tilde_v + c_ctr * m_ctr) / (tilde_e + c_ctr)
    img.ctr = current_ctr
```

This causes images that receive attention but fail to convert to have their CTR downgraded immediately within the round, before the end-of-round EWMA update. The effect is that consistently unclicked images fall in rank faster.

---

## 12. Aggressive Forgetting (rho)

**Why changed:** The original `ρ = 0.10` (slow EWMA decay) caused the system to be sluggish in adapting to new signals. When new images were injected, their engagement data took many rounds to meaningfully influence their EWMA accumulators, since the accumulated history of old images heavily dominated the smoothed values.

**Fix:** `ρ = 0.20` was chosen. This means each round's new signal contributes 20% to the smoothed estimate, allowing the system to adapt roughly twice as fast. For a 1000-round simulation with image injection at round 501, this is necessary so the second half of the simulation (500 rounds) can meaningfully reflect the new content landscape.

---

## 13. Image Injection Experiment

The second half of the project focuses on a **three-phase experiment**: run the system for 500 rounds to establish a stable baseline, inject three new images at round 501, then observe how the ranking system adapts over the next 500 rounds.

### 13.1 Injection Design

Three images are injected into the `cat` category at round 501:

| Image ID | Quality | Description |
|---|---|---|
| 50 | 0.40 | High quality — should rise in rank as engagement builds |
| 51 | 0.25 | Mid quality — should stabilise in the middle ranks |
| 52 | 0.01 | Low quality — should be demoted as poor engagement is observed |

**Cosine similarity assignment (fixing a critical bug):**

Initially, all injected images were given `cosine = 0.9`, which is roughly **10× higher** than the real average for cat images (~0.084). This made injected images dominate Stage 1 filtering entirely, crowding out original images regardless of quality.

The fix computes the actual average cosine of existing category images and assigns values proportionally:
```python
avg_cat_cos = mean(cosine_by_query["cat"] for existing cat images)
cos_high = avg_cat_cos + 0.02   # image 50
cos_mid  = avg_cat_cos + 0.01   # image 51
cos_low  = avg_cat_cos − 0.02   # image 52
```

This gives injected images cosine values that are realistic and proportional to their relative quality, without artificially dominating Stage 1.

### 13.2 Cold-Start Fix: Seeded Engagement

**Problem discovered:** Without any engagement history, a newly injected image has CTR = 0.10 (prior), LR = 0.05 (prior), AWT = 0.0. Its Stage 2 score is too low to enter the displayed top-10. But without being displayed, it can never build engagement. This **cold-start deadlock** meant mid-quality image 51 was always ranked 11–12 and never appeared in the top-10.

**Fix:** At injection time, each image is seeded with quality-proportional synthetic engagement history via virtual impressions:
```python
def _seed_engagement(img, virtual_exams, p_ctr, p_lr, awt_val):
    img.tilde_e = virtual_exams           # 50 virtual impressions
    img.tilde_v = virtual_exams * p_ctr   # virtual views
    img.tilde_l = tilde_v * p_lr          # virtual likes
    img.ctr = (tilde_v + c_ctr*m_ctr) / (tilde_e + c_ctr)
    img.lr  = (tilde_l + c_lr*m_lr)  / (tilde_v + c_lr)
    img.awt = awt_val
```

**Seeding parameters:**
| Image | virtual_exams | p_ctr | p_lr | awt |
|---|---|---|---|---|
| High (q=0.4) | 50 | 0.28 | 0.25 | 0.45 |
| Mid (q=0.25) | 50 | 0.20 | 0.17 | 0.30 |
| Low (q=0.01) | 50 | 0.06 | 0.03 | 0.05 |

**Rationale:** In a real system, a newly published image would typically come with some metadata-based quality assessment or initial engagement from curators. Seeding with quality-proportional virtual data reflects this prior knowledge. It also ensures the image can enter the display pool immediately, after which real user engagement takes over and the seeded values are gradually overwritten via EWMA.

**Result after fix:** Image 51 now appears at ranks 5–10 (was always 11–12), and the seeded values are naturally overridden by real engagement within ~100 rounds.

### 13.3 Recovery Boost for Old Images

**Problem:** After injecting new images with high freshness (= 1.0), old high-quality images (which had freshness ≈ 0 after 500 rounds of decay) were temporarily suppressed in ranking. Even after the injected images' freshness decayed, some original images that had been pushed out of the top-10 were stuck there — they were no longer being shown, so their engagement accumulators were stale, and stale engagement signals produced low scores.

**Fix — Recovery Boost:** Every 50 rounds, any image that has not been shown in the last 50 rounds receives a temporary freshness boost:
```python
if t % recovery_boost_interval == 0:
    for img in images:
        if (t - img.last_shown_round) >= recovery_boost_interval:
            img.freshness = min(1.0, img.freshness + 0.5)
```

The boost is +0.5 freshness (capped at 1.0). Since freshness appears in the Stage 2 quality score with weight 0.5, this adds up to 0.25 to the score — enough to push a borderline-excluded image back into the visible pool. Once the image is shown again, real user feedback resumes and the image either recovers its rank organically or falls back if its engagement is genuinely low.

---

## 14. Debugging Journey & Root Cause Fixes

This section documents the iterative debugging process that led to the final design.

### Issue 1: Low-quality image 52 stuck in top-5
**Symptom:** After injection, image 52 (q=0.01) consistently ranked in positions 3–5.  
**Root cause:** `cosine_by_query = 0.9` (hardcoded). Stage 1 always selected image 52 first, and its fresh high freshness score (1.0) combined with the match bonus (2.0) gave it a Stage 2 score that outcompeted old images with real engagement.  
**Fix:** Replace hardcoded cosine with `avg_cat_cos − 0.02`, matching realistic values for a low-quality image.

### Issue 2: Old images never recovering rank
**Symptom:** After round 501, original cat images gradually fell out of the top-10 and never returned, even after the injected images' freshness decayed.  
**Root cause 1:** `freshness` was tracked in `ImageState` but was **not included in the Stage 2 quality_score formula**. The recovery boost added freshness to the image's state, but the Stage 2 formula never read it. Thus the boost had zero effect on ranking.  
**Fix:** Added `args.quality_weight_fresh * img.freshness` to the Stage 2 formula.  
**Root cause 2:** Old images with zero freshness and stale engagement could not compete against even low-quality fresh images because their accumulated CTR/LR/AWT, while real, were still bounded by the prior.  
**Fix:** Recovery Boost (Section 13.3) gives excluded images a freshness re-entry ticket, enabling them to re-enter the pool and rebuild engagement.

### Issue 3: Mid-quality image 51 always outside top-10
**Symptom:** Image 51 consistently ranked 11–12 and was invisible in the injected images plot.  
**Root cause:** Cold-start deadlock — zero engagement history meant the Stage 2 score was too low to display, but without display, no engagement history could build.  
**Fix:** Seeded engagement + cosine = avg+0.01 (Section 13.2).

### Issue 4: Low-quality image 52 invisible in plots
**Symptom:** After fixing the cosine bug, image 52's realistic cosine (avg−0.02) placed it exactly at position 11 in Stage 1 (outside the top-10 cutoff), so it was never logged.  
**Fix:** Increased Stage 1 top-K from 10 to **13** (to include all 13 cat images: 10 original + 3 injected). Only top-10 are displayed, but all 13 are logged to `query_ranks.csv` with their full Stage 2 rank.

---

## 15. Output Files & Plots

All output is written to:
```
backend/data/final_search_first_v3_twostage_injection_v5_fixed/
```

| File | Description |
|---|---|
| `simulation_rounds.csv` | Per-round: reward, CTR, LR, AWT, all 6 weights |
| `item_stats.csv` | Every 10 rounds: per-image freshness, CTR, LR, AWT, score |
| `query_ranks.csv` | Every 10 rounds: full Stage 1+2 rankings for all 5 queries |
| `reward_curve.png` | Average reward over 1000 rounds |
| `weight_trajectory.png` | Evolution of all 6 policy gradient weights on simplex |
| `figure1_diff_quality_6lines.png` | Rank trajectories for cat images at quality positions 1st/2nd, 5th/6th, 9th/10th |
| `figure2_similar_quality_planes_6lines.png` | Rank trajectories for 6 similarly-quality-scored plane images |
| `figure3_injected_images.png` | Rank trajectories of injected images 50/51/52 from rounds 500–1000 |
| `figure4_quality_distribution.png` | Strip plot of quality distribution per category (dots = individual images) |

---

## 16. How to Run

### Step 1: Build the cosine cache (only once)
```bash
python backend/build_dataset_cosine_cache.py \
  --image-dir dataset/image_add_midway \
  --out dataset/cosine_cache_siglip.json
```

### Step 2: Run the simulation
```bash
python backend/simulate_ranking.py \
  --w-init 0.25 0.15 0.15 0.15 0.15 0.15 \
  --tau 0.40 --beta 0.005 --eta 0.004 --batch-size 64 \
  --rounds 1000 \
  --dataset-cache dataset/cosine_cache_siglip.json \
  --lambda-v 1.2 --lambda-t 2.2 --lambda-l 1.2 \
  --m-ctr 0.10 --m-lr 0.05 --c-ctr 10 --c-lr 10 \
  --rho 0.20 --omega-l 1.5 --gamma-pos 0.90 \
  --alpha -1.40 1.40 3.40 0.05 0.10 --alpha-q 0.30 \
  --gamma -0.90 0.25 0.90 0.08 0.05 --gamma-q 4.50 \
  --kappa 12 \
  --delta -2.60 0.05 0.25 0.03 0.05 --delta-t 3.00 --delta-q 4.00 \
  --user-random-sd 0.10 --seed 42 \
  --item-diagnostics \
  --two-stage-top-k 13 \
  --quality-weight-cos 1.0 --quality-weight-match 2.0 \
  --quality-weight-fresh 0.5 --quality-weight-ctr 0.5 \
  --quality-weight-lr 5.0 --quality-weight-awt 2.0 \
  --normalize-features \
  --inject-images --inject-round 501 --inject-category cat \
  --min-fresh-weight 0.05 --freshness-decay-scale 5.0 \
  --recovery-boost-interval 50 --recovery-boost-strength 0.5 \
  --out-dir backend/data/final_search_first_v3_twostage_injection_v5_fixed
```

### Step 3: Generate plots
```bash
python backend/plot_rank_dynamics_v3.py \
  backend/data/final_search_first_v3_twostage_injection_v5_fixed
```

---

## 17. Final Configuration Parameters

### Oracle (User Behaviour)
| Parameter | Value | Meaning |
|---|---|---|
| `alpha` | `[-1.40, 1.40, 3.40, 0.05, 0.10]` | View probability: intercept, match, cosine, freshness, social |
| `alpha_q` | `0.30` | Quality boost on view probability |
| `gamma` | `[-0.90, 0.25, 0.90, 0.08, 0.05]` | Watch time mean: same structure |
| `gamma_q` | `4.50` | Quality boost on watch time (stronger effect) |
| `delta` | `[-2.60, 0.05, 0.25, 0.03, 0.05]` | Like probability: high intercept penalty means likes are rare |
| `delta_t` | `3.00` | Watch time drives like probability |
| `delta_q` | `4.00` | Quality is the strongest driver of likes |
| `kappa` | `12` | Beta distribution concentration (watch time sharpness) |
| `gamma_pos` | `0.90` | Position bias exponent |

### Simulation
| Parameter | Value | Meaning |
|---|---|---|
| `rounds` | `1000` | Total simulation rounds |
| `batch_size` | `64` | Users sampled per round |
| `top_k` | `10` | Images displayed per query |
| `seed` | `42` | Reproducibility seed |

### Ranking & Learning
| Parameter | Value | Meaning |
|---|---|---|
| `w_init` | `[0.25, 0.15, 0.15, 0.15, 0.15, 0.15]` | Initial policy weights (cos, fresh, lr, ctr, awt, social) |
| `eta` | `0.004` | Policy gradient learning rate |
| `tau` | `0.40` | Softmax temperature |
| `beta` | `0.005` | L2 regularisation |
| `min_fresh_weight` | `0.05` | Floor on freshness weight |
| `two_stage_top_k` | `13` | Stage 1 candidate pool size |
| `quality_weight_cos` | `1.0` | Stage 2 cosine weight |
| `quality_weight_match` | `2.0` | Stage 2 category match bonus |
| `quality_weight_fresh` | `0.5` | Stage 2 freshness weight |
| `quality_weight_ctr` | `0.5` | Stage 2 CTR weight |
| `quality_weight_lr` | `5.0` | Stage 2 like rate weight |
| `quality_weight_awt` | `2.0` | Stage 2 watch time weight |

### Engagement Smoothing
| Parameter | Value | Meaning |
|---|---|---|
| `rho` | `0.20` | EWMA decay rate (aggressive forgetting) |
| `c_ctr` | `10` | Bayesian CTR prior strength |
| `m_ctr` | `0.10` | Bayesian CTR prior mean |
| `c_lr` | `10` | Bayesian LR prior strength |
| `m_lr` | `0.05` | Bayesian LR prior mean |
| `omega_l` | `1.5` | Like weight in social proof |

### Injection Experiment
| Parameter | Value | Meaning |
|---|---|---|
| `inject_images` | `True` | Enable injection experiment |
| `inject_round` | `501` | Injection at round 501 |
| `inject_category` | `cat` | Category for injected images |
| `freshness_decay_scale` | `5.0` | Half-life of freshness in rounds |
| `recovery_boost_interval` | `50` | Frequency of boost for unseen images |
| `recovery_boost_strength` | `0.5` | Freshness delta added by boost |
| `lambda_v` | `1.2` | View reward weight |
| `lambda_t` | `2.2` | Watch time reward weight |
| `lambda_l` | `1.2` | Like reward weight |
