# ESTR3302 Multimedia Search — Image Ranking Simulation

A lightweight multimedia search engine with a two-stage ranking pipeline
that infers latent image quality from noisy user interaction signals.
Content–query matching is handled by SigLIP embeddings; the ranking
system uses fixed-weight scoring with Bayesian-smoothed engagement
statistics and integrity defenses (velocity clamp + trust ramp).

## Project Structure

```
ESTR3302-Project/
├── backend/
│   ├── simulate_ranking.py          ← Core simulation (user behaviour + ranking)
│   ├── run_experiments.py           ← Reproduces all report experiments & figures
│   ├── main.py                      ← FastAPI server for the GUI demo
│   ├── embeddings.py                ← SigLIP embedding & cosine cache
│   ├── build_dataset_cosine_cache.py
│   ├── plot_rank_dynamics_v3.py     ← Standalone plotting utility
│   ├── requirements.txt
│   ├── tests/
│   └── data/
│       ├── gen_figures.py           ← Ranking-performance & appendix figures
│       ├── exp_bayesian_defended/   ← Canonical Bayesian run (report Section 6 + appendix)
│       ├── exp_bayesian_vs_freq/    ← Exp 1: Bayesian vs Frequency (report Section 7.1)
│       ├── exp_bayesian_vs_freq_w10/← Exp 1: Frequency W=10 data
│       ├── exp_bayesian_vs_freq_w30/← Exp 1: Frequency W=30 data
│       ├── exp_bayesian_trust_ramp/ ← Bayesian + trust ramp illustration (appendix)
│       └── exp_synthetic/           ← Exp 2: Synthetic attack (report Section 7.2)
├── dataset/
│   ├── cosine_cache_siglip.json     ← Pre-computed SigLIP cosines (50 images)
│   └── images…
├── frontend/
│   ├── index.html, app.js, styles.css  ← GUI demo (not used for reported results)
└── models/
    └── siglip-base-patch16-224/     ← Local SigLIP weights (download once)
```

---

## Installation

### Prerequisites

- Python 3.11+ (tested on 3.12)
- ~400 MB disk for the SigLIP model (only needed for the GUI; experiments use a pre-computed cosine cache)

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. (Optional) Download SigLIP model — only for GUI / re-generating cosine cache

```bash
pip install huggingface_hub
huggingface-cli login           # paste your HF token when prompted
huggingface-cli download google/siglip-base-patch16-224 \
    --local-dir ./models/siglip-base-patch16-224
```

The cosine cache (`dataset/cosine_cache_siglip.json`) is already included,
so experiments can run without the model download.

---

## Replicating All Report Results (for TA)

All reported figures and data can be reproduced with a single command:

```bash
source .venv/bin/activate
python backend/run_experiments.py all
```

This runs (~5–10 min total):

| Step | What it does | Report section |
|------|--------------|----------------|
| `exp1` | Bayesian smoothing run + Frequency W=30/W=10 runs (cold-start injection at round 501) | Section 7.1 |
| `exp1_defended` | Bayesian + trust ramp run | Appendix figure |
| `exp3` | Synthetic attack runs (undefended + defended, 5 intensities each) | Section 7.2 |
| `plots` | Generates all experiment figures from the CSV data | Figures 2–6 |
| `figures` | Generates ranking-performance & appendix figures (Precision@10 table, per-category rank dynamics) | Section 6 + Appendix |

### Running individual experiments

```bash
python backend/run_experiments.py exp1          # Exp 1 only + its plots
python backend/run_experiments.py exp3          # Exp 2 (attacks) only + its plots
python backend/run_experiments.py figures       # Ranking-performance figures only
```

### Output locations

| Report figure | File path |
|---------------|-----------|
| Fig 1: Cat rank trajectories | `backend/data/exp_bayesian_defended/figure1_diff_quality_6lines.png` |
| Fig 2: Bayesian vs Frequency ranks | `backend/data/exp_bayesian_vs_freq/fig_rank_comparison.png` |
| Fig 3: CTR/LR cold-start metrics | `backend/data/exp_bayesian_vs_freq/fig_cold_start_metrics.png` |
| Fig 4: Attack trajectories | `backend/data/exp_synthetic/fig_attack_defended.png` |
| Appendix: Bayesian + trust ramp | `backend/data/exp_bayesian_trust_ramp/fig_bayesian_trust_ramp.png` |
| Appendix: Quality distribution | `backend/data/exp_bayesian_defended/figure4_quality_distribution.png` |
| Appendix: Per-category dynamics | `backend/data/exp_bayesian_defended/figure_*_rank_dynamics.png` |

### Running tests

```bash
python -m pytest backend/tests -v
```

---

## Two-Stage Ranking (Mathematical Core)

**Stage 1** filters the full catalog to the top K₁=13 images by cosine
similarity, guaranteeing semantic relevance.

**Stage 2** re-ranks these candidates by a fixed weighted sum:

```
r_i = 1.0·cos + 2.0·match + 0.5·fresh + 0.5·CTR + 5.0·LR + 2.0·AWT
```

Engagement statistics (CTR, LR, AWT) are updated each round via
EWMA smoothing (ρ=0.20) with Bayesian-style priors (c=10).

**Defenses:**
- Velocity clamp (k=2): caps per-image engagement at 2× the population mean
- Trust ramp (M=200): attenuates engagement for images younger than 200 rounds

---

## GUI Demo (Optional)

The GUI is functional but is **not used to produce the reported results**.

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
# Open frontend/index.html in a browser
```

---

## Notes

- All experiments use seed 42 for reproducibility.
- The cosine cache (`dataset/cosine_cache_siglip.json`) is pre-computed and
  included in the repo — no GPU or model download needed for experiments.
- Frontend sandbox state (`backend/data/frontend/`) is separate from experiment data.
