# ESTR3302 Image Ranking Simulation

A math-focused image ranking simulation project with an interactive GUI frontend.
The core is a policy-gradient ranking algorithm operating over SigLIP image embeddings.

## What this project does

- **Ranking simulation** — two-stage ranking (cosine filter → quality re-rank) with REINFORCE policy gradient weight learning, freshness decay, exploration injection, and recovery boost.
- **Interactive GUI** — open `frontend/index.html` to become a real user in the simulation. Upload images, like them, watch the ranking weights evolve in real time.
- **Experiment mode** — run `simulate_ranking.py` directly for the controlled 1000-round experiment with full CSV/plot output.

## Project Structure

```
ESTR3302-Project/
├── backend/
│   ├── main.py                  ← FastAPI server (no SQLite; JSON storage)
│   ├── embeddings.py            ← SigLIP only: image + text embed, cosine cache
│   ├── preprocess.py            ← Image resize helper
│   ├── debug_log.py             ← In-memory log for frontend debug panel
│   ├── simulate_ranking.py      ← Full CLI simulation script (experiment mode)
│   ├── plot_rank_dynamics_v3.py ← Post-simulation plotting
│   ├── build_dataset_cosine_cache.py
│   ├── evaluate_siglip_accuracy.py
│   ├── requirements.txt
│   ├── tests/
│   └── data/
│       ├── frontend/            ← GUI sandbox state (gallery.json, sim_state.json, …)
│       └── final_search_first_v3_twostage_injection_v5_fixed/  ← experiment results
├── dataset/
│   ├── cosine_cache_siglip.json ← Pre-computed SigLIP cosines for 50 images
│   └── …image folders…
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── models/
│   └── siglip-base-patch16-224/ ← Local SigLIP weights (download once)
└── IMPLEMENTATION_REPORT.md     ← Detailed design & rationale document
```

## Installation

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Download SigLIP model (once)

```bash
.venv/bin/hf auth login        # login with Hugging Face token
.venv/bin/hf download google/siglip-base-patch16-224 \
    --local-dir ./models/siglip-base-patch16-224
```

The model is ~400 MB. Once downloaded, the backend uses it fully offline.

---

## Running the Web Demo (GUI mode)

### Start the backend

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

### Open the frontend

Open `frontend/index.html` in a browser (Live Server or directly via `file://`).

### First-time setup

On first open you will see a pool selector:
- **Load 50-Image Dataset** — initialises with the 50 pre-embedded images from `dataset/cosine_cache_siglip.json`. Simulation can start immediately.
- **Start Empty** — blank sandbox; upload your own images.

### Interacting

| Action | Effect on simulation |
|---|---|
| Click image thumbnail | Counts as a **view** (increases CTR) |
| Click 👍 Like | Counts as a **like** (increases LR and CTR) |
| Delete 🗑 | Removes image from pool |
| Upload image | SigLIP embeds it, auto-detects category, joins simulation pool |
| Search `cat` / `dog` / … | Ranks pool by cosine + quality score for that query |

### Simulation controls (sidebar)

| Control | Description |
|---|---|
| **▶ Start / ⏸ Pause** | Toggle the background simulation thread |
| **+1 Step** | Manually advance one round (works while paused) |
| **Reset** | Revert to round 0 (keeps images) |
| **Interval slider** | Set seconds between automatic rounds (1–30 s) |
| **📊 View Figures** | Show pre-computed experiment plots in a modal |
| **⬇ Load Round-1000 State** | Restore weights and per-image stats from the finished 1000-round experiment, then continue running |

### Backend API reference

| Endpoint | Method | Description |
|---|---|---|
| `/images` | GET | List images with current ranks (`?q=` for query) |
| `/images/upload` | POST | Upload image (multipart) |
| `/images/{id}` | DELETE | Remove image |
| `/search?q=` | GET | Alias for `/images?q=` |
| `/interact` | POST | Record `{image_id, action}` where action=`view`/`like` |
| `/sim/status` | GET | Round, running, weights, interval |
| `/sim/init` | POST | Init pool: `{mode: "empty"|"dataset"}` |
| `/sim/toggle` | POST | Start / pause background sim |
| `/sim/step` | POST | Manual single round |
| `/sim/reset` | POST | Reset round counter and engagement stats |
| `/sim/set_interval` | POST | `{interval: N}` seconds |
| `/sim/preset` | GET | List experiment figures + summary |
| `/sim/load_preset` | POST | Load round-1000 state |
| `/preset_figures/{name}` | GET | Serve experiment PNG |
| `/debug/logs?since=N` | GET | Debug log entries |

---

## Running the Experiment (CLI simulation)

### Build cosine cache (once, if re-generating)

```bash
python backend/build_dataset_cosine_cache.py \
  --image-dir dataset/image_add_midway \
  --out dataset/cosine_cache_siglip.json
```

### Run the 1000-round simulation

```bash
source .venv/bin/activate
python backend/simulate_ranking.py \
  --w-init 0.25 0.15 0.15 0.15 0.15 0.15 \
  --tau 0.40 --beta 0.005 --eta 0.004 --batch-size 64 \
  --rounds 1000 \
  --dataset-cache dataset/cosine_cache_siglip.json \
  --lambda-v 1.2 --lambda-t 2.2 --lambda-l 1.2 \
  --rho 0.20 --gamma-pos 0.90 \
  --two-stage-top-k 13 \
  --quality-weight-cos 1.0 --quality-weight-match 2.0 \
  --quality-weight-fresh 0.5 --quality-weight-ctr 0.5 \
  --quality-weight-lr 5.0 --quality-weight-awt 2.0 \
  --normalize-features \
  --inject-images --inject-round 501 --inject-category cat \
  --min-fresh-weight 0.05 --freshness-decay-scale 5.0 \
  --recovery-boost-interval 50 --recovery-boost-strength 0.5 \
  --seed 42 \
  --item-diagnostics \
  --out-dir backend/data/final_search_first_v3_twostage_injection_v5_fixed
```

### Generate plots

```bash
python backend/plot_rank_dynamics_v3.py \
  backend/data/final_search_first_v3_twostage_injection_v5_fixed
```

### Run tests

```bash
source .venv/bin/activate
python -m pytest backend/tests -v
```

---

## Mathematical Core

**Feature vector** (dim 6):
$$\mathbf{x}_i = [\cos_i,\ f_i,\ \widetilde{\text{LR}}_i,\ \widetilde{\text{CTR}}_i,\ \widetilde{\text{AWT}}_i,\ b_i]^\top$$

**Policy** (softmax with temperature $\tau$):
$$a_i = \frac{\exp(\tau\, \mathbf{w}^\top\mathbf{x}_i)}{\sum_j \exp(\tau\, \mathbf{w}^\top\mathbf{x}_j)}, \quad \mathbf{w} \in \Delta^5$$

**Reward**:
$$R_i = \lambda_v V_i + \lambda_t T_i + \lambda_l L_i$$

**REINFORCE gradient** (projected onto simplex):
$$\mathbf{w} \leftarrow \Pi_\Delta\!\left[\mathbf{w} - \eta\left(-\tau \sum_i a_i(R_i - \bar{R})\mathbf{x}_i + 2\beta\mathbf{w}\right)\right]$$

See `IMPLEMENTATION_REPORT.md` for the full design rationale, debugging history, and parameter table.

---

## Notes

- Frontend sandbox state (`backend/data/frontend/`) is completely separate from experiment data.
- To reset frontend state: delete `backend/data/frontend/` and restart the server.
- The SigLIP model is loaded lazily on first use; first search/upload will be slow (~5–20 s depending on machine).
- Query text embeddings and image cosines are cached in `query_cache.json` — no re-computation on subsequent searches.
