# ESTR3302 Image Search and Ranking Simulation

Course project refactored to an image-only semantic retrieval system with a mathematically focused ranking simulation module.

## What this project does now
- Upload and index image files only.
- Embed images and text queries with SigLIP.
- Search images by cosine similarity with ranking signals.
- Keep lightweight feedback actions: view, like, delete, info.
- Run backend-only user behavior simulation with policy-gradient weight learning.

## Project structure

```
ESTR3302-Project/
├── backend/
│   ├── main.py
│   ├── preprocess.py
│   ├── embeddings.py
│   ├── db.py
│   ├── simulate_ranking.py
│   ├── requirements.txt
│   ├── migrations/001_init.sql
│   ├── tests/
│   └── data/
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── docs/
│   └── schema.md
└── models/
    └── siglip-base-patch16-224/
```

## Backend API (used by frontend)
- `POST /media/upload` image upload
- `GET /search?q=...` image search
- `POST /media/{id}/like` like counter
- `POST /media/{id}/view` view counter
- `GET /media/{id}` media detail
- `DELETE /media/{id}` soft delete + embedding cleanup
- `GET /debug/logs?since=N` debug panel polling
- `GET /uploads/{filename}` static uploaded files

## Installation

1. Create virtual environment and activate:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

3. Ensure local SigLIP model exists at:
- `models/siglip-base-patch16-224/`

If you need to download:

```bash
.venv/bin/hf auth login
.venv/bin/hf download google/siglip-base-patch16-224 --local-dir ./models/siglip-base-patch16-224
```

## Run web demo

1. Start backend:

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

2. Open frontend:
- open `frontend/index.html` directly, or serve it via Live Server.

3. Demo flow:
- upload image
- wait for status `UPLOADED -> PROCESSING -> INDEXED`
- search by text query
- test like/info/delete buttons

## Run ranking simulation (backend only)

Default run:

```bash
source .venv/bin/activate
python backend/simulate_ranking.py
```

Useful overrides:

```bash
python backend/simulate_ranking.py --rounds 300 --batch-size 32 --beta 0 --out-dir backend/data/simulation_run1
```

Outputs:
- `simulation_rounds.csv`
- `reward_curve.png` (if matplotlib available)
- `weight_trajectory.png` (if matplotlib available)

## Mathematical core in simulation
- Feature vector:
  $\mathbf{x}_{ui}^{(t)} = [\mathrm{cos}_{ui}, f_i, \widetilde{\mathrm{LR}}_i, \widetilde{\mathrm{CTR}}_i, \widetilde{\mathrm{AWT}}_i, b_i]^\top$
- Score:
  $S_{ui}^{(t)} = \mathbf{w}^\top\mathbf{x}_{ui}^{(t)}$, with $\mathbf{w} \in \Delta$
- Policy:
  $a_{ui}^{(t)}(\mathbf{w}) = \frac{\exp(\tau S_{ui}^{(t)})}{\sum_j\exp(\tau S_{uj}^{(t)})}$
- Reward:
  $R_{ui}^{(t)} = \lambda_v V_{ui}^{(t)} + \lambda_t T_{ui}^{(t)} + \lambda_l L_{ui}^{(t)}$
- Update:
  projected gradient descent on simplex, default $\beta=0$ with CLI override.

## Tests

```bash
source .venv/bin/activate
python -m pytest backend/tests -v
```

## Notes
- Non-image modalities were intentionally removed from runtime scope.
- Persistent vectors are kept for indexed images; extra cache layers are removed.
- To reset local state, delete `backend/data/` and restart backend.
