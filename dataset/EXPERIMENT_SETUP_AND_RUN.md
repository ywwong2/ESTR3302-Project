# Experiment Setup and Run Log (2026-04-27)

## 1. Dataset and preprocessing state
- Dataset root: `dataset/`
- Category folders used:
  - `dataset/cats` -> query label `cat`
  - `dataset/dogs` -> query label `dog`
  - `dataset/planes` -> query label `plane`
  - `dataset/clothes` -> query label `clothes`
  - `dataset/cars` -> query label `car`
- Verified image counts:
  - cats: 10
  - dogs: 10
  - planes: 10
  - clothes: 10
  - cars: 10
  - total: 50
- Old backend preprocessed artifacts were deleted from:
  - `backend/data/preprocessed`
- Verified current preprocessed file count: 0

## 2. Cosine cache generation (250 pairs)
### Model and method
- Model: `google/siglip-base-patch16-224`
- Script: `backend/build_dataset_cosine_cache.py`
- Queries used (fixed order): `[cat, dog, plane, clothes, car]`
- Similarity method:
  1. Encode each query with SigLIP text encoder.
  2. Encode each image with SigLIP image encoder.
  3. L2-normalize both vectors.
  4. Compute cosine similarity for each query-image pair.

### Output cache
- Cache file: `dataset/cosine_cache_siglip.json`
- Verified cache content:
  - `image_count = 50`
  - `pair_count = 250`
  - each image contains cosine values for all 5 queries

### Command used
```bash
"/Users/wong/Desktop/this semester/ESTR3302/ESTR3302-Project/.venv/bin/python" backend/build_dataset_cosine_cache.py --dataset-root dataset --output dataset/cosine_cache_siglip.json
```

## 3. Simulation with dataset cache
### Script and mode
- Script: `backend/simulate_ranking.py`
- Dataset mode flag: `--dataset-cache dataset/cosine_cache_siglip.json`
- In dataset mode, the simulation uses cached cosine scores from `cosine_by_query` per image instead of synthetic embedding cosine.

### Ranking and training formulation implemented
- Feature vector:
  - `x = [cos, freshness, LR_tilde, CTR_tilde, AWT_tilde, social_proof]`
- Score:
  - `S = w^T x`
- Policy:
  - softmax with temperature `tau`
- Reward:
  - `R = lambda_v * view + lambda_t * watch + lambda_l * like`
- Update:
  - empirical policy gradient with projected gradient descent onto simplex

### User behavior oracle setup
- Latent quality per image: `q_i ~ Beta(2, 5)`
- Examination probability at rank `r`:
  - `eta_r = (1 + r)^(-gamma_pos)`
- Logistic coefficients used:
  - view (`alpha`): `[-0.8, 0.6, 2.0, 0.3, 1.0]`, `alpha_q = 0.5`
  - watch (`gamma`): `[-0.6, 0.4, 1.0, 0.2, 0.4]`, `gamma_q = 2.5`
  - like (`delta`): `[-2.0, 0.3, 0.8, 0.1, 0.5]`, `delta_t = 2.0`, `delta_q = 2.0`
- User random effects:
  - `xi_view, xi_watch, xi_like ~ N(0, 0.1^2)`

### Smoothing and social-proof setup
- EWMA rate: `rho = 0.1`
- Bayesian priors:
  - `m_ctr = 0.1`, `c_ctr = 10`
  - `m_lr = 0.05`, `c_lr = 10`
- Social-proof:
  - `omega_l = 5.0`
  - `M = ln(1 + 500)`

### Optimization setup
- Initial weights: uniform simplex `[1/6, ..., 1/6]`
- `eta = 0.01`
- `tau = 5.0`
- `beta = 0.0` (no regularization default)

### Scale setup
- `num_users = 1000`
- `rounds = 1000`
- `batch_size = 32`
- `top_k = 10`

### Reward weights
- `lambda_v = 1.0`
- `lambda_t = 2.0`
- `lambda_l = 5.0`

### Command used
```bash
"/Users/wong/Desktop/this semester/ESTR3302/ESTR3302-Project/.venv/bin/python" backend/simulate_ranking.py --dataset-cache dataset/cosine_cache_siglip.json --out-dir backend/data/simulation_dataset --no-plots
```

### Main simulation artifact
- CSV output: `backend/data/simulation_dataset/simulation_rounds.csv`

## 4. Implementation changes made for this run
- Added cache builder script:
  - `backend/build_dataset_cosine_cache.py`
- Updated simulation to support dataset-cache mode:
  - `backend/simulate_ranking.py`
  - new argument: `--dataset-cache`
  - new loader: cache -> image states with `cosine_by_query`
  - simulation branch uses cache cosine per query

## 5. Notes
- Plot generation is optional and disabled in this run via `--no-plots`.
- The run is now directly tied to your dataset folder content and query set.
