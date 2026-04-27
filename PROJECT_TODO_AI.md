# Image Retrieval and Ranking Simulation — AI TODO Plan

## Scope lock
- Project is now image-only.
- Frontend-backend demo must keep working.
- Research focus is mathematical ranking and simulation.
- No embedding cache layer beyond persistent indexed vectors.

---

## 0) Baseline cleanup

### 0.1 Remove non-image code paths
- [ ] Remove audio/video/text preprocessing branches.
- [ ] Remove CLAP and non-image embedding branches.
- [ ] Remove pure text upload endpoint and UI controls.

**Definition of Done**
- Backend accepts `image/*` only.
- Frontend only exposes image upload.
- Search still works end-to-end.

### 0.2 Keep essential runtime paths
- [ ] Keep routes: `GET /search`, `POST /media/upload`, `POST /media/{id}/like`, `POST /media/{id}/view`, `GET /media/{id}`, `DELETE /media/{id}`, `GET /debug/logs`.
- [ ] Keep upload -> background processing -> indexed lifecycle.
- [ ] Keep result card fields used by frontend (`id`, `title`, `media_type`, `url`, `status`, metrics).

**Definition of Done**
- All frontend buttons still function.
- No broken API calls from frontend console.

---

## 1) Image embedding and retrieval core

### 1.1 SigLIP pipeline
- [ ] Keep SigLIP image embedding for indexed images.
- [ ] Keep SigLIP text embedding for query.
- [ ] Keep cosine similarity search and ranking output fields.

### 1.2 Persistent vector storage
- [ ] Keep stored vectors for indexed images.
- [ ] Remove extra cache logic and unused cache docs/tasks.
- [ ] Ensure delete also removes embedding metadata and vector file.

**Definition of Done**
- Uploading an image eventually becomes `INDEXED`.
- Query returns ranked image list with stable scores.
- Deleted image never appears in search.

---

## 2) Ranking function implementation (online system)

At round $t$, for user $u$ and image $i$, use:
$$
\mathbf{x}_{ui}^{(t)} = [\mathrm{cos}_{ui}, f_i^{(t)}, \widetilde{\mathrm{LR}}_i^{(t-1)}, \widetilde{\mathrm{CTR}}_i^{(t-1)}, \widetilde{\mathrm{AWT}}_i^{(t-1)}, b_i^{(t-1)}]^\top \in \mathbb{R}^6,
$$
$$
S_{ui}^{(t)} = \mathbf{w}^\top \mathbf{x}_{ui}^{(t)}, \quad \mathbf{w} \in \Delta.
$$

### 2.1 Historical signal updates
- [ ] Maintain EWMA counts for examinations/views/likes:
  $\tilde{N}_i^{(t)}=(1-\rho)\tilde{N}_i^{(t-1)}+\rho N_i^{(t)}$.
- [ ] Compute Bayesian-smoothed metrics:
  $\widetilde{\mathrm{CTR}}_i^{(t)}=\frac{\tilde{V}_i^{(t)}+c_{\mathrm{ctr}}m_{\mathrm{ctr}}}{\tilde{E}_i^{(t)}+c_{\mathrm{ctr}}}$,
  $\widetilde{\mathrm{LR}}_i^{(t)}=\frac{\tilde{L}_i^{(t)}+c_{\mathrm{lr}}m_{\mathrm{lr}}}{\tilde{V}_i^{(t)}+c_{\mathrm{lr}}}$.
- [ ] Update AWT and social proof:
  $\widetilde{\mathrm{AWT}}_i^{(t)}=(1-\rho)\widetilde{\mathrm{AWT}}_i^{(t-1)}+\rho\bar{T}_i^{(t)}$,
  $b_i^{(t)}=\min\left(1,\frac{\ln(1+\tilde{V}_i^{(t)}+\omega_l\tilde{L}_i^{(t)})}{M}\right)$.

**Definition of Done**
- Backend ranking state is deterministic for fixed seed/data.
- Metrics remain bounded and numerically stable.

---

## 3) User behavior simulation (backend only)

### 3.1 Simulation engine
- [ ] Assign latent quality $q_i \sim \mathrm{Beta}(2,5)$ for each image.
- [ ] Apply examination model with position bias:
  $\eta_r=(1+r)^{-\gamma_{\mathrm{pos}}}$.
- [ ] Sample user events for displayed images:
  - view $V$ via Bernoulli on logistic score,
  - watch time $T$ via Beta,
  - like $L$ via Bernoulli with watch-time gating.
- [ ] Include user-specific random effects $\xi_u$.

### 3.2 Training with empirical policy gradient
- [ ] Use observed reward:
  $R_{ui}^{(t)}=\lambda_v V_{ui}^{(t)}+\lambda_t T_{ui}^{(t)}+\lambda_l L_{ui}^{(t)}$.
- [ ] Use soft-max exposure policy:
  $a_{ui}^{(t)}(\mathbf{w})=\frac{\exp(\tau S_{ui}^{(t)})}{\sum_j \exp(\tau S_{uj}^{(t)})}$.
- [ ] Optimize objective:
  $F_t(\mathbf{w})=-\frac{1}{|\mathcal{B}_t|}\sum_u\sum_i a_{ui}^{(t)}R_{ui}^{(t)} + \beta\|\mathbf{w}\|_2^2$,
  with default $\beta=0$ and CLI override.
- [ ] Update with projected gradient descent on simplex:
  $\mathbf{w}^{(t+1)}=\Pi_\Delta(\mathbf{w}^{(t)}-\eta\nabla F_t(\mathbf{w}^{(t)}))$.

### 3.3 Experiment defaults
- [ ] Dataset scale: 50 images, 5 categories, balanced 10 each.
- [ ] Simulation scale: $|U|=1000$, $T=1000$, batch size 32, top-$K=10$.
- [ ] Initialize weights uniformly: $\mathbf{w}^{(0)}=[1/6,\dots,1/6]^\top$.
- [ ] Use priors/hyperparameters from latest plan (CTR/LR priors, $\rho$, $\omega_l$, $M$, $\gamma_{\mathrm{pos}}$, $\kappa$, $\eta$, $\tau$).

### 3.4 Artifacts
- [ ] Save per-round CSV logs.
- [ ] Save reward and weight trajectory PNG plots.
- [ ] Print final simplex-valid weights and key aggregate metrics.

**Definition of Done**
- One command runs simulation without frontend.
- CSV and PNG outputs generated in output folder.
- Weight vector stays on simplex across rounds.

---

## 4) Frontend and API demo integrity

### 4.1 Keep demo interactive
- [ ] Search button still triggers backend search.
- [ ] Upload button supports image files only.
- [ ] Like, info, delete buttons still work correctly.
- [ ] Status overlays still update via polling.

### 4.2 Keep response compatibility
- [ ] Preserve fields used by cards/metrics.
- [ ] Preserve debug panel polling endpoint.

**Definition of Done**
- Manual smoke test passes for: upload image -> indexed -> search -> like -> info -> delete.

---

## 5) Test and documentation updates

### 5.1 Tests
- [ ] Rewrite preprocessing tests for image-only behavior.
- [ ] Rewrite upload tests for image accept / non-image reject.
- [ ] Keep/adjust embedding utility tests for image-only mapping.
- [ ] Add simulation smoke test (short rounds) if runtime permits.

### 5.2 Docs
- [ ] Rewrite README to image-only architecture and runbook.
- [ ] Update schema doc to minimal entities currently used.
- [ ] Document simulation command and outputs.

**Definition of Done**
- `pytest backend/tests -v` passes.
- README instructions match actual commands and routes.

---

## 6) Milestone schedule

### Milestone 1
- [ ] Complete sections 0-1 (core simplification + image retrieval path).

### Milestone 2
- [ ] Complete section 3 (simulation + training loop + outputs).

### Milestone 3
- [ ] Complete sections 4-5 (UI/API regression + tests/docs finalization).

---

## Optional future appendix (out of current scope)
- Optional migration from JSON vector files to dedicated vector DB.
- Optional offline evaluation expansion (`Recall@K`, `nDCG@K`, category fairness).
- Optional learned ranker baseline comparison (e.g., XGBoost) after simulation baseline is stable.
- Optional auth/security hardening for production deployment.
