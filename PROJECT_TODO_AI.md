# Multimedia Search Engine — AI-Executable TODO Plan

## How to use this file
- Each task is small and testable.
- Complete tasks in order.
- Do not move to next task until **Definition of Done** is satisfied.
- Keep all code, configs, and docs under version control.

---

## 0) Project setup and architecture

### 0.1 Create repository structure
- [ ] Create folders:
  - `backend/`
  - `frontend/`
  - `worker/`
  - `infra/`
  - `docs/`
  - `data/` (local dev only)
- [ ] Add top-level files:
  - `.gitignore`
  - `README.md`
  - `.env.example`
  - `docker-compose.yml`

**Definition of Done**
- Folder structure exists.
- Project starts with placeholder services.

### 0.2 Choose baseline stack
- [ ] Backend API: FastAPI (Python)
- [ ] Frontend: Next.js or React + Vite
- [ ] Worker queue: Celery/RQ (or simple background tasks first)
- [ ] Metadata DB: PostgreSQL
- [ ] Vector DB: Chroma
- [ ] Cache (future): Redis (design now, implement later)

**Definition of Done**
- Tech stack recorded in `docs/architecture.md`.
- Initial dependency files created (`requirements.txt`, `package.json`).

### 0.3 Define system boundaries
- [ ] Draw high-level flow:
  1) upload media → 2) preprocess → 3) embed → 4) store vector + metadata → 5) query → 6) rank → 7) display
- [ ] Define API boundaries between frontend, backend, workers.

**Definition of Done**
- Sequence diagram saved in `docs/architecture.md`.

---

## 1) Data model and schema

### 1.1 Define core entities
- [x] `MediaItem` (id, type, uri/path, title, description, tags, uploader, timestamps)
- [x] `EmbeddingRecord` (media_id, modality, model_name, vector_dim, vector_id, created_at)
- [x] `QualitySignal` (media_id, quality_score, source)
- [x] `InteractionLog` (user_id/session_id, media_id, action, dwell_time, timestamp)
- [x] `StatsSnapshot` (media_id, views, likes, ctr, avg_watch_time)

**Definition of Done**
- ER diagram and SQL schema in `docs/schema.md`.
- Migration scripts prepared.

### 1.2 Define media lifecycle
- [x] Status states: `UPLOADED`, `PROCESSING`, `INDEXED`, `FAILED`, `DELETED`
- [x] Soft-delete + hard-delete policy
- [x] Re-index policy when model changes

**Definition of Done**
- Lifecycle states documented and represented in DB.

---

## 2) Ingestion and preprocessing pipeline

### 2.1 File upload API
- [x] `POST /media/upload` for image/video/audio/text
- [x] Insert metadata row + status `UPLOADED`

**Definition of Done**
- Upload works for all modalities.
- Invalid files rejected with clear error messages.

### 2.2 Preprocessing per modality

#### Video preprocessing
- [x] Uniform frame sampling:
  - target 8–16 frames per video
  - roughly every 1–2 seconds
- [x] Resize/normalize frames for image model input
- [x] Keep frame timestamps for optional explainability

#### Audio preprocessing
- [x] Split audio into ~10 equal chunks
- [x] Convert to required sample rate/channels for CLAP
- [x] Handle short audio gracefully (pad/merge strategy)

#### Image preprocessing
- [x] Resize/downsample to model input (e.g., 224x224 for SigLIP base patch16)
- [x] Normalize with model-specific preprocessing

#### Text preprocessing
- [x] Normalize Unicode
- [x] Trim extreme length

**Definition of Done**
- Each modality has deterministic preprocessing and unit tests.

---

## 3) Embedding generation

### 3.1 Model integration
- [x] Image/video frames: `google/siglip-base-patch16-224`
- [x] Audio chunks: CLAP model
- [x] Text: `google/embeddinggemma-300m`

**Definition of Done**
- Model wrappers return fixed-length vectors and pass smoke tests.

### 3.2 Pooling strategies
- [x] Video: mean pooling over sampled frame embeddings (support max pooling option)
- [x] Audio: mean pooling over 10 chunk embeddings
- [x] Track per-item pooling metadata (`pooling_type`, `num_segments`)

**Definition of Done**
- Final vector produced for every media item.
- Pooling strategy configurable.

### 3.3 Batch indexing worker
- [ ] Worker consumes pending media (`UPLOADED`/`PROCESSING`)
- [ ] Retry on transient failures
- [ ] Write failure reason for debugging

**Definition of Done**
- End-to-end indexing completes asynchronously.

---

## 4) Storage layer (metadata + vectors)

### 4.1 Chroma integration
- [ ] Create collection strategy (single collection with modality metadata OR per modality)
- [ ] Store:
  - `id` (media_id or embedding_id)
  - `embedding`
  - metadata: `media_type`, `title`, `tags`, `created_at`, `quality_score`, etc.
- [ ] Implement upsert and delete-by-id

**Definition of Done**
- Search and delete work correctly in Chroma.

### 4.2 Relational DB integration
- [ ] Persist media metadata and ranking signals in PostgreSQL
- [ ] Ensure vector record references are consistent with Chroma IDs

**Definition of Done**
- DB and vector store stay consistent after upload/update/delete.

### 4.3 Delete flow
- [ ] `DELETE /media/{id}`
- [ ] Remove from Chroma + metadata DB + file storage
- [ ] Add tombstone/audit log

**Definition of Done**
- Deleted items never appear in search.

---

## 5) Search API (retrieval)

### 5.1 Query embedding
- [ ] `POST /search` accepts text query
- [ ] Convert query text to embedding (text model)

### 5.2 Candidate retrieval
- [ ] Run vector similarity search (top-K, e.g., 100)
- [ ] Return candidate list with raw similarity score

### 5.3 Filtering
- [ ] Add filters: modality, date range, tags, uploader
- [ ] Exclude deleted/blocked content

**Definition of Done**
- API returns stable candidate set with filters.

---

## 6) Ranking system

### 6.1 Baseline weighted ranking
- [ ] Implement score formula:

  `final_score = 0.6*cosine_sim + 0.1*log(1+views+2*likes) + 0.1*(1/(1+days_since_upload)) + 0.1*quality_score_norm + 0.1*(ctr + 0.5*avg_watch_time_norm)`

- [ ] Normalize all non-similarity signals to consistent ranges
- [ ] Add guardrails for missing values

**Definition of Done**
- Ranking service outputs deterministic sorted list.

### 6.2 Quality score pipeline
- [ ] Define `quality_score` source:
  - manual annotation OR heuristic model
- [ ] Backfill for existing content

**Definition of Done**
- Every searchable item has a quality score.

### 6.3 Interaction logging
- [ ] Log events: impression, click, like, watch_time
- [ ] Aggregate periodic stats (ctr, avg_watch_time)

**Definition of Done**
- Ranking reads fresh user-behavior features from DB.

### 6.4 Learning-to-rank (optional phase)
- [ ] Export training dataset from logs
- [ ] Train XGBoost reg/rank model
- [ ] Compare against weighted baseline with offline metrics

**Definition of Done**
- Documented A/B decision for baseline vs learned model.

---

## 7) Web UI

### 7.1 Core search interface
- [ ] Search bar with submit + keyboard support
- [ ] Filter panel (modality/date/tags)
- [ ] Sort options (relevance/newest/popularity)

### 7.2 Result rendering
- [ ] Card/grid view with thumbnail, title, snippet, score indicators
- [ ] Media-specific preview:
  - image thumbnail
  - video preview frame and duration
  - audio waveform/icon and duration
  - text snippet highlight

### 7.3 Detail page
- [ ] Show metadata + engagement info
- [ ] Play/view content
- [ ] Emit interaction events (click/view/watch time)

**Definition of Done**
- User can search, open results, and interactions are logged.

---

## 8) Evaluation and quality assurance

### 8.1 Build evaluation dataset
- [ ] Collect representative queries
- [ ] Label relevant items (topical relevance)

### 8.2 Offline retrieval metrics
- [ ] Compute Recall@K, Precision@K, nDCG@K
- [ ] Measure per-modality performance

### 8.3 Ranking metrics
- [ ] CTR uplift proxy (offline replay if available)
- [ ] Diversity and freshness checks

### 8.4 Latency and scalability
- [ ] Measure p50/p95 for upload/index/search
- [ ] Test with increasing corpus size

**Definition of Done**
- Evaluation report in `docs/evaluation.md`.

---

## 9) Reliability, observability, and security

### 9.1 Observability
- [ ] Structured logging for API + workers
- [ ] Metrics: indexing throughput, search latency, failure rates
- [ ] Basic dashboard and alert thresholds

### 9.2 Reliability
- [ ] Idempotent indexing jobs
- [ ] Dead-letter handling for failed tasks
- [ ] Backup/restore for DB and vector store

### 9.3 Security and governance
- [ ] File scanning, MIME validation
- [ ] Auth for upload/delete endpoints
- [ ] Access control for private media

**Definition of Done**
- System can recover from common failures and has auditability.

---

## 10) Deployment

### 10.1 Containerization
- [ ] Dockerfiles for backend/frontend/worker
- [ ] Compose for local development (Postgres + Chroma + app services)

### 10.2 Environment configuration
- [ ] `.env` keys documented
- [ ] Separate dev/staging/prod configs

### 10.3 CI/CD
- [ ] Lint + test + build pipeline
- [ ] Auto migration and deployment strategy

**Definition of Done**
- One command deploy for dev; documented production path.

---

## 11) Future cache design (Redis-ready)

### 11.1 Query result caching
- [ ] Cache key design: hash(query + filters + page + model_version)
- [ ] TTL strategy and invalidation conditions

### 11.2 Embedding cache
- [ ] Cache repeated query embeddings
- [ ] Invalidate on embedding model update

**Definition of Done**
- Redis integration design doc in `docs/cache-design.md`.

---

## 12) AI execution checklist (strict order)

- [ ] A1: Create project scaffolding and docs skeleton
- [ ] A2: Implement upload + metadata persistence
- [ ] A3: Implement preprocessing for all modalities
- [ ] A4: Integrate embedding models and pooling
- [ ] A5: Write vectors to Chroma + metadata to Postgres
- [ ] A6: Implement search API and candidate retrieval
- [ ] A7: Implement ranking formula and signal normalization
- [ ] A8: Build frontend search UI and result pages
- [ ] A9: Add interaction logging and aggregation jobs
- [ ] A10: Add evaluation scripts and report generation
- [ ] A11: Add monitoring, retries, and delete consistency checks
- [ ] A12: Containerize and document deployment

**Definition of Done**
- Each A-task has:
  - code
  - tests
  - docs update
  - demo evidence (API response screenshot/log/UI capture)

---

## Suggested milestone schedule

### Milestone 1 (Week 1–2)
- A1–A3 complete

### Milestone 2 (Week 3–4)
- A4–A6 complete

### Milestone 3 (Week 5–6)
- A7–A9 complete

### Milestone 4 (Week 7–8)
- A10–A12 complete + final polish

---

## Notes for model/version control
- Track embedding model version in every indexed row.
- Re-index content when model or preprocessing changes.
- Keep reproducibility metadata: random seeds, sampling strategy, chunk sizes.
