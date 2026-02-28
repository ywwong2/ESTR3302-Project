# ESTR3302 Multimedia Search Engine (Course Project)

A course-project multimedia search engine with **semantic search** powered by
contrastive vision-language models. Upload images, audio, video, or plain text,
and search across all modalities with natural-language queries.

---

## Project Structure

```
ESTR3302-Project/
├── backend/
│   ├── main.py              # FastAPI application (all endpoints)
│   ├── db.py                # SQLite access layer (media_items + embedding_records)
│   ├── embeddings.py        # Model loading, embedding generation, vector search helpers
│   ├── preprocess.py        # Per-modality preprocessing (resize, chunk, trim)
│   ├── debug_log.py         # In-memory debug log (thread-safe, polled by frontend)
│   ├── requirements.txt     # Python dependencies
│   ├── migrations/
│   │   └── 001_init.sql     # DB schema (media_items, embedding_records)
│   ├── tests/               # pytest test suite
│   └── data/                # (auto-created at runtime)
│       ├── app.db           # SQLite database
│       ├── uploads/         # Original uploaded files
│       ├── preprocessed/    # Preprocessing manifests (JSON)
│       └── vectors/         # Stored embedding vectors (JSON)
├── frontend/
│   ├── index.html           # Main HTML page
│   ├── app.js               # All frontend logic (search, upload, render, debug panel)
│   └── styles.css           # Styles (cards, modal, metrics, debug panel)
├── models/                  # Local model weights (downloaded from Hugging Face)
│   ├── siglip-base-patch16-224/
│   └── clap-htsat-unfused/
└── docs/                    # Planning and project notes
```

## What is connected

| Frontend action | Backend endpoint | Purpose |
|---|---|---|
| Search bar | `GET /search?q=...` | Semantic vector search with ranking |
| Upload file | `POST /media/upload` | Upload image / video / audio / text file |
| Upload text | `POST /media/upload/text` | Upload raw text (no file required) |
| Like button | `POST /media/{id}/like` | Increment like counter |
| Click card | `POST /media/{id}/view` | Increment view counter |
| Info button | `GET /media/{id}` | Retrieve full metadata + embedding records |
| Delete button | `DELETE /media/{id}` | Soft-delete media item |
| Debug panel | `GET /debug/logs?since=N` | Poll in-memory debug log (1 s interval) |

## Embedding architecture

### Unified vector spaces

A critical design choice: **text queries must be embedded with the same model
that produced the stored content vectors**, so cosine similarity is meaningful.

| Content modality | Content encoder | Query encoder | Vector dim | Shared space |
|---|---|---|---|---|
| Image | SigLIP vision tower | SigLIP text tower | 768 | SigLIP |
| Video | SigLIP vision tower (per-frame) | SigLIP text tower | 768 | SigLIP |
| Text | SigLIP text tower | SigLIP text tower | 768 | SigLIP |
| Audio | CLAP audio tower | CLAP text tower | 512 | CLAP |

- **SigLIP** (`google/siglip-base-patch16-224`): A contrastive vision-language model.
  Its image encoder and text encoder are aligned during training, so cosine
  similarity between their outputs is directly meaningful.
- **CLAP** (`laion/clap-htsat-unfused`): A contrastive language-audio model.
  Same principle — its audio encoder and text encoder share a common space.

> **Important**: SigLIP requires `padding="max_length"` (64 tokens) for text
> tokenization — this is how the model was trained. Using `padding=True`
> produces broken embeddings with near-zero cosine similarity.

### Models summary

| Model | HF ID | Role | Output dim |
|---|---|---|---|
| SigLIP | `google/siglip-base-patch16-224` | Image/video/text embedding + text query embedding | 768 |
| CLAP | `laion/clap-htsat-unfused` | Audio embedding + audio-space text query embedding | 512 |

Models are loaded lazily (on first use) as singletons and cached in memory.

---

## Multimedia data flow (upload → preprocess → embed → search)

This section shows the exact runtime sequence, including script and function names.

### A) File upload flow (`POST /media/upload`)

1. **Frontend sends upload request**
    - Script: `frontend/app.js`
    - Trigger: upload button click handler (`uploadBtn.addEventListener('click', ...)`)
    - Action: sends `FormData(file, title)` to backend endpoint `POST /media/upload`

2. **Backend receives upload**
    - Script: `backend/main.py`
    - Function: `upload_media()`
    - Key steps:
       - Call `detect_media_type(content_type)` from `backend/preprocess.py`
       - Generate UUID `media_id`
       - Save uploaded binary to `backend/data/uploads/<uuid>_<filename>`

3. **Initial metadata stored in DB**
    - Script: `backend/main.py` → `backend/db.py`
    - Function: `insert_media(media)` — inserts row with status `"UPLOADED"`
    - DB table: `media_items` (14 columns: id, media_type, file_path, stored_name, title, status, error_message, is_deleted, views, likes, ctr, avg_watch_time, created_at, updated_at)

4. **Background thread launched**
    - Script: `backend/main.py`
    - Function: `_process_in_background(mid, mtype, fpath)` — runs as daemon thread
    - Status changes to `"PROCESSING"`
    - Frontend polls `GET /search` every 3 s to detect status change

5. **Step 1/3: Preprocessing**
    - Script: `backend/preprocess.py`
    - Function: `run_preprocessing(media_type, file_path, out_dir)`
    - Routes to one of:
       - `preprocess_image()` — resize to 224×224, save as JPEG
       - `preprocess_video()` — generate uniform timestamp list
       - `preprocess_audio()` — compute duration, generate chunk boundaries
       - `preprocess_text()` — trim to 4000 chars, generate preview
    - Output: preprocessing manifest JSON saved under `backend/data/preprocessed/`

6. **Step 2/3: Embedding generation**
    - Script: `backend/embeddings.py`
    - Function: `generate_embedding(media_type, file_path, preprocess_result)`
    - Routes to one of:
       - `_embed_image()` — loads SigLIP model+processor, runs `model.get_image_features()`, extracts `pooler_output[0]`
       - `_embed_video()` — applies `_embed_image()` per frame timestamp
       - `_embed_audio()` — loads CLAP model+processor, chunks audio at 48 kHz, runs `model.get_audio_features()` per chunk
       - `_embed_text()` — loads SigLIP model+processor, runs `model.get_text_features()` with `padding="max_length"`
    - Segment vectors are pooled (`mean_pool` or `max_pool`)
    - Final vector is L2-normalized
    - `vector_id` = `vec_<sha256_hash[:20]>`

7. **Step 3/3: Vector and embedding metadata stored**
    - Scripts: `backend/embeddings.py` + `backend/db.py`
    - Function calls:
       - `save_vector_file(vector_id, vector, vector_dir)` → `backend/data/vectors/<vector_id>.json`
       - `insert_embedding_record(...)` → `INSERT OR REPLACE` into `embedding_records` table
       - `update_media_status(mid, "INDEXED")`
    - DB table: `embedding_records` (media_id, modality, model_name, vector_dim, vector_id, pooling_type, num_segments, created_at)

8. **API response returned to frontend**
    - Script: `backend/main.py`
    - Response payload: `{ "message": "uploaded", "item": { ... } }`
    - Item includes `url` (for `/uploads/...` static file serving)

### B) Pure text upload flow (`POST /media/upload/text`)

1. **Frontend sends JSON request**
    - Script: `frontend/app.js`
    - Trigger: text upload button click (`textUploadBtn.addEventListener('click', ...)`)
    - Action: sends `{ "text": "...", "title": "..." }` as JSON body

2. **Backend saves as .txt file**
    - Script: `backend/main.py`
    - Function: `upload_pure_text()`
    - Writes text to `backend/data/uploads/<uuid>_text.txt`

3. **Same pipeline as file upload (steps 3–8 above)**
    - `media_type = "text"`, so `_embed_text()` uses SigLIP text encoder

### C) Search flow (`GET /search?q=...`)

1. **Frontend triggers search**
    - Script: `frontend/app.js`
    - Function: `doSearch()`
    - Action: calls `GET /search?q=<query>` (URL-encoded)

2. **Empty query → return all items**
    - If `q` is empty or whitespace, returns all non-deleted items ordered by `created_at DESC`
    - Each item gets `cosine_sim: null`, `final_score: null`

3. **Non-empty query → dual-space vector search**
    - Script: `backend/main.py` → `backend/embeddings.py`
    - **SigLIP query embedding**: `embed_query_text(q)` — always computed
       - Uses `SiglipProcessor` + `model.get_text_features()` with `padding="max_length"`
       - Returns L2-normalized 768-dim vector in SigLIP's shared space
    - **CLAP query embedding**: `embed_query_audio(q)` — lazy, only computed if audio items exist
       - Uses CLAP's `AutoProcessor` + `model.get_text_features()`
       - Returns L2-normalized 512-dim vector in CLAP's shared space

4. **Per-item scoring**
    - For each `INDEXED` item with a stored embedding:
       - Load stored vector from `backend/data/vectors/<vector_id>.json`
       - Select query vector by modality: SigLIP for image/video/text, CLAP for audio
       - Compute `cosine_similarity(query_vec, stored_vec)`
       - Compute final ranking score:

    ```
    final_score = 0.6 × cosine_sim
                + 0.1 × log(1 + views + 2 × likes)
                + 0.1 × (1 / (1 + days_since_upload))
                + 0.1 × (ctr + 0.5 × avg_watch_time)
    ```

5. **Results returned sorted by `final_score` DESC**
    - Scored items (INDEXED) sorted descending, followed by unscored items (PROCESSING / FAILED)
    - Each item includes: `cosine_sim`, `final_score`, `days_since_upload`, `views`, `likes`, `ctr`, `avg_watch_time`

6. **Frontend renders ranked result cards**
    - Script: `frontend/app.js`
    - Functions: `normalizeItem()` → `render()` → `buildMetrics()` → `createPreview()`
    - Each card shows: preview thumbnail, title, metrics row (cos_sim, score, views, likes, days, ctr, avg_watch_time), action buttons (like, info, delete)
    - Processing items show an animated overlay

### D) Engagement tracking

| Endpoint | Trigger | Effect |
|---|---|---|
| `POST /media/{id}/like` | Click 👍 button | `likes += 1` in DB |
| `POST /media/{id}/view` | Open modal preview | `views += 1` in DB |

These counters feed into the ranking formula for future searches.

### E) Debug log system

- `backend/debug_log.py` provides an in-memory thread-safe log
- All backend modules call `log(msg)` instead of `print()`
- Frontend polls `GET /debug/logs?since=N` every 1 second
- Entries are displayed in a dark terminal-style panel with color-coded messages (red for errors, green for success)

### Current storage map

| Data | Location | Format |
|---|---|---|
| Uploaded original files | `backend/data/uploads/` | Original binary |
| Preprocessing manifests | `backend/data/preprocessed/` | JSON |
| Embedding vectors | `backend/data/vectors/` | JSON (`{id, vector}`) |
| SQLite database | `backend/data/app.db` | SQLite |
| — media metadata | table `media_items` | 14 columns |
| — embedding metadata | table `embedding_records` | 9 columns |

---

## Installation

### Prerequisites

- Python 3.12+ (tested with 3.14)
- macOS / Linux (Windows should work but untested)
- ~2 GB disk space for model weights

### 1. Create and activate Python virtual environment

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

This installs:
- **fastapi** + **uvicorn** — web framework and ASGI server
- **transformers** — Hugging Face model loading (SigLIP, CLAP)
- **torch** + **torchaudio** — tensor operations and audio loading
- **sentence-transformers** — sentence embedding utilities
- **Pillow** — image preprocessing
- **sentencepiece** + **protobuf** — required by SigLIP's tokenizer
- **accelerate** + **huggingface_hub** — model downloading
- **python-multipart** — form upload parsing
- **pytest** + **httpx** — testing

### 3. Authenticate Hugging Face CLI

Create a Hugging Face access token with **read** permission:

1. Go to https://huggingface.co/settings/tokens
2. Create a token (or copy existing one)
3. Log in:

```bash
.venv/bin/hf auth login
```

Verify:

```bash
.venv/bin/hf auth whoami
```

### 4. Download models into local `models/` folder

```bash
.venv/bin/hf download google/siglip-base-patch16-224 --local-dir ./models/siglip-base-patch16-224
.venv/bin/hf download laion/clap-htsat-unfused --local-dir ./models/clap-htsat-unfused
```

Expected directories after download:

```
models/
├── siglip-base-patch16-224/   (~400 MB)
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── spiece.model
│   ├── tokenizer.json
│   └── ...
└── clap-htsat-unfused/        (~1.2 GB)
    ├── config.json
    ├── model.safetensors
    ├── preprocessor_config.json
    └── ...
```

> **Note**: EmbeddingGemma (`google/embeddinggemma-300m`) is listed in the code
> but is **not used** in the current architecture. All text is now embedded with
> SigLIP's text encoder for cross-modal alignment. You do not need to download it.

### 5. Security note

- Do **not** commit tokens to git.
- If a token is exposed, revoke it immediately and create a new one.

---

### (Optional) Conda environment

If you prefer conda:

```bash
conda create -n estr3302 python=3.14 -y
conda activate estr3302
pip install -r backend/requirements.txt
```

---

## How to run the project

### 1. Start backend API

From project root (with virtual env activated):

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

The API will be available at: http://127.0.0.1:8000

The `--reload` flag enables auto-restart when source files change.

### 2. Open the frontend

Open `frontend/index.html` directly in your browser, or use VS Code's Live
Server extension. The frontend connects to the backend at `http://127.0.0.1:8000`.

### 3. Use the application

- **Search**: Type a natural-language query (e.g. "a cute cat") and click Search.
  The query is embedded with SigLIP's text encoder and compared against all
  stored vectors using cosine similarity. Results are ranked by the scoring formula.
- **Upload file**: Select an image/video/audio/text file, optionally add a title,
  click "Upload File". Embedding runs in a background thread (~1–5 s depending on
  model cold-start).
- **Upload text**: Paste text directly into the textarea, click "Upload Text".
  No file needed.
- **Like**: Click 👍 on any card. Persistent — stored in DB and affects future
  search ranking.
- **View**: Click any card preview to open the enlarged modal. Each open counts
  as a view.
- **Debug log**: Scroll down to the debug panel to see real-time model loading
  progress, embedding status, and any errors.

## Current UI features

- Semantic search bar (embeds your query with SigLIP / CLAP)
- File upload (image / video / audio / text)
- Pure text upload (textarea, no file needed)
- Result cards with preview thumbnails
- Metrics row per card: cosine similarity, final score, views, likes, days since upload, CTR, avg watch time
- Like button (persistent, feeds into ranking)
- View counting (on modal open)
- Info button (shows full metadata + embedding records as JSON)
- Delete button (soft-delete)
- Click-to-enlarge modal preview
- Real-time debug log panel (polls every 1 s)
- Processing status overlays (queued → embedding → indexed)

## API endpoints reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/search?q=...` | Semantic search (empty q = list all) |
| `POST` | `/media/upload` | Upload file (multipart form) |
| `POST` | `/media/upload/text` | Upload raw text (JSON body) |
| `GET` | `/media/{id}` | Get full media detail + embeddings |
| `POST` | `/media/{id}/like` | Increment likes |
| `POST` | `/media/{id}/view` | Increment views |
| `DELETE` | `/media/{id}` | Soft-delete |
| `GET` | `/debug/logs?since=N` | Poll debug log entries |
| `GET` | `/uploads/{filename}` | Static file serving |

## Running tests

```bash
source .venv/bin/activate
python -m pytest backend/tests/ -v
```

## Notes

- Models are loaded lazily on first request — the first search or upload after
  server start may take 5–10 s for model loading.
- All data is stored locally (SQLite + filesystem). No external services required
  at runtime.
- The `backend/data/` directory is auto-created. To reset all data, delete it and
  restart the server.
