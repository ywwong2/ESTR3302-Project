# ESTR3302 Multimedia Search Engine (Course Project)

A simple and manageable project scaffold for a multimedia search engine.

## Project Structure

- backend/: minimal API for health, upload, and search (starter)
- frontend/: simple HTML/CSS/JS UI
- docs/: planning and project notes

## What is connected now

- Frontend search uses backend `GET /search`
- Frontend upload uses backend `POST /media/upload`
- Uploaded files are served by backend at `/uploads/...`
- Like/dislike is kept simple on frontend (demo only)

## Multimedia data flow (upload → preprocess → embed → search)

This section shows the exact runtime sequence, including script and function names.

### A) Upload flow (`POST /media/upload`)

1. Frontend sends upload request
    - Script: `frontend/app.js`
    - Trigger: upload button click handler (`uploadBtn.addEventListener('click', ...)`)
    - Action: sends `FormData(file, title)` to backend endpoint `POST /media/upload`

2. Backend receives upload
    - Script: `backend/main.py`
    - Function: `upload_media()`
    - Key steps:
       - Validate filename/content type/size
       - Call `detect_media_type()` from `backend/preprocess.py`
       - Generate `media_id`
       - Save uploaded binary with `_save_upload_file()`

3. Initial metadata is stored in DB
    - Script: `backend/main.py` + `backend/db.py`
    - Function calls:
       - `insert_media(...)`
       - `update_media_status(..., status="PROCESSING")`
    - DB file: `backend/data/app.db`
    - Main table: `media_items`

4. Preprocessing is executed
    - Script: `backend/preprocess.py`
    - Function chain:
       - `run_preprocessing(media_type, file_path, out_dir)`
       - internally routes to one of:
          - `preprocess_image()`
          - `preprocess_video()`
          - `preprocess_audio()`
          - `preprocess_text()`
    - Output:
       - A preprocessing manifest JSON saved under `backend/data/preprocessed/`

5. Embedding is generated
    - Script: `backend/embeddings.py`
    - Function: `generate_embedding(...)`
    - Internal behavior:
       - resolves local model paths with `_resolve_model_source()`
       - uses modality-specific pipeline:
          - image/video via SigLIP
          - audio via CLAP
          - text via EmbeddingGemma
       - pools segment vectors (`mean` or `max`)
       - returns `vector`, `vector_id`, `model_name`, `vector_dim`, `num_segments`

6. Vector and embedding metadata are stored
    - Scripts: `backend/main.py` + `backend/db.py` + `backend/embeddings.py`
    - Function calls:
       - `save_vector_file(vector_id, vector, vector_dir)`
       - `insert_embedding_record(...)`
       - `update_media_status(..., status="INDEXED")`
    - Storage:
       - Vector JSON file: `backend/data/vectors/<vector_id>.json`
       - DB table: `embedding_records`

7. API response returned to frontend
    - Script: `backend/main.py`
    - Function: `serialize_media()` creates public `url`
    - Response payload: `{ message: "uploaded", item: ... }`

### B) Search flow (`GET /search?q=...`)

1. Frontend triggers search
    - Script: `frontend/app.js`
    - Function: `doSearch()`
    - Action: calls `GET /search?q=<query>`

2. Backend retrieves records
    - Script: `backend/main.py`
    - Function: `search()`
    - DB access function: `list_media(query=q)` from `backend/db.py`
    - Current retrieval logic:
       - SQL `LIKE` match on `title` or `stored_name`
       - excludes soft-deleted rows (`is_deleted = 0`)
       - sorted by newest (`created_at DESC`)

3. Backend returns result list
    - Script: `backend/main.py`
    - Function: `serialize_media()` converts each row into API-friendly item with `/uploads/...` URL

4. Frontend renders cards
    - Script: `frontend/app.js`
    - Functions:
       - `normalizeItem()`
       - `render()`
       - `createPreview()`

### Current storage map

- Uploaded original files: `backend/data/uploads/`
- Preprocessing manifests: `backend/data/preprocessed/`
- Embedding vector files: `backend/data/vectors/`
- SQLite database: `backend/data/app.db`
   - media metadata: `media_items`
   - embedding metadata: `embedding_records`

## Install dependencies (only inside this project)

You have 2 simple options.


### How this environment was set up

From project root:

1. Create a local Python virtual environment (in this folder):
   python3 -m venv .venv
2. Activate the environment:
   source .venv/bin/activate
3. Install backend dependencies:
   pip install -r backend/requirements.txt

This keeps all packages inside `.venv/` under this project (not global).

## Install local embedding models (required)

This project uses three local models for embedding generation:

- `google/siglip-base-patch16-224` (image/video)
- `laion/clap-htsat-unfused` (audio)
- `google/embeddinggemma-300m` (text)

### 1. Authenticate Hugging Face CLI

Create a Hugging Face access token with **read** permission, then log in:

1. Go to Hugging Face → Settings → Access Tokens
2. Create/copy token
3. Run:

   .venv/bin/hf auth login

Then verify:

   .venv/bin/hf auth whoami

### 2. Request access for gated model (one-time)

`google/embeddinggemma-300m` is gated/manual access.

Open and accept terms first:

- https://huggingface.co/google/embeddinggemma-300m

If access is not approved yet, download will fail with `403 GatedRepoError`.

### 3. Download models into local `models/` folder

From project root:

   .venv/bin/hf download google/siglip-base-patch16-224 --local-dir ./models/siglip-base-patch16-224
   .venv/bin/hf download laion/clap-htsat-unfused --local-dir ./models/clap-htsat-unfused
   .venv/bin/hf download google/embeddinggemma-300m --local-dir ./models/embeddinggemma-300m

Expected directories:

- `models/siglip-base-patch16-224/`
- `models/clap-htsat-unfused/`
- `models/embeddinggemma-300m/`

### 4. Security note

- Do **not** commit tokens to git.
- If a token is exposed, revoke it immediately and create a new one.

---

#### (Optional) Conda environment

If you prefer conda:
1. Create env:
   conda create -n estr3302-course python=3.14.2 -y
2. Activate:
   conda activate estr3302-course
3. Install backend libraries:
   pip install -r backend/requirements.txt

## Run the project


## How to run the project

### 1. Start backend API

From project root:

      source .venv/bin/activate
      uvicorn backend.main:app --reload

The API will be available at: http://127.0.0.1:8000

### 2. Open the frontend

Open `frontend/index.html`(http://127.0.0.1:3000/frontend/index.html?serverWindowId=6d887715-f282-43a6-9a27-e14477d36f0a) directly in your browser (double-click or drag into browser window).

You can now:
- Upload content
- Search content
- Click card to enlarge preview
- Like/dislike each result

## Current UI Features

- Search bar + Search button
- Upload content (image/video/audio/text file)
- Result cards
- Like and Dislike buttons
- Click card to enlarge preview in a modal

## Notes

This is intentionally kept simple for course use. You can extend it step-by-step later.
