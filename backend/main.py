from pathlib import Path
from uuid import uuid4
import math
import threading

from fastapi import FastAPI, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.db import (
    get_all_indexed_media,
    get_embedding_record_by_media,
    get_media,
    increment_likes,
    increment_views,
    init_db,
    insert_embedding_record,
    insert_media,
    list_media,
    now_iso,
    soft_delete_media,
    update_media_status,
)
from backend.debug_log import log, get_logs
from backend.embeddings import (
    cosine_similarity,
    embed_query_audio,
    embed_query_text,
    generate_embedding,
    load_vector_file,
    save_vector_file,
)
from backend.preprocess import detect_media_type, run_preprocessing

app = FastAPI(title="Course Multimedia Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

PREPROCESS_DIR = Path(__file__).parent / "data" / "preprocessed"
PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_DIR = Path(__file__).parent / "data" / "vectors"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def _process_in_background(mid: str, mtype: str, fpath: Path) -> None:
    """Preprocess + embed a media item. Runs in a daemon thread."""
    import time as _t, traceback as _tb
    log(f"▶ Starting background processing: id={mid}, type={mtype}, file={fpath.name}")
    try:
        update_media_status(mid, "PROCESSING")

        log(f"Step 1/3: Preprocessing {mtype}...")
        t0 = _t.time()
        preprocess_result = run_preprocessing(mtype, fpath, PREPROCESS_DIR)
        log(f"Step 1/3: Preprocessing done in {_t.time()-t0:.1f}s")

        log(f"Step 2/3: Generating embedding (loading model if needed)...")
        t1 = _t.time()
        embedding_result = generate_embedding(mtype, fpath, preprocess_result)
        log(f"Step 2/3: Embedding done in {_t.time()-t1:.1f}s")

        log(f"Step 3/3: Saving vector + DB record...")
        save_vector_file(embedding_result["vector_id"], embedding_result["vector"], VECTOR_DIR)

        insert_embedding_record({
            "media_id": mid,
            "modality": embedding_result["modality"],
            "model_name": embedding_result["model_name"],
            "vector_dim": embedding_result["vector_dim"],
            "vector_id": embedding_result["vector_id"],
            "pooling_type": embedding_result["pooling_type"],
            "num_segments": embedding_result["num_segments"],
            "created_at": now_iso(),
        })
        update_media_status(mid, "INDEXED")
        log(f"✅ DONE: {fpath.name} → INDEXED")
    except Exception as exc:
        log(f"❌ FAILED: {fpath.name} → {exc}")
        log(_tb.format_exc())
        update_media_status(mid, "FAILED", error_message=str(exc))


@app.post("/media/upload")
async def upload_media(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(""),
) -> dict:
    try:
        media_type = detect_media_type(file.content_type or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data = await file.read()

    media_id = str(uuid4())
    safe_name = f"{media_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_name
    file_path.write_bytes(data)
    log(f"📥 Upload received: {file.filename} ({media_type}, {len(data)} bytes)")

    media = {
        "id": media_id,
        "media_type": media_type,
        "title": title or file.filename,
        "stored_name": safe_name,
        "file_path": str(file_path),
        "status": "UPLOADED",
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    insert_media(media)

    threading.Thread(
        target=_process_in_background,
        args=(media_id, media_type, file_path),
        daemon=True,
    ).start()

    media["content_type"] = file.content_type or "application/octet-stream"
    media["url"] = str(request.url_for("uploads", path=safe_name))
    return {"message": "uploaded", "item": media}


@app.post("/media/upload/text")
async def upload_pure_text(
    request: Request,
    text: str = Body(..., embed=True),
    title: str = Body("", embed=True),
) -> dict:
    """Upload raw text (no file needed). Saves as .txt and runs embedding pipeline."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    media_id = str(uuid4())
    safe_name = f"{media_id}_text.txt"
    file_path = UPLOAD_DIR / safe_name
    file_path.write_text(text, encoding="utf-8")
    log(f"📥 Pure-text upload: '{(title or 'untitled')[:40]}' ({len(text)} chars)")

    media = {
        "id": media_id,
        "media_type": "text",
        "title": title.strip() or f"Text ({len(text)} chars)",
        "stored_name": safe_name,
        "file_path": str(file_path),
        "status": "UPLOADED",
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    insert_media(media)

    threading.Thread(
        target=_process_in_background,
        args=(media_id, "text", file_path),
        daemon=True,
    ).start()

    media["content_type"] = "text/plain"
    media["url"] = str(request.url_for("uploads", path=safe_name))
    return {"message": "uploaded", "item": media}


@app.get("/search")
def search(request: Request, q: str = "") -> dict:
    """
    If q is empty → return all items ordered by created_at DESC.
    If q is given → embed the query, compute cosine similarity with each
    indexed item, then rank by:
      final_score = 0.6*cosine_sim + 0.1*log(1+views+2*likes)
                  + 0.1*(1/(1+days_since_upload))
                  + 0.1*(ctr + 0.5*avg_watch_time)
    Non-indexed items are appended at the end (still processing / failed).
    """
    from datetime import datetime, timezone

    q = (q or "").strip()

    # No query → return everything (title-filtered if needed)
    if not q:
        rows = list_media(query="")
        for row in rows:
            row["content_type"] = f"{row['media_type']}/*"
            row["url"] = str(request.url_for("uploads", path=row["stored_name"]))
            row["cosine_sim"] = None
            row["final_score"] = None
            _attach_days(row)
        return {"items": rows}

    # Embed the query with SigLIP text encoder (for image/video/text matching)
    log(f"🔎 Search query: '{q}'")
    query_vec_siglip = embed_query_text(q)
    query_vec_clap = None  # lazy: only embed with CLAP if audio items exist

    # Retrieve all non-deleted items
    all_items = list_media(query="")
    scored = []
    unscored = []

    now = datetime.now(timezone.utc)

    for item in all_items:
        item["content_type"] = f"{item['media_type']}/*"
        item["url"] = str(request.url_for("uploads", path=item["stored_name"]))

        # Only score INDEXED items that have an embedding
        emb_rec = get_embedding_record_by_media(item["id"])
        if item["status"] != "INDEXED" or not emb_rec:
            item["cosine_sim"] = None
            item["final_score"] = None
            _attach_days(item)
            unscored.append(item)
            continue

        # Load stored vector
        stored_vec = load_vector_file(emb_rec["vector_id"], VECTOR_DIR)
        if stored_vec is None:
            item["cosine_sim"] = None
            item["final_score"] = None
            _attach_days(item)
            unscored.append(item)
            continue

        # Pick the right query embedding for this item's modality
        if item["media_type"] == "audio":
            # CLAP text encoder → same space as CLAP audio embeddings
            if query_vec_clap is None:
                query_vec_clap = embed_query_audio(q)
            query_vec = query_vec_clap
        else:
            # SigLIP text encoder → same space as SigLIP image/text embeddings
            query_vec = query_vec_siglip

        cos_sim = cosine_similarity(query_vec, stored_vec)

        views = item.get("views", 0)
        likes = item.get("likes", 0)
        ctr = item.get("ctr", 0.0)
        avg_wt = item.get("avg_watch_time", 0.0)

        # Days since upload
        try:
            created = datetime.fromisoformat(item["created_at"])
            days_since = max((now - created).total_seconds() / 86400, 0)
        except Exception:
            days_since = 0

        final_score = (
            0.6 * cos_sim
            + 0.1 * math.log(1 + views + 2 * likes)
            + 0.1 * (1 / (1 + days_since))
            + 0.1 * (ctr + 0.5 * avg_wt)
        )

        item["cosine_sim"] = round(cos_sim, 4)
        item["final_score"] = round(final_score, 4)
        item["days_since_upload"] = round(days_since, 2)
        scored.append(item)

    # Sort scored items by final_score descending
    scored.sort(key=lambda x: x["final_score"], reverse=True)
    log(f"🔎 Ranked {len(scored)} items, {len(unscored)} unscored")

    return {"items": scored + unscored}


def _attach_days(item: dict) -> None:
    """Attach days_since_upload to an item dict."""
    from datetime import datetime, timezone
    try:
        created = datetime.fromisoformat(item["created_at"])
        item["days_since_upload"] = round(
            max((datetime.now(timezone.utc) - created).total_seconds() / 86400, 0), 2
        )
    except Exception:
        item["days_since_upload"] = 0


@app.post("/media/{media_id}/like")
def like_media(media_id: str) -> dict:
    item = get_media(media_id)
    if not item:
        raise HTTPException(status_code=404, detail="Media not found")
    result = increment_likes(media_id)
    return {"likes": result["likes"] if result else 0}


@app.post("/media/{media_id}/view")
def view_media(media_id: str) -> dict:
    item = get_media(media_id)
    if not item:
        raise HTTPException(status_code=404, detail="Media not found")
    increment_views(media_id)
    return {"ok": True}


@app.get("/media/{media_id}")
def get_media_detail(media_id: str, request: Request) -> dict:
    item = get_media(media_id)
    if not item:
        raise HTTPException(status_code=404, detail="Media not found")
    item["url"] = str(request.url_for("uploads", path=item["stored_name"]))
    # Attach embedding records if any
    from backend.db import get_conn
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM embedding_records WHERE media_id = ?", (media_id,)
    ).fetchall()
    conn.close()
    item["embeddings"] = [dict(r) for r in rows]
    return item


@app.get("/debug/logs")
def debug_logs(since: int = 0) -> dict:
    """Return debug log entries for the frontend debug panel."""
    entries = get_logs(since)
    return {"entries": entries}


@app.delete("/media/{media_id}")
def delete_media(media_id: str) -> dict:
    item = get_media(media_id)
    if not item:
        raise HTTPException(status_code=404, detail="Media not found")

    file_path = Path(item["file_path"])
    if file_path.exists():
        file_path.unlink()

    soft_delete_media(media_id)
    return {"message": "deleted"}
