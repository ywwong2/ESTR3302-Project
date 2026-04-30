"""
FastAPI backend for the Image Ranking Simulation GUI.

Storage: JSON files in backend/data/frontend/ (no SQLite).
Simulation: background thread running the same two-stage ranking + policy
            gradient logic from simulate_ranking.py, operating on gallery.json.
"""
from __future__ import annotations

import json
import math
import random
import threading
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.debug_log import log, get_logs
from backend.embeddings import (
    cosine_similarity,
    embed_image,
    embed_query_text,
    compute_cosines_for_image,
    detect_category,
    QUERY_LABELS,
)

# ── Paths ─────────────────────────────────────────────────────
_BASE = Path(__file__).parent
_FRONTEND_DIR = _BASE / "data" / "frontend"
_GALLERY_DIR = _FRONTEND_DIR / "gallery"
_GALLERY_JSON = _FRONTEND_DIR / "gallery.json"
_SIM_STATE_JSON = _FRONTEND_DIR / "sim_state.json"
_QUERY_CACHE_JSON = _FRONTEND_DIR / "query_cache.json"
_CATEGORIES_JSON = _FRONTEND_DIR / "categories.json"

_EXPERIMENT_DIR = _BASE / "data" / "exp_bayesian_defended"
_DATASET_CACHE = _BASE.parent / "dataset" / "cosine_cache_siglip.json"

# ── Gallery helpers ────────────────────────────────────────────

_gallery_lock = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_gallery() -> list[dict]:
    if _GALLERY_JSON.exists():
        return json.loads(_GALLERY_JSON.read_text(encoding="utf-8"))
    return []


def _save_gallery(items: list[dict]) -> None:
    _GALLERY_JSON.write_text(json.dumps(items, indent=2), encoding="utf-8")


_CATEGORIES_LOCK = threading.Lock()


def _load_categories() -> list[str]:
    if _CATEGORIES_JSON.exists():
        return json.loads(_CATEGORIES_JSON.read_text(encoding="utf-8"))
    return list(QUERY_LABELS)


def _save_categories(cats: list[str]) -> None:
    _CATEGORIES_JSON.write_text(json.dumps(cats, indent=2), encoding="utf-8")


def _gallery_get(image_id: str) -> dict | None:
    for item in _load_gallery():
        if item["id"] == image_id:
            return item
    return None


def _gallery_update(image_id: str, **kwargs: Any) -> None:
    items = _load_gallery()
    for item in items:
        if item["id"] == image_id:
            item.update(kwargs)
    _save_gallery(items)


# ── Simulation state helpers ───────────────────────────────────

_DEFAULT_SIM_STATE: dict = {
    "round": 0,
    "running": False,
    "interval": 5,
    "pool_mode": None,
    "weights": [0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
}

_sim_lock = threading.Lock()
_sim_thread: threading.Thread | None = None


def _load_sim_state() -> dict:
    if _SIM_STATE_JSON.exists():
        return json.loads(_SIM_STATE_JSON.read_text(encoding="utf-8"))
    return dict(_DEFAULT_SIM_STATE)


def _save_sim_state(state: dict) -> None:
    _SIM_STATE_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── Simulation math (mirrors simulate_ranking.py) ─────────────

_SIM_PARAMS = {
    "tau": 0.40, "eta": 0.004, "beta": 0.005,
    "rho": 0.20, "c_ctr": 10.0, "m_ctr": 0.10,
    "c_lr": 10.0, "m_lr": 0.05, "omega_l": 1.5,
    "gamma_pos": 0.90,
    "alpha": [-1.40, 1.40, 3.40, 0.05, 0.10], "alpha_q": 0.30,
    "gamma": [-0.90, 0.25, 0.90, 0.08, 0.05], "gamma_q": 4.50,
    "kappa": 12,
    "delta": [-2.60, 0.05, 0.25, 0.03, 0.05], "delta_t": 3.00, "delta_q": 4.00,
    "lambda_v": 1.2, "lambda_t": 2.2, "lambda_l": 1.2,
    "top_k": 10, "two_stage_top_k": 13, "batch_size": 30,
    "qw_cos": 1.0, "qw_match": 2.0, "qw_fresh": 0.5,
    "qw_ctr": 0.5, "qw_lr": 5.0, "qw_awt": 2.0,
    "min_fresh_weight": 0.05,
    "freshness_decay_scale": 5.0,
}


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(vals: list[float]) -> list[float]:
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps)
    return [v / s for v in exps] if s > 0 else [1.0 / len(vals)] * len(vals)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _project_simplex(v: list[float]) -> list[float]:
    n = len(v)
    u = sorted(v, reverse=True)
    cssv = 0.0
    rho = -1
    for i, ui in enumerate(u):
        cssv += ui
        if ui - (cssv - 1.0) / (i + 1) > 0:
            rho = i
    if rho == -1:
        return [1.0 / n] * n
    theta = (sum(u[:rho + 1]) - 1.0) / (rho + 1)
    w = [max(x - theta, 0.0) for x in v]
    s = sum(w)
    return [x / s for x in w] if s > 0 else [1.0 / n] * n


def _run_one_round(
    items: list[dict],
    w: list[float],
    rng: random.Random,
    p: dict,
    real_user_feedback: dict[str, dict] | None = None,
) -> tuple[list[dict], list[float], float]:
    """Run one simulation round. Returns (updated items, updated weights, avg_reward)."""
    n = len(items)
    if n == 0:
        return items, w, 0.0

    t_now = p.get("t", 1)
    scale = p["freshness_decay_scale"]
    for img in items:
        age = t_now - img.get("arrival_time", 0)
        img["freshness"] = 2.0 ** (-age / scale)

    total_reward = 0.0
    grad = [0.0] * 6

    users_per_round = p["batch_size"]
    cats = _load_categories()
    num_query_cats = len(cats)

    all_interactions = list(range(users_per_round))
    if real_user_feedback:
        all_interactions.append("real_user")

    for uid in all_interactions:
        is_real = uid == "real_user"
        q_label = cats[rng.randrange(num_query_cats)]
        q_cat = cats.index(q_label)

        cos_vals = [float(img.get("cosines", {}).get(q_label, 0.0)) for img in items]

        stage1 = sorted(range(n), key=lambda i: cos_vals[i], reverse=True)[:p["two_stage_top_k"]]

        stage2_scores = []
        for idx in stage1:
            img = items[idx]
            match = 1.0 if img.get("category_idx", 0) == q_cat else 0.0
            qs = (
                p["qw_cos"] * cos_vals[idx] +
                p["qw_match"] * match +
                p["qw_fresh"] * img.get("freshness", 0.5) +
                p["qw_ctr"] * img.get("ctr", p["m_ctr"]) +
                p["qw_lr"] * img.get("lr", p["m_lr"]) +
                p["qw_awt"] * img.get("awt", 0.0)
            )
            stage2_scores.append(qs)

        stage2_ranked = sorted(range(len(stage1)), key=lambda i: stage2_scores[i], reverse=True)
        displayed_local = stage2_ranked[:p["top_k"]]
        displayed = [stage1[i] for i in displayed_local]

        x_vecs = []
        for img in items:
            cos_ui = float(img.get("cosines", {}).get(q_label, 0.0))
            x_vecs.append([cos_ui, img.get("freshness", 0.5), img.get("lr", p["m_lr"]),
                           img.get("ctr", p["m_ctr"]), img.get("awt", 0.0), img.get("social_proof", 0.0)])

        policy_scores = [0.0] * n
        for li, gi in enumerate(stage1):
            policy_scores[gi] = stage2_scores[li]

        logits = [p["tau"] * s for s in policy_scores]
        a = _softmax(logits)
        rewards = [0.0] * n

        user_sigma = p.get("user_sigma", 0.0)
        xi_view = rng.gauss(0.0, user_sigma) if user_sigma > 0 else 0.0
        xi_like = rng.gauss(0.0, user_sigma) if user_sigma > 0 else 0.0

        for rank, idx in enumerate(displayed):
            img = items[idx]
            if is_real and real_user_feedback:
                fb = real_user_feedback.get(img["id"], {})
                view = float(fb.get("view", 0))
                like = float(fb.get("like", 0))
                watch = float(fb.get("watch", 0.3 * view))
            else:
                h = [1.0, 1.0 if img.get("category_idx", 0) == q_cat else 0.0,
                     cos_vals[idx], img.get("freshness", 0.5), img.get("social_proof", 0.0)]
                q_val = img.get("quality", 0.3)
                eta_r = (1.0 + rank) ** (-p["gamma_pos"])
                examined = 1.0 if rng.random() < eta_r else 0.0
                if examined == 0.0:
                    continue
                view = 1.0 if rng.random() < _sigmoid(_dot(p["alpha"], h) + p["alpha_q"] * q_val + xi_view) else 0.0
                watch = 0.0
                like = 0.0
                if view:
                    mu = _sigmoid(_dot(p["gamma"], h) + p["gamma_q"] * q_val)
                    a_b = max(p["kappa"] * mu, 1e-6)
                    b_b = max(p["kappa"] * (1 - mu), 1e-6)
                    watch = rng.betavariate(a_b, b_b)
                    like = 1.0 if rng.random() < _sigmoid(_dot(p["delta"], h) + p["delta_t"] * watch + p["delta_q"] * q_val + xi_like) else 0.0

            r_ui = p["lambda_v"] * view + p["lambda_t"] * watch + p["lambda_l"] * like
            rewards[idx] = r_ui
            total_reward += r_ui

            agg = img.setdefault("_agg", {"e": 0.0, "v": 0.0, "l": 0.0, "ws": 0.0, "wc": 0.0})
            agg["e"] += 1.0
            agg["v"] += view
            agg["l"] += like
            if view:
                agg["ws"] += watch
                agg["wc"] += 1.0

        exp_r = sum(a[i] * rewards[i] for i in range(n))
        for idx in displayed:
            coeff = -p["tau"] * a[idx] * (rewards[idx] - exp_r)
            x = x_vecs[idx]
            for j in range(6):
                grad[j] += coeff * x[j]

    bs = max(users_per_round, 1)
    grad = [g / bs + 2.0 * p["beta"] * wj for g, wj in zip(grad, w)]
    w = _project_simplex([wj - p["eta"] * gj for wj, gj in zip(w, grad)])
    w = [max(wj, 0.0) for wj in w]
    w[1] = max(w[1], p["min_fresh_weight"])

    rho = p["rho"]
    for img in items:
        agg = img.pop("_agg", {"e": 0.0, "v": 0.0, "l": 0.0, "ws": 0.0, "wc": 0.0})
        te, tv, tl = agg["e"], agg["v"], agg["l"]
        img["tilde_e"] = (1 - rho) * img.get("tilde_e", 0.0) + rho * te
        img["tilde_v"] = (1 - rho) * img.get("tilde_v", 0.0) + rho * tv
        img["tilde_l"] = (1 - rho) * img.get("tilde_l", 0.0) + rho * tl
        img["ctr"] = (img["tilde_v"] + p["c_ctr"] * p["m_ctr"]) / (img["tilde_e"] + p["c_ctr"])
        img["lr"] = (img["tilde_l"] + p["c_lr"] * p["m_lr"]) / (img["tilde_v"] + p["c_lr"])
        mw = agg["ws"] / agg["wc"] if agg["wc"] > 0 else 0.0
        img["awt"] = (1 - rho) * img.get("awt", 0.0) + rho * mw
        img["total_views"] = img.get("total_views", 0) + round(tv)
        img["total_likes"] = img.get("total_likes", 0) + round(tl)

    return items, w, total_reward / max(bs, 1)


# ── Background simulation thread ──────────────────────────────

def _sim_loop() -> None:
    rng = random.Random(42)
    log("[Sim] Background simulation thread started")
    while True:
        state = _load_sim_state()
        if not state.get("running"):
            time.sleep(0.5)
            continue

        interval = max(float(state.get("interval", 5)), 1.0)
        time.sleep(interval)

        state = _load_sim_state()
        if not state.get("running"):
            continue

        with _gallery_lock:
            items = _load_gallery()
            if not items:
                continue
            w = state.get("weights", _DEFAULT_SIM_STATE["weights"])
            p = dict(_SIM_PARAMS)
            p["t"] = state.get("round", 0) + 1
            custom = state.get("custom_params", {})
            if custom:
                p.update(custom)

            pending_fb = state.pop("pending_feedback", {})
            updated_items, w, avg_r = _run_one_round(items, w, rng, p, real_user_feedback=pending_fb if pending_fb else None)
            _save_gallery(updated_items)

        state["round"] = p["t"]
        state["weights"] = w
        state["last_reward"] = round(avg_r, 4)
        state.pop("pending_feedback", None)
        _save_sim_state(state)
        log(f"[Sim] Round {p['t']} done, reward={avg_r:.4f}, w={[round(x,3) for x in w]}")


# ── FastAPI app ────────────────────────────────────────────────

app = FastAPI(title="Image Ranking Simulation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DATASET_IMAGES_DIR = _BASE.parent / "dataset"
_GALLERY_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(_GALLERY_DIR)), name="uploads")
app.mount("/dataset", StaticFiles(directory=str(_DATASET_IMAGES_DIR)), name="dataset")


@app.on_event("startup")
def on_startup() -> None:
    _FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    _GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    if not _GALLERY_JSON.exists():
        _save_gallery([])
    if not _SIM_STATE_JSON.exists():
        _save_sim_state(dict(_DEFAULT_SIM_STATE))
    global _sim_thread
    _sim_thread = threading.Thread(target=_sim_loop, daemon=True)
    _sim_thread.start()
    log("✅ Backend started. Simulation thread running.")


# ── Image CRUD ─────────────────────────────────────────────────

@app.get("/images")
def list_images(request: Request, q: str = "") -> dict:
    """List all images. If q given, rank by cosine similarity + quality score."""
    with _gallery_lock:
        items = _load_gallery()
    state = _load_sim_state()
    w = state.get("weights", _DEFAULT_SIM_STATE["weights"])

    q = (q or "").strip().lower()

    for item in items:
        if item.get("is_dataset") and item.get("relative_path"):
            item["url"] = str(request.base_url) + "dataset/" + item["relative_path"]
        else:
            item["url"] = str(request.url_for("uploads", path=item["filename"])) if item.get("filename") else ""

    if not q:
        for i, item in enumerate(items):
            item["rank"] = i + 1
        return {"items": items, "weights": w, "round": state.get("round", 0)}

    q_vec = embed_query_text(q, cache_path=_QUERY_CACHE_JSON)
    cats = _load_categories()
    q_cat_idx = cats.index(q) if q in cats else -1

    # Stage 1: score all by cosine, keep top-K
    for item in items:
        cos_ui = float(item.get("cosines", {}).get(q, 0.0))
        if cos_ui == 0.0 and item.get("vector"):
            cos_ui = cosine_similarity(item["vector"], q_vec)
        item["_cos"] = cos_ui

    two_stage_k = _SIM_PARAMS["two_stage_top_k"]
    items_by_cos = sorted(items, key=lambda x: x["_cos"], reverse=True)
    stage1 = items_by_cos[:two_stage_k]
    rest   = items_by_cos[two_stage_k:]

    # Stage 2: re-rank stage1 by quality score (with match bonus)
    qw_cos = float(state.get("custom_params", {}).get("qw_cos", _SIM_PARAMS["qw_cos"]))
    qw_match = float(state.get("custom_params", {}).get("qw_match", _SIM_PARAMS["qw_match"]))
    qw_fresh = float(state.get("custom_params", {}).get("qw_fresh", _SIM_PARAMS["qw_fresh"]))
    qw_ctr = float(state.get("custom_params", {}).get("qw_ctr", _SIM_PARAMS["qw_ctr"]))
    qw_lr = float(state.get("custom_params", {}).get("qw_lr", _SIM_PARAMS["qw_lr"]))
    qw_awt = float(state.get("custom_params", {}).get("qw_awt", _SIM_PARAMS["qw_awt"]))
    for item in stage1:
        cos_ui = item["_cos"]
        match = 1.0 if item.get("category_idx", -1) == q_cat_idx else 0.0
        qs = (
            qw_cos   * cos_ui +
            qw_match * match +
            qw_fresh * item.get("freshness", 0.5) +
            qw_ctr   * item.get("ctr", 0.1) +
            qw_lr    * item.get("lr", 0.05) +
            qw_awt   * item.get("awt", 0.0)
        )
        item["query_score"] = round(qs, 4)
        item["cosine_sim"]  = round(cos_ui, 4)

    stage1.sort(key=lambda x: x["query_score"], reverse=True)
    for i, item in enumerate(stage1):
        item["rank"] = i + 1
    for i, item in enumerate(rest):
        item["rank"] = two_stage_k + i + 1

    return {"items": stage1 + rest, "weights": w, "round": state.get("round", 0)}


@app.post("/images/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(""),
    quality: float = Form(0.3),
) -> dict:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    data = await file.read()
    image_id = str(uuid4())
    safe_name = f"{image_id}_{file.filename}"
    file_path = _GALLERY_DIR / safe_name
    file_path.write_bytes(data)
    log(f"📥 Uploaded: {file.filename} ({len(data)} bytes)")

    item: dict = {
        "id": image_id,
        "title": title or file.filename,
        "filename": safe_name,
        "status": "PROCESSING",
        "category": "unknown",
        "category_idx": 0,
        "quality": round(max(0.0, min(1.0, quality)), 4),
        "cosines": {},
        "vector": [],
        "freshness": 1.0,
        "arrival_time": _load_sim_state().get("round", 0),
        "tilde_e": 0.0, "tilde_v": 0.0, "tilde_l": 0.0,
        "ctr": 0.1, "lr": 0.05, "awt": 0.0, "social_proof": 0.0,
        "created_at": _now(),
    }

    with _gallery_lock:
        items = _load_gallery()
        items.append(item)
        _save_gallery(items)

    def _bg(iid: str, fpath: Path, sname: str) -> None:
        try:
            vec = embed_image(fpath)
            cats = _load_categories()
            cosines: dict[str, float] = {}
            for cat in cats:
                q_vec = embed_query_text(cat, cache_path=_QUERY_CACHE_JSON)
                cosines[cat] = round(cosine_similarity(vec, q_vec), 6)
            best_cat = max(cosines, key=cosines.get) if cosines else "unknown"
            cat_idx = cats.index(best_cat) if best_cat in cats else 0
            with _gallery_lock:
                imgs = _load_gallery()
                for img in imgs:
                    if img["id"] == iid:
                        img["vector"] = vec
                        img["cosines"] = cosines
                        img["category"] = best_cat
                        img["category_idx"] = cat_idx
                        img["status"] = "INDEXED"
                        break
                _save_gallery(imgs)
            log(f"✅ Indexed: {sname} → category={best_cat}")
        except Exception as exc:
            log(f"❌ Embedding failed for {sname}: {exc}")
            with _gallery_lock:
                imgs = _load_gallery()
                for img in imgs:
                    if img["id"] == iid:
                        img["status"] = "FAILED"
                        break
                _save_gallery(imgs)

    threading.Thread(target=_bg, args=(image_id, file_path, safe_name), daemon=True).start()
    item["url"] = str(request.url_for("uploads", path=safe_name))
    return {"message": "uploaded", "item": item}


@app.delete("/images/{image_id}")
def delete_image(image_id: str) -> dict:
    with _gallery_lock:
        items = _load_gallery()
        target = next((it for it in items if it["id"] == image_id), None)
        if not target:
            raise HTTPException(status_code=404, detail="Image not found")
        if target.get("filename"):
            file_path = _GALLERY_DIR / target["filename"]
            if file_path.exists():
                file_path.unlink()
        items = [it for it in items if it["id"] != image_id]
        _save_gallery(items)
    log(f"🗑️ Deleted image {image_id}")
    return {"message": "deleted"}


# ── User interaction ───────────────────────────────────────────

@app.post("/interact")
def record_interaction(body: dict) -> dict:
    """Record a real-user view/like. Queued into next simulation round."""
    image_id = body.get("image_id", "")
    action = body.get("action", "view")

    with _sim_lock:
        state = _load_sim_state()
        fb = state.setdefault("pending_feedback", {})
        entry = fb.setdefault(image_id, {"view": 0, "like": 0, "watch": 0.0})
        if action == "examine":
            entry["examine"] = entry.get("examine", 0) + 1
        elif action == "view":
            entry["view"] = 1
            entry["examine"] = entry.get("examine", 0) + 1
            entry["watch"] = max(entry["watch"], 0.3)
        elif action == "like":
            entry["like"] = 1
            entry["view"] = 1
            entry["examine"] = entry.get("examine", 0) + 1
            entry["watch"] = max(entry["watch"], 0.6)
        _save_sim_state(state)

    # Update display counters in gallery (separate from sim feedback)
    if action in ("view", "like"):
        with _gallery_lock:
            imgs = _load_gallery()
            for img in imgs:
                if img["id"] == image_id:
                    if action == "view":
                        img["total_views"] = img.get("total_views", 0) + 1
                    elif action == "like":
                        img["total_likes"] = img.get("total_likes", 0) + 1
                        img["total_views"] = img.get("total_views", 0) + 1
                    break
            _save_gallery(imgs)

    return {"ok": True}


# ── Search ─────────────────────────────────────────────────────

@app.get("/search")
def search(request: Request, q: str = "") -> dict:
    return list_images(request, q=q)


@app.get("/categories")
def get_categories() -> dict:
    return {"categories": _load_categories()}


@app.post("/categories")
def add_category(body: dict) -> dict:
    cat = body.get("category", "").strip().lower()
    if not cat:
        raise HTTPException(status_code=400, detail="Category name required")
    with _CATEGORIES_LOCK:
        cats = _load_categories()
        if cat in cats:
            return {"ok": True, "categories": cats}
        # Embed and cache the new category text so future searches can use it
        try:
            embed_query_text(cat, cache_path=_QUERY_CACHE_JSON)
        except Exception:
            pass
        cats.append(cat)
        _save_categories(cats)
        log(f"[Categories] Added '{cat}'")
    return {"ok": True, "categories": cats}


# ── Simulation control ─────────────────────────────────────────

@app.get("/sim/status")
def sim_status() -> dict:
    state = _load_sim_state()
    return {
        "round": state.get("round", 0),
        "running": state.get("running", False),
        "interval": state.get("interval", 5),
        "pool_mode": state.get("pool_mode"),
        "weights": state.get("weights", _DEFAULT_SIM_STATE["weights"]),
    }


@app.post("/sim/init")
def sim_init(body: dict) -> dict:
    """Initialise image pool. mode='empty' or mode='dataset'."""
    mode = body.get("mode", "empty")
    with _gallery_lock:
        if mode == "dataset" and _DATASET_CACHE.exists():
            cache = json.loads(_DATASET_CACHE.read_text(encoding="utf-8"))
            rng = random.Random(42)
            new_items = []
            for idx, raw in enumerate(cache.get("images", [])):
                cat = str(raw.get("category", "cat")).lower()
                cat_idx = QUERY_LABELS.index(cat) if cat in QUERY_LABELS else 0
                cosines = {str(k): float(v) for k, v in raw.get("cosine_by_query", {}).items()}
                rel_path = raw.get("relative_path", "")
                new_items.append({
                    "id": f"ds_{idx}",
                    "title": rel_path.split("/")[-1] if rel_path else f"{cat} #{idx}",
                    "filename": "",
                    "relative_path": rel_path,
                    "status": "INDEXED",
                    "category": cat,
                    "category_idx": cat_idx,
                    "quality": rng.betavariate(2.0, 5.0),
                    "cosines": cosines,
                    "vector": [],
                    "freshness": rng.uniform(0.2, 1.0),
                    "arrival_time": 0,
                    "tilde_e": 0.0, "tilde_v": 0.0, "tilde_l": 0.0,
                    "ctr": 0.1, "lr": 0.05, "awt": 0.0, "social_proof": 0.0,
                    "total_views": 0, "total_likes": 0,
                    "created_at": _now(),
                    "is_dataset": True,
                })
            _save_gallery(new_items)
            log(f"[Sim] Pool initialised with {len(new_items)} dataset images")
        else:
            _save_gallery([])
            log("[Sim] Pool initialised empty")

    state = _load_sim_state()
    state["pool_mode"] = mode
    state["round"] = 0
    state["weights"] = list(_DEFAULT_SIM_STATE["weights"])
    state["running"] = False
    _save_sim_state(state)
    return {"ok": True, "mode": mode}


@app.post("/sim/toggle")
def sim_toggle() -> dict:
    state = _load_sim_state()
    state["running"] = not state.get("running", False)
    _save_sim_state(state)
    log(f"[Sim] {'Started' if state['running'] else 'Paused'}")
    return {"running": state["running"]}


@app.post("/sim/set_interval")
def sim_set_interval(body: dict) -> dict:
    interval = max(float(body.get("interval", 5)), 1.0)
    state = _load_sim_state()
    state["interval"] = interval
    _save_sim_state(state)
    return {"interval": interval}


@app.post("/sim/step")
def sim_step() -> dict:
    """Manually run one simulation round (works even if paused)."""
    rng = random.Random()
    with _gallery_lock:
        items = _load_gallery()
        if not items:
            return {"ok": False, "reason": "No images in pool"}
        state = _load_sim_state()
        w = state.get("weights", _DEFAULT_SIM_STATE["weights"])
        p = dict(_SIM_PARAMS)
        p["t"] = state.get("round", 0) + 1
        custom = state.get("custom_params", {})
        if custom:
            p.update(custom)
        pending_fb = state.pop("pending_feedback", {})
        updated, w, avg_r = _run_one_round(items, w, rng, p, real_user_feedback=pending_fb if pending_fb else None)
        _save_gallery(updated)

    state["round"] = p["t"]
    state["weights"] = w
    state["last_reward"] = round(avg_r, 4)
    state.pop("pending_feedback", None)
    _save_sim_state(state)
    log(f"[Sim] Manual step round {p['t']}, reward={avg_r:.4f}")
    return {"ok": True, "round": p["t"], "reward": avg_r}


@app.post("/sim/set_params")
def sim_set_params(body: dict) -> dict:
    """Persist custom behaviour parameters and ranking weights."""
    allowed = {"alpha_q", "gamma_q", "delta_q", "batch_size", "user_sigma",
               "qw_cos", "qw_match", "qw_fresh", "qw_ctr", "qw_lr", "qw_awt"}
    state = _load_sim_state()
    custom = state.setdefault("custom_params", {})
    for k, v in body.items():
        if k in allowed:
            custom[k] = float(v) if k != "batch_size" else int(v)
    _save_sim_state(state)
    return {"ok": True, "custom_params": custom}


@app.post("/sim/reset")
def sim_reset() -> dict:
    state = _load_sim_state()
    state["round"] = 0
    state["running"] = False
    state["weights"] = list(_DEFAULT_SIM_STATE["weights"])
    state.pop("pending_feedback", None)
    _save_sim_state(state)
    with _gallery_lock:
        items = _load_gallery()
        for img in items:
            img.update({"tilde_e": 0.0, "tilde_v": 0.0, "tilde_l": 0.0,
                        "ctr": 0.1, "lr": 0.05, "awt": 0.0, "social_proof": 0.0,
                        "freshness": 1.0})
        _save_gallery(items)
    log("[Sim] Reset to round 0")
    return {"ok": True}


# ── Preset (experiment results) ────────────────────────────────

@app.get("/sim/preset")
def sim_preset() -> dict:
    """Return available figures and round-1000 summary from the experiment."""
    figures = []
    if _EXPERIMENT_DIR.exists():
        for png in sorted(_EXPERIMENT_DIR.glob("figure*.png")):
            figures.append({"name": png.name, "path": f"/preset_figures/{png.name}"})

    summary: dict = {}
    rounds_csv = _EXPERIMENT_DIR / "simulation_rounds.csv"
    if rounds_csv.exists():
        import csv
        with rounds_csv.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            summary = {
                "total_rounds": len(rows),
                "final_reward": float(last.get("avg_reward", 0)),
                "final_ctr": float(last.get("global_ctr", 0)),
                "final_lr": float(last.get("global_lr", 0)),
                "final_awt": float(last.get("global_awt", 0)),
                "weights": {
                    "cos": float(last.get("w0_cos", 0)),
                    "fresh": float(last.get("w1_fresh", 0)),
                    "lr": float(last.get("w2_lr", 0)),
                    "ctr": float(last.get("w3_ctr", 0)),
                    "awt": float(last.get("w4_awt", 0)),
                    "social": float(last.get("w5_social", 0)),
                },
            }
    return {"figures": figures, "summary": summary}


@app.post("/sim/load_preset")
def sim_load_preset() -> dict:
    """Load weights + per-image stats from the round-1000 experiment state."""
    rounds_csv = _EXPERIMENT_DIR / "simulation_rounds.csv"
    item_csv = _EXPERIMENT_DIR / "item_stats.csv"
    if not rounds_csv.exists():
        raise HTTPException(status_code=404, detail="Experiment data not found")

    import csv
    with rounds_csv.open() as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    w = [
        float(last.get("w0_cos", 0.25)),
        float(last.get("w1_fresh", 0.15)),
        float(last.get("w2_lr", 0.15)),
        float(last.get("w3_ctr", 0.15)),
        float(last.get("w4_awt", 0.15)),
        float(last.get("w5_social", 0.15)),
    ]

    img_stats: dict[int, dict] = {}
    if item_csv.exists():
        with item_csv.open() as f:
            for row in csv.DictReader(f):
                if int(row["round"]) == int(last["round"]):
                    iid = int(row["image_id"])
                    img_stats[iid] = {
                        "ctr": float(row.get("ctr", 0.1)),
                        "lr": float(row.get("lr", 0.05)),
                        "awt": float(row.get("awt", 0.0)),
                        "freshness": float(row.get("freshness", 0.0)),
                        "total_views": int(float(row.get("cum_views", 0))),
                        "total_likes": int(float(row.get("cum_likes", 0))),
                    }

    state = _load_sim_state()
    state["weights"] = w
    state["round"] = int(last["round"])
    state["running"] = False
    _save_sim_state(state)

    with _gallery_lock:
        items = _load_gallery()
        for img in items:
            iid_try = None
            try:
                iid_try = int(img["id"].replace("ds_", ""))
            except Exception:
                pass
            if iid_try is not None and iid_try in img_stats:
                img.update(img_stats[iid_try])
        _save_gallery(items)

    log(f"[Sim] Loaded preset: round={state['round']}, w={[round(x,3) for x in w]}")
    return {"ok": True, "round": state["round"], "weights": w}


# ── Serve preset figures ───────────────────────────────────────

@app.get("/preset_figures/{filename}")
def serve_preset_figure(filename: str) -> FileResponse:
    fpath = _EXPERIMENT_DIR / filename
    if not fpath.exists() or not fpath.suffix == ".png":
        raise HTTPException(status_code=404)
    return FileResponse(str(fpath), media_type="image/png")


# ── Debug logs ─────────────────────────────────────────────────

@app.get("/debug/logs")
def debug_logs(since: int = 0) -> dict:
    return {"entries": get_logs(since)}
