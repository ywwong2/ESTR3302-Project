from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "data" / "app.db"
MIGRATION_FILE = Path(__file__).parent / "migrations" / "001_init.sql"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    sql = MIGRATION_FILE.read_text(encoding="utf-8")
    conn.executescript(sql)
    # Add columns for existing databases (safe: ignores if already exists)
    for col, default in [
        ("views INTEGER NOT NULL DEFAULT 0", "0"),
        ("likes INTEGER NOT NULL DEFAULT 0", "0"),
        ("ctr REAL NOT NULL DEFAULT 0.0", "0.0"),
        ("avg_watch_time REAL NOT NULL DEFAULT 0.0", "0.0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE media_items ADD COLUMN {col}")
        except Exception:
            pass  # column already exists
    conn.commit()
    conn.close()


def insert_media(item: dict[str, Any]) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO media_items (
          id, media_type, file_path, stored_name, title,
          status, error_message, is_deleted,
          views, likes, ctr, avg_watch_time,
          created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item["id"],
            item["media_type"],
            item["file_path"],
            item["stored_name"],
            item["title"],
            item["status"],
            item.get("error_message", ""),
            item.get("is_deleted", 0),
            item.get("views", 0),
            item.get("likes", 0),
            item.get("ctr", 0.0),
            item.get("avg_watch_time", 0.0),
            item["created_at"],
            item["updated_at"],
        ),
    )
    conn.commit()
    conn.close()


def update_media_status(media_id: str, status: str, error_message: str = "") -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE media_items SET status = ?, error_message = ?, updated_at = ? WHERE id = ?",
        (status, error_message, now_iso(), media_id),
    )
    conn.commit()
    conn.close()


def soft_delete_media(media_id: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE media_items SET is_deleted = 1, status = 'DELETED', updated_at = ? WHERE id = ?",
        (now_iso(), media_id),
    )
    conn.commit()
    conn.close()


def list_media(query: str = "") -> list[dict[str, Any]]:
    conn = get_conn()
    q = (query or "").strip().lower()
    if q:
        rows = conn.execute(
            """
            SELECT * FROM media_items
            WHERE is_deleted = 0
              AND (LOWER(title) LIKE ? OR LOWER(stored_name) LIKE ?)
            ORDER BY created_at DESC
            """,
            (f"%{q}%", f"%{q}%"),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM media_items WHERE is_deleted = 0 ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_media(media_id: str) -> dict[str, Any] | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM media_items WHERE id = ?", (media_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def insert_embedding_record(item: dict[str, Any]) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO embedding_records (
          media_id, modality, model_name, vector_dim, vector_id,
          pooling_type, num_segments, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item["media_id"],
            item["modality"],
            item["model_name"],
            item["vector_dim"],
            item["vector_id"],
            item.get("pooling_type", "mean"),
            item.get("num_segments", 1),
            item["created_at"],
        ),
    )
    conn.commit()
    conn.close()


def increment_likes(media_id: str) -> dict[str, Any] | None:
    conn = get_conn()
    conn.execute(
        "UPDATE media_items SET likes = likes + 1, updated_at = ? WHERE id = ?",
        (now_iso(), media_id),
    )
    conn.commit()
    row = conn.execute("SELECT likes FROM media_items WHERE id = ?", (media_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def increment_views(media_id: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE media_items SET views = views + 1, updated_at = ? WHERE id = ?",
        (now_iso(), media_id),
    )
    conn.commit()
    conn.close()


def get_all_indexed_media() -> list[dict[str, Any]]:
    """Return all INDEXED (non-deleted) items for vector search."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM media_items WHERE is_deleted = 0 AND status = 'INDEXED' ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_embedding_record_by_media(media_id: str) -> dict[str, Any] | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM embedding_records WHERE media_id = ? LIMIT 1", (media_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_embedding_records_by_media(media_id: str) -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM embedding_records WHERE media_id = ?", (media_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_embedding_records_by_media(media_id: str) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM embedding_records WHERE media_id = ?", (media_id,))
    conn.commit()
    conn.close()