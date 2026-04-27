# Data Model and Schema (Image-Only)

## Current scope
The runtime system keeps only the minimum entities needed for image upload, indexing, search, and feedback-aware ranking.

## ER Diagram

```mermaid
erDiagram
    MEDIA_ITEMS ||--o{ EMBEDDING_RECORDS : has

    MEDIA_ITEMS {
        string id PK
        string media_type
        string file_path
        string stored_name
        string title
        string status
        string error_message
        int is_deleted
        int views
        int likes
        float ctr
        float avg_watch_time
        string created_at
        string updated_at
    }

    EMBEDDING_RECORDS {
        int id PK
        string media_id FK
        string modality
        string model_name
        int vector_dim
        string vector_id
        string pooling_type
        int num_segments
        string created_at
    }
```

## SQL source
Migration script: [backend/migrations/001_init.sql](../backend/migrations/001_init.sql)

## Media lifecycle
Allowed statuses:
- `UPLOADED`
- `PROCESSING`
- `INDEXED`
- `FAILED`
- `DELETED`

Transition policy:
1. Upload creates row with `UPLOADED`.
2. Background indexing sets `PROCESSING`.
3. On success set `INDEXED` and insert embedding metadata.
4. On failure set `FAILED` and store `error_message`.
5. Delete sets `DELETED`, marks `is_deleted = 1`, and removes stored vector artifacts.

## Notes
- `media_type` is restricted to image in current implementation.
- Historical ranking signals (`views`, `likes`, `ctr`, `avg_watch_time`) are kept in `media_items` for simplified online scoring.
- Advanced entities from earlier multimedia plan (quality table, interaction logs, snapshot table) are intentionally deferred to keep this phase mathematically focused and implementation-light.
