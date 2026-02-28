# Data Model and Schema

## ER Diagram

```mermaid
erDiagram
    MEDIA_ITEMS ||--o{ EMBEDDING_RECORDS : has
    MEDIA_ITEMS ||--|| QUALITY_SIGNALS : has
    MEDIA_ITEMS ||--o{ INTERACTION_LOGS : receives
    MEDIA_ITEMS ||--|| STATS_SNAPSHOTS : aggregates

    MEDIA_ITEMS {
        string id PK
        string media_type
        string file_path
        string stored_name
        string title
        string description
        string tags_json
        string uploader
        string status
        string embedding_model_version
        string error_message
        int is_deleted
        string created_at
        string updated_at
        string deleted_at
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

    QUALITY_SIGNALS {
        string media_id PK, FK
        float quality_score
        string source
        string updated_at
    }

    INTERACTION_LOGS {
        int id PK
        string user_id
        string session_id
        string media_id FK
        string action
        float dwell_time
        string timestamp
    }

    STATS_SNAPSHOTS {
        string media_id PK, FK
        int views
        int likes
        float ctr
        float avg_watch_time
        string updated_at
    }
```

## SQL Schema

The migration script is in [backend/migrations/001_init.sql](../backend/migrations/001_init.sql).

## Media Lifecycle

Allowed status values:
- `UPLOADED`
- `PROCESSING`
- `INDEXED`
- `FAILED`
- `DELETED`

### Lifecycle policy

1. New upload starts as `UPLOADED`.
2. Preprocessing starts: status becomes `PROCESSING`.
3. If preprocessing succeeds: status becomes `INDEXED`.
4. If preprocessing fails: status becomes `FAILED` with `error_message`.
5. Delete action sets status to `DELETED` and `is_deleted = 1`.

### Soft-delete and hard-delete

- Soft-delete (current default): keep metadata row with `status='DELETED'` and `deleted_at`.
- Hard-delete (future admin action): remove file + DB row.

### Re-index policy

When model or preprocessing changes:
1. Update `embedding_model_version`.
2. Select items with outdated version.
3. Move those items to `UPLOADED` and queue re-index.
4. Re-run preprocessing/embedding and mark `INDEXED`.
