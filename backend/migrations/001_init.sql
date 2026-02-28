CREATE TABLE IF NOT EXISTS media_items (
  id TEXT PRIMARY KEY,
  media_type TEXT NOT NULL,
  file_path TEXT NOT NULL,
  stored_name TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'UPLOADED',
  error_message TEXT DEFAULT '',
  is_deleted INTEGER NOT NULL DEFAULT 0,
  views INTEGER NOT NULL DEFAULT 0,
  likes INTEGER NOT NULL DEFAULT 0,
  ctr REAL NOT NULL DEFAULT 0.0,
  avg_watch_time REAL NOT NULL DEFAULT 0.0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embedding_records (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  media_id TEXT NOT NULL,
  modality TEXT NOT NULL,
  model_name TEXT NOT NULL,
  vector_dim INTEGER NOT NULL,
  vector_id TEXT NOT NULL UNIQUE,
  pooling_type TEXT DEFAULT 'mean',
  num_segments INTEGER DEFAULT 1,
  created_at TEXT NOT NULL,
  FOREIGN KEY (media_id) REFERENCES media_items(id) ON DELETE CASCADE
);
