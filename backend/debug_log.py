"""In-memory debug log shared across backend modules.

Usage:
    from backend.debug_log import log, get_logs

    log("some message")       # stores + prints
    entries = get_logs(since)  # returns entries after given index
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

_lock = threading.Lock()
_entries: list[dict] = []  # {"i": int, "ts": str, "msg": str}


def log(msg: str) -> None:
    """Append a debug message and print it."""
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    with _lock:
        entry = {"i": len(_entries), "ts": ts, "msg": msg}
        _entries.append(entry)
    # Also print so terminal shows it
    print(f"[{ts}] {msg}", flush=True)


def get_logs(since: int = 0) -> list[dict]:
    """Return log entries with index >= since."""
    with _lock:
        return _entries[since:]
