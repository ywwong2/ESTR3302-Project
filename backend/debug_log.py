"""Simple in-memory debug log for the frontend debug panel."""
from __future__ import annotations
import threading
from datetime import datetime, UTC

_lock = threading.Lock()
_entries: list[dict] = []
_counter = 0


def log(msg: str) -> None:
    global _counter
    with _lock:
        _entries.append({"i": _counter, "ts": datetime.now(UTC).strftime("%H:%M:%S"), "msg": msg})
        _counter += 1
        if len(_entries) > 500:
            _entries.pop(0)


def get_logs(since: int = 0) -> list[dict]:
    with _lock:
        return [e for e in _entries if e["i"] >= since]
