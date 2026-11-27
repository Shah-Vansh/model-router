"""SQLite cache of model replies, so re-runs are instant and free."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, raw TEXT)")
        self._db.commit()
        self._lock = threading.Lock()

    @staticmethod
    def key(model_id: str, system: str, prompt: str) -> str:
        h = hashlib.sha256()
        for part in (model_id, system, prompt):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT raw FROM replies WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, raw: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO replies VALUES (?, ?)", (key, raw))
            self._db.commit()

    def close(self) -> None:
        self._db.close()
