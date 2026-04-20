"""
src/cache.py  –  SHA-256-keyed SQLite response cache.

Checks the hash of each query before entering the expensive multi-LLM pipeline.
Cache hits return immediately, reducing latency and OpenAI API cost.
"""

import hashlib
import json
import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


class ResponseCache:
    """Persist query→response pairs in a local SQLite database."""

    def __init__(self, db_path: str = "howdyai_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key        TEXT PRIMARY KEY,
                    query      TEXT NOT NULL,
                    response   TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        logger.debug(f"Cache DB initialised at {self.db_path}")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _hash(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> Optional[dict]:
        """Return cached response dict, or None on miss."""
        key = self._hash(query)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT response FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row:
            logger.info(f"Cache HIT for query: {query[:60]!r}")
            return json.loads(row[0])
        logger.debug(f"Cache MISS for query: {query[:60]!r}")
        return None

    def set(self, query: str, response: dict) -> None:
        """Insert or replace a cache entry."""
        key = self._hash(query)
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, query, response) VALUES (?, ?, ?)",
                (key, query, json.dumps(response)),
            )
            conn.commit()
        logger.debug(f"Cache SET for query: {query[:60]!r}")

    def clear(self) -> None:
        """Wipe all cached entries."""
        with self._conn() as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
        logger.info("Cache cleared.")

    def stats(self) -> dict:
        with self._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        return {"entries": count, "db_path": self.db_path}
