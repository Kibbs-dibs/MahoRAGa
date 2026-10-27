"""
SQLite Database Repository for Project Aletheia Backend (backend/database.py)

Thread-safe session persistence, chat history, and pipeline execution audits.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("backend.database")


class DatabaseManager:
    """Manages SQLite database connections and table schemas."""

    def __init__(self, db_path: str | Path = "data/aletheia.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection with foreign keys enabled."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=20.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            self._local.conn = conn
        return self._local.conn

    def close(self) -> None:
        """Close and release the calling thread's SQLite connection (Issue 28).

        Thread-local connections otherwise live for the thread's lifetime with no
        teardown, leaking handles under worker-thread pools. Call from a worker's
        shutdown / FastAPI lifespan, or rely on it in a context-manager exit.
        """
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def init_db(self) -> None:
        """Create database tables if they do not exist."""
        conn = self.get_connection()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    rubric_score INTEGER,
                    compression_ratio REAL,
                    critic_verdict TEXT,
                    latency REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_audits (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    reformulated_query TEXT,
                    routing_decision TEXT,
                    vector_chunks_count INTEGER DEFAULT 0,
                    lexical_chunks_count INTEGER DEFAULT 0,
                    graph_triples_count INTEGER DEFAULT 0,
                    external_count INTEGER DEFAULT 0,
                    rubric_score INTEGER DEFAULT 0,
                    compression_ratio REAL DEFAULT 1.0,
                    critic_verdict TEXT,
                    total_latency REAL DEFAULT 0.0,
                    timestamp TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);"
            )
        logger.info("Database schema initialized at %s", self.db_path)

    # ── Session Operations ───────────────────────────────────────────────

    def create_session(self, title: Optional[str] = None) -> Dict[str, Any]:
        """Create a new research chat session."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        session_title = title or "New Research Session"

        conn = self.get_connection()
        with conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?);",
                (session_id, session_title, now, now),
            )
        return {
            "id": session_id,
            "title": session_title,
            "created_at": now,
            "updated_at": now,
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by its unique ID."""
        conn = self.get_connection()
        cursor = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?;",
            (session_id,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent sessions ordered by latest update."""
        conn = self.get_connection()
        cursor = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?;",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its associated messages."""
        conn = self.get_connection()
        with conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?;", (session_id,))
            return cursor.rowcount > 0

    # ── Message Operations ───────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        rubric_score: Optional[int] = None,
        compression_ratio: Optional[float] = None,
        critic_verdict: Optional[str] = None,
        latency: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Insert a chat message into a session and update session timestamp."""
        # Ensure session exists or create it with the SAME id (Issue 29).
        # Previously this called create_session() which minted a *new* random id,
        # leaving the message's session_id dangling and violating the FK.
        if not self.get_session(session_id):
            title = content[:40].strip() + ("..." if len(content) > 40 else "")
            now_ts = datetime.now(timezone.utc).isoformat()
            conn = self.get_connection()
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?);",
                    (session_id, title or "New Research Session", now_ts, now_ts),
                )

        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        conn = self.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO messages (
                    id, session_id, role, content, rubric_score,
                    compression_ratio, critic_verdict, latency, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    rubric_score,
                    compression_ratio,
                    critic_verdict,
                    latency,
                    now,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?;",
                (now, session_id),
            )

        return {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "rubric_score": rubric_score,
            "compression_ratio": compression_ratio,
            "critic_verdict": critic_verdict,
            "latency": latency,
            "created_at": now,
        }

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all chronological messages in a session."""
        conn = self.get_connection()
        cursor = conn.execute(
            """
            SELECT id, session_id, role, content, rubric_score,
                   compression_ratio, critic_verdict, latency, created_at
            FROM messages WHERE session_id = ? ORDER BY created_at ASC;
            """,
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ── Audit Operations ─────────────────────────────────────────────────

    def record_audit(
        self,
        query: str,
        session_id: Optional[str] = None,
        reformulated_query: Optional[str] = None,
        routing_decision: Optional[str] = None,
        vector_chunks_count: int = 0,
        lexical_chunks_count: int = 0,
        graph_triples_count: int = 0,
        external_count: int = 0,
        rubric_score: int = 0,
        compression_ratio: float = 1.0,
        critic_verdict: Optional[str] = None,
        total_latency: float = 0.0,
    ) -> str:
        """Log pipeline execution audit record."""
        audit_id = f"audit_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        conn = self.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO query_audits (
                    id, session_id, query, reformulated_query, routing_decision,
                    vector_chunks_count, lexical_chunks_count, graph_triples_count,
                    external_count, rubric_score, compression_ratio,
                    critic_verdict, total_latency, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    audit_id,
                    session_id,
                    query,
                    reformulated_query,
                    routing_decision,
                    vector_chunks_count,
                    lexical_chunks_count,
                    graph_triples_count,
                    external_count,
                    rubric_score,
                    compression_ratio,
                    critic_verdict,
                    total_latency,
                    now,
                ),
            )
        return audit_id

    def get_audit_summary(self) -> Dict[str, Any]:
        """Aggregate observability metrics from query_audits (Feature 8.1).

        Surfaces the data the pipeline already records: query volume, average
        rubric/compression/latency, external-search hit rate, and
        self-correction (regeneration) rate.
        """
        conn = self.get_connection()
        row = conn.execute(
            """
            SELECT
                COUNT(*)                                               AS total_queries,
                COALESCE(AVG(rubric_score), 0)                         AS avg_rubric_score,
                COALESCE(AVG(compression_ratio), 0)                    AS avg_compression_ratio,
                COALESCE(AVG(total_latency), 0)                        AS avg_latency_seconds,
                COALESCE(SUM(CASE WHEN external_count > 0 THEN 1 ELSE 0 END), 0) AS external_hits,
                COALESCE(SUM(CASE WHEN critic_verdict LIKE '%Regenerated%'
                                   OR critic_verdict LIKE '%Self-Corrected%' THEN 1 ELSE 0 END), 0) AS regenerations
            FROM query_audits;
            """
        ).fetchone()
        total = row["total_queries"] or 0

        # Route distribution
        routes = {}
        for r in conn.execute(
            "SELECT routing_decision AS route, COUNT(*) AS n FROM query_audits "
            "WHERE routing_decision IS NOT NULL AND routing_decision != '' GROUP BY routing_decision;"
        ).fetchall():
            routes[r["route"]] = r["n"]

        return {
            "total_queries": total,
            "avg_rubric_score": round(float(row["avg_rubric_score"]), 2),
            "avg_compression_ratio": round(float(row["avg_compression_ratio"]), 2),
            "avg_latency_seconds": round(float(row["avg_latency_seconds"]), 3),
            "external_search_hit_rate": round((row["external_hits"] / total), 3) if total else 0.0,
            "regeneration_rate": round((row["regenerations"] / total), 3) if total else 0.0,
            "route_distribution": routes,
        }


# Default singleton instance
db_manager = DatabaseManager()
