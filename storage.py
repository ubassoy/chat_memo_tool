"""
storage.py — Persistent SQLite storage layer.

Replaces the flat JSON files with a proper database.
Two tables:
  - processed_ids  : tracks which ChatGPT node IDs have been handled
  - results        : stores classification outputs + original text
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

from logger import log


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def _get_conn(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

def init_db(db_path: str) -> None:
    """Creates tables if they don't already exist."""
    with _get_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS processed_ids (
                node_id TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS results (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id    TEXT    NOT NULL UNIQUE,
                title         TEXT,
                original_text TEXT,
                timestamp     REAL,
                summary       TEXT,
                category      TEXT,
                priority      TEXT,
                raw_analysis  TEXT,
                created_at    TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_results_title     ON results(title);
            CREATE INDEX IF NOT EXISTS idx_results_category  ON results(category);
            CREATE INDEX IF NOT EXISTS idx_results_timestamp ON results(timestamp);
            """
        )
    log.debug("Database schema verified at: %s", db_path)


# ---------------------------------------------------------------------------
# Processed IDs
# ---------------------------------------------------------------------------

def load_processed_ids(db_path: str) -> set[str]:
    with _get_conn(db_path) as conn:
        rows = conn.execute("SELECT node_id FROM processed_ids").fetchall()
    return {row["node_id"] for row in rows}


def mark_id_processed(db_path: str, node_id: str) -> None:
    with _get_conn(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed_ids (node_id, processed_at) VALUES (?, ?)",
            (node_id, datetime.utcnow().isoformat()),
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def save_result(db_path: str, entry: dict) -> None:
    """
    Upserts a single classified result.
    `entry` must have keys: message_id, title, original_text, timestamp, analysis (dict)
    """
    analysis: dict = entry.get("analysis", {})
    now = datetime.utcnow().isoformat()

    with _get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO results
                (message_id, title, original_text, timestamp,
                 summary, category, priority, raw_analysis, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                summary      = excluded.summary,
                category     = excluded.category,
                priority     = excluded.priority,
                raw_analysis = excluded.raw_analysis
            """,
            (
                entry["message_id"],
                entry.get("title"),
                entry.get("original_text"),
                entry.get("timestamp"),
                analysis.get("summary"),
                analysis.get("category"),
                analysis.get("priority"),
                json.dumps(analysis, ensure_ascii=False),
                now,
            ),
        )


def load_results(
    db_path: str,
    limit: Optional[int] = None,
    category: Optional[str] = None,
) -> list[dict]:
    """Returns results sorted newest-first, with optional filters."""
    query = "SELECT * FROM results WHERE 1=1"
    params: list = []

    if category:
        query += " AND category = ?"
        params.append(category)

    query += " ORDER BY timestamp DESC"

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    with _get_conn(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def get_result_count(db_path: str) -> int:
    with _get_conn(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM results").fetchone()
    return row["cnt"]


def delete_failed_results(db_path: str) -> int:
    """Removes rows where the summary is missing or flagged as an error."""
    with _get_conn(db_path) as conn:
        cursor = conn.execute(
            """
            DELETE FROM results
            WHERE summary IS NULL
               OR summary = ''
               OR summary = 'Error parsing JSON'
            """
        )
        deleted = cursor.rowcount

    # Also un-mark those IDs so they get re-processed next run
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            DELETE FROM processed_ids
            WHERE node_id NOT IN (SELECT message_id FROM results)
            """
        )

    log.info("Cleanup removed %d failed/empty rows.", deleted)
    return deleted
