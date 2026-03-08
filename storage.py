"""
storage.py — Persistent SQLite storage layer.

Tables:
  - processed_ids  : tracks which ChatGPT node IDs have been handled
  - results        : stores classification outputs + original text + cluster
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
                cluster       INTEGER DEFAULT NULL,
                cluster_label TEXT    DEFAULT NULL,
                raw_analysis  TEXT,
                created_at    TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_results_title     ON results(title);
            CREATE INDEX IF NOT EXISTS idx_results_category  ON results(category);
            CREATE INDEX IF NOT EXISTS idx_results_timestamp ON results(timestamp);
            CREATE INDEX IF NOT EXISTS idx_results_cluster   ON results(cluster);
            """
        )
        # Add cluster columns if upgrading from old schema
        try:
            conn.execute("ALTER TABLE results ADD COLUMN cluster INTEGER DEFAULT NULL")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE results ADD COLUMN cluster_label TEXT DEFAULT NULL")
        except Exception:
            pass

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
    """Upserts a single classified result."""
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


def save_cluster_assignments(db_path: str, assignments: list[dict]) -> None:
    """
    Saves cluster labels back to the results table.
    Each assignment dict must have: message_id, cluster (int), cluster_label (str)
    """
    with _get_conn(db_path) as conn:
        for item in assignments:
            conn.execute(
                """
                UPDATE results
                SET cluster = ?, cluster_label = ?
                WHERE message_id = ?
                """,
                (item["cluster"], item.get("cluster_label", f"Cluster {item['cluster']}"), item["message_id"]),
            )
    log.info("Saved %d cluster assignments to DB.", len(assignments))


def load_results(
    db_path: str,
    limit: Optional[int] = None,
    category: Optional[str] = None,
    cluster: Optional[int] = None,
) -> list[dict]:
    """Returns results sorted newest-first, with optional filters."""
    query = "SELECT * FROM results WHERE 1=1"
    params: list = []

    if category:
        query += " AND category = ?"
        params.append(category)

    if cluster is not None:
        query += " AND cluster = ?"
        params.append(cluster)

    query += " ORDER BY timestamp DESC"

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    with _get_conn(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def load_results_grouped_by_cluster(db_path: str) -> dict[int, list[dict]]:
    """
    Returns all results grouped by cluster number.
    { 0: [...], 1: [...], 2: [...] }
    Rows with no cluster assigned go into key -1.
    """
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM results ORDER BY cluster ASC, timestamp DESC"
        ).fetchall()

    groups: dict[int, list[dict]] = {}
    for row in rows:
        d = dict(row)
        key = d.get("cluster") if d.get("cluster") is not None else -1
        groups.setdefault(key, []).append(d)

    return groups


def get_cluster_ids(db_path: str) -> list[int]:
    """Returns a sorted list of unique cluster IDs that exist in the DB."""
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT cluster FROM results WHERE cluster IS NOT NULL ORDER BY cluster"
        ).fetchall()
    return [row["cluster"] for row in rows]


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

    with _get_conn(db_path) as conn:
        conn.execute(
            """
            DELETE FROM processed_ids
            WHERE node_id NOT IN (SELECT message_id FROM results)
            """
        )

    log.info("Cleanup removed %d failed/empty rows.", deleted)
    return deleted
