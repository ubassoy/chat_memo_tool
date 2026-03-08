"""
clustering.py — Semantic clustering of classified results.

Uses sentence-transformers + KMeans to group similar messages.
Embeddings are cached to disk so re-runs don't re-encode everything.
Cluster assignments are saved back to the database.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from config import cfg
from logger import log
from storage import load_results, save_cluster_assignments


CACHE_PATH = Path(".embedding_cache.pkl")


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with CACHE_PATH.open("wb") as f:
        pickle.dump(cache, f)


# ---------------------------------------------------------------------------
# Auto-label clusters using most common category
# ---------------------------------------------------------------------------

def _auto_label_cluster(items: list[dict]) -> str:
    """Returns the most common category in a cluster as its label."""
    categories = [i.get("category") or "Other" for i in items]
    return max(set(categories), key=categories.count)


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------

def run_clustering(db_path: str, num_clusters: Optional[int] = None) -> Optional[list[dict]]:
    """
    Loads all results, embeds their summaries (with disk cache),
    runs KMeans, saves cluster assignments to DB, and returns
    a list of annotated dicts.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
    except ImportError:
        log.warning(
            "Clustering skipped: install with: pip install sentence-transformers scikit-learn"
        )
        return None

    rows = load_results(db_path)

    if len(rows) < 5:
        log.warning("Clustering skipped: only %d results (minimum 5 required).", len(rows))
        return None

    log.info("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    cache = _load_cache()
    embeddings = []
    cache_hits = 0

    for row in rows:
        text = row.get("summary") or row.get("original_text") or ""
        key = _cache_key(text)
        if key in cache:
            embeddings.append(cache[key])
            cache_hits += 1
        else:
            vec = model.encode(text)
            cache[key] = vec
            embeddings.append(vec)

    _save_cache(cache)
    log.info("Embeddings ready — %d cached, %d new.", cache_hits, len(rows) - cache_hits)

    n_clusters = min(num_clusters or cfg.num_clusters, len(rows))
    X = np.vstack(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    # Group rows by cluster to auto-label
    groups: dict[int, list[dict]] = {}
    for row, label in zip(rows, labels):
        groups.setdefault(int(label), []).append(row)

    # Build assignments with auto-labels
    assignments = []
    for row, label in zip(rows, labels):
        cluster_id = int(label)
        cluster_label = _auto_label_cluster(groups[cluster_id])
        assignments.append({
            "message_id": row["message_id"],
            "cluster": cluster_id,
            "cluster_label": cluster_label,
        })

    # Save back to DB
    save_cluster_assignments(db_path, assignments)

    # Print summary
    _print_cluster_summary(groups, n_clusters)

    log.info("Clustering complete — %d items in %d clusters.", len(rows), n_clusters)

    clustered = []
    for row, label in zip(rows, labels):
        clustered.append({**row, "cluster": int(label)})
    return clustered


def _print_cluster_summary(groups: dict, n_clusters: int) -> None:
    log.info("=" * 60)
    log.info("CLUSTER SUMMARY")
    log.info("=" * 60)
    for cluster_id, items in sorted(groups.items()):
        label = _auto_label_cluster(items)
        log.info("Cluster %d — '%s' (%d items):", cluster_id, label, len(items))
        for item in items[:3]:
            log.info("  • %s", (item.get("summary") or "—")[:100])
        if len(items) > 3:
            log.info("  ... and %d more", len(items) - 3)
    log.info("=" * 60)
