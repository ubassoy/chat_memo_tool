"""
clustering.py — Optional semantic clustering of classified results.

Uses sentence-transformers + KMeans to group similar messages.
Embeddings are cached to disk so re-runs don't re-encode everything.
Requires: sentence-transformers, scikit-learn, numpy
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from config import cfg
from logger import log
from storage import load_results


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

CACHE_PATH = Path(".embedding_cache.pkl")


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _load_cache() -> dict[str, np.ndarray]:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict[str, np.ndarray]) -> None:
    with CACHE_PATH.open("wb") as f:
        pickle.dump(cache, f)


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------

def run_clustering(db_path: str, num_clusters: Optional[int] = None) -> Optional[list[dict]]:
    """
    Loads all results, embeds their summaries (with disk cache),
    runs KMeans, and returns a list of cluster assignments.

    Returns None if clustering is skipped (too few items or deps missing).
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
    except ImportError:
        log.warning(
            "Clustering skipped: 'sentence-transformers' or 'scikit-learn' not installed. "
            "Run: pip install sentence-transformers scikit-learn"
        )
        return None

    rows = load_results(db_path)

    if len(rows) < 5:
        log.warning(
            "Clustering skipped: only %d results found (minimum 5 required).",
            len(rows),
        )
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
    log.info(
        "Embeddings ready — %d from cache, %d newly encoded.",
        cache_hits,
        len(rows) - cache_hits,
    )

    n_clusters = min(num_clusters or cfg.num_clusters, len(rows))
    X = np.vstack(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    # Annotate results with cluster labels
    clustered = []
    for row, label in zip(rows, labels):
        clustered.append({**row, "cluster": int(label)})

    # Print cluster summary to console
    _print_cluster_summary(clustered, n_clusters)

    log.info("Clustering complete — %d items in %d clusters.", len(rows), n_clusters)
    return clustered


def _print_cluster_summary(clustered: list[dict], n_clusters: int) -> None:
    """Prints a readable summary of each cluster to the log."""
    groups: dict[int, list[str]] = {i: [] for i in range(n_clusters)}

    for item in clustered:
        groups[item["cluster"]].append(
            item.get("summary") or item.get("title") or "—"
        )

    log.info("=" * 60)
    log.info("CLUSTER SUMMARY")
    log.info("=" * 60)

    for cluster_id, summaries in groups.items():
        log.info("Cluster %d (%d items):", cluster_id, len(summaries))
        for s in summaries[:3]:
            log.info("  • %s", s[:100])
        if len(summaries) > 3:
            log.info("  ... and %d more", len(summaries) - 3)

    log.info("=" * 60)
