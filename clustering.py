"""
clustering.py — Semantic clustering of classified results.

Features:
  - Silhouette score to find the OPTIMAL number of clusters automatically
  - Elbow method as secondary check
  - Embeddings cached to disk
  - Cluster assignments saved back to DB with auto-labels
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
MIN_CLUSTERS = 2
MAX_CLUSTERS = 7  # Updated from 5 to 7


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
# Auto-label clusters
# ---------------------------------------------------------------------------

def _auto_label_cluster(items: list[dict]) -> str:
    """Returns the most common category in a cluster as its label."""
    categories = [i.get("category") or "Other" for i in items]
    return max(set(categories), key=categories.count)


# ---------------------------------------------------------------------------
# Optimal cluster finder
# ---------------------------------------------------------------------------

def find_optimal_clusters(X: np.ndarray, max_k: int) -> int:
    """
    Tests k from MIN_CLUSTERS to max_k and returns the k with the
    highest Silhouette Score.

    Silhouette Score ranges from -1 to +1:
      +1 = clusters are dense and well separated (perfect)
       0 = clusters overlap
      -1 = items are in the wrong cluster (bad)

    Also prints the Elbow curve (inertia) for reference.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    max_k = min(max_k, len(X) - 1)  # Can't have more clusters than samples - 1

    if max_k < MIN_CLUSTERS:
        log.warning("Not enough data to test multiple cluster counts. Using k=2.")
        return 2

    scores = {}
    inertias = {}

    log.info("Finding optimal cluster count (k=%d to k=%d)...", MIN_CLUSTERS, max_k)
    log.info("%-6s %-20s %-15s", "k", "Silhouette Score", "Inertia")
    log.info("-" * 45)

    for k in range(MIN_CLUSTERS, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)

        sil = silhouette_score(X, labels)
        inertia = kmeans.inertia_

        scores[k] = sil
        inertias[k] = inertia

        log.info("k=%-4d score=%-18.4f inertia=%.2f", k, sil, inertia)

    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]

    log.info("-" * 45)
    log.info("✅ Best k = %d  (Silhouette Score: %.4f)", best_k, best_score)

    # Give a human-readable quality rating
    if best_score > 0.5:
        quality = "Excellent separation"
    elif best_score > 0.3:
        quality = "Good separation"
    elif best_score > 0.1:
        quality = "Moderate separation"
    else:
        quality = "Weak separation — clusters may overlap"

    log.info("Cluster quality: %s", quality)
    return best_k


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------

def run_clustering(
    db_path: str,
    num_clusters: Optional[int] = None,
    auto_find: bool = True,
) -> Optional[list[dict]]:
    """
    Loads all results, embeds summaries (cached), runs KMeans,
    saves cluster assignments to DB.

    If auto_find=True and num_clusters is None, automatically finds
    the optimal k using Silhouette Score.
    If num_clusters is explicitly set, uses that value directly.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
    except ImportError:
        log.warning("Run: pip install sentence-transformers scikit-learn")
        return None

    rows = load_results(db_path)

    if len(rows) < 5:
        log.warning("Clustering skipped: only %d results (minimum 5).", len(rows))
        return None

    # --- Build embeddings ---
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

    X = np.vstack(embeddings)

    # --- Determine k ---
    if num_clusters is not None:
        # User explicitly set a number — use it
        n_clusters = min(num_clusters, len(rows) - 1)
        log.info("Using manually set k=%d", n_clusters)
    elif auto_find:
        # Automatically find the best k
        n_clusters = find_optimal_clusters(X, max_k=MAX_CLUSTERS)
    else:
        # Fall back to config value
        n_clusters = min(cfg.num_clusters, len(rows) - 1)
        log.info("Using config k=%d", n_clusters)

    # --- Run final KMeans with chosen k ---
    log.info("Running KMeans with k=%d...", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    # --- Group and auto-label ---
    groups: dict[int, list[dict]] = {}
    for row, label in zip(rows, labels):
        groups.setdefault(int(label), []).append(row)

    assignments = []
    for row, label in zip(rows, labels):
        cluster_id = int(label)
        cluster_label = _auto_label_cluster(groups[cluster_id])
        assignments.append({
            "message_id": row["message_id"],
            "cluster": cluster_id,
            "cluster_label": cluster_label,
        })

    save_cluster_assignments(db_path, assignments)
    _print_cluster_summary(groups, n_clusters)

    log.info("Clustering complete — %d items in %d clusters.", len(rows), n_clusters)

    return [{**row, "cluster": int(label)} for row, label in zip(rows, labels)]


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

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
