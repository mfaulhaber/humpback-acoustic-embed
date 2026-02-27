"""Cluster evaluation metrics and parameter sweep."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | None]:
    """Compute internal cluster evaluation metrics.

    Excludes noise points (label == -1).  Returns None values when fewer
    than 2 non-noise clusters exist.
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    mask = labels != -1
    clean_labels = labels[mask]
    clean_embeddings = embeddings[mask]

    n_clusters = len(set(clean_labels.tolist()))
    if n_clusters < 2 or len(clean_embeddings) < 2:
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "calinski_harabasz_score": None,
            "n_clusters": n_clusters,
            "noise_count": int((~mask).sum()),
        }

    return {
        "silhouette_score": float(silhouette_score(clean_embeddings, clean_labels)),
        "davies_bouldin_index": float(davies_bouldin_score(clean_embeddings, clean_labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(clean_embeddings, clean_labels)),
        "n_clusters": n_clusters,
        "noise_count": int((~mask).sum()),
    }


def extract_category_from_folder_path(folder_path: str) -> str | None:
    """Extract category from a folder path following the social-sounds convention.

    Example: ``"social-sounds/calls/subset1"`` -> ``"calls"``

    Returns None if the path does not contain ``social-sounds/`` or has no
    child folder beneath it.
    """
    if not folder_path:
        return None

    parts = folder_path.replace("\\", "/").split("/")
    try:
        idx = parts.index("social-sounds")
    except ValueError:
        return None

    if idx + 1 < len(parts) and parts[idx + 1]:
        return parts[idx + 1]
    return None


def compute_category_metrics(
    labels: np.ndarray,
    category_labels: list[str | None],
) -> dict[str, float | None]:
    """Compute semi-supervised metrics comparing clusters to folder categories.

    Returns None values when fewer than 2 distinct categories are present
    (after filtering out None entries and noise labels).
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Build aligned arrays excluding noise and None categories
    cluster_vals = []
    cat_vals = []
    for lab, cat in zip(labels.tolist(), category_labels):
        if lab == -1 or cat is None:
            continue
        cluster_vals.append(lab)
        cat_vals.append(cat)

    n_categories = len(set(cat_vals))
    if n_categories < 2 or len(cluster_vals) < 2:
        return {
            "adjusted_rand_index": None,
            "normalized_mutual_info": None,
            "n_categories": n_categories,
        }

    return {
        "adjusted_rand_index": float(adjusted_rand_score(cat_vals, cluster_vals)),
        "normalized_mutual_info": float(normalized_mutual_info_score(cat_vals, cluster_vals)),
        "n_categories": n_categories,
    }


def run_parameter_sweep(
    embeddings: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Sweep ``min_cluster_size`` over already-reduced embeddings.

    Returns a list of dicts with keys: ``min_cluster_size``,
    ``silhouette_score``, ``n_clusters``, ``noise_fraction``.
    """
    from sklearn.metrics import silhouette_score

    from humpback.clustering.clusterer import cluster_hdbscan

    params = parameters or {}
    min_samples = params.get("min_samples", None)

    n = len(embeddings)
    max_mcs = min(n // 2, 50)
    if max_mcs < 2:
        return []

    results: list[dict[str, Any]] = []
    for mcs in range(2, max_mcs + 1):
        labels = cluster_hdbscan(embeddings, min_cluster_size=mcs, min_samples=min_samples)

        mask = labels != -1
        clean_labels = labels[mask]
        n_clusters = len(set(clean_labels.tolist()))
        noise_fraction = float((~mask).sum()) / n if n > 0 else 0.0

        sil = None
        if n_clusters >= 2 and mask.sum() >= 2:
            sil = float(silhouette_score(embeddings[mask], clean_labels))

        results.append({
            "min_cluster_size": mcs,
            "silhouette_score": sil,
            "n_clusters": n_clusters,
            "noise_fraction": noise_fraction,
        })

    return results
