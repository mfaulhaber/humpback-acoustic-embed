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
    """Extract category label from an audio file's folder path.

    Uses the **last non-empty path component** as the category label.
    This works for structures like ``"Emily-Vierling-Orcasound-data/Grunt"``
    → ``"Grunt"`` and ``"social-sounds/calls"`` → ``"calls"``.

    Returns None if the path is empty or has no usable component.
    """
    if not folder_path:
        return None

    parts = [p for p in folder_path.replace("\\", "/").split("/") if p]
    if not parts:
        return None

    return parts[-1]


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


def compute_detailed_category_metrics(
    labels: np.ndarray,
    category_labels: list[str | None],
) -> dict[str, Any]:
    """Compute detailed supervised metrics comparing clusters to categories.

    Returns ARI, NMI, homogeneity, completeness, v_measure, per-category purity,
    and a confusion matrix (category → cluster label → count).
    """
    from sklearn.metrics import (
        adjusted_rand_score,
        completeness_score,
        homogeneity_score,
        normalized_mutual_info_score,
        v_measure_score,
    )

    # Build aligned arrays excluding noise and None categories
    cluster_vals: list[int] = []
    cat_vals: list[str] = []
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
            "homogeneity": None,
            "completeness": None,
            "v_measure": None,
            "n_categories": n_categories,
            "per_category_purity": {},
            "confusion_matrix": {},
        }

    # Core metrics
    ari = float(adjusted_rand_score(cat_vals, cluster_vals))
    nmi = float(normalized_mutual_info_score(cat_vals, cluster_vals))
    homogeneity = float(homogeneity_score(cat_vals, cluster_vals))
    completeness = float(completeness_score(cat_vals, cluster_vals))
    v_measure = float(v_measure_score(cat_vals, cluster_vals))

    # Confusion matrix: category -> {cluster_label: count}
    confusion: dict[str, dict[str, int]] = {}
    for cat, cl in zip(cat_vals, cluster_vals):
        if cat not in confusion:
            confusion[cat] = {}
        cl_str = str(cl)
        confusion[cat][cl_str] = confusion[cat].get(cl_str, 0) + 1

    # Per-category purity: fraction of each category in its majority cluster
    per_category_purity: dict[str, float] = {}
    for cat, cluster_counts in confusion.items():
        total = sum(cluster_counts.values())
        majority = max(cluster_counts.values())
        per_category_purity[cat] = round(majority / total, 4) if total > 0 else 0.0

    return {
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
        "n_categories": n_categories,
        "per_category_purity": per_category_purity,
        "confusion_matrix": confusion,
    }


def compute_dendrogram_data(
    confusion_matrix: dict[str, dict[str, int]],
) -> dict[str, Any] | None:
    """Compute hierarchical clustering dendrograms for a confusion matrix.

    Takes the confusion matrix (category → {cluster_label: count}) and returns
    reordered matrix data with dendrogram coordinates for both axes.

    Returns None if fewer than 2 clusters or 2 categories.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage

    categories = sorted(confusion_matrix.keys())
    cluster_set: set[str] = set()
    for counts in confusion_matrix.values():
        cluster_set.update(counts.keys())
    cluster_labels = sorted(cluster_set, key=lambda x: int(x) if x.lstrip("-").isdigit() else x)

    if len(categories) < 2 or len(cluster_labels) < 2:
        return None

    # Build matrix: rows=clusters, cols=categories
    matrix = np.zeros((len(cluster_labels), len(categories)))
    for j, cat in enumerate(categories):
        for i, cl in enumerate(cluster_labels):
            matrix[i, j] = confusion_matrix.get(cat, {}).get(cl, 0)

    raw_counts = matrix.copy()

    # Row-normalize (each row sums to 1)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    normalized = matrix / row_sums

    # Compute linkage for rows (clusters) and columns (categories)
    row_linkage = linkage(normalized, method="average", metric="euclidean")
    col_linkage = linkage(normalized.T, method="average", metric="euclidean")

    # Get dendrogram info (no_plot=True returns coordinates only)
    row_dendro = dendrogram(row_linkage, no_plot=True)
    col_dendro = dendrogram(col_linkage, no_plot=True)

    # Reorder by dendrogram leaf order
    row_order = row_dendro["leaves"]
    col_order = col_dendro["leaves"]

    reordered_normalized = normalized[np.ix_(row_order, col_order)]
    reordered_raw = raw_counts[np.ix_(row_order, col_order)]

    return {
        "categories": [categories[i] for i in col_order],
        "cluster_labels": [cluster_labels[i] for i in row_order],
        "values": reordered_normalized.tolist(),
        "raw_counts": reordered_raw.astype(int).tolist(),
        "row_dendrogram": {
            "icoord": row_dendro["icoord"],
            "dcoord": row_dendro["dcoord"],
        },
        "col_dendrogram": {
            "icoord": col_dendro["icoord"],
            "dcoord": col_dendro["dcoord"],
        },
    }


def run_parameter_sweep(
    embeddings: np.ndarray,
    parameters: dict[str, Any] | None = None,
    category_labels: list[str | None] | None = None,
) -> list[dict[str, Any]]:
    """Sweep clustering parameters over already-reduced embeddings.

    Sweeps HDBSCAN (min_cluster_size × selection_method) and K-Means (k=2..30).
    When ``category_labels`` are provided, includes ARI and NMI in results.
    """
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    from humpback.clustering.clusterer import cluster_hdbscan, cluster_kmeans

    params = parameters or {}
    min_samples = params.get("min_samples", None)

    n = len(embeddings)
    if n < 4:
        return []

    # Prepare category labels for ARI/NMI
    has_categories = category_labels is not None and any(
        c is not None for c in category_labels
    )

    def _category_metrics(labels: np.ndarray) -> dict[str, float | None]:
        if not has_categories:
            return {}
        cluster_vals = []
        cat_vals = []
        for lab, cat in zip(labels.tolist(), category_labels):
            if lab == -1 or cat is None:
                continue
            cluster_vals.append(lab)
            cat_vals.append(cat)
        if len(set(cat_vals)) < 2 or len(cluster_vals) < 2:
            return {"adjusted_rand_index": None, "normalized_mutual_info": None}
        return {
            "adjusted_rand_index": float(adjusted_rand_score(cat_vals, cluster_vals)),
            "normalized_mutual_info": float(
                normalized_mutual_info_score(cat_vals, cluster_vals)
            ),
        }

    results: list[dict[str, Any]] = []

    # --- HDBSCAN sweep: min_cluster_size × selection_method ---
    max_mcs = min(n // 2, 50)
    for selection_method in ["leaf", "eom"]:
        for mcs in range(2, max(max_mcs + 1, 3)):
            labels = cluster_hdbscan(
                embeddings,
                min_cluster_size=mcs,
                min_samples=min_samples,
                cluster_selection_method=selection_method,
            )

            mask = labels != -1
            clean_labels = labels[mask]
            n_clusters = len(set(clean_labels.tolist()))
            noise_fraction = float((~mask).sum()) / n if n > 0 else 0.0

            sil = None
            if n_clusters >= 2 and mask.sum() >= 2:
                sil = float(silhouette_score(embeddings[mask], clean_labels))

            entry: dict[str, Any] = {
                "algorithm": "hdbscan",
                "min_cluster_size": mcs,
                "cluster_selection_method": selection_method,
                "silhouette_score": sil,
                "n_clusters": n_clusters,
                "noise_fraction": noise_fraction,
            }
            entry.update(_category_metrics(labels))
            results.append(entry)

    # --- K-Means sweep: k=2..min(30, n) ---
    max_k = min(30, n)
    for k in range(2, max_k + 1):
        labels = cluster_kmeans(embeddings, n_clusters=k)

        n_clusters = len(set(labels.tolist()))
        sil = None
        if 2 <= n_clusters < len(embeddings):
            sil = float(silhouette_score(embeddings, labels))

        entry = {
            "algorithm": "kmeans",
            "n_clusters": k,
            "silhouette_score": sil,
            "noise_fraction": 0.0,
        }
        entry.update(_category_metrics(labels))
        results.append(entry)

    return results
