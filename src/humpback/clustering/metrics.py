"""Cluster evaluation metrics and parameter sweep."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy in nats from a 1-D array of non-negative counts."""
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def _gini_coefficient(counts: np.ndarray) -> float:
    """Gini coefficient (0 = equal, 1 = concentrated)."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    n = len(counts)
    if total == 0 or n <= 1:
        return 0.0
    sorted_x = np.sort(counts)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_x)) / (n * total) - (n + 1.0) / n)


def _top_k_mass(counts: np.ndarray, k: int) -> float:
    """Fraction of the total in the top-*k* elements."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    top_k = np.sort(counts)[-k:]
    return float(top_k.sum() / total)


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


def compute_category_fragmentation(
    labels: np.ndarray,
    category_labels: list[str | None],
) -> dict[str, dict[str, float]]:
    """Per-category fragmentation metrics.

    For each non-None category, computes noise stats, top-k cluster mass,
    Shannon entropy (normalized against global cluster count), effective
    number of clusters, and Gini coefficient.
    """
    labels = np.asarray(labels)
    # Global non-noise cluster count (used for entropy normalization)
    non_noise_mask = labels != -1
    global_clusters = set(labels[non_noise_mask].tolist())
    n_clusters_global = len(global_clusters)
    log_n = float(np.log(n_clusters_global)) if n_clusters_global > 1 else 1.0

    # Group indices by category
    cat_to_indices: dict[str, list[int]] = {}
    for i, cat in enumerate(category_labels):
        if cat is not None:
            cat_to_indices.setdefault(cat, []).append(i)

    result: dict[str, dict[str, float]] = {}
    for cat, indices in sorted(cat_to_indices.items()):
        cat_labels = labels[indices]
        n_total = len(cat_labels)
        noise_mask = cat_labels == -1
        n_noise = int(noise_mask.sum())
        n_non_noise = n_total - n_noise

        noise_rate = n_noise / n_total if n_total > 0 else 0.0

        # Cluster distribution (non-noise only)
        non_noise_labels = cat_labels[~noise_mask]
        if len(non_noise_labels) > 0:
            unique, counts = np.unique(non_noise_labels, return_counts=True)
            entropy = _shannon_entropy(counts)
            neff = float(np.exp(entropy))
            gini = _gini_coefficient(counts)
            top1 = _top_k_mass(counts, 1)
            top2 = _top_k_mass(counts, 2)
            top3 = _top_k_mass(counts, 3)
            norm_entropy = entropy / log_n if log_n > 0 else 0.0
        else:
            entropy = 0.0
            neff = 0.0
            gini = 0.0
            top1 = top2 = top3 = 0.0
            norm_entropy = 0.0

        result[cat] = {
            "n_total": float(n_total),
            "n_non_noise": float(n_non_noise),
            "n_noise": float(n_noise),
            "noise_rate": noise_rate,
            "top1_mass": top1,
            "top2_mass": top2,
            "top3_mass": top3,
            "entropy": entropy,
            "normalized_entropy": norm_entropy,
            "neff": neff,
            "gini": gini,
        }

    return result


def compute_cluster_fragmentation(
    labels: np.ndarray,
    category_labels: list[str | None],
) -> dict[str, dict[str, Any]]:
    """Per-cluster composition metrics.

    For each non-noise cluster, computes size, dominant category, dominant mass,
    and Shannon entropy across categories.
    """
    labels = np.asarray(labels)
    # Count distinct non-None categories for entropy normalization
    valid_cats = set(c for c in category_labels if c is not None)
    n_categories = len(valid_cats)
    log_n_cat = float(np.log(n_categories)) if n_categories > 1 else 1.0

    # Group by cluster
    cluster_to_cats: dict[int, list[str]] = {}
    for lab, cat in zip(labels.tolist(), category_labels):
        if lab == -1 or cat is None:
            continue
        cluster_to_cats.setdefault(lab, []).append(cat)

    result: dict[str, dict[str, Any]] = {}
    for cl in sorted(cluster_to_cats.keys()):
        cats = cluster_to_cats[cl]
        size = len(cats)
        # Category distribution
        cat_counts: dict[str, int] = {}
        for c in cats:
            cat_counts[c] = cat_counts.get(c, 0) + 1
        counts_arr = np.array(list(cat_counts.values()), dtype=np.float64)
        dominant_cat = max(cat_counts, key=cat_counts.get)  # type: ignore[arg-type]
        dominant_mass = float(cat_counts[dominant_cat] / size) if size > 0 else 0.0
        entropy = _shannon_entropy(counts_arr)
        norm_entropy = entropy / log_n_cat if log_n_cat > 0 else 0.0

        result[str(cl)] = {
            "size": size,
            "dominant_category": dominant_cat,
            "dominant_mass": dominant_mass,
            "cluster_entropy": entropy,
            "cluster_entropy_norm": norm_entropy,
        }

    return result


def compute_global_fragmentation(
    category_frag: dict[str, dict[str, float]],
    cluster_frag: dict[str, dict[str, Any]],
    labels: np.ndarray,
    category_labels: list[str | None],
) -> dict[str, float]:
    """Weighted-average global fragmentation indices."""
    # Category-weighted averages (weight = n_non_noise)
    cat_weights = []
    cat_entropy_norm = []
    cat_neff = []
    cat_noise_rate = []
    for cat, m in category_frag.items():
        w = m["n_non_noise"]
        cat_weights.append(w)
        cat_entropy_norm.append(m["normalized_entropy"])
        cat_neff.append(m["neff"])
        cat_noise_rate.append(m["noise_rate"])

    total_cat_w = sum(cat_weights) or 1.0
    mean_entropy_norm = sum(w * v for w, v in zip(cat_weights, cat_entropy_norm)) / total_cat_w
    mean_neff = sum(w * v for w, v in zip(cat_weights, cat_neff)) / total_cat_w
    mean_noise_rate = sum(w * v for w, v in zip(cat_weights, cat_noise_rate)) / total_cat_w

    # Cluster-weighted average (weight = size)
    cl_weights = []
    cl_entropy_norm = []
    for cl, m in cluster_frag.items():
        cl_weights.append(m["size"])
        cl_entropy_norm.append(m["cluster_entropy_norm"])

    total_cl_w = sum(cl_weights) or 1.0
    mean_cluster_entropy_norm = sum(w * v for w, v in zip(cl_weights, cl_entropy_norm)) / total_cl_w

    return {
        "mean_entropy_norm": mean_entropy_norm,
        "mean_neff": mean_neff,
        "mean_noise_rate": mean_noise_rate,
        "mean_cluster_entropy_norm": mean_cluster_entropy_norm,
    }


def compute_fragmentation_report(
    labels: np.ndarray,
    category_labels: list[str | None],
    job_id: str,
) -> dict[str, Any] | None:
    """Build the full fragmentation report (``report.json``).

    Returns None if no valid (non-None) categories exist.
    """
    if not any(c is not None for c in category_labels):
        return None

    labels = np.asarray(labels)
    cat_frag = compute_category_fragmentation(labels, category_labels)
    cl_frag = compute_cluster_fragmentation(labels, category_labels)
    global_frag = compute_global_fragmentation(cat_frag, cl_frag, labels, category_labels)

    n_total = len(labels)
    n_noise_total = int((labels == -1).sum())

    non_noise_clusters = set(labels[labels != -1].tolist())
    n_clusters = len(non_noise_clusters)
    n_categories = len(cat_frag)

    return {
        "job_id": job_id,
        "category_fragmentation": cat_frag,
        "cluster_fragmentation": cl_frag,
        "global_fragmentation": global_frag,
        "summary": {
            "n_categories": n_categories,
            "n_clusters": n_clusters,
            "n_total": n_total,
            "n_noise_total": n_noise_total,
            "overall_noise_rate": n_noise_total / n_total if n_total > 0 else 0.0,
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
