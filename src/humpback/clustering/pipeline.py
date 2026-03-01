"""Clustering pipeline: load embeddings -> optional reduce -> cluster."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from humpback.clustering.clusterer import cluster_agglomerative, cluster_hdbscan, cluster_kmeans
from humpback.clustering.reducer import reduce_pca, reduce_umap


@dataclass
class ClusteringResult:
    """Result of the clustering pipeline."""

    labels: np.ndarray
    reduced_embeddings: np.ndarray | None
    cluster_input: np.ndarray  # what the clusterer ran on (needed for metrics)


def run_clustering_pipeline(
    embeddings: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> ClusteringResult:
    """Run the full clustering pipeline.

    Parameters (all optional, via ``parameters`` dict):
    - ``reduction_method``: ``"umap"`` (default), ``"pca"``, or ``"none"``
    - ``distance_metric``: ``"euclidean"`` (default) or ``"cosine"``
    - ``clustering_algorithm``: ``"hdbscan"`` (default), ``"kmeans"``, ``"agglomerative"``
    - ``n_clusters``: int (default 15, for kmeans/agglomerative)
    - ``linkage``: str (default ``"ward"``, for agglomerative)
    - ``use_umap``: bool (backward compat, overridden by ``reduction_method``)
    """
    params = parameters or {}

    distance_metric = params.get("distance_metric", "euclidean")
    reduction_method = params.get("reduction_method", None)

    # Backward compat: use_umap=False maps to reduction_method="none"
    if reduction_method is None:
        reduction_method = "none" if not params.get("use_umap", True) else "umap"

    reduced = None
    cluster_input = embeddings

    if reduction_method != "none" and embeddings.shape[1] > 2:
        cluster_n_components = params.get(
            "umap_cluster_n_components",
            params.get("umap_n_components", 5),
        )
        cluster_n_components = max(2, min(cluster_n_components, embeddings.shape[1]))

        if reduction_method == "pca":
            if cluster_n_components == 2:
                reduced = reduce_pca(embeddings, n_components=2)
                cluster_input = reduced
            else:
                cluster_input = reduce_pca(embeddings, n_components=cluster_n_components)
                reduced = reduce_pca(embeddings, n_components=2)
        else:
            # UMAP (default)
            n_neighbors = params.get("umap_n_neighbors", 15)
            min_dist = params.get("umap_min_dist", 0.1)
            umap_kwargs = dict(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=distance_metric,
            )

            if cluster_n_components == 2:
                reduced = reduce_umap(embeddings, n_components=2, **umap_kwargs)
                cluster_input = reduced
            else:
                cluster_input = reduce_umap(
                    embeddings, n_components=cluster_n_components, **umap_kwargs,
                )
                reduced = reduce_umap(embeddings, n_components=2, **umap_kwargs)

    # --- Clustering ---
    algorithm = params.get("clustering_algorithm", "hdbscan")

    if algorithm == "kmeans":
        n_clusters = params.get("n_clusters", 15)
        labels = cluster_kmeans(cluster_input, n_clusters=n_clusters)
    elif algorithm == "agglomerative":
        n_clusters = params.get("n_clusters", 15)
        linkage = params.get("linkage", "ward")
        labels = cluster_agglomerative(
            cluster_input,
            n_clusters=n_clusters,
            linkage=linkage,
            metric=distance_metric,
        )
    else:
        # HDBSCAN (default)
        min_cluster_size = params.get("min_cluster_size", 5)
        min_samples = params.get("min_samples", None)
        cluster_selection_method = params.get("cluster_selection_method", "leaf")
        labels = cluster_hdbscan(
            cluster_input,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric=distance_metric,
        )

    return ClusteringResult(
        labels=labels,
        reduced_embeddings=reduced,
        cluster_input=cluster_input,
    )


def compute_cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Compute size of each cluster from label array."""
    return dict(Counter(labels.tolist()))
