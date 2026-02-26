"""Clustering pipeline: load embeddings → optional reduce → cluster."""

from collections import Counter
from typing import Any

import numpy as np

from humpback.clustering.clusterer import cluster_hdbscan
from humpback.clustering.reducer import reduce_umap


def run_clustering_pipeline(
    embeddings: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run the full clustering pipeline.

    Returns (labels, reduced_embeddings).
    reduced_embeddings is None if UMAP was not applied.
    """
    params = parameters or {}

    reduced = None
    cluster_input = embeddings

    if params.get("use_umap", True) and embeddings.shape[1] > 2:
        n_components = params.get("umap_n_components", 2)
        n_neighbors = params.get("umap_n_neighbors", 15)
        min_dist = params.get("umap_min_dist", 0.1)
        reduced = reduce_umap(
            embeddings,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        cluster_input = reduced

    min_cluster_size = params.get("min_cluster_size", 5)
    min_samples = params.get("min_samples", None)
    labels = cluster_hdbscan(
        cluster_input,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    return labels, reduced


def compute_cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Compute size of each cluster from label array."""
    return dict(Counter(labels.tolist()))
