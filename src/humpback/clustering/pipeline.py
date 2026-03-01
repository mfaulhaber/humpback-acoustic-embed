"""Clustering pipeline: load embeddings -> optional reduce -> cluster."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from humpback.clustering.clusterer import cluster_hdbscan
from humpback.clustering.reducer import reduce_umap


@dataclass
class ClusteringResult:
    """Result of the clustering pipeline."""

    labels: np.ndarray
    reduced_embeddings: np.ndarray | None
    cluster_input: np.ndarray  # what HDBSCAN clustered on (needed for metrics)


def run_clustering_pipeline(
    embeddings: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> ClusteringResult:
    """Run the full clustering pipeline.

    Returns a ClusteringResult with labels, optional UMAP-reduced embeddings,
    and the array that HDBSCAN actually clustered on.
    """
    params = parameters or {}

    reduced = None
    cluster_input = embeddings

    if params.get("use_umap", True) and embeddings.shape[1] > 2:
        # Backward compat: fall back to umap_n_components if new key absent
        cluster_n_components = params.get(
            "umap_cluster_n_components",
            params.get("umap_n_components", 5),
        )
        n_neighbors = params.get("umap_n_neighbors", 15)
        min_dist = params.get("umap_min_dist", 0.1)

        # Clamp to valid range
        cluster_n_components = max(2, min(cluster_n_components, embeddings.shape[1]))

        umap_kwargs = dict(n_neighbors=n_neighbors, min_dist=min_dist)

        if cluster_n_components == 2:
            # Single pass: 2D serves both clustering and visualization
            reduced = reduce_umap(
                embeddings, n_components=2, **umap_kwargs,
            )
            cluster_input = reduced
        else:
            # Two passes: high-dim for HDBSCAN, 2D for visualization
            cluster_input = reduce_umap(
                embeddings, n_components=cluster_n_components, **umap_kwargs,
            )
            reduced = reduce_umap(
                embeddings, n_components=2, **umap_kwargs,
            )

    min_cluster_size = params.get("min_cluster_size", 5)
    min_samples = params.get("min_samples", None)
    cluster_selection_method = params.get("cluster_selection_method", "leaf")
    labels = cluster_hdbscan(
        cluster_input,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    )

    return ClusteringResult(
        labels=labels,
        reduced_embeddings=reduced,
        cluster_input=cluster_input,
    )


def compute_cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Compute size of each cluster from label array."""
    return dict(Counter(labels.tolist()))
