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

    return ClusteringResult(
        labels=labels,
        reduced_embeddings=reduced,
        cluster_input=cluster_input,
    )


def compute_cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Compute size of each cluster from label array."""
    return dict(Counter(labels.tolist()))
