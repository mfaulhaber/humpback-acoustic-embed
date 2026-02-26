import numpy as np


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
) -> np.ndarray:
    """Cluster embeddings with HDBSCAN. Returns integer labels (-1 = noise)."""
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min(min_cluster_size, len(embeddings)),
            min_samples=min_samples,
        )
        return clusterer.fit_predict(embeddings)
    except ImportError:
        # Fallback: simple k-means-like clustering for testing
        return _simple_cluster(embeddings, min_cluster_size)


def _simple_cluster(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Simple fallback clustering: assign to bins based on first component."""
    if len(embeddings) < min_cluster_size:
        return np.zeros(len(embeddings), dtype=np.intp)

    n_clusters = max(1, len(embeddings) // min_cluster_size)
    n_clusters = min(n_clusters, 10)

    # Use first principal component for splitting
    centered = embeddings - embeddings.mean(axis=0)
    if centered.shape[1] > 1:
        # Use first component
        scores = centered[:, 0]
    else:
        scores = centered.ravel()

    # Assign to equal-sized bins
    order = np.argsort(scores)
    labels = np.zeros(len(embeddings), dtype=np.intp)
    chunk_size = max(1, len(embeddings) // n_clusters)
    for i, idx in enumerate(order):
        labels[idx] = min(i // chunk_size, n_clusters - 1)

    return labels
