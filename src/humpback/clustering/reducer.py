import numpy as np


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Reduce dimensionality with UMAP."""
    try:
        import umap

        effective_neighbors = min(n_neighbors, len(embeddings) - 1)
        if effective_neighbors < 2:
            # Too few samples for UMAP, fall back to SVD
            return _simple_reduce(embeddings, n_components)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    except ImportError:
        return _simple_reduce(embeddings, n_components)


def _simple_reduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Simple PCA-like reduction using SVD."""
    from numpy.linalg import svd

    centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = svd(centered, full_matrices=False)
    return (U[:, :n_components] * S[:n_components]).astype(np.float32)
