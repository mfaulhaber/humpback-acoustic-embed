import numpy as np


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
) -> np.ndarray:
    """Reduce dimensionality with UMAP."""
    try:
        import umap

        effective_neighbors = min(n_neighbors, len(embeddings) - 1)
        if effective_neighbors < 2:
            # Too few samples for UMAP, fall back to PCA
            return _svd_reduce(embeddings, n_components)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=effective_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    except ImportError:
        return _svd_reduce(embeddings, n_components)


def reduce_pca(
    embeddings: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """Reduce dimensionality with PCA (deterministic, good for small datasets)."""
    from sklearn.decomposition import PCA

    n_components = min(n_components, embeddings.shape[1], len(embeddings))
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings).astype(np.float32)


def _svd_reduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Simple PCA-like reduction using SVD (no sklearn dependency)."""
    from numpy.linalg import svd

    centered = embeddings - embeddings.mean(axis=0)
    U, S, Vt = svd(centered, full_matrices=False)
    return (U[:, :n_components] * S[:n_components]).astype(np.float32)
