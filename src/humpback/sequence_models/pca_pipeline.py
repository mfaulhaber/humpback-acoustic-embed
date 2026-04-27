"""PCA dimensionality reduction pipeline for HMM sequence modeling.

Optionally L2-normalizes embeddings before fitting PCA. All operations
are deterministic given a fixed ``random_state``.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def l2_normalize_sequences(sequences: list[np.ndarray]) -> list[np.ndarray]:
    """L2-normalize each row of each sequence in-place-safe."""
    return [
        np.asarray(normalize(seq, norm="l2"), dtype=np.float32) for seq in sequences
    ]


def fit_pca(
    sequences: list[np.ndarray],
    pca_dims: int,
    *,
    whiten: bool = False,
    l2_norm: bool = True,
    random_state: int = 42,
) -> tuple[PCA, list[np.ndarray]]:
    """Fit PCA on concatenated sequences and return (model, preprocessed_seqs).

    The returned ``preprocessed_seqs`` are L2-normalized (if requested) but
    **not** PCA-transformed — call ``transform_sequences`` separately. This
    separation lets the caller inspect or persist the preprocessed data.
    """
    if l2_norm:
        sequences = l2_normalize_sequences(sequences)

    concatenated = np.concatenate(sequences, axis=0)
    effective_dims = min(pca_dims, concatenated.shape[1], concatenated.shape[0])

    pca = PCA(n_components=effective_dims, whiten=whiten, random_state=random_state)
    pca.fit(concatenated)

    return pca, sequences


def transform_sequences(pca: PCA, sequences: list[np.ndarray]) -> list[np.ndarray]:
    """Project each sequence through a fitted PCA model."""
    return [pca.transform(seq).astype(np.float32) for seq in sequences]
