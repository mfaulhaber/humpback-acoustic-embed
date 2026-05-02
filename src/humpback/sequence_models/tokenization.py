"""K-means tokenization with softmax-temperature confidence (ADR-061).

Given a stack of contextual embeddings ``Z`` from the masked-transformer
encoder, this module fits one ``KMeans`` per requested ``k`` and decodes
per-chunk token labels with a [0,1] confidence score derived from a
softmax over negative squared distances scaled by ``tau``.

The temperature ``tau`` is auto-fit per job from the median pairwise
distance between centroids; this keeps confidences calibrated against
the spread of clusters in the contextual-embedding space rather than
some absolute scale.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or k < 2:
        raise ValueError(f"k must be an integer >= 2, got {k!r}")


def fit_kmeans_token_model(Z: np.ndarray, k: int, seed: int) -> tuple[KMeans, float]:
    """Fit a k-means tokenizer over contextual embeddings.

    Parameters
    ----------
    Z
        Contextual embeddings stacked along the first axis with shape
        ``(N, d_model)``.
    k
        Number of clusters / token vocabulary size.
    seed
        Random state passed to scikit-learn for reproducibility.

    Returns
    -------
    kmeans
        Fitted ``KMeans`` instance.
    tau
        Median pairwise distance between centroids. Always positive
        (clamped to a small floor when k=2 collapses both centroids).
    """
    _validate_k(k)
    Z_arr = np.asarray(Z, dtype=np.float64)
    if Z_arr.ndim != 2:
        raise ValueError(f"Z must be 2-D (N, D); got shape {Z_arr.shape}")
    if Z_arr.shape[0] < k:
        raise ValueError(f"Z has only {Z_arr.shape[0]} samples, fewer than k={k}")

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    kmeans.fit(Z_arr)

    centroids = kmeans.cluster_centers_
    if centroids.shape[0] < 2:
        tau = 1.0
    else:
        pairwise = pdist(centroids)
        if pairwise.size == 0:
            tau = 1.0
        else:
            tau = float(np.median(pairwise))
    if tau <= 0:
        tau = 1e-6
    return kmeans, tau


def decode_tokens(
    Z: np.ndarray, kmeans: KMeans, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """Assign per-chunk token labels and softmax-temperature confidences.

    Confidence at frame ``t`` is
    ``max_c softmax(-||z_t - mu_c||^2 / tau)``.

    Returns ``(labels, confidences)`` where labels are int32 and
    confidences are float32 in ``[0, 1]``.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    Z_arr = np.asarray(Z, dtype=np.float64)
    if Z_arr.ndim != 2:
        raise ValueError(f"Z must be 2-D (N, D); got shape {Z_arr.shape}")
    centroids = kmeans.cluster_centers_

    # Squared distances (N, k).
    sq_dists = np.sum((Z_arr[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    # Stable softmax over -d^2 / tau with temperature scaling.
    scaled = -sq_dists / float(tau)
    scaled -= scaled.max(axis=1, keepdims=True)
    exps = np.exp(scaled)
    denom = exps.sum(axis=1, keepdims=True)
    probs = exps / denom

    labels = np.argmin(sq_dists, axis=1).astype(np.int32)
    confidences = probs.max(axis=1).astype(np.float32)
    # Clip into [0, 1] in case of float drift.
    confidences = np.clip(confidences, 0.0, 1.0)
    return labels, confidences


def compute_run_lengths(
    token_sequences: Iterable[Iterable[int]], k: int
) -> dict[str, list[int]]:
    """Per-token run-length arrays.

    Returns a dict keyed by ``str(token_index)`` matching the existing
    HMM dwell-histogram convention. Empty sequences contribute nothing;
    a single-token sequence of length L contributes one run of length L
    to that token's bucket.
    """
    _validate_k(k)
    runs: dict[int, list[int]] = {i: [] for i in range(k)}
    for seq in token_sequences:
        seq_list = list(seq)
        if not seq_list:
            continue
        run_token = int(seq_list[0])
        run_length = 1
        for tok in seq_list[1:]:
            tok_int = int(tok)
            if tok_int == run_token:
                run_length += 1
            else:
                runs.setdefault(run_token, []).append(run_length)
                run_token = tok_int
                run_length = 1
        runs.setdefault(run_token, []).append(run_length)
    return {str(i): runs.get(i, []) for i in range(k)}


__all__ = [
    "compute_run_lengths",
    "decode_tokens",
    "fit_kmeans_token_model",
]
