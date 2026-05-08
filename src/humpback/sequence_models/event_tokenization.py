"""Preprocessing and k-means tokenization for Event Encoder jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class PreprocessResult:
    event_vectors: np.ndarray
    embedding_vectors: np.ndarray
    descriptor_vectors: np.ndarray
    pca_model: PCA | None
    effective_pca_dim: int
    descriptor_median: np.ndarray
    descriptor_scale: np.ndarray


@dataclass(frozen=True)
class KMeansTokenization:
    k: int
    model: KMeans
    token_ids: np.ndarray
    distances: np.ndarray
    second_distances: np.ndarray
    confidences: np.ndarray
    inertia: float
    token_counts: dict[int, int]


@dataclass(frozen=True)
class TokenizationResult:
    tokenizations: dict[int, KMeansTokenization]
    invalid_k_values: list[int]


def preprocess_event_features(
    pool_matrix: np.ndarray,
    descriptor_matrix: np.ndarray,
    *,
    pool_dim: int,
    pool_count: int,
    l2_normalize_pools: bool = True,
    pca_dim: int = 128,
    embedding_weight: float = 1.0,
    descriptor_weight: float = 1.0,
    descriptor_clip_value: float | None = 3.0,
    random_seed: int = 0,
    eps: float = 1e-12,
) -> PreprocessResult:
    """Normalize, reduce, scale, and concatenate event feature blocks."""
    pools = np.asarray(pool_matrix, dtype=np.float32)
    descriptors = np.asarray(descriptor_matrix, dtype=np.float32)
    if pools.ndim != 2:
        raise ValueError("pool_matrix must be 2D")
    if descriptors.ndim != 2:
        raise ValueError("descriptor_matrix must be 2D")
    if pools.shape[0] != descriptors.shape[0]:
        raise ValueError("pool_matrix and descriptor_matrix row counts must match")
    if pool_dim <= 0 or pool_count <= 0:
        raise ValueError("pool_dim and pool_count must be > 0")
    if pools.shape[1] != pool_dim * pool_count:
        raise ValueError("pool_matrix width must equal pool_dim * pool_count")
    if embedding_weight < 0 or descriptor_weight < 0:
        raise ValueError("feature weights must be >= 0")
    if embedding_weight == 0 and descriptor_weight == 0:
        raise ValueError("at least one feature weight must be > 0")
    if descriptor_clip_value is not None and descriptor_clip_value < 0:
        raise ValueError("descriptor_clip_value must be >= 0")

    normalized_pools = (
        _l2_normalize_pool_blocks(
            pools, pool_dim=pool_dim, pool_count=pool_count, eps=eps
        )
        if l2_normalize_pools
        else pools
    )
    n_events, embedding_dim = normalized_pools.shape
    effective_pca_dim = min(int(pca_dim), embedding_dim)
    if n_events > 1:
        effective_pca_dim = max(1, min(effective_pca_dim, n_events - 1))
    else:
        effective_pca_dim = max(1, effective_pca_dim)

    if n_events > 1 and effective_pca_dim < embedding_dim:
        pca = PCA(n_components=effective_pca_dim, random_state=random_seed)
        embedding_vectors = pca.fit_transform(normalized_pools).astype(np.float32)
    else:
        pca = None
        embedding_vectors = normalized_pools[:, :effective_pca_dim].astype(np.float32)

    descriptor_vectors, medians, scales = robust_zscore(descriptors, eps=eps)
    if descriptor_clip_value is not None:
        descriptor_vectors = np.clip(
            descriptor_vectors,
            -float(descriptor_clip_value),
            float(descriptor_clip_value),
        ).astype(np.float32)
    event_vectors = np.concatenate(
        [
            embedding_vectors * float(embedding_weight),
            descriptor_vectors * float(descriptor_weight),
        ],
        axis=1,
    ).astype(np.float32)
    return PreprocessResult(
        event_vectors=event_vectors,
        embedding_vectors=embedding_vectors,
        descriptor_vectors=descriptor_vectors,
        pca_model=pca,
        effective_pca_dim=effective_pca_dim,
        descriptor_median=medians,
        descriptor_scale=scales,
    )


def robust_zscore(
    values: np.ndarray, *, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust z-score columns using median and scaled MAD."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("values must be 2D")
    medians = np.median(matrix, axis=0).astype(np.float32)
    mad = np.median(np.abs(matrix - medians[None, :]), axis=0).astype(np.float32)
    scales = np.maximum(1.4826 * mad, eps).astype(np.float32)
    return (
        ((matrix - medians[None, :]) / scales[None, :]).astype(np.float32),
        medians,
        scales,
    )


def fit_kmeans_tokenizers(
    feature_matrix: np.ndarray,
    k_values: list[int],
    *,
    random_seed: int = 0,
) -> TokenizationResult:
    """Fit deterministic KMeans tokenizers for every feasible k."""
    features = np.asarray(feature_matrix, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("feature_matrix must be 2D")
    n_events = features.shape[0]
    invalid = [int(k) for k in k_values if k <= 0 or k > n_events]
    tokenizations: dict[int, KMeansTokenization] = {}
    for k in sorted({int(k) for k in k_values if 0 < k <= n_events}):
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=cast(Any, 10))
        raw_labels = model.fit_predict(features)
        tokenizations[k] = _build_tokenization(k, model, raw_labels, features)
    return TokenizationResult(
        tokenizations=tokenizations,
        invalid_k_values=sorted(invalid),
    )


def tokenization_summary(result: TokenizationResult) -> dict[str, Any]:
    """Return JSON-friendly tokenization diagnostics."""
    return {
        str(k): {
            "inertia": tokenization.inertia,
            "token_counts": {
                str(token_id): count
                for token_id, count in tokenization.token_counts.items()
            },
        }
        for k, tokenization in sorted(result.tokenizations.items())
    }


def _build_tokenization(
    k: int,
    model: KMeans,
    raw_labels: np.ndarray,
    features: np.ndarray,
) -> KMeansTokenization:
    centroids = model.cluster_centers_.astype(np.float32)
    label_to_token = _centroid_label_order(centroids)
    token_ids = np.asarray(
        [label_to_token[int(label)] for label in raw_labels], dtype=np.int32
    )
    ordered_centroids = np.empty_like(centroids)
    for raw_label, token_id in label_to_token.items():
        ordered_centroids[token_id] = centroids[raw_label]

    all_distances = np.linalg.norm(
        features[:, None, :] - ordered_centroids[None, :, :], axis=2
    ).astype(np.float32)
    distances = all_distances[np.arange(features.shape[0]), token_ids].astype(
        np.float32
    )
    if k == 1:
        second_distances = np.full(features.shape[0], np.nan, dtype=np.float32)
        confidences = np.ones(features.shape[0], dtype=np.float32)
    else:
        sorted_distances = np.sort(all_distances, axis=1)
        second_distances = sorted_distances[:, 1].astype(np.float32)
        confidences = np.clip(
            1.0 - (distances / (second_distances + 1e-12)), 0.0, 1.0
        ).astype(np.float32)

    token_counts = {
        int(token_id): int(np.sum(token_ids == token_id)) for token_id in range(k)
    }
    return KMeansTokenization(
        k=k,
        model=model,
        token_ids=token_ids,
        distances=distances,
        second_distances=second_distances,
        confidences=confidences,
        inertia=float(cast(Any, model.inertia_)),
        token_counts=token_counts,
    )


def _centroid_label_order(centroids: np.ndarray) -> dict[int, int]:
    order = sorted(
        range(centroids.shape[0]),
        key=lambda idx: (
            float(centroids[idx, 0]),
            float(np.linalg.norm(centroids[idx])),
        ),
    )
    return {raw_label: token_id for token_id, raw_label in enumerate(order)}


def _l2_normalize_pool_blocks(
    pools: np.ndarray, *, pool_dim: int, pool_count: int, eps: float
) -> np.ndarray:
    reshaped = pools.reshape(pools.shape[0], pool_count, pool_dim).astype(np.float32)
    norms = np.linalg.norm(reshaped, axis=2, keepdims=True)
    normalized = reshaped / np.maximum(norms, eps)
    return normalized.reshape(pools.shape).astype(np.float32)
