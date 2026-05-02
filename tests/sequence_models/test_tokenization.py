"""Tests for the per-k k-means tokenizer (ADR-061)."""

from __future__ import annotations

import numpy as np
import pytest

from humpback.sequence_models.tokenization import (
    compute_run_lengths,
    decode_tokens,
    fit_kmeans_token_model,
)


def _three_cluster_blob(seed: int = 0, n_per: int = 50, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            np.zeros(dim),
            np.full(dim, 5.0),
            np.concatenate([np.full(dim // 2, -3.0), np.full(dim - dim // 2, 3.0)]),
        ],
        dtype=np.float32,
    )
    blobs = []
    for c in centers:
        blobs.append(rng.standard_normal((n_per, dim)).astype(np.float32) * 0.2 + c)
    return np.concatenate(blobs, axis=0)


class TestFitKmeansTokenModel:
    def test_returns_positive_tau_equal_to_median_pairwise(self):
        Z = _three_cluster_blob(seed=1)
        kmeans, tau = fit_kmeans_token_model(Z, k=3, seed=42)
        assert tau > 0
        from scipy.spatial.distance import pdist

        expected_tau = float(np.median(pdist(kmeans.cluster_centers_)))
        assert abs(tau - expected_tau) < 1e-9

    def test_tau_scales_with_centroid_spread(self):
        Z = _three_cluster_blob(seed=2)
        _, tau1 = fit_kmeans_token_model(Z, k=3, seed=42)
        # Multiplying the data by 10 should multiply tau by 10
        # (centroids scale linearly).
        _, tau10 = fit_kmeans_token_model(Z * 10.0, k=3, seed=42)
        assert tau10 / tau1 == pytest.approx(10.0, rel=0.05)

    def test_too_few_samples_raises(self):
        Z = np.zeros((2, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="fewer than k"):
            fit_kmeans_token_model(Z, k=3, seed=0)

    def test_invalid_k_raises(self):
        Z = np.zeros((10, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="k must be"):
            fit_kmeans_token_model(Z, k=1, seed=0)

    def test_invalid_shape_raises(self):
        Z = np.zeros((10,), dtype=np.float32)
        with pytest.raises(ValueError, match="Z must be 2-D"):
            fit_kmeans_token_model(Z, k=2, seed=0)


class TestDecodeTokens:
    def test_confidences_are_probabilities(self):
        Z = _three_cluster_blob(seed=3)
        kmeans, tau = fit_kmeans_token_model(Z, k=3, seed=0)
        _, confidences = decode_tokens(Z, kmeans, tau)
        assert confidences.shape == (Z.shape[0],)
        assert confidences.dtype == np.float32
        assert np.all(confidences >= 0.0)
        assert np.all(confidences <= 1.0)

    def test_labels_match_nearest_centroid(self):
        Z = _three_cluster_blob(seed=4)
        kmeans, tau = fit_kmeans_token_model(Z, k=3, seed=0)
        labels, _ = decode_tokens(Z, kmeans, tau)
        # ``KMeans`` was fitted on float64; predict on float64 to match.
        expected = kmeans.predict(Z.astype(np.float64))
        np.testing.assert_array_equal(labels, expected)

    def test_deterministic_with_fixed_seed(self):
        Z = _three_cluster_blob(seed=5)
        k1, t1 = fit_kmeans_token_model(Z, k=3, seed=11)
        k2, t2 = fit_kmeans_token_model(Z, k=3, seed=11)
        l1, c1 = decode_tokens(Z, k1, t1)
        l2, c2 = decode_tokens(Z, k2, t2)
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_allclose(c1, c2, atol=1e-6)

    def test_invalid_tau_raises(self):
        Z = _three_cluster_blob(seed=6)
        kmeans, _ = fit_kmeans_token_model(Z, k=3, seed=0)
        with pytest.raises(ValueError, match="tau must be > 0"):
            decode_tokens(Z, kmeans, 0.0)
        with pytest.raises(ValueError, match="tau must be > 0"):
            decode_tokens(Z, kmeans, -0.1)


class TestComputeRunLengths:
    def test_empty_sequences_produce_empty_buckets(self):
        out = compute_run_lengths([], k=3)
        assert out == {"0": [], "1": [], "2": []}

    def test_all_same_token_one_run(self):
        out = compute_run_lengths([[1, 1, 1, 1]], k=3)
        assert out["1"] == [4]
        assert out["0"] == []
        assert out["2"] == []

    def test_alternating_tokens_individual_runs(self):
        out = compute_run_lengths([[0, 1, 0, 1, 0]], k=2)
        assert out["0"] == [1, 1, 1]
        assert out["1"] == [1, 1]

    def test_short_runs_at_sequence_boundaries(self):
        out = compute_run_lengths([[2, 0, 0, 1, 1, 1, 2]], k=3)
        assert out["2"] == [1, 1]
        assert out["0"] == [2]
        assert out["1"] == [3]

    def test_multiple_sequences_aggregate_buckets(self):
        out = compute_run_lengths(
            [
                [0, 0, 1, 1, 1],
                [1, 1, 0, 0],
                [],  # empty sequence skipped
                [2],  # single-token sequence
            ],
            k=3,
        )
        assert out["0"] == [2, 2]
        assert out["1"] == [3, 2]
        assert out["2"] == [1]

    def test_handles_token_index_above_k_minus_one(self):
        # Defensive: token-index strings keep keys within [0, k); a
        # corrupted upstream that emits a higher index should not crash.
        out = compute_run_lengths([[0, 5, 5, 0]], k=3)
        # Keys 0..k-1 always present; out-of-range tokens land in their
        # own buckets.
        assert out["0"] == [1, 1]
        assert out["1"] == []
        assert out["2"] == []

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be"):
            compute_run_lengths([[0, 0]], k=1)
