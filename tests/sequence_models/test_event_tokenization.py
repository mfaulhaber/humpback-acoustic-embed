"""Tests for Event Encoder preprocessing and tokenization."""

import numpy as np
import pytest

from humpback.sequence_models.event_tokenization import (
    fit_kmeans_tokenizers,
    preprocess_event_features,
    robust_zscore,
    tokenization_summary,
)


def test_preprocess_l2_normalizes_pool_blocks_and_clamps_pca_dim():
    pool_matrix = np.asarray(
        [
            [3.0, 4.0, 0.0, 5.0],
            [6.0, 8.0, 1.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    descriptors = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [100.0, 30.0],
        ],
        dtype=np.float32,
    )

    result = preprocess_event_features(
        pool_matrix,
        descriptors,
        pool_dim=2,
        pool_count=2,
        pca_dim=128,
        random_seed=7,
    )

    assert result.effective_pca_dim == 2
    assert result.embedding_vectors.shape == (3, 2)
    assert result.descriptor_vectors.shape == (3, 2)
    assert result.event_vectors.shape == (3, 4)
    assert result.pca_model is not None


def test_preprocess_supports_single_event_without_pca_fit():
    result = preprocess_event_features(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[5.0]], dtype=np.float32),
        pool_dim=1,
        pool_count=2,
        pca_dim=64,
    )

    assert result.effective_pca_dim == 2
    assert result.pca_model is None
    assert result.event_vectors.shape == (1, 3)


def test_robust_zscore_uses_median_and_mad_floor():
    scaled, medians, scales = robust_zscore(
        np.asarray(
            [
                [1.0, 5.0],
                [2.0, 5.0],
                [100.0, 5.0],
            ],
            dtype=np.float32,
        )
    )

    assert np.allclose(medians, [2.0, 5.0])
    assert scales[0] > 0
    assert scales[1] > 0
    assert scaled[1, 0] == pytest.approx(0.0)
    assert scaled[:, 1].tolist() == [0.0, 0.0, 0.0]


def test_fit_kmeans_tokenizers_skips_impossible_k_and_remaps_labels():
    features = np.asarray(
        [
            [-10.0, 0.0],
            [-9.0, 0.0],
            [9.0, 0.0],
            [10.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = fit_kmeans_tokenizers(features, [1, 2, 8], random_seed=0)

    assert result.invalid_k_values == [8]
    assert sorted(result.tokenizations) == [1, 2]
    one = result.tokenizations[1]
    assert np.isnan(one.second_distances).all()
    assert np.allclose(one.confidences, 1.0)

    two = result.tokenizations[2]
    assert two.token_ids[:2].tolist() == [0, 0]
    assert two.token_ids[2:].tolist() == [1, 1]
    assert np.all(two.distances >= 0)
    assert np.all((two.confidences >= 0) & (two.confidences <= 1))
    assert two.token_counts == {0: 2, 1: 2}


def test_tokenization_summary_is_json_friendly():
    result = fit_kmeans_tokenizers(
        np.asarray([[0.0], [1.0], [10.0]], dtype=np.float32),
        [2],
        random_seed=0,
    )

    summary = tokenization_summary(result)

    assert list(summary) == ["2"]
    assert "inertia" in summary["2"]
    assert set(summary["2"]["token_counts"]) == {"0", "1"}
