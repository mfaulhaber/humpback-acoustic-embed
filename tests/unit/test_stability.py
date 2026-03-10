"""Tests for clustering stability evaluation."""

import numpy as np
import pytest

from humpback.clustering.stability import (
    _aggregate_metric,
    _compute_pairwise_ari,
    _compute_run_metrics,
    _generate_seeds,
    run_stability_evaluation,
)


# ---------------------------------------------------------------------------
# Seed generation
# ---------------------------------------------------------------------------


def test_first_seed_is_base():
    seeds = _generate_seeds(5, base_seed=99)
    assert seeds[0] == 99


def test_deterministic_seeds():
    a = _generate_seeds(10, base_seed=42)
    b = _generate_seeds(10, base_seed=42)
    assert a == b


def test_correct_length():
    for n in [1, 3, 10]:
        assert len(_generate_seeds(n)) == n


# ---------------------------------------------------------------------------
# Pairwise ARI
# ---------------------------------------------------------------------------


def test_identical_labels_ari_one():
    labels = np.array([0, 0, 1, 1, 2, 2])
    result = _compute_pairwise_ari([labels, labels.copy(), labels.copy()])
    assert result["mean_pairwise_ari"] == pytest.approx(1.0)
    assert result["std_pairwise_ari"] == pytest.approx(0.0)


def test_single_run_returns_none():
    labels = np.array([0, 1, 2])
    result = _compute_pairwise_ari([labels])
    assert result["mean_pairwise_ari"] is None
    assert result["std_pairwise_ari"] is None
    assert result["min_pairwise_ari"] is None
    assert result["max_pairwise_ari"] is None


def test_random_labels_low_ari():
    rng = np.random.RandomState(42)
    runs = [rng.randint(0, 5, size=100) for _ in range(5)]
    result = _compute_pairwise_ari(runs)
    assert result["mean_pairwise_ari"] is not None
    assert result["mean_pairwise_ari"] < 0.5


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def test_aggregate_basic():
    runs = [
        {"score": 1.0},
        {"score": 3.0},
        {"score": 2.0},
    ]
    agg = _aggregate_metric(runs, "score")
    assert agg["score_mean"] == pytest.approx(2.0)
    assert agg["score_min"] == pytest.approx(1.0)
    assert agg["score_max"] == pytest.approx(3.0)
    assert agg["score_std"] is not None


def test_aggregate_all_none():
    runs = [{"score": None}, {"score": None}]
    agg = _aggregate_metric(runs, "score")
    assert agg["score_mean"] is None
    assert agg["score_std"] is None


def test_aggregate_mixed_none():
    runs = [{"score": 1.0}, {"score": None}, {"score": 3.0}]
    agg = _aggregate_metric(runs, "score")
    assert agg["score_mean"] == pytest.approx(2.0)
    assert agg["score_min"] == pytest.approx(1.0)
    assert agg["score_max"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Run metrics
# ---------------------------------------------------------------------------


def test_run_metrics_no_categories():
    rng = np.random.RandomState(42)
    embeddings = rng.randn(30, 8).astype(np.float32)
    labels = np.array([0] * 15 + [1] * 15)
    result = _compute_run_metrics(labels, embeddings, None)
    assert result["n_clusters"] == 2
    assert result["noise_fraction"] == 0.0
    assert result["adjusted_rand_index"] is None
    assert result["normalized_mutual_info"] is None
    assert result["fragmentation_index"] is None


def test_run_metrics_with_categories():
    rng = np.random.RandomState(42)
    embeddings = rng.randn(30, 8).astype(np.float32)
    labels = np.array([0] * 15 + [1] * 15)
    categories = ["catA"] * 15 + ["catB"] * 15
    result = _compute_run_metrics(labels, embeddings, categories)
    assert result["n_clusters"] == 2
    assert result["adjusted_rand_index"] is not None
    assert result["normalized_mutual_info"] is not None
    assert result["fragmentation_index"] is not None


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def test_basic_stability_runs():
    rng = np.random.RandomState(42)
    embeddings = rng.randn(40, 16).astype(np.float32)
    result = run_stability_evaluation(
        embeddings,
        {"clustering_algorithm": "kmeans", "n_clusters": 3, "reduction_method": "none"},
        None,
        n_runs=3,
    )
    assert result["n_runs"] == 3
    assert len(result["seeds"]) == 3
    assert len(result["per_run"]) == 3
    assert "pairwise_label_agreement" in result
    assert "aggregate_metrics" in result
    assert "n_clusters_mean" in result["aggregate_metrics"]


def test_too_few_runs_raises():
    rng = np.random.RandomState(42)
    embeddings = rng.randn(20, 8).astype(np.float32)
    with pytest.raises(ValueError, match="n_runs must be >= 2"):
        run_stability_evaluation(embeddings, None, None, n_runs=1)


def test_deterministic_reduction_perfect_stability():
    """PCA is deterministic, so all runs should produce identical labels -> ARI=1.0."""
    rng = np.random.RandomState(42)
    cluster_a = rng.randn(25, 32).astype(np.float32) + 5
    cluster_b = rng.randn(25, 32).astype(np.float32) - 5
    embeddings = np.vstack([cluster_a, cluster_b])

    result = run_stability_evaluation(
        embeddings,
        {
            "reduction_method": "pca",
            "clustering_algorithm": "hdbscan",
            "min_cluster_size": 5,
        },
        None,
        n_runs=3,
    )

    pla = result["pairwise_label_agreement"]
    assert pla["mean_pairwise_ari"] == pytest.approx(1.0)
    assert pla["std_pairwise_ari"] == pytest.approx(0.0)


def test_no_reduction_hdbscan_perfect_stability():
    """With reduction_method='none' and HDBSCAN (deterministic), ARI should be 1.0."""
    rng = np.random.RandomState(42)
    cluster_a = rng.randn(25, 8).astype(np.float32) + 5
    cluster_b = rng.randn(25, 8).astype(np.float32) - 5
    embeddings = np.vstack([cluster_a, cluster_b])

    result = run_stability_evaluation(
        embeddings,
        {
            "reduction_method": "none",
            "clustering_algorithm": "hdbscan",
            "min_cluster_size": 5,
        },
        None,
        n_runs=3,
    )

    pla = result["pairwise_label_agreement"]
    assert pla["mean_pairwise_ari"] == pytest.approx(1.0)
    assert pla["std_pairwise_ari"] == pytest.approx(0.0)
