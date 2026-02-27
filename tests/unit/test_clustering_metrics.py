import numpy as np

from humpback.clustering.metrics import (
    compute_category_metrics,
    compute_cluster_metrics,
    extract_category_from_folder_path,
    run_parameter_sweep,
)


def test_compute_cluster_metrics_two_clusters():
    """Well-separated clusters should have positive silhouette score."""
    rng = np.random.RandomState(42)
    embeddings = np.vstack([
        rng.randn(20, 8).astype(np.float32) + 10,
        rng.randn(20, 8).astype(np.float32) - 10,
    ])
    labels = np.array([0] * 20 + [1] * 20)

    result = compute_cluster_metrics(embeddings, labels)

    assert result["silhouette_score"] is not None
    assert result["silhouette_score"] > 0
    assert result["davies_bouldin_index"] is not None
    assert result["calinski_harabasz_score"] is not None
    assert result["n_clusters"] == 2
    assert result["noise_count"] == 0


def test_compute_cluster_metrics_single_cluster():
    """Single cluster should return None for all score metrics."""
    embeddings = np.random.randn(20, 8).astype(np.float32)
    labels = np.zeros(20, dtype=int)

    result = compute_cluster_metrics(embeddings, labels)

    assert result["silhouette_score"] is None
    assert result["davies_bouldin_index"] is None
    assert result["calinski_harabasz_score"] is None
    assert result["n_clusters"] == 1


def test_compute_cluster_metrics_all_noise():
    """All noise labels should return None for all score metrics."""
    embeddings = np.random.randn(20, 8).astype(np.float32)
    labels = np.full(20, -1, dtype=int)

    result = compute_cluster_metrics(embeddings, labels)

    assert result["silhouette_score"] is None
    assert result["davies_bouldin_index"] is None
    assert result["calinski_harabasz_score"] is None
    assert result["n_clusters"] == 0
    assert result["noise_count"] == 20


def test_extract_category_from_folder_path():
    assert extract_category_from_folder_path("social-sounds/calls/subset1") == "calls"
    assert extract_category_from_folder_path("social-sounds/songs") == "songs"
    assert extract_category_from_folder_path("data/social-sounds/moans/file1") == "moans"
    assert extract_category_from_folder_path("other/path") is None
    assert extract_category_from_folder_path("") is None
    assert extract_category_from_folder_path("social-sounds") is None
    assert extract_category_from_folder_path("social-sounds/") is None


def test_compute_category_metrics():
    """Semi-supervised metrics with matching clusters and categories."""
    labels = np.array([0, 0, 0, 1, 1, 1, -1])
    categories = ["calls", "calls", "calls", "songs", "songs", "songs", "calls"]

    result = compute_category_metrics(labels, categories)

    assert result["adjusted_rand_index"] is not None
    assert result["adjusted_rand_index"] == 1.0  # perfect agreement
    assert result["normalized_mutual_info"] is not None
    assert result["n_categories"] == 2


def test_compute_category_metrics_insufficient_categories():
    """Should return None when fewer than 2 categories."""
    labels = np.array([0, 0, 1, 1])
    categories = ["calls", "calls", "calls", "calls"]

    result = compute_category_metrics(labels, categories)

    assert result["adjusted_rand_index"] is None
    assert result["normalized_mutual_info"] is None
    assert result["n_categories"] == 1


def test_run_parameter_sweep_basic():
    """Sweep should return valid results with expected keys."""
    rng = np.random.RandomState(42)
    embeddings = np.vstack([
        rng.randn(30, 4).astype(np.float32) + 5,
        rng.randn(30, 4).astype(np.float32) - 5,
    ])

    results = run_parameter_sweep(embeddings)

    assert len(results) > 0
    for entry in results:
        assert "min_cluster_size" in entry
        assert "silhouette_score" in entry
        assert "n_clusters" in entry
        assert "noise_fraction" in entry
        assert 0 <= entry["noise_fraction"] <= 1
        assert entry["min_cluster_size"] >= 2


def test_run_parameter_sweep_too_small():
    """Sweep with fewer than 4 points should return empty."""
    embeddings = np.random.randn(3, 4).astype(np.float32)
    results = run_parameter_sweep(embeddings)
    assert results == []
