import numpy as np

from humpback.clustering.metrics import (
    _gini_coefficient,
    _shannon_entropy,
    _top_k_mass,
    compute_category_fragmentation,
    compute_category_metrics,
    compute_cluster_fragmentation,
    compute_cluster_metrics,
    compute_detailed_category_metrics,
    compute_fragmentation_report,
    compute_global_fragmentation,
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
    """Last path component is used as category label."""
    assert extract_category_from_folder_path("Emily-Vierling-Orcasound-data/Grunt") == "Grunt"
    assert extract_category_from_folder_path("social-sounds/calls") == "calls"
    assert extract_category_from_folder_path("data/social-sounds/moans") == "moans"
    assert extract_category_from_folder_path("Upsweep") == "Upsweep"
    assert extract_category_from_folder_path("a/b/c/Shriek") == "Shriek"
    assert extract_category_from_folder_path("") is None
    assert extract_category_from_folder_path("///") is None


def test_extract_category_trailing_slash():
    """Trailing slashes should be ignored."""
    assert extract_category_from_folder_path("data/Grunt/") == "Grunt"
    assert extract_category_from_folder_path("Buzz/") == "Buzz"


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


def test_compute_detailed_category_metrics_perfect():
    """Perfect clustering should have ARI=1, homogeneity=1, purity=1."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    categories = ["Grunt", "Grunt", "Grunt", "Buzz", "Buzz", "Buzz"]

    result = compute_detailed_category_metrics(labels, categories)

    assert result["adjusted_rand_index"] == 1.0
    assert result["normalized_mutual_info"] is not None
    assert result["homogeneity"] == 1.0
    assert result["completeness"] == 1.0
    assert result["v_measure"] == 1.0
    assert result["n_categories"] == 2
    assert result["per_category_purity"]["Grunt"] == 1.0
    assert result["per_category_purity"]["Buzz"] == 1.0
    assert result["confusion_matrix"]["Grunt"] == {"0": 3}
    assert result["confusion_matrix"]["Buzz"] == {"1": 3}


def test_compute_detailed_category_metrics_mixed():
    """Mixed clustering should have purity < 1 for split categories."""
    labels = np.array([0, 0, 1, 1, 1, 0])
    categories = ["Grunt", "Grunt", "Grunt", "Buzz", "Buzz", "Buzz"]

    result = compute_detailed_category_metrics(labels, categories)

    assert result["adjusted_rand_index"] is not None
    assert result["adjusted_rand_index"] < 1.0
    assert result["per_category_purity"]["Grunt"] is not None
    assert "confusion_matrix" in result


def test_compute_detailed_category_metrics_insufficient():
    """Should return None fields when < 2 categories."""
    labels = np.array([0, 0, 1])
    categories = ["Grunt", "Grunt", "Grunt"]

    result = compute_detailed_category_metrics(labels, categories)

    assert result["adjusted_rand_index"] is None
    assert result["homogeneity"] is None
    assert result["per_category_purity"] == {}
    assert result["confusion_matrix"] == {}


def test_compute_detailed_category_metrics_with_noise():
    """Noise points (label=-1) should be excluded from metrics."""
    labels = np.array([0, 0, -1, 1, 1, -1])
    categories = ["Grunt", "Grunt", "Grunt", "Buzz", "Buzz", "Buzz"]

    result = compute_detailed_category_metrics(labels, categories)

    assert result["adjusted_rand_index"] == 1.0
    assert result["n_categories"] == 2


def test_run_parameter_sweep_basic():
    """Sweep should return valid results with expected keys."""
    rng = np.random.RandomState(42)
    embeddings = np.vstack([
        rng.randn(30, 4).astype(np.float32) + 5,
        rng.randn(30, 4).astype(np.float32) - 5,
    ])

    results = run_parameter_sweep(embeddings)

    assert len(results) > 0
    # Should include both HDBSCAN and K-Means entries
    algorithms = {r["algorithm"] for r in results}
    assert "hdbscan" in algorithms
    assert "kmeans" in algorithms

    for entry in results:
        assert "algorithm" in entry
        assert "silhouette_score" in entry
        assert "n_clusters" in entry
        assert "noise_fraction" in entry
        assert 0 <= entry["noise_fraction"] <= 1

    # HDBSCAN entries should have selection_method
    hdbscan_entries = [r for r in results if r["algorithm"] == "hdbscan"]
    assert all("cluster_selection_method" in r for r in hdbscan_entries)
    methods = {r["cluster_selection_method"] for r in hdbscan_entries}
    assert "leaf" in methods
    assert "eom" in methods


def test_run_parameter_sweep_too_small():
    """Sweep with fewer than 4 points should return empty."""
    embeddings = np.random.randn(3, 4).astype(np.float32)
    results = run_parameter_sweep(embeddings)
    assert results == []


def test_run_parameter_sweep_with_category_labels():
    """Sweep with category labels should include ARI and NMI."""
    rng = np.random.RandomState(42)
    embeddings = np.vstack([
        rng.randn(15, 4).astype(np.float32) + 5,
        rng.randn(15, 4).astype(np.float32) - 5,
    ])
    categories = ["A"] * 15 + ["B"] * 15

    results = run_parameter_sweep(embeddings, category_labels=categories)

    assert len(results) > 0
    # At least some entries should have ARI/NMI
    entries_with_ari = [r for r in results if "adjusted_rand_index" in r]
    assert len(entries_with_ari) > 0
    for entry in entries_with_ari:
        assert "normalized_mutual_info" in entry


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_shannon_entropy_uniform():
    """Uniform distribution of 4 bins → ln(4)."""
    counts = np.array([10, 10, 10, 10])
    result = _shannon_entropy(counts)
    assert abs(result - np.log(4)) < 1e-10


def test_shannon_entropy_concentrated():
    """Single non-zero bin → 0."""
    counts = np.array([0, 0, 100, 0])
    assert _shannon_entropy(counts) == 0.0


def test_shannon_entropy_empty():
    """Empty array → 0."""
    assert _shannon_entropy(np.array([])) == 0.0


def test_gini_coefficient_equal():
    """Equal counts → Gini ≈ 0."""
    counts = np.array([25, 25, 25, 25])
    g = _gini_coefficient(counts)
    assert abs(g) < 0.01


def test_gini_coefficient_concentrated():
    """One non-zero in many → Gini > 0.5."""
    counts = np.array([0, 0, 0, 0, 100])
    g = _gini_coefficient(counts)
    assert g > 0.5


def test_top_k_mass():
    """Known distribution → verify top-k fractions."""
    counts = np.array([10, 20, 30, 40])
    assert abs(_top_k_mass(counts, 1) - 0.4) < 1e-10
    assert abs(_top_k_mass(counts, 2) - 0.7) < 1e-10
    assert abs(_top_k_mass(counts, 3) - 0.9) < 1e-10


def test_top_k_mass_empty():
    """All zeros → 0."""
    assert _top_k_mass(np.array([0, 0, 0]), 2) == 0.0


# ---------------------------------------------------------------------------
# Category fragmentation tests
# ---------------------------------------------------------------------------


def test_category_fragmentation_perfect_clustering():
    """Each category maps to exactly one cluster → top1=1, entropy=0, neff=1."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    categories = ["A", "A", "A", "B", "B", "B"]

    result = compute_category_fragmentation(labels, categories)

    assert result["A"]["top1_mass"] == 1.0
    assert result["A"]["entropy"] == 0.0
    assert result["A"]["neff"] == 1.0
    assert result["B"]["top1_mass"] == 1.0


def test_category_fragmentation_with_noise():
    """Noise rate correctly computed when noise points present."""
    labels = np.array([0, 0, -1, 1, 1, -1])
    categories = ["A", "A", "A", "B", "B", "B"]

    result = compute_category_fragmentation(labels, categories)

    assert result["A"]["n_total"] == 3
    assert result["A"]["n_noise"] == 1
    assert abs(result["A"]["noise_rate"] - 1.0 / 3) < 1e-10


def test_category_fragmentation_split_category():
    """Category split across clusters → top1<1, entropy>0, neff>1."""
    labels = np.array([0, 0, 1, 1, 2, 2])
    categories = ["A", "A", "A", "A", "B", "B"]

    result = compute_category_fragmentation(labels, categories)

    assert result["A"]["top1_mass"] < 1.0
    assert result["A"]["entropy"] > 0.0
    assert result["A"]["neff"] > 1.0


def test_category_fragmentation_none_categories_ignored():
    """None categories are excluded from results."""
    labels = np.array([0, 0, 1, 1])
    categories = ["A", "A", None, None]

    result = compute_category_fragmentation(labels, categories)

    assert "A" in result
    assert len(result) == 1  # None category not present


# ---------------------------------------------------------------------------
# Cluster fragmentation tests
# ---------------------------------------------------------------------------


def test_cluster_fragmentation_pure_clusters():
    """Pure clusters → dominant_mass=1, entropy=0."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    categories = ["A", "A", "A", "B", "B", "B"]

    result = compute_cluster_fragmentation(labels, categories)

    assert result["0"]["dominant_mass"] == 1.0
    assert result["0"]["cluster_entropy"] == 0.0
    assert result["1"]["dominant_category"] == "B"


def test_cluster_fragmentation_mixed_cluster():
    """Mixed cluster → dominant_mass<1, entropy>0."""
    labels = np.array([0, 0, 0, 0])
    categories = ["A", "A", "B", "B"]

    result = compute_cluster_fragmentation(labels, categories)

    assert result["0"]["dominant_mass"] == 0.5
    assert result["0"]["cluster_entropy"] > 0


def test_cluster_fragmentation_noise_excluded():
    """Noise cluster (-1) not in results."""
    labels = np.array([0, 0, -1, -1])
    categories = ["A", "A", "B", "B"]

    result = compute_cluster_fragmentation(labels, categories)

    assert "-1" not in result
    assert "0" in result


# ---------------------------------------------------------------------------
# Global fragmentation + full report tests
# ---------------------------------------------------------------------------


def test_global_fragmentation_basic():
    """Perfect clustering → low fragmentation indices."""
    labels = np.array([0, 0, 0, 1, 1, 1])
    categories = ["A", "A", "A", "B", "B", "B"]

    cat_frag = compute_category_fragmentation(labels, categories)
    cl_frag = compute_cluster_fragmentation(labels, categories)
    result = compute_global_fragmentation(cat_frag, cl_frag, labels, categories)

    assert result["mean_entropy_norm"] == 0.0
    assert result["mean_neff"] == 1.0
    assert result["mean_noise_rate"] == 0.0
    assert result["mean_cluster_entropy_norm"] == 0.0


def test_fragmentation_report_structure():
    """Report has correct top-level keys and summary."""
    labels = np.array([0, 0, 1, 1, -1])
    categories = ["A", "A", "B", "B", "A"]

    report = compute_fragmentation_report(labels, categories, "test-job-1")

    assert report is not None
    assert report["job_id"] == "test-job-1"
    assert "category_fragmentation" in report
    assert "cluster_fragmentation" in report
    assert "global_fragmentation" in report
    assert "summary" in report

    summary = report["summary"]
    assert summary["n_categories"] == 2
    assert summary["n_clusters"] == 2
    assert summary["n_total"] == 5
    assert summary["n_noise_total"] == 1
    assert abs(summary["overall_noise_rate"] - 0.2) < 1e-10


def test_fragmentation_report_no_categories():
    """Returns None when all categories are None."""
    labels = np.array([0, 0, 1, 1])
    categories = [None, None, None, None]

    report = compute_fragmentation_report(labels, categories, "test-job-2")

    assert report is None
