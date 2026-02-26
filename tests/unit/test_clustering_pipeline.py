import numpy as np

from humpback.clustering.pipeline import compute_cluster_sizes, run_clustering_pipeline


def test_pipeline_basic():
    # Create synthetic embeddings: two clear clusters
    rng = np.random.RandomState(42)
    cluster_a = rng.randn(20, 64).astype(np.float32) + 5
    cluster_b = rng.randn(20, 64).astype(np.float32) - 5
    embeddings = np.vstack([cluster_a, cluster_b])

    labels, reduced = run_clustering_pipeline(
        embeddings,
        parameters={"use_umap": True, "min_cluster_size": 5},
    )

    assert labels.shape == (40,)
    assert reduced is not None
    assert reduced.shape == (40, 2)


def test_pipeline_no_umap():
    rng = np.random.RandomState(42)
    embeddings = rng.randn(30, 64).astype(np.float32)

    labels, reduced = run_clustering_pipeline(
        embeddings,
        parameters={"use_umap": False, "min_cluster_size": 5},
    )

    assert labels.shape == (30,)
    assert reduced is None


def test_compute_cluster_sizes():
    labels = np.array([0, 0, 1, 1, 1, 2])
    sizes = compute_cluster_sizes(labels)
    assert sizes == {0: 2, 1: 3, 2: 1}
