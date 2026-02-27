import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from humpback.clustering.pipeline import ClusteringResult, compute_cluster_sizes, run_clustering_pipeline


def test_pipeline_basic():
    # Create synthetic embeddings: two clear clusters
    rng = np.random.RandomState(42)
    cluster_a = rng.randn(20, 64).astype(np.float32) + 5
    cluster_b = rng.randn(20, 64).astype(np.float32) - 5
    embeddings = np.vstack([cluster_a, cluster_b])

    result = run_clustering_pipeline(
        embeddings,
        parameters={"use_umap": True, "min_cluster_size": 5},
    )

    assert isinstance(result, ClusteringResult)
    assert result.labels.shape == (40,)
    assert result.reduced_embeddings is not None
    assert result.reduced_embeddings.shape == (40, 2)
    assert result.cluster_input is not None
    assert result.cluster_input.shape == (40, 2)


def test_pipeline_no_umap():
    rng = np.random.RandomState(42)
    embeddings = rng.randn(30, 64).astype(np.float32)

    result = run_clustering_pipeline(
        embeddings,
        parameters={"use_umap": False, "min_cluster_size": 5},
    )

    assert isinstance(result, ClusteringResult)
    assert result.labels.shape == (30,)
    assert result.reduced_embeddings is None
    # cluster_input should be the original embeddings when no UMAP
    assert result.cluster_input.shape == (30, 64)


def test_compute_cluster_sizes():
    labels = np.array([0, 0, 1, 1, 1, 2])
    sizes = compute_cluster_sizes(labels)
    assert sizes == {0: 2, 1: 3, 2: 1}


def test_umap_coords_parquet_schema(tmp_path):
    """Verify UMAP coords parquet has expected schema and roundtrips correctly."""
    n = 10
    reduced = np.random.randn(n, 2).astype(np.float32)
    labels = np.array([0, 0, 1, 1, 1, -1, 2, 2, 0, 1], dtype=np.int32)
    es_ids = [f"es-{i % 3}" for i in range(n)]
    row_indices = list(range(n))

    table = pa.table({
        "x": pa.array(reduced[:, 0], type=pa.float32()),
        "y": pa.array(reduced[:, 1], type=pa.float32()),
        "cluster_label": pa.array(labels.tolist(), type=pa.int32()),
        "embedding_set_id": pa.array(es_ids, type=pa.string()),
        "embedding_row_index": pa.array(row_indices, type=pa.int32()),
    })

    path = tmp_path / "umap_coords.parquet"
    pq.write_table(table, str(path))

    # Roundtrip
    loaded = pq.read_table(str(path))
    assert loaded.num_rows == n
    assert set(loaded.column_names) == {"x", "y", "cluster_label", "embedding_set_id", "embedding_row_index"}
    assert loaded.column("x").type == pa.float32()
    assert loaded.column("y").type == pa.float32()
    assert loaded.column("cluster_label").type == pa.int32()
    assert loaded.column("embedding_set_id").type == pa.string()
    assert loaded.column("embedding_row_index").type == pa.int32()
    np.testing.assert_array_almost_equal(
        loaded.column("x").to_numpy(), reduced[:, 0], decimal=5
    )
