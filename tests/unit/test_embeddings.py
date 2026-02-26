import numpy as np

from humpback.processing.embeddings import IncrementalParquetWriter, read_embeddings


def test_write_and_read(tmp_path):
    final = tmp_path / "embeddings" / "test.parquet"
    writer = IncrementalParquetWriter(final, vector_dim=4, batch_size=2)
    for i in range(5):
        writer.add(np.array([i, i + 1, i + 2, i + 3], dtype=np.float32))
    path = writer.close()

    assert path == final
    assert final.exists()
    assert not final.with_suffix(".tmp.parquet").exists()

    indices, embeddings = read_embeddings(final)
    assert len(indices) == 5
    assert embeddings.shape == (5, 4)
    np.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])


def test_atomic_rename(tmp_path):
    final = tmp_path / "out.parquet"
    writer = IncrementalParquetWriter(final, vector_dim=2, batch_size=10)
    writer.add(np.array([1.0, 2.0], dtype=np.float32))
    # Before close, tmp should exist after flush
    writer._flush()
    assert writer.tmp_path.exists()
    assert not final.exists()
    # After close, final should exist and tmp should not
    writer.close()
    assert final.exists()
    assert not writer.tmp_path.exists()


def test_total_rows(tmp_path):
    writer = IncrementalParquetWriter(tmp_path / "t.parquet", vector_dim=2, batch_size=100)
    assert writer.total_rows == 0
    writer.add(np.array([1.0, 2.0], dtype=np.float32))
    assert writer.total_rows == 1
    writer.add(np.array([3.0, 4.0], dtype=np.float32))
    assert writer.total_rows == 2
    writer.close()
