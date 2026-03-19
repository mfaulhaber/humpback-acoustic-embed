"""Unit tests for embedding similarity search service."""

from pathlib import Path

import numpy as np
import pytest

from humpback.processing.embeddings import IncrementalParquetWriter
from humpback.services.search_service import (
    _brute_force_search,
    _cached_read_embeddings,
    _cosine_scores,
    _euclidean_scores,
)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineScores:
    def test_identical_vectors(self):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        cand = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        scores = _cosine_scores(q, cand)
        assert scores[0] == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        cand = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        scores = _cosine_scores(q, cand)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        cand = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
        scores = _cosine_scores(q, cand)
        assert scores[0] == pytest.approx(-1.0)

    def test_batch(self):
        q = np.array([1.0, 0.0], dtype=np.float32)
        cand = np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        scores = _cosine_scores(q, cand)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0, abs=1e-6)
        assert scores[2] == pytest.approx(-1.0)

    def test_zero_query(self):
        q = np.zeros(3, dtype=np.float32)
        cand = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        scores = _cosine_scores(q, cand)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Euclidean distance
# ---------------------------------------------------------------------------


class TestEuclideanScores:
    def test_identical(self):
        q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cand = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        scores = _euclidean_scores(q, cand)
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_known_distance(self):
        q = np.array([0.0, 0.0], dtype=np.float32)
        cand = np.array([[3.0, 4.0]], dtype=np.float32)
        scores = _euclidean_scores(q, cand)
        # negative distance
        assert scores[0] == pytest.approx(-5.0)

    def test_ranking(self):
        q = np.zeros(2, dtype=np.float32)
        cand = np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        scores = _euclidean_scores(q, cand)
        # closer point has higher (less negative) score
        assert scores[0] > scores[1]


# ---------------------------------------------------------------------------
# Helper: create parquet with known embeddings
# ---------------------------------------------------------------------------


def _write_parquet(path: Path, vectors: np.ndarray) -> Path:
    """Write embeddings to a parquet file and return the path."""
    writer = IncrementalParquetWriter(path, vector_dim=vectors.shape[1])
    for v in vectors:
        writer.add(v)
    writer.close()
    return path


# ---------------------------------------------------------------------------
# Brute-force search
# ---------------------------------------------------------------------------


class TestBruteForceSearch:
    def test_basic_top_k(self, tmp_path: Path):
        # 5 vectors, query is closest to vector 2
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vectors = np.array(
            [
                [0.0, 1.0, 0.0],  # orthogonal
                [0.5, 0.5, 0.0],  # moderate
                [0.9, 0.1, 0.0],  # most similar
                [-1.0, 0.0, 0.0],  # opposite
                [0.7, 0.3, 0.0],  # fairly similar
            ],
            dtype=np.float32,
        )
        pq_path = _write_parquet(tmp_path / "set_a.parquet", vectors)

        hits, total = _brute_force_search(
            query,
            [("set_a", str(pq_path))],
            top_k=3,
            metric="cosine",
        )
        assert total == 5
        assert len(hits) == 3
        # Best match should be row_index=2 (the [0.9, 0.1, 0] vector)
        assert hits[0][2] == 2
        # Scores should be descending
        assert hits[0][0] >= hits[1][0] >= hits[2][0]

    def test_cross_set_search(self, tmp_path: Path):
        query = np.array([1.0, 0.0], dtype=np.float32)
        set_a = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        set_b = np.array([[0.99, 0.01], [-1.0, 0.0]], dtype=np.float32)
        set_c = np.array([[0.8, 0.2]], dtype=np.float32)

        pa = _write_parquet(tmp_path / "a.parquet", set_a)
        pb = _write_parquet(tmp_path / "b.parquet", set_b)
        pc = _write_parquet(tmp_path / "c.parquet", set_c)

        hits, total = _brute_force_search(
            query,
            [("a", str(pa)), ("b", str(pb)), ("c", str(pc))],
            top_k=2,
            metric="cosine",
        )
        assert total == 5
        assert len(hits) == 2
        # Best should be from set_b row 0 ([0.99, 0.01])
        assert hits[0][1] == "b"
        assert hits[0][2] == 0

    def test_top_k_larger_than_candidates(self, tmp_path: Path):
        query = np.array([1.0, 0.0], dtype=np.float32)
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        pq_path = _write_parquet(tmp_path / "small.parquet", vectors)

        hits, total = _brute_force_search(
            query,
            [("s", str(pq_path))],
            top_k=100,
            metric="cosine",
        )
        assert total == 2
        assert len(hits) == 2

    def test_missing_parquet_skipped(self, tmp_path: Path):
        query = np.array([1.0, 0.0], dtype=np.float32)
        hits, total = _brute_force_search(
            query,
            [("missing", str(tmp_path / "nonexistent.parquet"))],
            top_k=10,
            metric="cosine",
        )
        assert total == 0
        assert len(hits) == 0

    def test_euclidean_metric(self, tmp_path: Path):
        query = np.array([0.0, 0.0], dtype=np.float32)
        vectors = np.array([[1.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        pq_path = _write_parquet(tmp_path / "euc.parquet", vectors)

        hits, total = _brute_force_search(
            query,
            [("s", str(pq_path))],
            top_k=2,
            metric="euclidean",
        )
        assert total == 2
        # Closer point ([1,0] distance=1) ranked above ([3,4] distance=5)
        assert hits[0][2] == 0
        assert hits[1][2] == 1

    def test_empty_set(self, tmp_path: Path):
        """Empty parquet still readable but contributes no candidates."""
        # Write a parquet with zero rows
        writer = IncrementalParquetWriter(tmp_path / "empty.parquet", vector_dim=3)
        writer.close()

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        hits, total = _brute_force_search(
            query,
            [("e", str(tmp_path / "empty.parquet"))],
            top_k=10,
            metric="cosine",
        )
        assert total == 0
        assert len(hits) == 0


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    def test_cache_hit(self, tmp_path: Path):
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        pq_path = _write_parquet(tmp_path / "cached.parquet", vectors)
        path_str = str(pq_path)

        # Clear any prior cache state
        _cached_read_embeddings.cache_clear()

        r1_idx, r1_emb = _cached_read_embeddings(path_str)
        r2_idx, r2_emb = _cached_read_embeddings(path_str)

        np.testing.assert_array_equal(r1_idx, r2_idx)
        np.testing.assert_array_equal(r1_emb, r2_emb)

        info = _cached_read_embeddings.cache_info()
        assert info.hits >= 1

    def test_cache_distinct_paths(self, tmp_path: Path):
        _cached_read_embeddings.cache_clear()

        v1 = np.array([[1.0, 0.0]], dtype=np.float32)
        v2 = np.array([[0.0, 1.0]], dtype=np.float32)
        p1 = _write_parquet(tmp_path / "a.parquet", v1)
        p2 = _write_parquet(tmp_path / "b.parquet", v2)

        _, e1 = _cached_read_embeddings(str(p1))
        _, e2 = _cached_read_embeddings(str(p2))

        assert e1[0][0] == pytest.approx(1.0)
        assert e2[0][1] == pytest.approx(1.0)

        info = _cached_read_embeddings.cache_info()
        assert info.misses >= 2


# ---------------------------------------------------------------------------
# similarity_search_by_vector (async, requires DB session)
# ---------------------------------------------------------------------------


async def _seed_vector_search_data(session, tmp_path: Path):
    """Seed DB with audio files and embedding sets for vector search tests."""
    import uuid

    from humpback.models.audio import AudioFile
    from humpback.models.processing import EmbeddingSet

    # Write parquet with known dim=4 vectors
    pq_path = tmp_path / "vectors.parquet"
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    _write_parquet(pq_path, vectors)

    af = AudioFile(
        filename="test.wav",
        folder_path="test",
        checksum_sha256=f"test_{uuid.uuid4().hex[:8]}",
    )
    session.add(af)
    await session.flush()

    es = EmbeddingSet(
        audio_file_id=af.id,
        encoding_signature="sig_test",
        model_version="perch_v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        vector_dim=4,
        parquet_path=str(pq_path),
    )
    session.add(es)
    await session.flush()
    await session.commit()
    return es.id


class TestSimilaritySearchByVector:
    async def test_returns_results(self, session, tmp_path):
        """similarity_search_by_vector returns results for valid query."""
        from humpback.schemas.search import VectorSearchRequest
        from humpback.services.search_service import similarity_search_by_vector

        _cached_read_embeddings.cache_clear()
        await _seed_vector_search_data(session, tmp_path)

        request = VectorSearchRequest(
            vector=[1.0, 0.0, 0.0, 0.0],
            model_version="perch_v1",
            top_k=3,
        )
        response = await similarity_search_by_vector(session, request)

        assert response.model_version == "perch_v1"
        assert response.total_candidates == 3
        assert len(response.results) == 3
        # Best match should be row 0 (identical vector)
        assert response.results[0].row_index == 0
        assert response.results[0].score > 0.9

    async def test_dimension_mismatch_raises(self, session, tmp_path):
        """Vector dimension mismatch raises ValueError."""
        from humpback.schemas.search import VectorSearchRequest
        from humpback.services.search_service import similarity_search_by_vector

        _cached_read_embeddings.cache_clear()
        await _seed_vector_search_data(session, tmp_path)

        # Query with wrong dimension (3 instead of 4)
        request = VectorSearchRequest(
            vector=[1.0, 0.0, 0.0],
            model_version="perch_v1",
            top_k=3,
        )
        with pytest.raises(ValueError, match="dimension"):
            await similarity_search_by_vector(session, request)

    async def test_empty_model_version_raises(self, session, tmp_path):
        """Non-existent model_version raises ValueError."""
        from humpback.schemas.search import VectorSearchRequest
        from humpback.services.search_service import similarity_search_by_vector

        _cached_read_embeddings.cache_clear()
        await _seed_vector_search_data(session, tmp_path)

        request = VectorSearchRequest(
            vector=[1.0, 0.0, 0.0, 0.0],
            model_version="nonexistent_model",
            top_k=3,
        )
        with pytest.raises(ValueError, match="No embedding sets found"):
            await similarity_search_by_vector(session, request)
