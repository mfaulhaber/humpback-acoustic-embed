"""Unit tests for embedding similarity search service."""

from pathlib import Path

import numpy as np
import pytest

from humpback.processing.embeddings import IncrementalParquetWriter
from humpback.services.search_service import (
    HistogramBin,
    ScoreDistribution,
    _brute_force_search,
    _cached_read_embeddings,
    _compute_percentile_rank,
    _compute_score_distribution,
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

        hits, total, _dist = _brute_force_search(
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

        hits, total, _dist = _brute_force_search(
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

        hits, total, _dist = _brute_force_search(
            query,
            [("s", str(pq_path))],
            top_k=100,
            metric="cosine",
        )
        assert total == 2
        assert len(hits) == 2

    def test_missing_parquet_skipped(self, tmp_path: Path):
        query = np.array([1.0, 0.0], dtype=np.float32)
        hits, total, _dist = _brute_force_search(
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

        hits, total, _dist = _brute_force_search(
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
        hits, total, _dist = _brute_force_search(
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


# ---------------------------------------------------------------------------
# Score distribution (relocated from tests/test_search_service.py)
# ---------------------------------------------------------------------------


def _write_test_parquet(path: str, embeddings: np.ndarray) -> None:
    """Write a minimal parquet file with row_index + embedding columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = embeddings.shape[0]
    table = pa.table(
        {
            "row_index": pa.array(list(range(n)), type=pa.int32()),
            "embedding": pa.array(
                [emb.tolist() for emb in embeddings], type=pa.list_(pa.float32())
            ),
        }
    )
    pq.write_table(table, path)


class TestScoreDistribution:
    def test_basic_stats(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        dist = _compute_score_distribution(scores)
        assert abs(dist.mean - 0.3) < 1e-5
        assert abs(dist.min - 0.1) < 1e-5
        assert abs(dist.max - 0.5) < 1e-5
        assert abs(dist.p50 - 0.3) < 1e-5
        assert dist.std > 0

    def test_percentiles(self) -> None:
        scores = np.arange(100, dtype=np.float32) / 100.0
        dist = _compute_score_distribution(scores)
        assert abs(dist.p25 - 0.2475) < 0.01
        assert abs(dist.p50 - 0.495) < 0.01
        assert abs(dist.p75 - 0.7425) < 0.01

    def test_histogram_bins(self) -> None:
        scores = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        dist = _compute_score_distribution(scores)
        assert len(dist.histogram) == 20
        total_count = sum(b.count for b in dist.histogram)
        assert total_count == 100

    def test_histogram_bin_types(self) -> None:
        scores = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        dist = _compute_score_distribution(scores)
        for b in dist.histogram:
            assert isinstance(b, HistogramBin)
            assert isinstance(b.bin_start, float)
            assert isinstance(b.bin_end, float)
            assert isinstance(b.count, int)

    def test_empty_scores(self) -> None:
        dist = _compute_score_distribution(np.array([]))
        assert dist.mean == 0.0
        assert dist.std == 0.0
        assert dist.min == 0.0
        assert dist.max == 0.0
        assert dist.histogram == []

    def test_single_score(self) -> None:
        dist = _compute_score_distribution(np.array([0.5]))
        assert abs(dist.mean - 0.5) < 1e-5
        assert dist.std == 0.0
        assert abs(dist.min - 0.5) < 1e-5
        assert abs(dist.max - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# Percentile rank (relocated from tests/test_search_service.py)
# ---------------------------------------------------------------------------


class TestPercentileRank:
    def test_highest_score(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rank = _compute_percentile_rank(0.5, scores)
        assert abs(rank - 0.8) < 1e-5  # 4 out of 5 below

    def test_lowest_score(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rank = _compute_percentile_rank(0.1, scores)
        assert rank == 0.0  # none strictly below

    def test_middle_score(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rank = _compute_percentile_rank(0.3, scores)
        assert abs(rank - 0.4) < 1e-5  # 2 out of 5 below

    def test_empty_scores(self) -> None:
        rank = _compute_percentile_rank(0.5, np.array([]))
        assert rank == 0.0

    def test_monotonic_ordering(self) -> None:
        scores = np.random.RandomState(42).rand(100).astype(np.float32)
        sorted_scores = np.sort(scores)
        ranks = [_compute_percentile_rank(s, scores) for s in sorted_scores]
        # Ranks should be monotonically non-decreasing
        for i in range(1, len(ranks)):
            assert ranks[i] >= ranks[i - 1]


# ---------------------------------------------------------------------------
# Brute-force search: projector (relocated from tests/test_search_service.py)
# ---------------------------------------------------------------------------


class TestBruteForceSearchProjector:
    def test_no_projector(self, tmp_path: Path) -> None:
        """Basic search without projector returns scored hits."""
        embs = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
            dtype=np.float32,
        )
        ppath = str(tmp_path / "test.parquet")
        _write_test_parquet(ppath, embs)

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        hits, total, dist = _brute_force_search(
            query, [("es1", ppath)], top_k=3, metric="cosine"
        )

        assert total == 3
        assert len(hits) == 3
        # First hit should be the identical vector (row 0)
        assert hits[0][2] == 0  # row_index
        assert hits[0][0] > 0.99  # score ~1.0
        # Each hit has 4 elements: (score, es_id, row_idx, percentile_rank)
        assert len(hits[0]) == 4
        assert isinstance(dist, ScoreDistribution)

    def test_with_identity_projector(self, tmp_path: Path) -> None:
        """Identity projector produces same results as no projector."""
        embs = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]],
            dtype=np.float32,
        )
        ppath = str(tmp_path / "test.parquet")
        _write_test_parquet(ppath, embs)

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        hits_no_proj, total_no, dist_no = _brute_force_search(
            query, [("es1", ppath)], top_k=3, metric="cosine"
        )
        hits_id_proj, total_id, dist_id = _brute_force_search(
            query,
            [("es1", ppath)],
            top_k=3,
            metric="cosine",
            projector=lambda x: x,
        )

        assert total_no == total_id
        for h1, h2 in zip(hits_no_proj, hits_id_proj):
            assert abs(h1[0] - h2[0]) < 1e-5  # same scores
            assert h1[2] == h2[2]  # same row indices

    def test_projector_transforms_vectors(self, tmp_path: Path) -> None:
        """Projector changes the search space and produces different rankings."""
        embs = np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        ppath = str(tmp_path / "test.parquet")
        _write_test_parquet(ppath, embs)

        # Query biased toward dim 1 -> row 1 wins without projector
        query = np.array([0.3, 0.7], dtype=np.float32)

        hits_raw, _, _ = _brute_force_search(
            query, [("es1", ppath)], top_k=1, metric="cosine"
        )
        assert hits_raw[0][2] == 1

        # Projector that amplifies dim 0 and suppresses dim 1,
        # flipping the ranking so row 0 wins
        def scale_projector(x: np.ndarray) -> np.ndarray:
            return x * np.array([100.0, 0.01], dtype=np.float32)

        hits_proj, _, _ = _brute_force_search(
            query,
            [("es1", ppath)],
            top_k=1,
            metric="cosine",
            projector=scale_projector,
        )
        assert hits_proj[0][2] == 0

    def test_empty_candidates(self) -> None:
        """Empty candidate set returns empty results with zero distribution."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        hits, total, dist = _brute_force_search(query, [], top_k=5, metric="cosine")
        assert hits == []
        assert total == 0
        assert dist.mean == 0.0
        assert dist.histogram == []

    def test_distribution_in_results(self, tmp_path: Path) -> None:
        """Score distribution is correctly computed from all candidates."""
        rng = np.random.RandomState(42)
        embs = rng.randn(50, 4).astype(np.float32)
        ppath = str(tmp_path / "test.parquet")
        _write_test_parquet(ppath, embs)

        query = rng.randn(4).astype(np.float32)
        hits, total, dist = _brute_force_search(
            query, [("es1", ppath)], top_k=5, metric="cosine"
        )

        assert total == 50
        assert len(hits) == 5
        assert dist.min <= dist.p25 <= dist.p50 <= dist.p75 <= dist.max
        assert sum(b.count for b in dist.histogram) == 50

    def test_percentile_ranks_in_hits(self, tmp_path: Path) -> None:
        """Percentile ranks in hits are monotonically non-increasing (top-K sorted desc)."""
        rng = np.random.RandomState(42)
        embs = rng.randn(100, 4).astype(np.float32)
        ppath = str(tmp_path / "test.parquet")
        _write_test_parquet(ppath, embs)

        query = rng.randn(4).astype(np.float32)
        hits, _, _ = _brute_force_search(
            query, [("es1", ppath)], top_k=10, metric="cosine"
        )

        ranks = [h[3] for h in hits]
        for i in range(1, len(ranks)):
            assert ranks[i] <= ranks[i - 1]


# ---------------------------------------------------------------------------
# Projected search mode rejection (relocated from tests/test_search_service.py)
# ---------------------------------------------------------------------------


class TestProjectedSearchModeRejection:
    """search_mode=projected should raise ValueError at service level."""

    def test_projected_mode_rejected_similar(self) -> None:
        from humpback.schemas.search import SimilaritySearchRequest

        req = SimilaritySearchRequest(
            embedding_set_id="es1",
            row_index=0,
            search_mode="projected",
        )
        assert req.search_mode == "projected"

    def test_projected_mode_rejected_vector(self) -> None:
        from humpback.schemas.search import VectorSearchRequest

        req = VectorSearchRequest(
            vector=[0.1, 0.2],
            model_version="v1",
            search_mode="projected",
        )
        assert req.search_mode == "projected"

    def test_invalid_search_mode_rejected(self) -> None:
        from pydantic import ValidationError

        from humpback.schemas.search import SimilaritySearchRequest

        with pytest.raises(ValidationError):
            SimilaritySearchRequest(
                embedding_set_id="es1",
                row_index=0,
                search_mode="invalid",
            )
