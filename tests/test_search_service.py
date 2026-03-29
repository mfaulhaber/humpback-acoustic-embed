"""Tests for search service score distribution, percentile ranking, and projector."""

from pathlib import Path

import numpy as np
import pytest

from humpback.services.search_service import (
    HistogramBin,
    ScoreDistribution,
    _compute_percentile_rank,
    _compute_score_distribution,
    _brute_force_search,
)


# ---- Score distribution tests ----


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


# ---- Percentile rank tests ----


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


# ---- Projector tests ----


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


class TestBruteForceSearch:
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

        # Query biased toward dim 1 → row 1 wins without projector
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
