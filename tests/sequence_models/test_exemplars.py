"""Tests for per-state exemplar selection."""

from __future__ import annotations

import numpy as np

from humpback.sequence_models.exemplars import WindowMeta, select_exemplars


def _meta(idx: int, state_prob: float, audio_id: int = 1) -> WindowMeta:
    return WindowMeta(
        merged_span_id=0,
        window_index_in_span=idx,
        audio_file_id=audio_id,
        start_time_sec=100.0 + idx,
        end_time_sec=105.0 + idx,
        max_state_probability=state_prob,
    )


class TestSelectExemplars:
    def test_high_confidence_picks_highest_probability(self):
        rng = np.random.RandomState(0)
        n = 20
        embeddings = rng.randn(n, 10).astype(np.float32)
        states = np.zeros(n, dtype=np.intp)
        probs = np.linspace(0.5, 0.99, n)
        metas = [_meta(i, probs[i]) for i in range(n)]

        result = select_exemplars(
            embeddings, states, metas, n_states=1, n_exemplars_per_type=3
        )

        high_conf = [
            r for r in result["states"]["0"] if r["exemplar_type"] == "high_confidence"
        ]
        assert len(high_conf) == 3
        for r in high_conf:
            assert r["max_state_probability"] >= probs[-4]

    def test_boundary_picks_lowest_probability(self):
        rng = np.random.RandomState(0)
        n = 20
        embeddings = rng.randn(n, 10).astype(np.float32)
        states = np.zeros(n, dtype=np.intp)
        probs = np.linspace(0.1, 0.99, n)
        metas = [_meta(i, probs[i]) for i in range(n)]

        result = select_exemplars(
            embeddings, states, metas, n_states=1, n_exemplars_per_type=3
        )

        boundary = [
            r for r in result["states"]["0"] if r["exemplar_type"] == "boundary"
        ]
        assert len(boundary) == 3
        for r in boundary:
            assert r["max_state_probability"] <= probs[3]

    def test_mean_nearest_close_to_centroid(self):
        rng = np.random.RandomState(0)
        n = 30
        embeddings = rng.randn(n, 10).astype(np.float32)
        states = np.zeros(n, dtype=np.intp)
        metas = [_meta(i, 0.8) for i in range(n)]

        result = select_exemplars(
            embeddings, states, metas, n_states=1, n_exemplars_per_type=3
        )

        nearest = [
            r for r in result["states"]["0"] if r["exemplar_type"] == "mean_nearest"
        ]
        assert len(nearest) == 3

        centroid = embeddings.mean(axis=0)
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        nearest_indices = [r["window_index_in_span"] for r in nearest]
        top3_indices = set(np.argsort(dists)[:3].tolist())
        assert set(nearest_indices) == top3_indices

    def test_multi_state(self):
        rng = np.random.RandomState(0)
        n = 40
        embeddings = rng.randn(n, 10).astype(np.float32)
        states = np.array([0] * 20 + [1] * 20, dtype=np.intp)
        metas = [_meta(i, 0.8) for i in range(n)]

        result = select_exemplars(
            embeddings, states, metas, n_states=2, n_exemplars_per_type=2
        )

        assert result["n_states"] == 2
        assert len(result["states"]["0"]) > 0
        assert len(result["states"]["1"]) > 0

    def test_state_with_one_window(self):
        embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
        states = np.array([0], dtype=np.intp)
        metas = [_meta(0, 0.95)]

        result = select_exemplars(
            embeddings, states, metas, n_states=1, n_exemplars_per_type=3
        )

        records = result["states"]["0"]
        assert len(records) == 3
        types = {r["exemplar_type"] for r in records}
        assert types == {"high_confidence", "mean_nearest", "boundary"}

    def test_empty_state(self):
        rng = np.random.RandomState(0)
        embeddings = rng.randn(10, 5).astype(np.float32)
        states = np.zeros(10, dtype=np.intp)
        metas = [_meta(i, 0.8) for i in range(10)]

        result = select_exemplars(
            embeddings, states, metas, n_states=3, n_exemplars_per_type=2
        )

        assert result["states"]["1"] == []
        assert result["states"]["2"] == []

    def test_n_exemplars_per_type_respected(self):
        rng = np.random.RandomState(0)
        n = 50
        embeddings = rng.randn(n, 10).astype(np.float32)
        states = np.zeros(n, dtype=np.intp)
        metas = [_meta(i, rng.uniform(0.5, 1.0)) for i in range(n)]

        result = select_exemplars(
            embeddings, states, metas, n_states=1, n_exemplars_per_type=5
        )

        by_type: dict[str, int] = {}
        for r in result["states"]["0"]:
            by_type[r["exemplar_type"]] = by_type.get(r["exemplar_type"], 0) + 1
        assert by_type.get("high_confidence", 0) == 5
        assert by_type.get("mean_nearest", 0) == 5
        assert by_type.get("boundary", 0) == 5
