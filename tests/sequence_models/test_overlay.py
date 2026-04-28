"""Tests for PCA/UMAP overlay computation."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from humpback.sequence_models.overlay import OverlayMetadata, compute_overlay


def _make_pca(
    n_samples: int = 200, n_features: int = 64, n_components: int = 10
) -> PCA:
    rng = np.random.RandomState(0)
    data = rng.randn(n_samples, n_features).astype(np.float32)
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(data)
    return pca


def _make_inputs(
    n_spans: int = 2,
    windows_per_span: int = 30,
    n_features: int = 64,
    n_states: int = 3,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    raw_sequences = []
    viterbi_states = []
    max_probs = []
    span_ids = []
    win_indices = []
    starts = []
    ends = []

    for span in range(n_spans):
        emb = rng.randn(windows_per_span, n_features).astype(np.float32)
        states = rng.randint(0, n_states, size=windows_per_span).astype(np.int16)
        probs = rng.uniform(0.5, 1.0, size=windows_per_span).astype(np.float32)
        raw_sequences.append(emb)
        viterbi_states.append(states)
        max_probs.append(probs)
        for w in range(windows_per_span):
            span_ids.append(span)
            win_indices.append(w)
            starts.append(100.0 + span * 200 + w * 1.0)
            ends.append(105.0 + span * 200 + w * 1.0)

    meta = OverlayMetadata(
        merged_span_ids=span_ids,
        window_indices=win_indices,
        start_times=starts,
        end_times=ends,
    )
    return raw_sequences, viterbi_states, max_probs, meta


class TestComputeOverlay:
    def test_output_shape_and_columns(self):
        pca = _make_pca()
        raw, vs, mp, meta = _make_inputs()
        table = compute_overlay(pca, raw, vs, mp, meta)

        total_windows = sum(len(s) for s in raw)
        assert table.num_rows == total_windows
        expected_cols = {
            "merged_span_id",
            "window_index_in_span",
            "start_time_sec",
            "end_time_sec",
            "pca_x",
            "pca_y",
            "umap_x",
            "umap_y",
            "viterbi_state",
            "max_state_probability",
        }
        assert set(table.column_names) == expected_cols

    def test_viterbi_states_match_input(self):
        pca = _make_pca()
        raw, vs, mp, meta = _make_inputs()
        table = compute_overlay(pca, raw, vs, mp, meta)

        expected_states = np.concatenate(vs)
        actual_states = np.array(table.column("viterbi_state").to_pylist())
        np.testing.assert_array_equal(actual_states, expected_states)

    def test_determinism_on_fixed_seed(self):
        pca = _make_pca()
        raw, vs, mp, meta = _make_inputs()
        t1 = compute_overlay(pca, raw, vs, mp, meta, random_state=99)
        t2 = compute_overlay(pca, raw, vs, mp, meta, random_state=99)

        np.testing.assert_array_equal(
            t1.column("umap_x").to_pylist(),
            t2.column("umap_x").to_pylist(),
        )
        np.testing.assert_array_equal(
            t1.column("umap_y").to_pylist(),
            t2.column("umap_y").to_pylist(),
        )

    def test_single_span(self):
        pca = _make_pca()
        raw, vs, mp, meta = _make_inputs(n_spans=1, windows_per_span=20)
        table = compute_overlay(pca, raw, vs, mp, meta)
        assert table.num_rows == 20
        assert all(v == 0 for v in table.column("merged_span_id").to_pylist())

    def test_l2_normalize_flag(self):
        pca = _make_pca()
        raw, vs, mp, meta = _make_inputs()
        t_norm = compute_overlay(pca, raw, vs, mp, meta, l2_normalize=True)
        t_no_norm = compute_overlay(pca, raw, vs, mp, meta, l2_normalize=False)

        pca_x_norm = np.array(t_norm.column("pca_x").to_pylist())
        pca_x_raw = np.array(t_no_norm.column("pca_x").to_pylist())
        assert not np.allclose(pca_x_norm, pca_x_raw, atol=1e-5)
