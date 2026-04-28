"""Tests for the HMM summary statistics module."""

from __future__ import annotations

import numpy as np

from humpback.sequence_models.summary import compute_summary


class TestComputeSummary:
    def test_occupancy_sums_to_one(self):
        seqs = [np.array([0, 0, 1, 1, 2, 0, 0, 1], dtype=np.int16)]
        result = compute_summary(seqs, n_states=3)
        total_occ = sum(s.occupancy for s in result.states)
        assert abs(total_occ - 1.0) < 1e-9

    def test_occupancy_fractions(self):
        seqs = [np.array([0, 0, 0, 1, 1, 2], dtype=np.int16)]
        result = compute_summary(seqs, n_states=3)
        assert abs(result.states[0].occupancy - 3 / 6) < 1e-9
        assert abs(result.states[1].occupancy - 2 / 6) < 1e-9
        assert abs(result.states[2].occupancy - 1 / 6) < 1e-9

    def test_dwell_histogram_counts(self):
        seqs = [np.array([0, 0, 0, 1, 1, 0, 0], dtype=np.int16)]
        result = compute_summary(seqs, n_states=2)
        assert sorted(result.states[0].dwell_histogram) == [2, 3]
        assert result.states[1].dwell_histogram == [2]

    def test_mean_dwell(self):
        seqs = [np.array([0, 0, 0, 1, 0, 0], dtype=np.int16)]
        result = compute_summary(seqs, n_states=2)
        # State 0: runs of length 3, 2 → mean 2.5
        assert abs(result.states[0].mean_dwell_frames - 2.5) < 1e-9
        # State 1: run of length 1 → mean 1.0
        assert abs(result.states[1].mean_dwell_frames - 1.0) < 1e-9

    def test_transition_matrix_shape(self):
        seqs = [np.array([0, 1, 2, 0, 1], dtype=np.int16)]
        result = compute_summary(seqs, n_states=3)
        assert result.transition_matrix.shape == (3, 3)

    def test_transition_matrix_row_normalized(self):
        seqs = [
            np.array([0, 0, 1, 2, 0, 1, 1, 2], dtype=np.int16),
            np.array([2, 1, 0, 0, 2], dtype=np.int16),
        ]
        result = compute_summary(seqs, n_states=3)
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_transition_matrix_values(self):
        # [0→0, 0→1, 1→0] → state 0 row: [0.5, 0.5, 0.0], state 1 row: [1.0, 0.0, 0.0]
        seqs = [np.array([0, 0, 1, 0], dtype=np.int16)]
        result = compute_summary(seqs, n_states=3)
        np.testing.assert_allclose(result.transition_matrix[0], [1 / 2, 1 / 2, 0])
        np.testing.assert_allclose(result.transition_matrix[1], [1.0, 0.0, 0.0])
        # State 2 has no transitions — row should be [0, 0, 0] normalized to [1/3, 1/3, 1/3]?
        # No, our impl divides by 0 → row stays [0,0,0] / 1.0 = [0,0,0]
        np.testing.assert_allclose(result.transition_matrix[2], [0.0, 0.0, 0.0])

    def test_multiple_sequences(self):
        seqs = [
            np.array([0, 1], dtype=np.int16),
            np.array([1, 0], dtype=np.int16),
        ]
        result = compute_summary(seqs, n_states=2)
        total_occ = sum(s.occupancy for s in result.states)
        assert abs(total_occ - 1.0) < 1e-9
        assert abs(result.states[0].occupancy - 0.5) < 1e-9

    def test_unused_state_has_zero_occupancy(self):
        seqs = [np.array([0, 0, 1, 1], dtype=np.int16)]
        result = compute_summary(seqs, n_states=3)
        assert result.states[2].occupancy == 0.0
        assert result.states[2].dwell_histogram == []
        assert result.states[2].mean_dwell_frames == 0.0
