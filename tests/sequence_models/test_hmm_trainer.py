"""Tests for the HMM trainer module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from humpback.sequence_models.hmm_trainer import fit_hmm
from tests.fixtures.sequence_models.synthetic_sequences import (
    generate_synthetic_sequences,
)


def _hungarian_accuracy(
    true_states: list[np.ndarray],
    pred_states: list[np.ndarray],
    n_states: int,
) -> float:
    """Compute accuracy using Hungarian alignment of predicted to true states."""
    confusion = np.zeros((n_states, n_states), dtype=np.int64)
    for ts, ps in zip(true_states, pred_states):
        for t, p in zip(ts, ps):
            confusion[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    return float(confusion[row_ind, col_ind].sum()) / sum(len(s) for s in true_states)


@pytest.fixture()
def planted_data():
    n_states = 3
    seqs, gt = generate_synthetic_sequences(
        n_states=n_states,
        n_sequences=15,
        min_length=40,
        max_length=80,
        vector_dim=8,
        cluster_separation=8.0,
        seed=42,
    )
    return n_states, seqs, gt


class TestFitHMM:
    def test_recovers_planted_structure(self, planted_data):
        n_states, seqs, gt = planted_data
        result = fit_hmm(seqs, n_states=n_states, random_state=42)
        pred = [result.model.predict(s) for s in seqs]
        acc = _hungarian_accuracy(gt, pred, n_states)
        assert acc >= 0.85, f"Hungarian accuracy {acc:.3f} < 0.85"

    def test_training_metadata(self, planted_data):
        n_states, seqs, _ = planted_data
        result = fit_hmm(seqs, n_states=n_states, random_state=42)
        assert result.n_train_sequences == len(seqs)
        assert result.n_train_frames == sum(len(s) for s in seqs)
        assert isinstance(result.train_log_likelihood, float)

    def test_min_sequence_length_filter(self, planted_data):
        n_states, seqs, _ = planted_data
        high_threshold = max(len(s) for s in seqs) - 1
        result = fit_hmm(
            seqs,
            n_states=n_states,
            min_sequence_length_frames=high_threshold,
            random_state=42,
        )
        expected_count = sum(1 for s in seqs if len(s) >= high_threshold)
        assert result.n_train_sequences == expected_count
        assert result.n_train_sequences < len(seqs)
        assert len(result.training_mask) == len(seqs)

    def test_all_below_threshold_raises(self):
        seqs = [np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32)]
        with pytest.raises(ValueError, match="No sequences meet"):
            fit_hmm(seqs, n_states=2, min_sequence_length_frames=100)

    def test_deterministic_given_seed(self, planted_data):
        n_states, seqs, _ = planted_data
        r1 = fit_hmm(seqs, n_states=n_states, random_state=77)
        r2 = fit_hmm(seqs, n_states=n_states, random_state=77)
        pred1 = [r1.model.predict(s) for s in seqs]
        pred2 = [r2.model.predict(s) for s in seqs]
        for a, b in zip(pred1, pred2):
            np.testing.assert_array_equal(a, b)

    def test_transition_matrix_shape(self, planted_data):
        n_states, seqs, _ = planted_data
        result = fit_hmm(seqs, n_states=n_states, random_state=42)
        assert result.model.transmat_.shape == (n_states, n_states)
        np.testing.assert_allclose(result.model.transmat_.sum(axis=1), 1.0, atol=1e-6)
