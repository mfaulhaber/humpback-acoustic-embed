"""Tests for the HMM decoder module."""

from __future__ import annotations

import numpy as np
import pytest

from humpback.sequence_models.hmm_decoder import decode_sequence, decode_sequences
from humpback.sequence_models.hmm_trainer import fit_hmm
from tests.fixtures.sequence_models.synthetic_sequences import (
    generate_synthetic_sequences,
)


@pytest.fixture()
def fitted_model_and_seqs():
    n_states = 3
    seqs, _ = generate_synthetic_sequences(
        n_states=n_states,
        n_sequences=10,
        min_length=30,
        max_length=60,
        vector_dim=8,
        cluster_separation=8.0,
        seed=42,
    )
    result = fit_hmm(seqs, n_states=n_states, random_state=42)
    return result.model, seqs, n_states


class TestDecodeSequence:
    def test_viterbi_shape(self, fitted_model_and_seqs):
        model, seqs, _ = fitted_model_and_seqs
        dec = decode_sequence(model, seqs[0])
        assert dec.viterbi_states.shape == (len(seqs[0]),)

    def test_posterior_shape(self, fitted_model_and_seqs):
        model, seqs, n_states = fitted_model_and_seqs
        dec = decode_sequence(model, seqs[0])
        assert dec.posteriors.shape == (len(seqs[0]), n_states)

    def test_max_state_probability_matches_argmax(self, fitted_model_and_seqs):
        model, seqs, _ = fitted_model_and_seqs
        dec = decode_sequence(model, seqs[0])
        expected_max = dec.posteriors.max(axis=1)
        np.testing.assert_allclose(dec.max_state_probability, expected_max, atol=1e-6)

    def test_posteriors_sum_to_one(self, fitted_model_and_seqs):
        model, seqs, _ = fitted_model_and_seqs
        dec = decode_sequence(model, seqs[0])
        row_sums = dec.posteriors.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_viterbi_states_dtype(self, fitted_model_and_seqs):
        model, seqs, _ = fitted_model_and_seqs
        dec = decode_sequence(model, seqs[0])
        assert dec.viterbi_states.dtype == np.int16


class TestDecodeSequences:
    def test_decodes_all(self, fitted_model_and_seqs):
        model, seqs, _ = fitted_model_and_seqs
        results = decode_sequences(model, seqs)
        assert len(results) == len(seqs)

    def test_short_sequence_decoded(self, fitted_model_and_seqs):
        model, _, n_states = fitted_model_and_seqs
        short_seq = np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32)
        dec = decode_sequence(model, short_seq)
        assert dec.viterbi_states.shape == (3,)
        assert dec.posteriors.shape == (3, n_states)
