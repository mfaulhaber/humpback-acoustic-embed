"""Tests for the PCA pipeline module."""

import numpy as np
import pytest

from humpback.sequence_models.pca_pipeline import (
    fit_pca,
    l2_normalize_sequences,
    transform_sequences,
)
from tests.fixtures.sequence_models.synthetic_sequences import (
    generate_synthetic_sequences,
)


@pytest.fixture()
def synthetic_data():
    seqs, _ = generate_synthetic_sequences(
        n_states=3,
        n_sequences=5,
        min_length=30,
        max_length=50,
        vector_dim=16,
        seed=99,
    )
    return seqs


class TestL2Normalize:
    def test_rows_are_unit_norm(self, synthetic_data):
        normed = l2_normalize_sequences(synthetic_data)
        for seq in normed:
            norms = np.linalg.norm(seq, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_preserves_shape(self, synthetic_data):
        normed = l2_normalize_sequences(synthetic_data)
        for orig, n in zip(synthetic_data, normed):
            assert orig.shape == n.shape


class TestFitPCA:
    def test_shape_after_transform(self, synthetic_data):
        pca_dims = 8
        pca, preproc = fit_pca(synthetic_data, pca_dims, random_state=42)
        transformed = transform_sequences(pca, preproc)
        for seq in transformed:
            assert seq.shape[1] == pca_dims

    def test_deterministic_given_seed(self, synthetic_data):
        pca1, p1 = fit_pca(synthetic_data, 8, random_state=123)
        t1 = transform_sequences(pca1, p1)
        pca2, p2 = fit_pca(synthetic_data, 8, random_state=123)
        t2 = transform_sequences(pca2, p2)
        for a, b in zip(t1, t2):
            np.testing.assert_array_equal(a, b)

    def test_l2_norm_vs_no_l2_norm_differs(self, synthetic_data):
        pca1, p1 = fit_pca(synthetic_data, 8, l2_norm=True, random_state=42)
        t1 = transform_sequences(pca1, p1)
        pca2, p2 = fit_pca(synthetic_data, 8, l2_norm=False, random_state=42)
        t2 = transform_sequences(pca2, p2)
        concat1 = np.concatenate(t1)
        concat2 = np.concatenate(t2)
        assert not np.allclose(concat1, concat2)

    def test_whiten_changes_output(self, synthetic_data):
        _, p1 = fit_pca(synthetic_data, 8, whiten=False, random_state=42)
        pca_w, p2 = fit_pca(synthetic_data, 8, whiten=True, random_state=42)
        t_no = transform_sequences(
            fit_pca(synthetic_data, 8, whiten=False, random_state=42)[0], p1
        )
        t_yes = transform_sequences(pca_w, p2)
        concat_no = np.concatenate(t_no)
        concat_yes = np.concatenate(t_yes)
        assert not np.allclose(concat_no, concat_yes)

    def test_no_l2_norm(self, synthetic_data):
        pca, preproc = fit_pca(synthetic_data, 8, l2_norm=False, random_state=42)
        transformed = transform_sequences(pca, preproc)
        for seq in transformed:
            assert seq.shape[1] == 8

    def test_pca_dims_clamped_to_features(self):
        seqs = [np.random.default_rng(0).standard_normal((20, 4)).astype(np.float32)]
        pca, _ = fit_pca(seqs, pca_dims=100, random_state=42)
        assert pca.n_components_ <= 4
