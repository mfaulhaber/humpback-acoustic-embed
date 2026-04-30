"""Tests for the ``ChunkProjection`` implementations."""

from __future__ import annotations

import numpy as np
import pytest

from humpback.sequence_models.chunk_projection import (
    ChunkProjection,
    IdentityProjection,
    PCAProjection,
    RandomProjection,
    load_projection,
)


def _make_data(n: int = 64, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n, dim)).astype(np.float32)


def test_identity_passes_through_unchanged():
    proj = IdentityProjection(input_dim=8)
    X = _make_data(n=10, dim=8)
    proj.fit(X)
    out = proj.transform(X)
    np.testing.assert_array_equal(out, X)
    assert proj.output_dim == 8


def test_identity_rejects_wrong_input_dim():
    proj = IdentityProjection(input_dim=8)
    bad = _make_data(n=4, dim=4)
    with pytest.raises(ValueError, match="IdentityProjection expects"):
        proj.transform(bad)


def test_random_is_deterministic_for_fixed_seed():
    X = _make_data(n=32, dim=16)
    a = RandomProjection(output_dim=8, seed=123)
    a.fit(X)
    b = RandomProjection(output_dim=8, seed=123)
    b.fit(X)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))


def test_random_changes_with_different_seed():
    X = _make_data(n=32, dim=16)
    a = RandomProjection(output_dim=8, seed=1)
    a.fit(X)
    b = RandomProjection(output_dim=8, seed=2)
    b.fit(X)
    assert not np.array_equal(a.transform(X), b.transform(X))


def test_random_rejects_transform_before_fit():
    proj = RandomProjection(output_dim=4, seed=0)
    with pytest.raises(RuntimeError, match="fit must be called"):
        proj.transform(_make_data(n=2, dim=8))


def test_pca_fits_and_reduces_dimension():
    X = _make_data(n=64, dim=16)
    proj = PCAProjection(output_dim=4, whiten=False)
    proj.fit(X)
    out = proj.transform(X)
    assert out.shape == (64, 4)
    assert out.dtype == np.float32


def test_pca_whitening_changes_output():
    X = _make_data(n=64, dim=16)
    a = PCAProjection(output_dim=4, whiten=False)
    a.fit(X)
    b = PCAProjection(output_dim=4, whiten=True)
    b.fit(X)
    assert not np.allclose(a.transform(X), b.transform(X))


def test_pca_rejects_transform_before_fit():
    proj = PCAProjection(output_dim=2)
    with pytest.raises(RuntimeError, match="fit must be called"):
        proj.transform(_make_data(n=2, dim=8))


def test_identity_round_trip_save_load(tmp_path):
    proj = IdentityProjection(input_dim=8)
    path = tmp_path / "identity.joblib"
    proj.save(path)
    loaded = IdentityProjection.load(path)
    X = _make_data(n=4, dim=8)
    np.testing.assert_array_equal(loaded.transform(X), proj.transform(X))


def test_random_round_trip_save_load(tmp_path):
    X = _make_data(n=32, dim=16)
    proj = RandomProjection(output_dim=4, seed=42)
    proj.fit(X)
    path = tmp_path / "random.joblib"
    proj.save(path)
    loaded = RandomProjection.load(path)
    np.testing.assert_array_equal(loaded.transform(X), proj.transform(X))


def test_pca_round_trip_save_load(tmp_path):
    X = _make_data(n=32, dim=16)
    proj = PCAProjection(output_dim=4, whiten=True)
    proj.fit(X)
    path = tmp_path / "pca.joblib"
    proj.save(path)
    loaded = PCAProjection.load(path)
    np.testing.assert_array_equal(loaded.transform(X), proj.transform(X))


def test_load_projection_dispatches_on_kind(tmp_path):
    X = _make_data(n=32, dim=16)
    iden = IdentityProjection(input_dim=16)
    rnd = RandomProjection(output_dim=4, seed=7)
    rnd.fit(X)
    pca = PCAProjection(output_dim=4)
    pca.fit(X)

    p_iden = tmp_path / "i.joblib"
    p_rnd = tmp_path / "r.joblib"
    p_pca = tmp_path / "p.joblib"
    iden.save(p_iden)
    rnd.save(p_rnd)
    pca.save(p_pca)

    assert isinstance(load_projection(p_iden), IdentityProjection)
    assert isinstance(load_projection(p_rnd), RandomProjection)
    assert isinstance(load_projection(p_pca), PCAProjection)


def test_protocol_is_satisfied_at_runtime():
    """The Protocol's ``runtime_checkable`` decorator allows isinstance checks."""
    iden = IdentityProjection(input_dim=4)
    assert isinstance(iden, ChunkProjection)
