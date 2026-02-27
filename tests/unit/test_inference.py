import numpy as np

from humpback.processing.inference import FakeTF2Model, FakeTFLiteModel


def test_fake_model_output_shape():
    model = FakeTFLiteModel(vector_dim=1280)
    specs = np.random.randn(3, 128, 128).astype(np.float32)
    embeddings = model.embed(specs)
    assert embeddings.shape == (3, 1280)


def test_fake_model_deterministic():
    model = FakeTFLiteModel(vector_dim=256)
    specs = np.random.randn(2, 128, 128).astype(np.float32)
    e1 = model.embed(specs)
    e2 = model.embed(specs)
    np.testing.assert_array_equal(e1, e2)


def test_fake_model_different_inputs_different_outputs():
    model = FakeTFLiteModel(vector_dim=128)
    s1 = np.zeros((1, 128, 128), dtype=np.float32)
    s2 = np.ones((1, 128, 128), dtype=np.float32)
    e1 = model.embed(s1)
    e2 = model.embed(s2)
    assert not np.allclose(e1, e2)


def test_vector_dim_property():
    model = FakeTFLiteModel(vector_dim=1024)
    assert model.vector_dim == 1024


def test_fake_model_varying_vector_dims():
    """FakeTFLiteModel should work correctly with different vector_dim values."""
    for dim in [64, 128, 512, 1280, 2048]:
        model = FakeTFLiteModel(vector_dim=dim)
        specs = np.random.randn(2, 128, 128).astype(np.float32)
        embeddings = model.embed(specs)
        assert embeddings.shape == (2, dim)
        assert model.vector_dim == dim


# ---- FakeTF2Model tests ----


def test_fake_tf2_model_output_shape():
    model = FakeTF2Model(vector_dim=1280)
    waveforms = np.random.randn(3, 160000).astype(np.float32)
    embeddings = model.embed(waveforms)
    assert embeddings.shape == (3, 1280)


def test_fake_tf2_model_deterministic():
    model = FakeTF2Model(vector_dim=256)
    waveforms = np.random.randn(2, 160000).astype(np.float32)
    e1 = model.embed(waveforms)
    e2 = model.embed(waveforms)
    np.testing.assert_array_equal(e1, e2)


def test_fake_tf2_model_different_inputs_different_outputs():
    model = FakeTF2Model(vector_dim=128)
    w1 = np.zeros((1, 160000), dtype=np.float32)
    w2 = np.ones((1, 160000), dtype=np.float32)
    e1 = model.embed(w1)
    e2 = model.embed(w2)
    assert not np.allclose(e1, e2)


def test_fake_tf2_model_vector_dim_property():
    model = FakeTF2Model(vector_dim=1024)
    assert model.vector_dim == 1024


def test_fake_tf2_model_varying_dims():
    """FakeTF2Model should work with different vector_dim values."""
    for dim in [64, 128, 512, 1280]:
        model = FakeTF2Model(vector_dim=dim)
        waveforms = np.random.randn(2, 160000).astype(np.float32)
        embeddings = model.embed(waveforms)
        assert embeddings.shape == (2, dim)
        assert model.vector_dim == dim
