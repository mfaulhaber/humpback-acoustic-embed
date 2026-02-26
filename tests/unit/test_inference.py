import numpy as np

from humpback.processing.inference import FakeTFLiteModel


def test_fake_model_output_shape():
    model = FakeTFLiteModel(vector_dim=512)
    windows = np.random.randn(3, 16000).astype(np.float32)
    embeddings = model.embed(windows)
    assert embeddings.shape == (3, 512)


def test_fake_model_deterministic():
    model = FakeTFLiteModel(vector_dim=256)
    windows = np.random.randn(2, 16000).astype(np.float32)
    e1 = model.embed(windows)
    e2 = model.embed(windows)
    np.testing.assert_array_equal(e1, e2)


def test_fake_model_different_inputs_different_outputs():
    model = FakeTFLiteModel(vector_dim=128)
    w1 = np.zeros((1, 16000), dtype=np.float32)
    w2 = np.ones((1, 16000), dtype=np.float32)
    e1 = model.embed(w1)
    e2 = model.embed(w2)
    assert not np.allclose(e1, e2)


def test_vector_dim_property():
    model = FakeTFLiteModel(vector_dim=1024)
    assert model.vector_dim == 1024
