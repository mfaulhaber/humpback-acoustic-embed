import numpy as np

from humpback.processing.inference import FakeTFLiteModel


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
