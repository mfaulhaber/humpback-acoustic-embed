from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from humpback.processing.inference import (
    FakeTF2Model,
    FakeTFLiteModel,
    TF2SavedModel,
    _has_xla_must_compile,
    _strip_xla_must_compile,
)


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


# ---- configure_tf_gpu tests ----


def test_configure_tf_gpu_idempotent():
    """configure_tf_gpu should only configure GPUs on the first call."""
    import humpback.processing.inference as inf

    original_flag = inf._tf_gpu_configured
    try:
        inf._tf_gpu_configured = False

        mock_gpu = MagicMock()
        mock_gpu.name = "GPU:0"

        with patch.object(inf, "logger") as mock_logger:
            with patch("tensorflow.config.list_physical_devices", return_value=[mock_gpu]):
                with patch("tensorflow.config.experimental.set_memory_growth") as mock_growth:
                    inf.configure_tf_gpu()
                    assert inf._tf_gpu_configured is True
                    assert mock_growth.call_count == 1

                    # Second call should be a no-op
                    mock_growth.reset_mock()
                    inf.configure_tf_gpu()
                    assert mock_growth.call_count == 0
    finally:
        inf._tf_gpu_configured = original_flag


def test_configure_tf_gpu_handles_runtime_error():
    """configure_tf_gpu should handle RuntimeError when GPUs already initialized."""
    import humpback.processing.inference as inf

    original_flag = inf._tf_gpu_configured
    try:
        inf._tf_gpu_configured = False

        mock_gpu = MagicMock()
        mock_gpu.name = "GPU:0"

        with patch("tensorflow.config.list_physical_devices", return_value=[mock_gpu]):
            with patch(
                "tensorflow.config.experimental.set_memory_growth",
                side_effect=RuntimeError("GPUs already initialized"),
            ):
                # Should not raise
                inf.configure_tf_gpu()
                assert inf._tf_gpu_configured is True
    finally:
        inf._tf_gpu_configured = original_flag


# ---- TF2SavedModel tests ----


def test_tf2_saved_model_gpu_failed_class_default():
    """TF2SavedModel.gpu_failed class attribute should default to False."""
    assert TF2SavedModel.gpu_failed is False


# ---- _has_xla_must_compile / _strip_xla_must_compile tests ----


def _make_saved_model_pb(tmp_path: Path, xla_must_compile: bool) -> Path:
    """Create a minimal saved_model.pb with or without _XlaMustCompile."""
    from tensorflow.core.protobuf import saved_model_pb2
    from tensorflow.core.framework import graph_pb2, attr_value_pb2

    sm = saved_model_pb2.SavedModel()
    meta = sm.meta_graphs.add()
    node = meta.graph_def.node.add()
    node.name = "StatefulPartitionedCall"
    node.op = "StatefulPartitionedCall"
    if xla_must_compile:
        node.attr["_XlaMustCompile"].CopyFrom(
            attr_value_pb2.AttrValue(b=True)
        )

    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    (model_dir / "saved_model.pb").write_bytes(sm.SerializeToString())
    return model_dir


def test_has_xla_must_compile_true(tmp_path):
    """Detects _XlaMustCompile when present."""
    model_dir = _make_saved_model_pb(tmp_path, xla_must_compile=True)
    assert _has_xla_must_compile(str(model_dir)) is True


def test_has_xla_must_compile_false(tmp_path):
    """Returns False when _XlaMustCompile is absent."""
    model_dir = _make_saved_model_pb(tmp_path, xla_must_compile=False)
    assert _has_xla_must_compile(str(model_dir)) is False


def test_has_xla_must_compile_no_pb(tmp_path):
    """Returns False when saved_model.pb doesn't exist."""
    assert _has_xla_must_compile(str(tmp_path)) is False


def test_strip_xla_must_compile(tmp_path):
    """Stripping produces a copy without _XlaMustCompile."""
    model_dir = _make_saved_model_pb(tmp_path, xla_must_compile=True)
    # Add a dummy variable file to verify it's copied
    (model_dir / "variables").mkdir()
    (model_dir / "variables" / "variables.index").write_text("dummy")

    patched_dir = _strip_xla_must_compile(str(model_dir))

    assert Path(patched_dir).exists()
    assert _has_xla_must_compile(patched_dir) is False
    # Variable files are preserved
    assert (Path(patched_dir) / "variables" / "variables.index").exists()


def test_strip_xla_must_compile_reuses_existing(tmp_path):
    """Second call reuses the already-patched directory."""
    model_dir = _make_saved_model_pb(tmp_path, xla_must_compile=True)

    first = _strip_xla_must_compile(str(model_dir))
    second = _strip_xla_must_compile(str(model_dir))
    assert first == second
