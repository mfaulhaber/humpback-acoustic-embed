from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from humpback.processing.inference import (
    FakeTF2Model,
    FakeTFLiteModel,
    TFLiteModel,
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


# ---- TFLiteModel auto-detect vector_dim tests ----


def _make_tflite_model_stub(output_shape, vector_dim):
    """Create a TFLiteModel stub with mocked output details (bypasses real interpreter)."""
    model = object.__new__(TFLiteModel)
    model._output_details = [{"index": 0, "shape": np.array(output_shape)}]
    model._vector_dim = vector_dim
    return model


def test_tflite_auto_detects_vector_dim():
    """TFLiteModel._detect_output_dim should read actual dim from output details."""
    model = _make_tflite_model_stub([1, 1536], vector_dim=1280)
    actual = model._detect_output_dim()
    assert actual == 1536


def test_tflite_no_override_when_dims_match():
    """_detect_output_dim returns the matching dim (no mismatch to correct)."""
    model = _make_tflite_model_stub([1, 1280], vector_dim=1280)
    actual = model._detect_output_dim()
    assert actual == 1280


def test_tflite_fallback_when_detection_fails():
    """_detect_output_dim returns None for unusual output shapes."""
    model = _make_tflite_model_stub([1], vector_dim=512)
    actual = model._detect_output_dim()
    assert actual is None


# ---- TFLiteModel batch embed tests ----


def _make_batch_tflite_model(vector_dim=128):
    """Create a TFLiteModel with a mocked interpreter for batch testing."""
    model = object.__new__(TFLiteModel)
    model._vector_dim = vector_dim
    model._batch_resize_failed = False

    interpreter = MagicMock()
    model._interpreter = interpreter
    model._input_details = [{"index": 0, "shape": np.array([1, 128, 128])}]
    model._output_details = [{"index": 1, "shape": np.array([1, vector_dim])}]
    model._base_input_shape = [1, 128, 128]
    model._last_batch_size = 1
    model._num_threads = 4

    def fake_get_tensor(idx):
        # Return output shaped to current batch size
        return np.zeros((model._last_batch_size, vector_dim), dtype=np.float32)

    interpreter.get_tensor.side_effect = fake_get_tensor
    return model


def test_tflite_batch_embed_single_invoke():
    """Batch of 4 should call invoke() exactly once (not 4 times)."""
    model = _make_batch_tflite_model(vector_dim=128)
    spectrograms = np.random.randn(4, 128, 128).astype(np.float32)

    # Track batch size through resize
    def track_resize(idx, shape):
        model._last_batch_size = shape[0]

    model._interpreter.resize_tensor_input.side_effect = track_resize

    result = model.embed(spectrograms)
    assert result.shape == (4, 128)
    assert model._interpreter.invoke.call_count == 1


def test_tflite_batch_embed_resize_on_size_change():
    """resize_tensor_input should be called when batch size changes from default."""
    model = _make_batch_tflite_model(vector_dim=128)
    spectrograms = np.random.randn(4, 128, 128).astype(np.float32)

    def track_resize(idx, shape):
        model._last_batch_size = shape[0]

    model._interpreter.resize_tensor_input.side_effect = track_resize

    model.embed(spectrograms)
    model._interpreter.resize_tensor_input.assert_called_once_with(0, [4, 128, 128])


def test_tflite_batch_embed_no_resize_same_size():
    """No resize should happen when batch size matches the last batch size."""
    model = _make_batch_tflite_model(vector_dim=128)
    model._last_batch_size = 4  # pretend we already resized to 4

    spectrograms = np.random.randn(4, 128, 128).astype(np.float32)
    model.embed(spectrograms)
    model._interpreter.resize_tensor_input.assert_not_called()


def test_tflite_batch_embed_empty():
    """Empty input should return shape (0, vector_dim) without calling interpreter."""
    model = _make_batch_tflite_model(vector_dim=256)
    spectrograms = np.empty((0, 128, 128), dtype=np.float32)

    result = model.embed(spectrograms)
    assert result.shape == (0, 256)
    model._interpreter.invoke.assert_not_called()


def test_tflite_batch_fallback_on_resize_failure():
    """When resize raises, model should fall back to sequential with warning logged."""
    model = _make_batch_tflite_model(vector_dim=128)
    model._interpreter.resize_tensor_input.side_effect = RuntimeError("resize not supported")

    # For sequential fallback, get_tensor returns single-item output
    model._interpreter.get_tensor.side_effect = lambda idx: np.zeros(
        (1, 128), dtype=np.float32
    )

    spectrograms = np.random.randn(3, 128, 128).astype(np.float32)
    result = model.embed(spectrograms)

    assert result.shape == (3, 128)
    assert model._batch_resize_failed is True
    # Sequential: invoke called once per item
    assert model._interpreter.invoke.call_count == 3


def test_tflite_num_threads_passed():
    """num_threads should be passed to the Interpreter constructor."""
    import tensorflow as tf

    with patch.object(tf.lite, "Interpreter", wraps=None) as mock_interp_cls:
        mock_interp = MagicMock()
        mock_interp.get_input_details.return_value = [
            {"index": 0, "shape": np.array([1, 128, 128])}
        ]
        mock_interp.get_output_details.return_value = [
            {"index": 1, "shape": np.array([1, 512])}
        ]
        mock_interp_cls.return_value = mock_interp

        TFLiteModel("dummy.tflite", vector_dim=512, num_threads=8)

        mock_interp_cls.assert_called_once_with(
            model_path="dummy.tflite", num_threads=8
        )
