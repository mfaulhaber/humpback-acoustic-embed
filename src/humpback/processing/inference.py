import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

_tf_gpu_configured = False


def configure_tf_gpu() -> None:
    """Enable memory growth on all physical GPUs (idempotent).

    Prevents TensorFlow from pre-allocating all GPU memory at startup.
    Safe to call multiple times — only configures on the first invocation.
    """
    global _tf_gpu_configured
    if _tf_gpu_configured:
        return

    import tensorflow as tf

    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Enabled memory growth for GPU: %s", gpu.name)
    except RuntimeError as e:
        # GPUs must be configured before they're initialized — if we're
        # too late, log a warning and continue (the GPU will still work,
        # it just won't have memory growth enabled).
        logger.warning("Could not configure GPU memory growth: %s", e)

    _tf_gpu_configured = True


class EmbeddingModel(Protocol):
    @property
    def vector_dim(self) -> int: ...

    def embed(self, windows: np.ndarray) -> np.ndarray:
        """Embed a batch of windows. Output: (batch, vector_dim)."""
        ...


class FakeTFLiteModel:
    """Deterministic fake model for testing. Returns sin/cos embeddings based on spectrogram content."""

    def __init__(self, vector_dim: int = 1280):
        self._vector_dim = vector_dim

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    def embed(self, spectrograms: np.ndarray) -> np.ndarray:
        batch_size = spectrograms.shape[0]
        embeddings = np.zeros((batch_size, self._vector_dim), dtype=np.float32)
        for i in range(batch_size):
            # Deterministic embedding based on spectrogram content hash
            flat = spectrograms[i].flatten()
            seed = int(np.abs(flat[:8]).sum() * 10000) % (2**31)
            t = np.arange(self._vector_dim, dtype=np.float32)
            embeddings[i] = np.sin(t * (seed + 1) / self._vector_dim)
        return embeddings


class FakeTF2Model:
    """Deterministic fake model for testing TF2 SavedModel path. Accepts raw waveform input."""

    def __init__(self, vector_dim: int = 1280):
        self._vector_dim = vector_dim

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    def embed(self, waveforms: np.ndarray) -> np.ndarray:
        """Embed raw waveform windows. Input: (batch, n_samples). Output: (batch, vector_dim)."""
        batch_size = waveforms.shape[0]
        embeddings = np.zeros((batch_size, self._vector_dim), dtype=np.float32)
        for i in range(batch_size):
            flat = waveforms[i].flatten()
            seed = int(np.abs(flat[:8]).sum() * 10000) % (2**31)
            t = np.arange(self._vector_dim, dtype=np.float32)
            embeddings[i] = np.cos(t * (seed + 1) / self._vector_dim)
        return embeddings


class TFLiteModel:
    """Real TFLite model wrapper. Only used when USE_REAL_MODEL=true."""

    def __init__(self, model_path: str, vector_dim: int = 512):
        import tensorflow as tf
        # Full TF includes flex delegate support automatically for flex models
        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        actual_dim = self._detect_output_dim()
        if actual_dim is not None and actual_dim != vector_dim:
            logger.warning(
                "Model %s: configured vector_dim=%d but actual output dim=%d; using %d",
                model_path, vector_dim, actual_dim, actual_dim,
            )
            self._vector_dim = actual_dim
        else:
            self._vector_dim = vector_dim

    def _detect_output_dim(self) -> int | None:
        """Read the actual output dimension from the interpreter's output details."""
        try:
            if self._output_details and len(self._output_details[0]["shape"]) == 2:
                return int(self._output_details[0]["shape"][1])
        except (IndexError, KeyError):
            pass
        return None

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    def embed(self, spectrograms: np.ndarray) -> np.ndarray:
        """Embed a batch of spectrograms. Input: (batch, n_mels, time_frames). Output: (batch, vector_dim)."""
        results = []
        for i in range(spectrograms.shape[0]):
            inp = spectrograms[i : i + 1].astype(np.float32)
            self._interpreter.set_tensor(self._input_details[0]["index"], inp)
            self._interpreter.invoke()
            out = self._interpreter.get_tensor(self._output_details[0]["index"])
            results.append(out[0])
        return np.array(results, dtype=np.float32)


def _has_xla_must_compile(model_dir: str) -> bool:
    """Check if a SavedModel's graph contains ``_XlaMustCompile`` attributes.

    JAX-exported models (via ``jax2tf``) set this attribute, which forces XLA
    compilation.  XLA is unavailable on Metal GPU, so these models need the
    attribute stripped before they can run on Apple Silicon GPUs.
    """
    from pathlib import Path

    from tensorflow.core.protobuf import saved_model_pb2

    pb_path = Path(model_dir) / "saved_model.pb"
    if not pb_path.exists():
        return False

    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString(pb_path.read_bytes())

    for meta_graph in sm.meta_graphs:
        gd = meta_graph.graph_def
        for node in gd.node:
            if "_XlaMustCompile" in node.attr:
                return True
        for func in gd.library.function:
            if "_XlaMustCompile" in func.attr:
                return True
            for node in func.node_def:
                if "_XlaMustCompile" in node.attr:
                    return True
    return False


def _strip_xla_must_compile(model_dir: str) -> str:
    """Return a path to a copy of *model_dir* with ``_XlaMustCompile`` removed.

    The patched copy is placed next to the original as
    ``<model_dir>-no-xla-compile`` and reused on subsequent calls.  Only the
    ``saved_model.pb`` file is modified; variable shards are shared via
    hard-links (or copies if hard-linking fails) to avoid doubling disk usage.
    """
    import shutil
    from pathlib import Path

    from tensorflow.core.protobuf import saved_model_pb2

    src = Path(model_dir)
    dest = src.with_name(src.name + "-no-xla-compile")

    # Reuse existing patched copy if it exists.
    if (dest / "saved_model.pb").exists():
        logger.info("Reusing patched (no-XLA) model at %s", dest)
        return str(dest)

    logger.info("Stripping _XlaMustCompile from %s → %s", src, dest)

    # Copy the directory structure, preferring hard-links for large variable
    # shards to save disk space.
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, copy_function=_try_hardlink)

    # Patch the protobuf.
    pb_path = dest / "saved_model.pb"
    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString(pb_path.read_bytes())

    count = 0
    for meta_graph in sm.meta_graphs:
        gd = meta_graph.graph_def
        for node in gd.node:
            if "_XlaMustCompile" in node.attr:
                del node.attr["_XlaMustCompile"]
                count += 1
        for func in gd.library.function:
            if "_XlaMustCompile" in func.attr:
                del func.attr["_XlaMustCompile"]
                count += 1
            for node in func.node_def:
                if "_XlaMustCompile" in node.attr:
                    del node.attr["_XlaMustCompile"]
                    count += 1

    pb_path.write_bytes(sm.SerializeToString())
    logger.info("Stripped %d _XlaMustCompile attribute(s)", count)
    return str(dest)


def _try_hardlink(src: str, dst: str) -> None:
    """Copy helper: hard-link if possible, fall back to regular copy."""
    import shutil
    from pathlib import Path

    try:
        Path(dst).hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


class TF2SavedModel:
    """TensorFlow 2 SavedModel wrapper. Takes raw waveform input.

    Attempts GPU inference first; falls back to CPU if validation fails
    (e.g. Metal/XLA producing incorrect results).  The ``gpu_failed`` flag
    is set when fallback occurs so callers can surface a warning.

    JAX-exported models that embed ``_XlaMustCompile`` attributes are
    automatically detected and patched so they can run on Metal GPU (which
    has no XLA backend).
    """

    gpu_failed: bool = False

    def __init__(self, model_dir: str, vector_dim: int = 1280, *, force_cpu: bool = False):
        import tensorflow as tf

        self._vector_dim = vector_dim

        cpus = tf.config.list_logical_devices("CPU")
        cpu_device = cpus[0].name if cpus else "/device:CPU:0"

        # --- Force-CPU escape hatch ---
        if force_cpu:
            logger.info(
                "Loading TF2 SavedModel from %s (force_cpu=True, device=%s)",
                model_dir,
                cpu_device,
            )
            with tf.device(cpu_device):
                self._model = tf.saved_model.load(model_dir)
            self._serving_fn = self._model.signatures["serving_default"]
            self._device = cpu_device
            self._auto_correct_vector_dim(model_dir)
            return

        # --- Attempt GPU ---
        configure_tf_gpu()

        gpus = tf.config.list_logical_devices("GPU")
        if not gpus:
            logger.info(
                "Loading TF2 SavedModel from %s (no GPU found, using CPU: %s)",
                model_dir,
                cpu_device,
            )
            with tf.device(cpu_device):
                self._model = tf.saved_model.load(model_dir)
            self._serving_fn = self._model.signatures["serving_default"]
            self._device = cpu_device
            self._auto_correct_vector_dim(model_dir)
            return

        gpu_device = gpus[0].name

        # --- Auto-patch JAX models for Metal compatibility ---
        gpu_model_dir = model_dir
        if _has_xla_must_compile(model_dir):
            logger.info(
                "Model at %s contains _XlaMustCompile (JAX/XLA model). "
                "Patching for Metal GPU compatibility.",
                model_dir,
            )
            gpu_model_dir = _strip_xla_must_compile(model_dir)

        logger.info(
            "Loading TF2 SavedModel from %s (GPU available: %s, validating…)",
            model_dir,
            gpu_device,
        )

        # Load a CPU copy first as the validation baseline.
        with tf.device(cpu_device):
            cpu_model = tf.saved_model.load(model_dir)
        cpu_fn = cpu_model.signatures["serving_default"]

        # Load a GPU copy for validation (possibly from patched dir).
        try:
            with tf.device(gpu_device):
                gpu_model = tf.saved_model.load(gpu_model_dir)
            gpu_fn = gpu_model.signatures["serving_default"]
        except Exception:
            logger.exception(
                "Failed to load model on GPU. Falling back to CPU: %s",
                cpu_device,
            )
            self.gpu_failed = True
            self._model = cpu_model
            self._serving_fn = cpu_fn
            self._device = cpu_device
            self._auto_correct_vector_dim(model_dir)
            return

        # Validate GPU output matches CPU
        try:
            if self._validate_gpu(gpu_fn, cpu_fn, gpu_device, cpu_device):
                logger.info("GPU validation passed. Using GPU device: %s", gpu_device)
                self._model = gpu_model
                self._serving_fn = gpu_fn
                self._device = gpu_device
                del cpu_model
                self._auto_correct_vector_dim(model_dir)
                return
        except Exception:
            logger.exception(
                "GPU validation raised an exception. Falling back to CPU: %s",
                cpu_device,
            )

        # Fallback to CPU
        logger.warning(
            "GPU validation FAILED — outputs differ from CPU. "
            "Falling back to CPU: %s",
            cpu_device,
        )
        self.gpu_failed = True
        self._model = cpu_model
        self._serving_fn = cpu_fn
        self._device = cpu_device
        self._auto_correct_vector_dim(model_dir)
        del gpu_model

    def _auto_correct_vector_dim(self, model_dir: str) -> None:
        """Detect actual output dim from the serving function and correct if needed."""
        try:
            shape = self._serving_fn.structured_outputs["embedding"].shape
            if len(shape) == 2 and shape[1] is not None:
                actual_dim = int(shape[1])
                if actual_dim != self._vector_dim:
                    logger.warning(
                        "Model %s: configured vector_dim=%d but actual output dim=%d; using %d",
                        model_dir, self._vector_dim, actual_dim, actual_dim,
                    )
                    self._vector_dim = actual_dim
        except (KeyError, IndexError, TypeError, AttributeError):
            pass

    @staticmethod
    def _validate_gpu(gpu_fn, cpu_fn, gpu_device: str, cpu_device: str) -> bool:
        """Run a deterministic test input through both serving functions, compare.

        Each function's variables live on its own device, so we call each
        within the matching device scope to avoid cross-device access errors.

        Returns True if outputs match within tolerance.
        """
        import tensorflow as tf

        rng = np.random.RandomState(42)
        test_input = rng.randn(2, 160_000).astype(np.float32)
        tensor_input = tf.constant(test_input, dtype=tf.float32)

        with tf.device(gpu_device):
            gpu_result = gpu_fn(inputs=tensor_input)["embedding"].numpy()

        with tf.device(cpu_device):
            cpu_result = cpu_fn(inputs=tensor_input)["embedding"].numpy()

        if np.allclose(gpu_result, cpu_result, atol=1e-4, rtol=1e-3):
            return True

        max_diff = np.max(np.abs(gpu_result - cpu_result))
        logger.warning(
            "GPU vs CPU max absolute difference: %e (atol=1e-4, rtol=1e-3)",
            max_diff,
        )
        return False

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    def embed(self, waveforms: np.ndarray) -> np.ndarray:
        """Embed raw waveform windows. Input: (batch, n_samples). Output: (batch, vector_dim)."""
        import tensorflow as tf

        inputs = tf.constant(waveforms, dtype=tf.float32)
        with tf.device(self._device):
            result = self._serving_fn(inputs=inputs)
        return result["embedding"].numpy()
