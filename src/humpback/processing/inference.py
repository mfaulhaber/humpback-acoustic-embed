from typing import Protocol

import numpy as np


class EmbeddingModel(Protocol):
    @property
    def vector_dim(self) -> int: ...

    def embed(self, windows: np.ndarray) -> np.ndarray:
        """Embed a batch of spectrograms. Input: (batch, n_mels, time_frames). Output: (batch, vector_dim)."""
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


class TFLiteModel:
    """Real TFLite model wrapper. Only used when USE_REAL_MODEL=true."""

    def __init__(self, model_path: str, vector_dim: int = 512):
        import tensorflow as tf
        # Full TF includes flex delegate support automatically for flex models
        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._vector_dim = vector_dim

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
