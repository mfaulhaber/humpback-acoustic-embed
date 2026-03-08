import numpy as np

from humpback.processing.features import extract_logmel, extract_logmel_batch

SR = 32000
WINDOW_SAMPLES = SR * 5  # 5-second windows (160000 samples)


def _make_windows(n: int, seed: int = 42) -> list[np.ndarray]:
    """Generate deterministic test windows."""
    rng = np.random.RandomState(seed)
    return [rng.randn(WINDOW_SAMPLES).astype(np.float32) for _ in range(n)]


# ── existing tests ──────────────────────────────────────────────


def test_output_shape():
    window = np.random.randn(32000).astype(np.float32)
    features = extract_logmel(window, 32000, n_mels=128, n_fft=2048, hop_length=512)
    assert features.shape[0] == 128  # n_mels
    assert features.ndim == 2


def test_deterministic():
    window = np.sin(np.linspace(0, 100, 32000)).astype(np.float32)
    f1 = extract_logmel(window, 32000)
    f2 = extract_logmel(window, 32000)
    np.testing.assert_array_equal(f1, f2)


# ── batch equivalence tests ────────────────────────────────────


class TestBatchMatchesSingle:
    """Verify extract_logmel_batch matches per-window extract_logmel."""

    def test_per_window_max(self):
        windows = _make_windows(4)
        batch = extract_logmel_batch(windows, SR, normalization="per_window_max")
        for i, w in enumerate(windows):
            single = extract_logmel(w, SR, normalization="per_window_max")
            np.testing.assert_allclose(batch[i], single, atol=1e-4)

    def test_global_ref(self):
        windows = _make_windows(4)
        batch = extract_logmel_batch(windows, SR, normalization="global_ref")
        for i, w in enumerate(windows):
            single = extract_logmel(w, SR, normalization="global_ref")
            np.testing.assert_allclose(batch[i], single, atol=1e-4)

    def test_standardize(self):
        windows = _make_windows(4)
        batch = extract_logmel_batch(windows, SR, normalization="standardize")
        for i, w in enumerate(windows):
            single = extract_logmel(w, SR, normalization="standardize")
            np.testing.assert_allclose(batch[i], single, atol=1e-4)


class TestBatchEdgeCases:
    """Edge cases for extract_logmel_batch."""

    def test_empty_input(self):
        assert extract_logmel_batch([], SR) == []

    def test_single_window(self):
        windows = _make_windows(1)
        batch = extract_logmel_batch(windows, SR)
        single = extract_logmel(windows[0], SR)
        np.testing.assert_allclose(batch[0], single, atol=1e-4)

    def test_chunking(self):
        """Results are identical regardless of chunk_size."""
        windows = _make_windows(7)
        full = extract_logmel_batch(windows, SR, chunk_size=100)
        chunked = extract_logmel_batch(windows, SR, chunk_size=3)
        assert len(full) == len(chunked) == 7
        for i in range(7):
            np.testing.assert_array_equal(full[i], chunked[i])

    def test_target_frames(self):
        windows = _make_windows(2)
        batch = extract_logmel_batch(windows, SR, target_frames=128)
        for spec in batch:
            assert spec.shape == (128, 128)

    def test_output_shapes(self):
        windows = _make_windows(3)
        batch = extract_logmel_batch(windows, SR)
        assert len(batch) == 3
        for spec in batch:
            assert spec.ndim == 2
            assert spec.shape[0] == 128  # n_mels
