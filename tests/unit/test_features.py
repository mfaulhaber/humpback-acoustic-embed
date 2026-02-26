import numpy as np

from humpback.processing.features import extract_logmel


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
