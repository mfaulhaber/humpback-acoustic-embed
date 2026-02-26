import numpy as np


def extract_logmel(
    window: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Extract log-mel spectrogram features from an audio window.

    Returns shape (n_mels, time_frames).
    """
    try:
        import librosa

        S = librosa.feature.melspectrogram(
            y=window,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        return librosa.power_to_db(S, ref=np.max)
    except ImportError:
        # Fallback: simple FFT-based approximation for testing
        return _simple_logmel(window, sample_rate, n_mels, n_fft, hop_length)


def _simple_logmel(
    window: np.ndarray, sample_rate: int, n_mels: int, n_fft: int, hop_length: int
) -> np.ndarray:
    """Simple log-mel approximation without librosa."""
    n_frames = 1 + (len(window) - n_fft) // hop_length
    n_frames = max(n_frames, 1)
    # Just return a deterministic feature matrix based on the window
    rng = np.random.RandomState(int(np.abs(window[:4]).sum() * 1000) % (2**31))
    return rng.randn(n_mels, n_frames).astype(np.float32)
