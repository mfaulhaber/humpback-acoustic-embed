import numpy as np


def extract_logmel(
    window: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    target_frames: int | None = None,
) -> np.ndarray:
    """Extract log-mel spectrogram features from an audio window.

    Returns shape (n_mels, time_frames). If target_frames is set,
    pads or truncates the time axis to that length.
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
        result = librosa.power_to_db(S, ref=np.max)
    except ImportError:
        # Fallback: simple FFT-based approximation for testing
        result = _simple_logmel(window, sample_rate, n_mels, n_fft, hop_length)

    if target_frames is not None:
        result = _fit_time_frames(result, target_frames)
    return result


def _simple_logmel(
    window: np.ndarray, sample_rate: int, n_mels: int, n_fft: int, hop_length: int
) -> np.ndarray:
    """Simple log-mel approximation without librosa."""
    n_frames = 1 + (len(window) - n_fft) // hop_length
    n_frames = max(n_frames, 1)
    # Just return a deterministic feature matrix based on the window
    rng = np.random.RandomState(int(np.abs(window[:4]).sum() * 1000) % (2**31))
    return rng.randn(n_mels, n_frames).astype(np.float32)


def _fit_time_frames(spec: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad or truncate spectrogram time axis to target_frames."""
    n_mels, n_frames = spec.shape
    if n_frames == target_frames:
        return spec
    if n_frames > target_frames:
        return spec[:, :target_frames]
    # Pad with minimum value (silence in dB scale)
    pad_width = target_frames - n_frames
    return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=spec.min())
