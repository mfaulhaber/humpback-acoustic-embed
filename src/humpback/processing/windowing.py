from collections.abc import Iterator

import numpy as np


def slice_windows(
    audio: np.ndarray, sample_rate: int, window_seconds: float
) -> Iterator[np.ndarray]:
    """Slice audio into fixed-length windows. Zero-pads the final window if needed."""
    window_samples = int(sample_rate * window_seconds)
    if window_samples <= 0:
        raise ValueError(f"Invalid window: {window_seconds}s at {sample_rate}Hz")

    n_samples = len(audio)
    offset = 0
    while offset < n_samples:
        chunk = audio[offset : offset + window_samples]
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))
        yield chunk
        offset += window_samples


def count_windows(n_samples: int, sample_rate: int, window_seconds: float) -> int:
    """Return number of windows for given audio length."""
    window_samples = int(sample_rate * window_seconds)
    if window_samples <= 0:
        return 0
    return max(1, (n_samples + window_samples - 1) // window_samples)
