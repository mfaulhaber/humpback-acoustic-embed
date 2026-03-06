from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class WindowMetadata:
    """Per-window metadata emitted by slice_windows_with_metadata."""

    window_index: int
    offset_sec: float
    is_overlapped: bool
    original_samples: int


def slice_windows(
    audio: np.ndarray, sample_rate: int, window_seconds: float
) -> Iterator[np.ndarray]:
    """Slice audio into fixed-length windows.

    If the last chunk is shorter than a full window, the window is shifted
    backward so it ends at the audio boundary (overlapping with the previous
    window).  Audio shorter than one full window is skipped entirely (yields
    nothing).
    """
    window_samples = int(sample_rate * window_seconds)
    if window_samples <= 0:
        raise ValueError(f"Invalid window: {window_seconds}s at {sample_rate}Hz")

    n_samples = len(audio)
    if n_samples < window_samples:
        return  # too short — skip

    offset = 0
    while offset < n_samples:
        remaining = n_samples - offset
        if remaining < window_samples:
            # Overlap: shift back so the window ends at n_samples
            offset = n_samples - window_samples
        yield audio[offset : offset + window_samples]
        offset += window_samples


def slice_windows_with_metadata(
    audio: np.ndarray, sample_rate: int, window_seconds: float
) -> Iterator[tuple[np.ndarray, WindowMetadata]]:
    """Slice audio into windows, yielding (window, metadata) tuples.

    Same slicing logic as slice_windows but additionally tracks whether each
    window was shifted backward (overlapped) to avoid zero-padding.
    """
    window_samples = int(sample_rate * window_seconds)
    if window_samples <= 0:
        raise ValueError(f"Invalid window: {window_seconds}s at {sample_rate}Hz")

    n_samples = len(audio)
    if n_samples < window_samples:
        return  # too short — skip

    offset = 0
    idx = 0
    while offset < n_samples:
        remaining = n_samples - offset
        is_overlapped = False
        if remaining < window_samples:
            offset = n_samples - window_samples
            is_overlapped = True
        chunk = audio[offset : offset + window_samples]
        meta = WindowMetadata(
            window_index=idx,
            offset_sec=offset / sample_rate,
            is_overlapped=is_overlapped,
            original_samples=len(chunk),
        )
        yield chunk, meta
        offset += window_samples
        idx += 1


def count_windows(n_samples: int, sample_rate: int, window_seconds: float) -> int:
    """Return number of windows for given audio length."""
    window_samples = int(sample_rate * window_seconds)
    if window_samples <= 0:
        return 0
    if n_samples < window_samples:
        return 0
    # Full windows + 1 overlapped window if there's a remainder
    full = n_samples // window_samples
    remainder = n_samples % window_samples
    return full + (1 if remainder > 0 else 0)
