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
    audio: np.ndarray, sample_rate: int, window_seconds: float,
    hop_seconds: float | None = None,
) -> Iterator[np.ndarray]:
    """Slice audio into fixed-length windows.

    If the last chunk is shorter than a full window, the window is shifted
    backward so it ends at the audio boundary (overlapping with the previous
    window).  Audio shorter than one full window is skipped entirely (yields
    nothing).

    Args:
        hop_seconds: Stride between window starts. Defaults to window_seconds
            (no overlap). Must be > 0 and <= window_seconds.
    """
    window_samples = int(sample_rate * window_seconds)
    if window_samples <= 0:
        raise ValueError(f"Invalid window: {window_seconds}s at {sample_rate}Hz")

    if hop_seconds is not None:
        if hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive")
        if hop_seconds > window_seconds:
            raise ValueError("hop_seconds must be <= window_seconds")

    hop_samples = int(sample_rate * hop_seconds) if hop_seconds else window_samples

    n_samples = len(audio)
    if n_samples < window_samples:
        return  # too short — skip

    offset = 0
    while offset < n_samples:
        remaining = n_samples - offset
        is_overlap_back = False
        if remaining < window_samples:
            new_offset = n_samples - window_samples
            if new_offset <= offset - hop_samples:
                break
            offset = new_offset
            is_overlap_back = True
        yield audio[offset : offset + window_samples]
        if is_overlap_back:
            break
        offset += hop_samples


def slice_windows_with_metadata(
    audio: np.ndarray, sample_rate: int, window_seconds: float,
    hop_seconds: float | None = None,
) -> Iterator[tuple[np.ndarray, WindowMetadata]]:
    """Slice audio into windows, yielding (window, metadata) tuples.

    Same slicing logic as slice_windows but additionally tracks whether each
    window was shifted backward (overlapped) to avoid zero-padding.
    """
    window_samples = int(sample_rate * window_seconds)
    if window_samples <= 0:
        raise ValueError(f"Invalid window: {window_seconds}s at {sample_rate}Hz")

    if hop_seconds is not None:
        if hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive")
        if hop_seconds > window_seconds:
            raise ValueError("hop_seconds must be <= window_seconds")

    hop_samples = int(sample_rate * hop_seconds) if hop_seconds else window_samples

    n_samples = len(audio)
    if n_samples < window_samples:
        return  # too short — skip

    offset = 0
    idx = 0
    while offset < n_samples:
        remaining = n_samples - offset
        is_overlapped = False
        is_overlap_back = False
        if remaining < window_samples:
            new_offset = n_samples - window_samples
            if new_offset <= offset - hop_samples:
                break
            offset = new_offset
            is_overlapped = True
            is_overlap_back = True
        chunk = audio[offset : offset + window_samples]
        meta = WindowMetadata(
            window_index=idx,
            offset_sec=offset / sample_rate,
            is_overlapped=is_overlapped,
            original_samples=len(chunk),
        )
        yield chunk, meta
        if is_overlap_back:
            break
        offset += hop_samples
        idx += 1


def count_windows(n_samples: int, sample_rate: int, window_seconds: float,
                  hop_seconds: float | None = None) -> int:
    """Return number of windows for given audio length."""
    window_samples = int(sample_rate * window_seconds)
    if window_samples <= 0:
        return 0
    if n_samples < window_samples:
        return 0
    hop_samples = int(sample_rate * hop_seconds) if hop_seconds else window_samples
    # Hop-aligned windows + optional overlap-back window
    k = (n_samples - window_samples) // hop_samples
    remainder = (n_samples - window_samples) % hop_samples
    return k + 1 + (1 if remainder > 0 else 0)
