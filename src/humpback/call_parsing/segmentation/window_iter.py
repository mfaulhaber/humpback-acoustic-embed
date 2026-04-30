"""Shared windowed-inference iterator for region audio.

Pure helper used by Pass 2 segmentation inference and by the Sequence
Models CRNN region embedder. Slices a region's audio into overlapping
windows that match the sequence lengths the CRNN was trained on, and
also yields the frame offset of each window inside the region so the
caller can stitch per-frame outputs back together.

The boundary behavior matches the original Pass 2 logic byte-for-byte:
when the remaining tail at the end of the region is shorter than half a
window, the final window is pulled back so its right edge sits at the
end of the region.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

# Maximum audio duration for a single forward pass. Regions longer than
# this are processed with overlapping windows so the CRNN sees the same
# sequence lengths it was trained on (matches the 30-second crops used
# during feedback training).
MAX_WINDOW_SEC: float = 30.0
WINDOW_HOP_SEC: float = 15.0


def iter_inference_windows(
    audio: np.ndarray,
    sample_rate: int,
    frame_hop_samples: int,
    window_seconds: float = MAX_WINDOW_SEC,
    hop_seconds: float = WINDOW_HOP_SEC,
) -> Iterator[tuple[np.ndarray, int]]:
    """Yield ``(window_audio, frame_offset_in_region)`` tuples.

    For audio shorter than (or equal to) one window, yields the full
    audio with ``frame_offset_in_region == 0`` and stops.

    For longer audio, yields overlapping ``window_seconds``-long windows
    advancing by ``hop_seconds``. The final window is pulled back to end
    on the last sample whenever the tail would otherwise be shorter than
    ``window_samples // 2``.

    ``frame_offset_in_region`` is the spectrogram-frame index of the
    window's left edge inside the region, computed as
    ``round(sample_offset / frame_hop_samples)``.
    """
    total_samples = len(audio)
    if total_samples == 0:
        return
    total_duration = total_samples / sample_rate

    if total_duration <= window_seconds:
        yield audio, 0
        return

    window_samples = int(window_seconds * sample_rate)
    hop_samples = int(hop_seconds * sample_rate)

    offset = 0
    while offset < total_samples:
        end = min(offset + window_samples, total_samples)
        # If the remaining tail is too short, extend the window backwards
        if total_samples - offset < window_samples // 2 and offset > 0:
            offset = max(0, total_samples - window_samples)
            end = total_samples

        frame_offset = int(np.round(offset / frame_hop_samples))
        yield audio[offset:end], frame_offset

        if end >= total_samples:
            break
        offset += hop_samples
