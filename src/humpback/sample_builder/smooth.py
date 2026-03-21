"""Stage 8: Smooth splice joins with raised-cosine crossfade."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from humpback.processing.dsp import raised_cosine_fade


def smooth_joins(
    audio: NDArray[np.floating],
    splice_points: list[int],
    sr: int,
    crossfade_ms: float = 50.0,
) -> NDArray[np.floating]:
    """Apply raised-cosine crossfade at each splice boundary.

    At each splice point, a short crossfade window blends the audio on
    either side to eliminate discontinuity clicks.

    Parameters
    ----------
    audio:
        The assembled sample (1-D, mono).
    splice_points:
        Sample indices where fragments were joined.
    sr:
        Sample rate in Hz.
    crossfade_ms:
        Crossfade duration in milliseconds (applied symmetrically around
        each splice point).

    Returns
    -------
    Audio array with smoothed joins (same length as input).
    """
    result = audio.copy()
    half_fade = int((crossfade_ms / 1000.0) * sr / 2.0)

    if half_fade <= 0:
        return result

    for sp in splice_points:
        # Crossfade region: [sp - half_fade, sp + half_fade]
        fade_start = max(0, sp - half_fade)
        fade_end = min(len(result), sp + half_fade)
        fade_len = fade_end - fade_start

        if fade_len < 2:
            continue

        # Apply fade-out then fade-in ramp
        ramp = raised_cosine_fade(fade_len)
        # Before splice point: fade out (multiply by 1→0)
        # After splice point: fade in (multiply by 0→1)
        # Combined: smooth transition
        result[fade_start:fade_end] *= ramp

    return result
