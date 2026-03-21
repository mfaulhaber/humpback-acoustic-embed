"""Shared DSP utilities used by multiple pipelines."""

from __future__ import annotations

import numpy as np


def raised_cosine_fade(length: int) -> np.ndarray:
    """Return a half-cosine ramp from 0 to 1 of *length* samples.

    Used for crossfade blending at splice boundaries in both the
    score-based label processor and the sample builder pipeline.
    """
    if length <= 0:
        return np.array([], dtype=np.float32)
    return (0.5 * (1.0 - np.cos(np.pi * np.arange(length) / length))).astype(np.float32)
