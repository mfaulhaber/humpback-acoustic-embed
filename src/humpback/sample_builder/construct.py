"""Stage 7: Assemble the final sample from an AssemblyPlan."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from humpback.sample_builder.planner import AssemblyPlan


def construct_sample(
    plan: AssemblyPlan,
    full_audio: NDArray[np.floating],
    sr: int,
) -> tuple[NDArray[np.floating], list[int]]:
    """Assemble a sample by extracting and concatenating fragment audio.

    Parameters
    ----------
    plan:
        A successful AssemblyPlan (``can_assemble=True``).
    full_audio:
        The complete recording audio (1-D, mono).
    sr:
        Sample rate in Hz.

    Returns
    -------
    Tuple of (assembled audio array, splice point sample indices).
    The audio is padded or trimmed to exactly ``plan.window_size * sr`` samples.
    """
    target_samples = int(plan.window_size * sr)
    parts: list[NDArray[np.floating]] = []
    splice_points: list[int] = []
    cursor = 0

    # Left background fragments
    for frag in plan.left_fragments:
        start_sample = int(frag.start_sec * sr)
        end_sample = int(frag.end_sec * sr)
        end_sample = min(end_sample, len(full_audio))
        start_sample = min(start_sample, end_sample)
        chunk = full_audio[start_sample:end_sample]
        parts.append(chunk)
        cursor += len(chunk)
        splice_points.append(cursor)

    # Target vocalization
    target_start = int(plan.target_start_sec * sr)
    target_end = int(plan.target_end_sec * sr)
    target_end = min(target_end, len(full_audio))
    target_start = min(target_start, target_end)
    target_chunk = full_audio[target_start:target_end]
    parts.append(target_chunk)
    cursor += len(target_chunk)

    # Splice point after target (before right background)
    if plan.right_fragments:
        splice_points.append(cursor)

    # Right background fragments
    for frag in plan.right_fragments:
        start_sample = int(frag.start_sec * sr)
        end_sample = int(frag.end_sec * sr)
        end_sample = min(end_sample, len(full_audio))
        start_sample = min(start_sample, end_sample)
        chunk = full_audio[start_sample:end_sample]
        parts.append(chunk)
        cursor += len(chunk)

    # Concatenate all parts
    if parts:
        result = np.concatenate(parts)
    else:
        result = np.zeros(target_samples, dtype=full_audio.dtype)

    # Pad or trim to exact window size
    if len(result) < target_samples:
        padded = np.zeros(target_samples, dtype=result.dtype)
        padded[: len(result)] = result
        result = padded
    elif len(result) > target_samples:
        result = result[:target_samples]
        # Adjust splice points that fall beyond the trim
        splice_points = [sp for sp in splice_points if sp < target_samples]

    return result, splice_points
