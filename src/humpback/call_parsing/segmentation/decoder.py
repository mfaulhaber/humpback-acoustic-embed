"""Hysteresis decoder for Pass 2 segmentation.

Pure function that turns a framewise probability vector into a list of
``Event`` rows. Not coupled to the CRNN, audio, or I/O — just threshold
crossings, gap merging, minimum-duration filtering, and absolute
timestamp math. The caller (the event segmentation worker) supplies
``region_id`` and the region's audio-relative start time so the decoder
can write audio-timeline timestamps directly onto each ``Event``.
"""

from __future__ import annotations

import uuid

import numpy as np

from humpback.call_parsing.types import Event
from humpback.schemas.call_parsing import SegmentationDecoderConfig


def decode_events(
    frame_probs: np.ndarray,
    region_id: str,
    region_start_sec: float,
    hop_sec: float,
    config: SegmentationDecoderConfig,
) -> list[Event]:
    """Turn frame probabilities into ``Event`` rows via hysteresis.

    Walks ``frame_probs`` left-to-right, opens events on frames that
    cross ``config.high_threshold``, extends while frames stay at or
    above ``config.low_threshold``, and closes on dips below. Adjacent
    events whose frame-quantized gap is below ``config.merge_gap_sec``
    are fused, then events shorter than ``config.min_event_sec`` are
    dropped. Survivors are returned sorted by ``start_sec`` (already
    sorted by construction).
    """
    if frame_probs.ndim != 1:
        raise ValueError(
            f"decode_events expects 1-D frame_probs, got shape {frame_probs.shape}"
        )
    if hop_sec <= 0.0:
        raise ValueError(f"hop_sec must be positive, got {hop_sec}")

    n_frames = int(frame_probs.shape[0])
    high = float(config.high_threshold)
    low = float(config.low_threshold)

    spans: list[tuple[int, int]] = []
    inside = False
    first_idx = 0
    for i in range(n_frames):
        p = float(frame_probs[i])
        if not inside:
            if p >= high:
                inside = True
                first_idx = i
        else:
            if p < low:
                spans.append((first_idx, i - 1))
                inside = False
    if inside:
        spans.append((first_idx, n_frames - 1))

    merge_gap_frames = int(round(config.merge_gap_sec / hop_sec))
    merged: list[tuple[int, int]] = []
    for span in spans:
        if merged:
            prev_first, prev_last = merged[-1]
            gap_frames = span[0] - prev_last - 1
            if gap_frames < merge_gap_frames:
                merged[-1] = (prev_first, span[1])
                continue
        merged.append(span)

    surviving: list[tuple[int, int]] = [
        (first, last)
        for first, last in merged
        if (last - first + 1) * hop_sec >= config.min_event_sec
    ]

    events: list[Event] = []
    for first, last in surviving:
        start_sec = region_start_sec + first * hop_sec
        end_sec = region_start_sec + (last + 1) * hop_sec
        center_sec = (start_sec + end_sec) / 2.0
        confidence = float(np.asarray(frame_probs[first : last + 1]).max())
        events.append(
            Event(
                event_id=uuid.uuid4().hex,
                region_id=region_id,
                start_sec=start_sec,
                end_sec=end_sec,
                center_sec=center_sec,
                segmentation_confidence=confidence,
            )
        )
    return events
