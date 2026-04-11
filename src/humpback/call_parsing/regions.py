"""Pass 1 region decoder.

Pure transformation: hysteresis events + audio duration + config -> padded,
merged ``Region`` list. No I/O, no audio, no models. The worker calls this
once after ``merge_detection_events`` has run on the full concatenated trace.

The algorithm is documented in ADR-049 and
``docs/specs/2026-04-11-call-parsing-pass1-region-detector-design.md``.
"""

from __future__ import annotations

import uuid
from typing import Any

from humpback.call_parsing.types import Region
from humpback.schemas.call_parsing import RegionDetectionConfig


def decode_regions(
    events: list[dict[str, Any]],
    audio_duration_sec: float,
    config: RegionDetectionConfig,
) -> list[Region]:
    """Turn hysteresis events into padded, merged ``Region`` rows.

    Accepts the dict shape ``merge_detection_events`` returns:
    ``{start_sec, end_sec, avg_confidence, peak_confidence, n_windows}``.
    Returns frozen ``Region`` dataclasses sorted by ``start_sec`` with
    ``padded_start_sec`` / ``padded_end_sec`` clamped to
    ``[0.0, audio_duration_sec]``.
    """
    if not events:
        return []

    padding = config.padding_sec
    duration = audio_duration_sec

    # Build working dicts of (raw bounds, padded bounds, aggregates) sorted
    # by raw start. Everything downstream operates on this shape; we only
    # materialize ``Region`` at the very end.
    working: list[dict[str, Any]] = []
    for ev in sorted(events, key=lambda e: e["start_sec"]):
        raw_start = float(ev["start_sec"])
        raw_end = float(ev["end_sec"])
        padded_start = max(0.0, raw_start - padding)
        padded_end = min(duration, raw_end + padding)
        n_windows = int(ev["n_windows"])
        working.append(
            {
                "raw_start": raw_start,
                "raw_end": raw_end,
                "padded_start": padded_start,
                "padded_end": padded_end,
                "max_score": float(ev["peak_confidence"]),
                "mean_score": float(ev["avg_confidence"]),
                "n_windows": n_windows,
            }
        )

    merged: list[dict[str, Any]] = []
    for entry in working:
        if merged and entry["padded_start"] <= merged[-1]["padded_end"]:
            cur = merged[-1]
            total_n = cur["n_windows"] + entry["n_windows"]
            if total_n > 0:
                weighted_mean = (
                    cur["mean_score"] * cur["n_windows"]
                    + entry["mean_score"] * entry["n_windows"]
                ) / total_n
            else:
                weighted_mean = 0.0
            cur["raw_start"] = min(cur["raw_start"], entry["raw_start"])
            cur["raw_end"] = max(cur["raw_end"], entry["raw_end"])
            cur["padded_start"] = min(cur["padded_start"], entry["padded_start"])
            cur["padded_end"] = max(cur["padded_end"], entry["padded_end"])
            cur["max_score"] = max(cur["max_score"], entry["max_score"])
            cur["mean_score"] = weighted_mean
            cur["n_windows"] = total_n
        else:
            merged.append(dict(entry))

    min_duration = config.min_region_duration_sec
    regions: list[Region] = []
    for entry in merged:
        if (entry["raw_end"] - entry["raw_start"]) < min_duration:
            continue
        regions.append(
            Region(
                region_id=uuid.uuid4().hex,
                start_sec=entry["raw_start"],
                end_sec=entry["raw_end"],
                padded_start_sec=entry["padded_start"],
                padded_end_sec=entry["padded_end"],
                max_score=entry["max_score"],
                mean_score=entry["mean_score"],
                n_windows=entry["n_windows"],
            )
        )

    return regions
