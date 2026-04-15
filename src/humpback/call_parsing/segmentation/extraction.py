"""Shared extraction logic for segmentation training samples.

Converts human boundary corrections into training-ready samples.  Used by
both the ``from-corrections`` dataset-creation endpoint and the feedback
training worker.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.storage import (
    read_events,
    read_regions,
    region_job_dir,
    segmentation_job_dir,
)
from humpback.call_parsing.types import Event, Region
from humpback.models.call_parsing import (
    EventSegmentationJob,
    RegionDetectionJob,
)
from humpback.models.feedback_training import EventBoundaryCorrection

logger = logging.getLogger(__name__)

# Maximum crop duration in seconds.  Regions longer than this are
# subdivided into overlapping crops so the GRU sees manageable sequence
# lengths and the effective training set grows.
MAX_CROP_SEC: float = 30.0
CROP_HOP_SEC: float = 15.0


@dataclass
class CorrectedSample:
    """One audio crop with corrected event boundaries."""

    hydrophone_id: str
    start_timestamp: float
    end_timestamp: float
    crop_start_sec: float
    crop_end_sec: float
    events_json: str


def apply_corrections(
    original_events: list[Event],
    corrections: list[EventBoundaryCorrection],
) -> list[dict[str, float]]:
    """Apply boundary corrections to a region's events.

    Returns corrected event dicts with ``start_sec`` and ``end_sec`` keys.
    """
    events_by_id: dict[str, dict[str, float]] = {
        e.event_id: {"start_sec": e.start_sec, "end_sec": e.end_sec}
        for e in original_events
    }

    for c in corrections:
        if c.correction_type == "delete":
            events_by_id.pop(c.event_id, None)
        elif c.correction_type == "adjust":
            if (
                c.event_id in events_by_id
                and c.start_sec is not None
                and c.end_sec is not None
            ):
                events_by_id[c.event_id] = {
                    "start_sec": c.start_sec,
                    "end_sec": c.end_sec,
                }
        elif c.correction_type == "add":
            if c.start_sec is not None and c.end_sec is not None:
                events_by_id[c.event_id] = {
                    "start_sec": c.start_sec,
                    "end_sec": c.end_sec,
                }

    return list(events_by_id.values())


async def load_corrected_events(
    session: AsyncSession,
    segmentation_job_id: str,
    storage_root: Path,
) -> list[Event]:
    """Load events for a segmentation job with boundary corrections applied.

    Reads ``events.parquet``, queries ``event_boundary_corrections`` for the
    job, merges via :func:`apply_corrections`, and returns full ``Event``
    objects.  When no corrections exist the original events are returned
    unchanged.
    """
    seg_dir = segmentation_job_dir(storage_root, segmentation_job_id)
    events_path = seg_dir / "events.parquet"
    if not events_path.exists():
        raise ValueError(
            f"events.parquet not found for segmentation job {segmentation_job_id}"
        )

    original_events = read_events(events_path)

    corr_result = await session.execute(
        select(EventBoundaryCorrection).where(
            EventBoundaryCorrection.event_segmentation_job_id == segmentation_job_id
        )
    )
    corrections = list(corr_result.scalars().all())

    if not corrections:
        return original_events

    # Build lookup for original event metadata (region_id, confidence)
    originals_by_id: dict[str, Event] = {e.event_id: e for e in original_events}

    # apply_corrections returns dicts keyed by event_id
    events_by_id: dict[str, dict[str, float]] = {
        e.event_id: {"start_sec": e.start_sec, "end_sec": e.end_sec}
        for e in original_events
    }
    for c in corrections:
        if c.correction_type == "delete":
            events_by_id.pop(c.event_id, None)
        elif c.correction_type == "adjust":
            if (
                c.event_id in events_by_id
                and c.start_sec is not None
                and c.end_sec is not None
            ):
                events_by_id[c.event_id] = {
                    "start_sec": c.start_sec,
                    "end_sec": c.end_sec,
                }
        elif c.correction_type == "add":
            if c.start_sec is not None and c.end_sec is not None:
                events_by_id[c.event_id] = {
                    "start_sec": c.start_sec,
                    "end_sec": c.end_sec,
                }

    # Rebuild Event objects — preserve original metadata where available,
    # synthesize for added events.
    corr_region_lookup: dict[str, str] = {c.event_id: c.region_id for c in corrections}
    result: list[Event] = []
    for eid, bounds in events_by_id.items():
        start = bounds["start_sec"]
        end = bounds["end_sec"]
        orig = originals_by_id.get(eid)
        result.append(
            Event(
                event_id=eid,
                region_id=orig.region_id if orig else corr_region_lookup.get(eid, ""),
                start_sec=start,
                end_sec=end,
                center_sec=(start + end) / 2.0,
                segmentation_confidence=orig.segmentation_confidence if orig else 0.0,
            )
        )

    return result


def subdivide_region(
    crop_start: float,
    crop_end: float,
    corrected_events: list[dict[str, float]],
    hydrophone_id: str,
    start_timestamp: float,
    end_timestamp: float,
    max_crop_sec: float = MAX_CROP_SEC,
    crop_hop_sec: float = CROP_HOP_SEC,
) -> list[CorrectedSample]:
    """Split a long region into shorter crops, each with its own event subset."""
    region_dur = crop_end - crop_start
    if region_dur <= max_crop_sec:
        return [
            CorrectedSample(
                hydrophone_id=hydrophone_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                crop_start_sec=crop_start,
                crop_end_sec=crop_end,
                events_json=json.dumps(corrected_events),
            )
        ]

    samples: list[CorrectedSample] = []
    window_start = crop_start
    while window_start < crop_end:
        window_end = min(window_start + max_crop_sec, crop_end)
        if crop_end - window_start < max_crop_sec * 0.5:
            window_start = max(crop_start, crop_end - max_crop_sec)
            window_end = crop_end

        window_events = [
            e
            for e in corrected_events
            if e["end_sec"] > window_start and e["start_sec"] < window_end
        ]

        samples.append(
            CorrectedSample(
                hydrophone_id=hydrophone_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                crop_start_sec=window_start,
                crop_end_sec=window_end,
                events_json=json.dumps(window_events),
            )
        )

        if window_end >= crop_end:
            break
        window_start += crop_hop_sec

    return samples


async def collect_corrected_samples(
    session: AsyncSession,
    segmentation_job_id: str,
    storage_root: Path,
) -> list[CorrectedSample]:
    """Collect training samples from corrected regions of one segmentation job.

    Only regions with at least one boundary correction are included.
    Returns an empty list if no corrections exist.
    """
    seg_job = await session.get(EventSegmentationJob, segmentation_job_id)
    if seg_job is None:
        raise ValueError(f"Segmentation job {segmentation_job_id} not found")

    upstream = await session.get(RegionDetectionJob, seg_job.region_detection_job_id)
    if upstream is None:
        raise ValueError(
            f"Upstream region detection job {seg_job.region_detection_job_id} not found"
        )
    if not upstream.hydrophone_id:
        raise ValueError(
            f"Region detection job {upstream.id} is not hydrophone-sourced"
        )

    hydro_id = upstream.hydrophone_id
    job_start_ts = upstream.start_timestamp or 0.0
    job_end_ts = upstream.end_timestamp or 0.0

    seg_dir = segmentation_job_dir(storage_root, segmentation_job_id)
    events_path = seg_dir / "events.parquet"
    regions_path = region_job_dir(storage_root, upstream.id) / "regions.parquet"

    if not events_path.exists():
        logger.warning(
            "events.parquet missing for segmentation job %s", segmentation_job_id
        )
        return []
    if not regions_path.exists():
        logger.warning("regions.parquet missing for region job %s", upstream.id)
        return []

    all_events = read_events(events_path)
    all_regions = read_regions(regions_path)
    regions_by_id: dict[str, Region] = {r.region_id: r for r in all_regions}

    events_by_region: dict[str, list[Event]] = defaultdict(list)
    for e in all_events:
        events_by_region[e.region_id].append(e)

    corr_result = await session.execute(
        select(EventBoundaryCorrection).where(
            EventBoundaryCorrection.event_segmentation_job_id == segmentation_job_id
        )
    )
    corrections = list(corr_result.scalars().all())
    corrections_by_region: dict[str, list[EventBoundaryCorrection]] = defaultdict(list)
    for c in corrections:
        corrections_by_region[c.region_id].append(c)

    samples: list[CorrectedSample] = []
    for region_id, region in regions_by_id.items():
        region_corrections = corrections_by_region.get(region_id, [])
        if not region_corrections:
            continue
        region_events = events_by_region.get(region_id, [])
        corrected = apply_corrections(region_events, region_corrections)

        samples.extend(
            subdivide_region(
                crop_start=region.padded_start_sec,
                crop_end=region.padded_end_sec,
                corrected_events=corrected,
                hydrophone_id=hydro_id,
                start_timestamp=job_start_ts,
                end_timestamp=job_end_ts,
            )
        )

    return samples
