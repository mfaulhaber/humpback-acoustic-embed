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
    EventBoundaryCorrection,
    EventSegmentationJob,
    RegionDetectionJob,
)

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
    Matches events by ``(original_start_sec, original_end_sec)`` time-range
    identity rather than event_id.
    """
    events: list[dict[str, float]] = [
        {"start_sec": e.start_sec, "end_sec": e.end_sec} for e in original_events
    ]

    for c in corrections:
        if c.correction_type == "delete":
            events = [
                e
                for e in events
                if not (
                    e["start_sec"] == c.original_start_sec
                    and e["end_sec"] == c.original_end_sec
                )
            ]
        elif c.correction_type == "adjust":
            for e in events:
                if (
                    e["start_sec"] == c.original_start_sec
                    and e["end_sec"] == c.original_end_sec
                    and c.corrected_start_sec is not None
                    and c.corrected_end_sec is not None
                ):
                    e["start_sec"] = c.corrected_start_sec
                    e["end_sec"] = c.corrected_end_sec
                    break
        elif c.correction_type == "add":
            if c.corrected_start_sec is not None and c.corrected_end_sec is not None:
                events.append(
                    {
                        "start_sec": c.corrected_start_sec,
                        "end_sec": c.corrected_end_sec,
                    }
                )

    return events


def _find_source_event(
    events_by_id: dict[str, Event],
    events: list[Event],
    correction: EventBoundaryCorrection,
) -> Event | None:
    if correction.source_event_id:
        return events_by_id.get(correction.source_event_id)
    if correction.original_start_sec is None or correction.original_end_sec is None:
        return None
    for event in events:
        if (
            event.region_id == correction.region_id
            and event.start_sec == correction.original_start_sec
            and event.end_sec == correction.original_end_sec
        ):
            return event
    return None


def build_effective_events(
    original_events: list[Event],
    corrections: list[EventBoundaryCorrection],
) -> list[Event]:
    """Overlay scoped boundary corrections onto raw event rows."""
    if not corrections:
        return sorted(
            original_events, key=lambda e: (e.start_sec, e.end_sec, e.event_id)
        )

    events_by_id: dict[str, Event] = {
        event.event_id: event for event in original_events
    }
    effective_by_id: dict[str, Event] = dict(events_by_id)

    for correction in corrections:
        if correction.correction_type == "delete":
            source = _find_source_event(events_by_id, original_events, correction)
            if source is not None:
                effective_by_id.pop(source.event_id, None)
        elif correction.correction_type == "adjust":
            source = _find_source_event(events_by_id, original_events, correction)
            if (
                source is not None
                and correction.corrected_start_sec is not None
                and correction.corrected_end_sec is not None
            ):
                effective_by_id[source.event_id] = Event(
                    event_id=source.event_id,
                    region_id=source.region_id,
                    start_sec=correction.corrected_start_sec,
                    end_sec=correction.corrected_end_sec,
                    center_sec=(
                        correction.corrected_start_sec + correction.corrected_end_sec
                    )
                    / 2.0,
                    segmentation_confidence=source.segmentation_confidence,
                )
        elif (
            correction.correction_type == "add"
            and correction.corrected_start_sec is not None
            and correction.corrected_end_sec is not None
        ):
            event_id = f"added-{correction.id}"
            effective_by_id[event_id] = Event(
                event_id=event_id,
                region_id=correction.region_id,
                start_sec=correction.corrected_start_sec,
                end_sec=correction.corrected_end_sec,
                center_sec=(
                    correction.corrected_start_sec + correction.corrected_end_sec
                )
                / 2.0,
                segmentation_confidence=0.0,
            )

    return sorted(
        effective_by_id.values(), key=lambda e: (e.start_sec, e.end_sec, e.event_id)
    )


async def load_effective_events(
    session: AsyncSession,
    *,
    event_segmentation_job_id: str,
    storage_root: Path,
    include_boundary_corrections: bool = True,
) -> list[Event]:
    """Load canonical effective events for one segmentation job.

    Raw ``events.parquet`` rows are immutable. Boundary corrections are
    overlaid at read time, scoped to the selected segmentation job.
    Adjusted events keep their source ``event_id``; added events receive
    a stable synthetic ID based on the correction row.
    """
    seg_job = await session.get(EventSegmentationJob, event_segmentation_job_id)
    if seg_job is None:
        raise ValueError(f"EventSegmentationJob {event_segmentation_job_id} not found")

    seg_dir = segmentation_job_dir(storage_root, event_segmentation_job_id)
    events_path = seg_dir / "events.parquet"
    if not events_path.exists():
        raise ValueError(
            f"events.parquet not found for segmentation job {event_segmentation_job_id}"
        )

    original_events = read_events(events_path)
    if not include_boundary_corrections:
        return sorted(
            original_events, key=lambda e: (e.start_sec, e.end_sec, e.event_id)
        )

    corr_result = await session.execute(
        select(EventBoundaryCorrection)
        .where(
            EventBoundaryCorrection.event_segmentation_job_id
            == event_segmentation_job_id
        )
        .order_by(EventBoundaryCorrection.created_at, EventBoundaryCorrection.id)
    )
    corrections = list(corr_result.scalars().all())
    return build_effective_events(original_events, corrections)


async def load_corrected_events(
    session: AsyncSession,
    region_detection_job_id: str,
    segmentation_job_id: str,
    storage_root: Path,
) -> list[Event]:
    """Compatibility wrapper for legacy region-scoped correction overlays.

    New code should call :func:`load_effective_events` with
    ``event_segmentation_job_id``. If scoped correction rows exist for the
    segmentation job, this wrapper delegates to the canonical loader.
    Otherwise it preserves the old region-scoped behavior for older callers.
    """
    scoped_result = await session.execute(
        select(EventBoundaryCorrection.id)
        .where(EventBoundaryCorrection.event_segmentation_job_id == segmentation_job_id)
        .limit(1)
    )
    if scoped_result.scalar_one_or_none() is not None:
        return await load_effective_events(
            session,
            event_segmentation_job_id=segmentation_job_id,
            storage_root=storage_root,
        )

    seg_dir = segmentation_job_dir(storage_root, segmentation_job_id)
    events_path = seg_dir / "events.parquet"
    if not events_path.exists():
        raise ValueError(
            f"events.parquet not found for segmentation job {segmentation_job_id}"
        )

    original_events = read_events(events_path)

    corr_result = await session.execute(
        select(EventBoundaryCorrection).where(
            EventBoundaryCorrection.region_detection_job_id == region_detection_job_id,
            EventBoundaryCorrection.event_segmentation_job_id.is_(None),
        )
    )
    corrections = list(corr_result.scalars().all())

    if not corrections:
        return original_events

    # Group corrections and events by region for scoped application
    corrections_by_region: dict[str, list[EventBoundaryCorrection]] = defaultdict(list)
    for c in corrections:
        corrections_by_region[c.region_id].append(c)

    events_by_region: dict[str, list[Event]] = defaultdict(list)
    uncorrected_events: list[Event] = []
    for e in original_events:
        if e.region_id in corrections_by_region:
            events_by_region[e.region_id].append(e)
        else:
            uncorrected_events.append(e)

    originals_by_bounds: dict[tuple[str, float, float], Event] = {
        (e.region_id, e.start_sec, e.end_sec): e for e in original_events
    }

    result: list[Event] = list(uncorrected_events)

    for region_id, region_corrections in corrections_by_region.items():
        region_events = events_by_region.get(region_id, [])
        corrected = apply_corrections(region_events, region_corrections)

        for bounds in corrected:
            start = bounds["start_sec"]
            end = bounds["end_sec"]
            orig = originals_by_bounds.get((region_id, start, end))
            result.append(
                Event(
                    event_id=orig.event_id if orig else f"added-{start:.3f}-{end:.3f}",
                    region_id=region_id,
                    start_sec=start,
                    end_sec=end,
                    center_sec=(start + end) / 2.0,
                    segmentation_confidence=(
                        orig.segmentation_confidence if orig else 0.0
                    ),
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
    region_detection_job_id: str,
    segmentation_job_id: str,
    storage_root: Path,
) -> list[CorrectedSample]:
    """Collect training samples from corrected regions.

    Queries corrections by *region_detection_job_id*, reads events from
    the segmentation parquet via *segmentation_job_id*.  Only regions with
    at least one boundary correction are included.  Returns an empty list
    if no corrections exist.
    """
    upstream = await session.get(RegionDetectionJob, region_detection_job_id)
    if upstream is None:
        raise ValueError(f"Region detection job {region_detection_job_id} not found")
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
