"""Tests for ``load_effective_event_labels`` against real DB + parquet fixtures.

Exercises the async loader path that bridges Pass 3 ``typed_events.parquet``
through ``load_effective_events`` (boundary corrections, ADR-054) and
``VocalizationCorrection`` (type-level corrections, region-scoped) into
the per-event type set consumed by HMM/MT label distribution.
"""

from __future__ import annotations

import pytest

from humpback.call_parsing.storage import (
    classification_job_dir,
    segmentation_job_dir,
    write_events,
    write_typed_events,
)
from humpback.call_parsing.types import Event, TypedEvent
from humpback.models.call_parsing import (
    EventBoundaryCorrection,
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
    VocalizationCorrection,
)
from humpback.models.processing import JobStatus
from humpback.sequence_models.label_distribution import load_effective_event_labels

REGION_START_UTC = 1_000.0


async def _seed_classify_chain(session, storage_root) -> tuple[str, str, str]:
    """Create RegionDetectionJob → EventSegmentationJob → EventClassificationJob.

    Writes an empty ``events.parquet`` for the segmentation job; tests
    add their own events via ``write_events`` before invoking the loader.
    Returns the three job IDs.
    """
    rdj = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=REGION_START_UTC,
        end_timestamp=REGION_START_UTC + 300.0,
    )
    session.add(rdj)
    await session.flush()

    seg = EventSegmentationJob(
        status=JobStatus.complete.value,
        region_detection_job_id=rdj.id,
    )
    session.add(seg)
    await session.flush()

    cls = EventClassificationJob(
        status="complete",
        event_segmentation_job_id=seg.id,
    )
    session.add(cls)
    await session.commit()
    await session.refresh(cls)
    await session.refresh(seg)
    await session.refresh(rdj)

    seg_dir = segmentation_job_dir(storage_root, seg.id)
    seg_dir.mkdir(parents=True, exist_ok=True)

    return rdj.id, seg.id, cls.id


def _event(event_id: str, region_id: str, start: float, end: float) -> Event:
    return Event(
        event_id=event_id,
        region_id=region_id,
        start_sec=start,
        end_sec=end,
        center_sec=(start + end) / 2.0,
        segmentation_confidence=0.9,
    )


def _typed(event_id: str, type_name: str, score: float, above: bool) -> TypedEvent:
    return TypedEvent(
        event_id=event_id,
        start_sec=0.0,  # not used by loader
        end_sec=0.0,
        type_name=type_name,
        score=score,
        above_threshold=above,
    )


async def test_keeps_above_threshold_drops_below(session, tmp_storage):
    rdj_id, seg_id, cls_id = await _seed_classify_chain(session, tmp_storage)
    write_events(
        segmentation_job_dir(tmp_storage, seg_id) / "events.parquet",
        [_event("E1", "R1", 10.0, 12.0)],
    )
    cls_dir = classification_job_dir(tmp_storage, cls_id)
    cls_dir.mkdir(parents=True, exist_ok=True)
    write_typed_events(
        cls_dir / "typed_events.parquet",
        [
            _typed("E1", "moan", 0.95, above=True),
            _typed("E1", "whup", 0.40, above=False),
        ],
    )

    result = await load_effective_event_labels(
        session,
        event_classification_job_id=cls_id,
        storage_root=tmp_storage,
    )

    assert len(result) == 1
    assert result[0].event_id == "E1"
    assert result[0].types == frozenset({"moan"})
    assert result[0].confidences == {"moan": 0.95}
    # UTC bridging: start_sec 10.0 + REGION_START_UTC = 1010.0
    assert result[0].start_utc == REGION_START_UTC + 10.0
    assert result[0].end_utc == REGION_START_UTC + 12.0


async def test_user_added_type_overrides_threshold(session, tmp_storage):
    """VocalizationCorrection 'add' includes a below-threshold type."""
    rdj_id, seg_id, cls_id = await _seed_classify_chain(session, tmp_storage)
    write_events(
        segmentation_job_dir(tmp_storage, seg_id) / "events.parquet",
        [_event("E1", "R1", 10.0, 12.0)],
    )
    cls_dir = classification_job_dir(tmp_storage, cls_id)
    cls_dir.mkdir(parents=True, exist_ok=True)
    write_typed_events(
        cls_dir / "typed_events.parquet",
        [_typed("E1", "moan", 0.20, above=False)],
    )
    session.add(
        VocalizationCorrection(
            region_detection_job_id=rdj_id,
            start_sec=10.0,
            end_sec=12.0,
            type_name="moan",
            correction_type="add",
        )
    )
    await session.commit()

    result = await load_effective_event_labels(
        session,
        event_classification_job_id=cls_id,
        storage_root=tmp_storage,
    )

    assert result[0].types == frozenset({"moan"})


async def test_user_remove_drops_above_threshold(session, tmp_storage):
    """VocalizationCorrection 'remove' subtracts an above-threshold type."""
    rdj_id, seg_id, cls_id = await _seed_classify_chain(session, tmp_storage)
    write_events(
        segmentation_job_dir(tmp_storage, seg_id) / "events.parquet",
        [_event("E1", "R1", 10.0, 12.0)],
    )
    cls_dir = classification_job_dir(tmp_storage, cls_id)
    cls_dir.mkdir(parents=True, exist_ok=True)
    write_typed_events(
        cls_dir / "typed_events.parquet",
        [
            _typed("E1", "moan", 0.95, above=True),
            _typed("E1", "song", 0.85, above=True),
        ],
    )
    session.add(
        VocalizationCorrection(
            region_detection_job_id=rdj_id,
            start_sec=10.0,
            end_sec=12.0,
            type_name="moan",
            correction_type="remove",
        )
    )
    await session.commit()

    result = await load_effective_event_labels(
        session,
        event_classification_job_id=cls_id,
        storage_root=tmp_storage,
    )

    assert result[0].types == frozenset({"song"})
    assert "moan" not in result[0].confidences
    assert result[0].confidences == {"song": 0.85}


async def test_empty_types_after_corrections(session, tmp_storage):
    """All types removed → empty type set, but event is still returned."""
    rdj_id, seg_id, cls_id = await _seed_classify_chain(session, tmp_storage)
    write_events(
        segmentation_job_dir(tmp_storage, seg_id) / "events.parquet",
        [_event("E1", "R1", 10.0, 12.0)],
    )
    cls_dir = classification_job_dir(tmp_storage, cls_id)
    cls_dir.mkdir(parents=True, exist_ok=True)
    write_typed_events(
        cls_dir / "typed_events.parquet",
        [_typed("E1", "moan", 0.95, above=True)],
    )
    session.add(
        VocalizationCorrection(
            region_detection_job_id=rdj_id,
            start_sec=10.0,
            end_sec=12.0,
            type_name="moan",
            correction_type="remove",
        )
    )
    await session.commit()

    result = await load_effective_event_labels(
        session,
        event_classification_job_id=cls_id,
        storage_root=tmp_storage,
    )

    assert len(result) == 1
    assert result[0].types == frozenset()
    assert result[0].confidences == {}


async def test_boundary_correction_shifts_event_bounds(session, tmp_storage):
    """EventBoundaryCorrection (adjust) changes the effective UTC window."""
    rdj_id, seg_id, cls_id = await _seed_classify_chain(session, tmp_storage)
    write_events(
        segmentation_job_dir(tmp_storage, seg_id) / "events.parquet",
        [_event("E1", "R1", 10.0, 12.0)],
    )
    cls_dir = classification_job_dir(tmp_storage, cls_id)
    cls_dir.mkdir(parents=True, exist_ok=True)
    write_typed_events(
        cls_dir / "typed_events.parquet",
        [_typed("E1", "moan", 0.95, above=True)],
    )
    session.add(
        EventBoundaryCorrection(
            region_detection_job_id=rdj_id,
            event_segmentation_job_id=seg_id,
            region_id="R1",
            source_event_id="E1",
            correction_type="adjust",
            original_start_sec=10.0,
            original_end_sec=12.0,
            corrected_start_sec=15.0,
            corrected_end_sec=17.0,
        )
    )
    await session.commit()

    result = await load_effective_event_labels(
        session,
        event_classification_job_id=cls_id,
        storage_root=tmp_storage,
    )

    assert len(result) == 1
    assert result[0].event_id == "E1"
    assert result[0].start_utc == REGION_START_UTC + 15.0
    assert result[0].end_utc == REGION_START_UTC + 17.0
    assert result[0].types == frozenset({"moan"})


async def test_returns_events_sorted_by_start_utc(session, tmp_storage):
    rdj_id, seg_id, cls_id = await _seed_classify_chain(session, tmp_storage)
    write_events(
        segmentation_job_dir(tmp_storage, seg_id) / "events.parquet",
        [
            _event("E2", "R1", 50.0, 52.0),
            _event("E1", "R1", 10.0, 12.0),
            _event("E3", "R1", 100.0, 102.0),
        ],
    )
    cls_dir = classification_job_dir(tmp_storage, cls_id)
    cls_dir.mkdir(parents=True, exist_ok=True)
    write_typed_events(
        cls_dir / "typed_events.parquet",
        [
            _typed("E1", "moan", 0.9, above=True),
            _typed("E2", "song", 0.9, above=True),
            _typed("E3", "whup", 0.9, above=True),
        ],
    )

    result = await load_effective_event_labels(
        session,
        event_classification_job_id=cls_id,
        storage_root=tmp_storage,
    )

    assert [e.event_id for e in result] == ["E1", "E2", "E3"]


async def test_missing_classify_job_raises(session, tmp_storage):
    with pytest.raises(ValueError, match="EventClassificationJob"):
        await load_effective_event_labels(
            session,
            event_classification_job_id="does-not-exist",
            storage_root=tmp_storage,
        )
