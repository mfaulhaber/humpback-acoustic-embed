"""Shared helper to seed an EventClassificationJob for Sequence Models tests.

After spec ``2026-05-04-sequence-models-classify-label-source-design.md``,
HMM and Masked Transformer interpretation generation requires a bound
``event_classification_job_id`` on the job row. Most worker / integration
tests don't care about specific Classify content; they just need a
completed Classify job whose ``typed_events.parquet`` exists (even if
empty) so the loader bridges to absolute UTC and the per-event type set
resolves to background.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.storage import (
    classification_job_dir,
    segmentation_job_dir,
    write_events,
    write_typed_events,
)
from humpback.call_parsing.types import Event
from humpback.models.call_parsing import EventClassificationJob


async def seed_classify_for_segmentation(
    session: AsyncSession,
    storage_root: Path,
    *,
    event_segmentation_job_id: str,
    events: list[Event] | None = None,
) -> str:
    """Create a completed EventClassificationJob with empty parquet artifacts.

    Also ensures the upstream segmentation job has an ``events.parquet``
    on disk (writing an empty one if missing) so ``load_effective_events``
    doesn't ValueError. Returns the new Classify job id; the caller sets
    ``event_classification_job_id`` on whichever HMM/MT row needs it.
    """
    seg_dir = segmentation_job_dir(storage_root, event_segmentation_job_id)
    seg_dir.mkdir(parents=True, exist_ok=True)
    events_path = seg_dir / "events.parquet"
    if events is not None or not events_path.exists():
        write_events(events_path, events or [])

    cls_job = EventClassificationJob(
        status="complete",
        event_segmentation_job_id=event_segmentation_job_id,
    )
    session.add(cls_job)
    await session.commit()
    await session.refresh(cls_job)

    cls_dir = classification_job_dir(storage_root, cls_job.id)
    cls_dir.mkdir(parents=True, exist_ok=True)
    write_typed_events(cls_dir / "typed_events.parquet", [])

    return cls_job.id
