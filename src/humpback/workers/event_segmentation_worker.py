"""Pass 2 worker shell — Phase 0 stub.

Claims a queued ``EventSegmentationJob`` and immediately marks it
``failed``. Pass 2 replaces this body with the real framewise
segmentation inference that decodes per-event onset/offset bounds
inside each region.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.call_parsing import EventSegmentationJob
from humpback.workers.queue import claim_event_segmentation_job


async def run_event_segmentation_job(
    session: AsyncSession,
    job: EventSegmentationJob,
    _settings: Settings,
) -> None:
    """Phase 0 stub: mark the claimed job as failed."""
    job.status = "failed"
    job.error_message = (
        "NotImplementedError: Pass 2 (event segmentation) not yet implemented in "
        "Phase 0"
    )
    job.updated_at = datetime.now(timezone.utc)
    job.completed_at = datetime.now(timezone.utc)
    await session.commit()


async def run_one_iteration(
    session: AsyncSession, settings: Settings
) -> EventSegmentationJob | None:
    """Claim and process at most one event segmentation job. Returns it or None."""
    job = await claim_event_segmentation_job(session)
    if job is None:
        return None
    await run_event_segmentation_job(session, job, settings)
    return job
