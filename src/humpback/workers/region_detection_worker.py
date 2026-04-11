"""Pass 1 worker shell — Phase 0 stub.

Claims a queued ``RegionDetectionJob`` and immediately marks it
``failed`` with a ``NotImplementedError`` message. Pass 1 replaces this
body with the real region-detection logic: dense Perch inference +
hysteresis + padded region emission. The claim/dispatch pattern is the
same as every other worker in the project (ADR-009).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.call_parsing import RegionDetectionJob
from humpback.workers.queue import claim_region_detection_job


async def run_region_detection_job(
    session: AsyncSession,
    job: RegionDetectionJob,
    _settings: Settings,
) -> None:
    """Phase 0 stub: mark the claimed job as failed."""
    job.status = "failed"
    job.error_message = (
        "NotImplementedError: Pass 1 (region detection) not yet implemented in Phase 0"
    )
    job.updated_at = datetime.now(timezone.utc)
    job.completed_at = datetime.now(timezone.utc)
    await session.commit()


async def run_one_iteration(
    session: AsyncSession, settings: Settings
) -> RegionDetectionJob | None:
    """Claim and process at most one region detection job. Returns it or None."""
    job = await claim_region_detection_job(session)
    if job is None:
        return None
    await run_region_detection_job(session, job, settings)
    return job
