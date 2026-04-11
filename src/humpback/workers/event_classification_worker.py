"""Pass 3 worker shell — Phase 0 stub.

Claims a queued ``EventClassificationJob`` and immediately marks it
``failed``. Pass 3 replaces this body with per-event multi-label
classification using a ``vocalization_models`` row whose
``model_family`` is ``pytorch_event_cnn``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.call_parsing import EventClassificationJob
from humpback.workers.queue import claim_event_classification_job


async def run_event_classification_job(
    session: AsyncSession,
    job: EventClassificationJob,
    _settings: Settings,
) -> None:
    """Phase 0 stub: mark the claimed job as failed."""
    job.status = "failed"
    job.error_message = (
        "NotImplementedError: Pass 3 (event classification) not yet implemented in "
        "Phase 0"
    )
    job.updated_at = datetime.now(timezone.utc)
    job.completed_at = datetime.now(timezone.utc)
    await session.commit()


async def run_one_iteration(
    session: AsyncSession, settings: Settings
) -> EventClassificationJob | None:
    """Claim and process at most one event classification job. Returns it or None."""
    job = await claim_event_classification_job(session)
    if job is None:
        return None
    await run_event_classification_job(session, job, settings)
    return job
