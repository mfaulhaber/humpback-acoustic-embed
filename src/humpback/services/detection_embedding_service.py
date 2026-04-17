"""Service functions for detection re-embedding jobs.

Re-embedding jobs are keyed by ``(detection_job_id, model_version)``. Callers
request a job for a given pair; if a live row exists (``queued``, ``running``,
or ``complete``) it is returned as-is; a ``failed`` row is reset to ``queued``;
otherwise a new row is inserted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.detection_embedding_job import DetectionEmbeddingJob


async def get_reembedding_job(
    session: AsyncSession,
    detection_job_id: str,
    model_version: str,
) -> Optional[DetectionEmbeddingJob]:
    """Return the existing job row for ``(detection_job_id, model_version)`` if any."""
    result = await session.execute(
        select(DetectionEmbeddingJob).where(
            DetectionEmbeddingJob.detection_job_id == detection_job_id,
            DetectionEmbeddingJob.model_version == model_version,
        )
    )
    return result.scalar_one_or_none()


async def list_reembedding_jobs(
    session: AsyncSession,
    detection_job_ids: list[str],
    model_version: str,
) -> dict[str, DetectionEmbeddingJob]:
    """Return a mapping ``detection_job_id -> job`` for the given pairs."""
    if not detection_job_ids:
        return {}
    result = await session.execute(
        select(DetectionEmbeddingJob).where(
            DetectionEmbeddingJob.detection_job_id.in_(detection_job_ids),
            DetectionEmbeddingJob.model_version == model_version,
        )
    )
    return {j.detection_job_id: j for j in result.scalars().all()}


async def create_reembedding_job(
    session: AsyncSession,
    detection_job_id: str,
    model_version: str,
    *,
    mode: str = "full",
) -> DetectionEmbeddingJob:
    """Idempotently enqueue a re-embedding job for ``(detection_job_id, model_version)``.

    - Returns the existing row unchanged when it is ``queued``, ``running``, or
      ``complete``.
    - Resets a ``failed`` row to ``queued`` and clears ``error_message`` /
      progress fields.
    - Otherwise creates and persists a new row with status ``queued``.
    """
    existing = await get_reembedding_job(session, detection_job_id, model_version)
    if existing is not None:
        if existing.status in ("queued", "running", "complete"):
            return existing
        # failed or other terminal-but-retryable state → reset.
        existing.status = "queued"
        existing.error_message = None
        existing.mode = mode
        existing.progress_current = None
        existing.progress_total = None
        existing.rows_processed = 0
        existing.rows_total = None
        existing.result_summary = None
        existing.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(existing)
        return existing

    job = DetectionEmbeddingJob(
        detection_job_id=detection_job_id,
        model_version=model_version,
        mode=mode,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job
