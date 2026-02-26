"""SQL-backed job queue with claim semantics."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.clustering import ClusteringJob
from humpback.models.processing import EmbeddingSet, JobStatus, ProcessingJob

logger = logging.getLogger(__name__)

STALE_JOB_TIMEOUT = timedelta(minutes=10)


async def recover_stale_jobs(session: AsyncSession) -> int:
    """Reset jobs stuck in 'running' past the stale timeout back to 'queued'."""
    cutoff = datetime.now(timezone.utc) - STALE_JOB_TIMEOUT
    result = await session.execute(
        update(ProcessingJob)
        .where(
            ProcessingJob.status == JobStatus.running.value,
            ProcessingJob.updated_at < cutoff,
        )
        .values(
            status=JobStatus.queued.value,
            updated_at=datetime.now(timezone.utc),
        )
    )
    count = result.rowcount
    if count:
        logger.warning(f"Recovered {count} stale processing job(s)")

    result2 = await session.execute(
        update(ClusteringJob)
        .where(
            ClusteringJob.status == "running",
            ClusteringJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count2 = result2.rowcount
    if count2:
        logger.warning(f"Recovered {count2} stale clustering job(s)")

    if count or count2:
        await session.commit()
    return count + count2


async def claim_processing_job(session: AsyncSession) -> Optional[ProcessingJob]:
    """Claim a queued processing job atomically.

    Skips jobs whose encoding_signature already has a running job
    (prevents concurrent processing of same config).
    """
    # Find encoding_signatures that are currently running
    running_sigs = (
        select(ProcessingJob.encoding_signature)
        .where(ProcessingJob.status == JobStatus.running.value)
        .scalar_subquery()
    )

    # Find a queued job not blocked by a running job with same signature
    result = await session.execute(
        select(ProcessingJob)
        .where(
            ProcessingJob.status == JobStatus.queued.value,
            ~ProcessingJob.encoding_signature.in_(running_sigs),
        )
        .order_by(ProcessingJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None

    job.status = JobStatus.running.value
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return job


async def complete_processing_job(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.id == job_id)
        .values(
            status=JobStatus.complete.value,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()


async def fail_processing_job(
    session: AsyncSession, job_id: str, error: str
) -> None:
    await session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.id == job_id)
        .values(
            status=JobStatus.failed.value,
            error_message=error,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()


async def claim_clustering_job(session: AsyncSession) -> Optional[ClusteringJob]:
    result = await session.execute(
        select(ClusteringJob)
        .where(ClusteringJob.status == "queued")
        .order_by(ClusteringJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None

    job.status = "running"
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return job


async def complete_clustering_job(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(ClusteringJob)
        .where(ClusteringJob.id == job_id)
        .values(status="complete", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def fail_clustering_job(
    session: AsyncSession, job_id: str, error: str
) -> None:
    await session.execute(
        update(ClusteringJob)
        .where(ClusteringJob.id == job_id)
        .values(
            status="failed",
            error_message=error,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()
