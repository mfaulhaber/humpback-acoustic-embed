"""Service layer for HMM sequence jobs.

Validates that the source ``ContinuousEmbeddingJob`` is complete before
creating an ``HMMSequenceJob``. No idempotency key — HMM training is
stochastic and comparing configs requires multiple runs.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob, HMMSequenceJob
from humpback.schemas.sequence_models import HMMSequenceJobCreate


class CancelTerminalJobError(Exception):
    """Raised when caller attempts to cancel a job in a terminal state."""


async def create_hmm_sequence_job(
    session: AsyncSession,
    payload: HMMSequenceJobCreate,
) -> HMMSequenceJob:
    source = await session.get(
        ContinuousEmbeddingJob, payload.continuous_embedding_job_id
    )
    if source is None:
        raise ValueError(
            f"continuous_embedding_job not found: {payload.continuous_embedding_job_id}"
        )
    if source.status != JobStatus.complete.value:
        raise ValueError(
            "HMM sequence job requires a completed continuous_embedding_job "
            f"(current status: {source.status!r})"
        )

    job = HMMSequenceJob(
        continuous_embedding_job_id=payload.continuous_embedding_job_id,
        n_states=payload.n_states,
        pca_dims=payload.pca_dims,
        pca_whiten=payload.pca_whiten,
        l2_normalize=payload.l2_normalize,
        covariance_type=payload.covariance_type,
        n_iter=payload.n_iter,
        random_seed=payload.random_seed,
        min_sequence_length_frames=payload.min_sequence_length_frames,
        tol=payload.tol,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def list_hmm_sequence_jobs(
    session: AsyncSession,
    *,
    status: Optional[str] = None,
    continuous_embedding_job_id: Optional[str] = None,
) -> list[HMMSequenceJob]:
    stmt = select(HMMSequenceJob).order_by(HMMSequenceJob.created_at.desc())
    if status is not None:
        stmt = stmt.where(HMMSequenceJob.status == status)
    if continuous_embedding_job_id is not None:
        stmt = stmt.where(
            HMMSequenceJob.continuous_embedding_job_id == continuous_embedding_job_id
        )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_hmm_sequence_job(
    session: AsyncSession, job_id: str
) -> Optional[HMMSequenceJob]:
    return await session.get(HMMSequenceJob, job_id)


async def cancel_hmm_sequence_job(
    session: AsyncSession, job_id: str
) -> Optional[HMMSequenceJob]:
    """Flip ``queued`` or ``running`` to ``canceled``.

    Returns the (possibly updated) job, ``None`` if not found,
    or raises ``CancelTerminalJobError`` for terminal states.
    """
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        return None
    if job.status in (JobStatus.queued.value, JobStatus.running.value):
        job.status = JobStatus.canceled.value
        await session.commit()
        return job
    raise CancelTerminalJobError(
        f"hmm_sequence_job {job_id} is in terminal state {job.status!r}"
    )
