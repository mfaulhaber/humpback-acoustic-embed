"""Service layer for motif extraction jobs."""

from __future__ import annotations

import shutil
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import (
    ContinuousEmbeddingJob,
    HMMSequenceJob,
    MotifExtractionJob,
)
from humpback.sequence_models.motifs import MotifExtractionConfig, config_signature
from humpback.services.continuous_embedding_service import source_kind_for
from humpback.storage import motif_extraction_dir


class CancelTerminalJobError(Exception):
    """Raised when caller attempts to cancel a terminal motif job."""


def config_from_payload(payload: Any) -> MotifExtractionConfig:
    config = MotifExtractionConfig(
        min_ngram=int(getattr(payload, "min_ngram", 2)),
        max_ngram=int(getattr(payload, "max_ngram", 8)),
        minimum_occurrences=int(getattr(payload, "minimum_occurrences", 5)),
        minimum_event_sources=int(getattr(payload, "minimum_event_sources", 2)),
        frequency_weight=float(getattr(payload, "frequency_weight", 0.40)),
        event_source_weight=float(getattr(payload, "event_source_weight", 0.30)),
        event_core_weight=float(getattr(payload, "event_core_weight", 0.20)),
        low_background_weight=float(getattr(payload, "low_background_weight", 0.10)),
        call_probability_weight=getattr(payload, "call_probability_weight", None),
    )
    config.validate()
    return config


async def create_motif_extraction_job(
    session: AsyncSession,
    payload: Any,
) -> tuple[MotifExtractionJob, bool]:
    hmm_job_id = str(getattr(payload, "hmm_sequence_job_id"))
    hmm_job = await session.get(HMMSequenceJob, hmm_job_id)
    if hmm_job is None:
        raise ValueError(f"hmm_sequence_job not found: {hmm_job_id}")
    if hmm_job.status != JobStatus.complete.value:
        raise ValueError(
            "motif extraction requires a completed hmm_sequence_job "
            f"(current status: {hmm_job.status!r})"
        )

    cej = await session.get(ContinuousEmbeddingJob, hmm_job.continuous_embedding_job_id)
    if cej is None:
        raise ValueError(
            f"continuous_embedding_job not found: {hmm_job.continuous_embedding_job_id}"
        )

    config = config_from_payload(payload)
    signature = config_signature(hmm_job_id, config)
    stmt = (
        select(MotifExtractionJob)
        .where(MotifExtractionJob.config_signature == signature)
        .where(
            MotifExtractionJob.status.in_(
                [
                    JobStatus.queued.value,
                    JobStatus.running.value,
                    JobStatus.complete.value,
                ]
            )
        )
        .order_by(MotifExtractionJob.created_at.desc())
        .limit(1)
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing is not None:
        return existing, False

    job = MotifExtractionJob(
        hmm_sequence_job_id=hmm_job_id,
        source_kind=source_kind_for(cej.model_version),
        min_ngram=config.min_ngram,
        max_ngram=config.max_ngram,
        minimum_occurrences=config.minimum_occurrences,
        minimum_event_sources=config.minimum_event_sources,
        frequency_weight=config.frequency_weight,
        event_source_weight=config.event_source_weight,
        event_core_weight=config.event_core_weight,
        low_background_weight=config.low_background_weight,
        call_probability_weight=config.call_probability_weight,
        config_signature=signature,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job, True


async def list_motif_extraction_jobs(
    session: AsyncSession,
    *,
    status: Optional[str] = None,
    hmm_sequence_job_id: Optional[str] = None,
) -> list[MotifExtractionJob]:
    stmt = select(MotifExtractionJob).order_by(MotifExtractionJob.created_at.desc())
    if status is not None:
        stmt = stmt.where(MotifExtractionJob.status == status)
    if hmm_sequence_job_id is not None:
        stmt = stmt.where(MotifExtractionJob.hmm_sequence_job_id == hmm_sequence_job_id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_motif_extraction_job(
    session: AsyncSession, job_id: str
) -> Optional[MotifExtractionJob]:
    return await session.get(MotifExtractionJob, job_id)


async def cancel_motif_extraction_job(
    session: AsyncSession, job_id: str
) -> Optional[MotifExtractionJob]:
    job = await get_motif_extraction_job(session, job_id)
    if job is None:
        return None
    if job.status in (JobStatus.queued.value, JobStatus.running.value):
        job.status = JobStatus.canceled.value
        await session.commit()
        return job
    raise CancelTerminalJobError(
        f"motif_extraction_job {job_id} is in terminal state {job.status!r}"
    )


async def delete_motif_extraction_job(
    session: AsyncSession,
    job_id: str,
    settings: Settings,
) -> bool:
    job = await session.get(MotifExtractionJob, job_id)
    if job is None:
        return False
    artifact_dir = motif_extraction_dir(settings.storage_root, job_id)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    await session.delete(job)
    await session.commit()
    return True
