"""Service layer for retained Sequence Models Event Encoder jobs."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.call_parsing import EventSegmentationJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob, EventEncoderJob
from humpback.schemas.sequence_models import EventEncoderJobCreate
from humpback.services.continuous_embedding_service import (
    SOURCE_KIND_REGION_CRNN,
    compute_effective_event_correction_revision,
    source_kind_for,
)
from humpback.storage import event_encoder_dir

_RESETTABLE_STATUSES = {JobStatus.failed.value, JobStatus.canceled.value}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _config_dict(payload: EventEncoderJobCreate) -> dict[str, Any]:
    return {
        "pooling": payload.pooling.model_dump(),
        "descriptor": payload.descriptor.model_dump(),
        "preprocessing": payload.preprocessing.model_dump(),
        "k_values": sorted(payload.k_values),
        "random_seed": payload.random_seed,
    }


def compute_event_encoder_signature(
    *,
    tokenizer_version: str,
    event_segmentation_job_id: str,
    event_source_mode: str,
    correction_revision: Optional[str],
    continuous_embedding_signature: str,
    continuous_embedding_provenance: dict[str, Any],
    pooling_config: dict[str, Any],
    descriptor_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    k_values: list[int],
    random_seed: int,
) -> str:
    """SHA-256 idempotency key for event-level tokenization jobs."""
    payload = {
        "tokenizer_version": tokenizer_version,
        "event_segmentation_job_id": event_segmentation_job_id,
        "event_source_mode": event_source_mode,
        "correction_revision": correction_revision,
        "continuous_embedding_signature": continuous_embedding_signature,
        "continuous_embedding_provenance": continuous_embedding_provenance,
        "pooling_config": pooling_config,
        "descriptor_config": descriptor_config,
        "preprocessing_config": preprocessing_config,
        "k_values": sorted(k_values),
        "random_seed": random_seed,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


async def create_event_encoder_job(
    session: AsyncSession,
    payload: EventEncoderJobCreate,
) -> tuple[EventEncoderJob, bool]:
    """Create or reuse an Event Encoder job."""
    seg_job = await session.get(EventSegmentationJob, payload.event_segmentation_job_id)
    if seg_job is None:
        raise ValueError(
            f"event_segmentation_job not found: {payload.event_segmentation_job_id}"
        )
    if seg_job.status != JobStatus.complete.value:
        raise ValueError("event encoder requires a completed event_segmentation_job")

    continuous = await session.get(
        ContinuousEmbeddingJob, payload.continuous_embedding_job_id
    )
    if continuous is None:
        raise ValueError(
            f"continuous_embedding_job not found: {payload.continuous_embedding_job_id}"
        )
    if continuous.status != JobStatus.complete.value:
        raise ValueError("event encoder requires a completed continuous_embedding_job")
    if source_kind_for(continuous.model_version) != SOURCE_KIND_REGION_CRNN:
        raise ValueError(
            "event encoder requires a region_crnn continuous embedding job"
        )
    if continuous.event_segmentation_job_id != payload.event_segmentation_job_id:
        raise ValueError(
            "continuous_embedding_job event_segmentation_job_id does not match "
            "the selected event_segmentation_job"
        )
    if not continuous.encoding_signature:
        raise ValueError("continuous_embedding_job is missing encoding_signature")

    correction_revision = None
    if payload.event_source_mode == "effective":
        correction_revision = await compute_effective_event_correction_revision(
            session, payload.event_segmentation_job_id
        )

    config = _config_dict(payload)
    pooling_config = config["pooling"]
    descriptor_config = config["descriptor"]
    preprocessing_config = config["preprocessing"]
    k_values = config["k_values"]
    continuous_provenance = {
        "model_version": continuous.model_version,
        "crnn_checkpoint_sha256": continuous.crnn_checkpoint_sha256,
        "chunk_size_seconds": continuous.chunk_size_seconds,
        "chunk_hop_seconds": continuous.chunk_hop_seconds,
        "projection_kind": continuous.projection_kind,
        "projection_dim": continuous.projection_dim,
    }
    signature = compute_event_encoder_signature(
        tokenizer_version=payload.tokenizer_version,
        event_segmentation_job_id=payload.event_segmentation_job_id,
        event_source_mode=payload.event_source_mode,
        correction_revision=correction_revision,
        continuous_embedding_signature=continuous.encoding_signature,
        continuous_embedding_provenance=continuous_provenance,
        pooling_config=pooling_config,
        descriptor_config=descriptor_config,
        preprocessing_config=preprocessing_config,
        k_values=k_values,
        random_seed=payload.random_seed,
    )

    reusable = await _find_reusable(session, signature)
    if reusable is not None:
        return reusable

    job = EventEncoderJob(
        event_segmentation_job_id=payload.event_segmentation_job_id,
        event_source_mode=payload.event_source_mode,
        continuous_embedding_job_id=payload.continuous_embedding_job_id,
        continuous_embedding_signature=continuous.encoding_signature,
        tokenizer_version=payload.tokenizer_version,
        pooling_config_json=_canonical_json(pooling_config),
        descriptor_config_json=_canonical_json(descriptor_config),
        preprocessing_config_json=_canonical_json(preprocessing_config),
        k_values_json=_canonical_json(k_values),
        random_seed=payload.random_seed,
        tokenization_signature=signature,
    )
    return await _persist_with_race_recovery(session, job, signature)


async def list_event_encoder_jobs(
    session: AsyncSession,
    *,
    status: Optional[str] = None,
) -> list[EventEncoderJob]:
    stmt = select(EventEncoderJob).order_by(EventEncoderJob.created_at.desc())
    if status is not None:
        stmt = stmt.where(EventEncoderJob.status == status)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_event_encoder_job(
    session: AsyncSession, job_id: str
) -> Optional[EventEncoderJob]:
    return await session.get(EventEncoderJob, job_id)


class CancelEventEncoderTerminalJobError(Exception):
    """Raised when caller attempts to cancel a terminal Event Encoder job."""


async def cancel_event_encoder_job(
    session: AsyncSession, job_id: str
) -> Optional[EventEncoderJob]:
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        return None
    if job.status in (JobStatus.queued.value, JobStatus.running.value):
        job.status = JobStatus.canceled.value
        await session.commit()
        return job
    raise CancelEventEncoderTerminalJobError(
        f"event_encoder_job {job_id} is in terminal state {job.status!r}"
    )


async def delete_event_encoder_job(
    session: AsyncSession, job_id: str, settings: Settings
) -> bool:
    job = await session.get(EventEncoderJob, job_id)
    if job is None:
        return False
    artifact_dir = event_encoder_dir(settings.storage_root, job_id)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    await session.delete(job)
    await session.commit()
    return True


async def _find_reusable(
    session: AsyncSession, signature: str
) -> Optional[tuple[EventEncoderJob, bool]]:
    existing = await _get_existing_jobs_by_signature(session, signature)
    reusable = _pick_active_or_complete_job(existing)
    if reusable is not None:
        return reusable, False

    resettable = _pick_resettable_job(existing)
    if resettable is not None:
        _reset_job_for_retry(resettable)
        await session.commit()
        return resettable, False
    return None


async def _persist_with_race_recovery(
    session: AsyncSession, job: EventEncoderJob, signature: str
) -> tuple[EventEncoderJob, bool]:
    session.add(job)
    try:
        await session.commit()
        return job, True
    except IntegrityError:
        await session.rollback()
        recovered = await _find_reusable(session, signature)
        if recovered is not None:
            return recovered
        raise


async def _get_existing_jobs_by_signature(
    session: AsyncSession, signature: str
) -> list[EventEncoderJob]:
    result = await session.execute(
        select(EventEncoderJob)
        .where(EventEncoderJob.tokenization_signature == signature)
        .order_by(EventEncoderJob.created_at.desc())
    )
    return list(result.scalars().all())


def _pick_active_or_complete_job(
    jobs: list[EventEncoderJob],
) -> Optional[EventEncoderJob]:
    for status in (
        JobStatus.complete.value,
        JobStatus.running.value,
        JobStatus.queued.value,
    ):
        for job in jobs:
            if job.status == status:
                return job
    return None


def _pick_resettable_job(jobs: list[EventEncoderJob]) -> Optional[EventEncoderJob]:
    for job in jobs:
        if job.status in _RESETTABLE_STATUSES:
            return job
    return None


def _reset_job_for_retry(job: EventEncoderJob) -> None:
    now = datetime.now(timezone.utc)
    job.status = JobStatus.queued.value
    job.event_vector_dim = None
    job.total_events = None
    job.encoded_events = None
    job.skipped_events = None
    job.event_vectors_path = None
    job.event_tokens_path = None
    job.token_sequences_path = None
    job.manifest_path = None
    job.report_path = None
    job.error_message = None
    job.updated_at = now
