"""Service layer for the Sequence Models continuous embedding producer.

Implements idempotent job creation keyed by ``encoding_signature`` and the
list/get/cancel surface used by the API. Workers pick up ``queued`` rows;
this module never invokes worker code directly.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.schemas.sequence_models import ContinuousEmbeddingJobCreate

# Per-model_version producer constants. ``ModelConfig`` rows in the
# registry intentionally do not yet carry ``window_size_seconds`` /
# ``target_sample_rate`` / ``feature_config`` — they live here until the
# registry refactor lands. Adding a new entry here is the only knob for
# enabling a new SurfPerch-like backbone for this producer.
SUPPORTED_MODEL_VERSIONS: dict[str, dict[str, Any]] = {
    "surfperch-tensorflow2": {
        "window_size_seconds": 5.0,
        "target_sample_rate": 32000,
        "feature_config": None,
    },
}


def _resolve_model_version(model_version: str) -> dict[str, Any]:
    if model_version not in SUPPORTED_MODEL_VERSIONS:
        raise ValueError(
            f"Unsupported model_version for continuous embedding: {model_version!r}. "
            f"Supported: {sorted(SUPPORTED_MODEL_VERSIONS)}"
        )
    return SUPPORTED_MODEL_VERSIONS[model_version]


def _serialize_feature_config(feature_config: Any) -> Optional[str]:
    if feature_config is None:
        return None
    return json.dumps(feature_config, sort_keys=True, separators=(",", ":"))


def compute_continuous_embedding_signature(
    *,
    region_detection_job_id: str,
    model_version: str,
    hop_seconds: float,
    window_size_seconds: float,
    pad_seconds: float,
    target_sample_rate: int,
    feature_config: Any,
) -> str:
    """SHA-256 idempotency key over the exact inputs from spec §5.1."""
    payload = {
        "region_detection_job_id": region_detection_job_id,
        "model_version": model_version,
        "hop_seconds": hop_seconds,
        "window_size_seconds": window_size_seconds,
        "pad_seconds": pad_seconds,
        "target_sample_rate": target_sample_rate,
        "feature_config": feature_config,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def create_continuous_embedding_job(
    session: AsyncSession,
    payload: ContinuousEmbeddingJobCreate,
) -> tuple[ContinuousEmbeddingJob, bool]:
    """Create or reuse a continuous-embedding job.

    Returns ``(job, created)`` where ``created=False`` when an existing
    complete or in-flight (queued/running) row was returned.
    """
    if payload.hop_seconds <= 0:
        raise ValueError("hop_seconds must be > 0")
    if payload.pad_seconds < 0:
        raise ValueError("pad_seconds must be >= 0")

    region_job = await session.get(RegionDetectionJob, payload.region_detection_job_id)
    if region_job is None:
        raise ValueError(
            f"region_detection_job not found: {payload.region_detection_job_id}"
        )

    model_constants = _resolve_model_version(payload.model_version)
    window_size_seconds = float(model_constants["window_size_seconds"])
    target_sample_rate = int(model_constants["target_sample_rate"])
    feature_config = model_constants["feature_config"]
    feature_config_json = _serialize_feature_config(feature_config)

    signature = compute_continuous_embedding_signature(
        region_detection_job_id=payload.region_detection_job_id,
        model_version=payload.model_version,
        hop_seconds=payload.hop_seconds,
        window_size_seconds=window_size_seconds,
        pad_seconds=payload.pad_seconds,
        target_sample_rate=target_sample_rate,
        feature_config=feature_config,
    )

    existing_complete = await session.execute(
        select(ContinuousEmbeddingJob).where(
            ContinuousEmbeddingJob.encoding_signature == signature,
            ContinuousEmbeddingJob.status == JobStatus.complete.value,
        )
    )
    job = existing_complete.scalars().first()
    if job is not None:
        return job, False

    in_flight = await session.execute(
        select(ContinuousEmbeddingJob).where(
            ContinuousEmbeddingJob.encoding_signature == signature,
            ContinuousEmbeddingJob.status.in_(
                [JobStatus.queued.value, JobStatus.running.value]
            ),
        )
    )
    job = in_flight.scalars().first()
    if job is not None:
        return job, False

    job = ContinuousEmbeddingJob(
        region_detection_job_id=payload.region_detection_job_id,
        model_version=payload.model_version,
        window_size_seconds=window_size_seconds,
        hop_seconds=payload.hop_seconds,
        pad_seconds=payload.pad_seconds,
        target_sample_rate=target_sample_rate,
        feature_config_json=feature_config_json,
        encoding_signature=signature,
    )
    session.add(job)
    await session.commit()
    return job, True


async def list_continuous_embedding_jobs(
    session: AsyncSession,
    *,
    status: Optional[str] = None,
) -> list[ContinuousEmbeddingJob]:
    stmt = select(ContinuousEmbeddingJob).order_by(
        ContinuousEmbeddingJob.created_at.desc()
    )
    if status is not None:
        stmt = stmt.where(ContinuousEmbeddingJob.status == status)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_continuous_embedding_job(
    session: AsyncSession, job_id: str
) -> Optional[ContinuousEmbeddingJob]:
    return await session.get(ContinuousEmbeddingJob, job_id)


class CancelTerminalJobError(Exception):
    """Raised when caller attempts to cancel a job in a terminal state."""


async def cancel_continuous_embedding_job(
    session: AsyncSession, job_id: str
) -> Optional[ContinuousEmbeddingJob]:
    """Flip ``queued`` or ``running`` to ``canceled``.

    Returns the (possibly updated) job, ``None`` if the job does not
    exist, or raises ``CancelTerminalJobError`` if the job is already in
    a terminal state.
    """
    job = await get_continuous_embedding_job(session, job_id)
    if job is None:
        return None
    if job.status in (JobStatus.queued.value, JobStatus.running.value):
        job.status = JobStatus.canceled.value
        await session.commit()
        return job
    raise CancelTerminalJobError(
        f"continuous_embedding_job {job_id} is in terminal state {job.status!r}"
    )
