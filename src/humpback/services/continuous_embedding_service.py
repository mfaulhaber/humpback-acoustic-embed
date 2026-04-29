"""Service layer for the Sequence Models continuous embedding producer.

Implements idempotent job creation keyed by ``encoding_signature`` and the
list/get/cancel surface used by the API. Workers pick up ``queued`` rows;
this module never invokes worker code directly.

Two source families share the table (ADR-057):

- ``surfperch-tensorflow2`` — event-padded SurfPerch chunks driven by a
  Pass 2 ``EventSegmentationJob``.
- ``crnn-call-parsing-pytorch`` — Pass-1-region-scoped chunks driven by a
  Pass 2 segmentation CRNN. Adds new request fields and an extended
  encoding signature; the SurfPerch path stays byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.call_parsing import (
    EventSegmentationJob,
    RegionDetectionJob,
    SegmentationModel,
)
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.schemas.sequence_models import ContinuousEmbeddingJobCreate
from humpback.sequence_models.crnn_features import compute_checkpoint_sha256
from humpback.storage import continuous_embedding_dir

# Source family discriminators. ``model_version`` carries the family;
# the worker and signature computation dispatch on it.
SOURCE_KIND_SURFPERCH = "surfperch"
SOURCE_KIND_REGION_CRNN = "region_crnn"

# Per-model_version producer constants. ``ModelConfig`` rows in the
# registry intentionally do not yet carry ``window_size_seconds`` /
# ``target_sample_rate`` / ``feature_config`` — they live here until the
# registry refactor lands. Adding a new entry here is the only knob for
# enabling a new producer family.
SUPPORTED_MODEL_VERSIONS: dict[str, dict[str, Any]] = {
    "surfperch-tensorflow2": {
        "source_kind": SOURCE_KIND_SURFPERCH,
        "window_size_seconds": 5.0,
        "target_sample_rate": 32000,
        "feature_config": None,
    },
    "crnn-call-parsing-pytorch": {
        "source_kind": SOURCE_KIND_REGION_CRNN,
        # Sample rate is fixed to the segmentation CRNN's feature
        # extractor (16 kHz mono); ``feature_config`` is read from the
        # checkpoint at worker time.
        "target_sample_rate": 16000,
        "feature_config": None,
    },
}


def source_kind_for(model_version: str) -> str:
    """Return the source family discriminator for ``model_version``."""
    constants = _resolve_model_version(model_version)
    return str(constants["source_kind"])


_RESETTABLE_STATUSES = {JobStatus.failed.value, JobStatus.canceled.value}


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
    event_segmentation_job_id: str,
    model_version: str,
    hop_seconds: float,
    window_size_seconds: float,
    pad_seconds: float,
    target_sample_rate: int,
    feature_config: Any,
) -> str:
    """SHA-256 idempotency key for the SurfPerch event-padded source."""
    payload = {
        "event_segmentation_job_id": event_segmentation_job_id,
        "model_version": model_version,
        "hop_seconds": hop_seconds,
        "window_size_seconds": window_size_seconds,
        "pad_seconds": pad_seconds,
        "target_sample_rate": target_sample_rate,
        "feature_config": feature_config,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_region_crnn_signature(
    *,
    region_detection_job_id: str,
    event_segmentation_job_id: str,
    model_version: str,
    crnn_checkpoint_sha256: str,
    chunk_size_seconds: float,
    chunk_hop_seconds: float,
    projection_kind: str,
    projection_dim: int,
    target_sample_rate: int,
    feature_config: Any,
) -> str:
    """SHA-256 idempotency key for the CRNN region-based source."""
    payload = {
        "region_detection_job_id": region_detection_job_id,
        "event_segmentation_job_id": event_segmentation_job_id,
        "model_version": model_version,
        "crnn_checkpoint_sha256": crnn_checkpoint_sha256,
        "chunk_size_seconds": chunk_size_seconds,
        "chunk_hop_seconds": chunk_hop_seconds,
        "projection_kind": projection_kind,
        "projection_dim": projection_dim,
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
    model_constants = _resolve_model_version(payload.model_version)
    source_kind = str(model_constants["source_kind"])

    if source_kind == SOURCE_KIND_REGION_CRNN:
        return await _create_region_crnn_job(session, payload, model_constants)
    return await _create_surfperch_job(session, payload, model_constants)


async def _create_surfperch_job(
    session: AsyncSession,
    payload: ContinuousEmbeddingJobCreate,
    model_constants: dict[str, Any],
) -> tuple[ContinuousEmbeddingJob, bool]:
    if payload.event_segmentation_job_id is None:
        raise ValueError(
            "event_segmentation_job_id is required for the SurfPerch source"
        )
    if payload.region_detection_job_id is not None:
        raise ValueError(
            "region_detection_job_id must not be set for the SurfPerch source"
        )
    if payload.hop_seconds <= 0:
        raise ValueError("hop_seconds must be > 0")
    if payload.pad_seconds < 0:
        raise ValueError("pad_seconds must be >= 0")

    seg_job = await session.get(EventSegmentationJob, payload.event_segmentation_job_id)
    if seg_job is None:
        raise ValueError(
            f"event_segmentation_job not found: {payload.event_segmentation_job_id}"
        )
    if seg_job.status != JobStatus.complete.value:
        raise ValueError(
            "continuous embedding requires a completed event_segmentation_job"
        )

    window_size_seconds = float(model_constants["window_size_seconds"])
    target_sample_rate = int(model_constants["target_sample_rate"])
    feature_config = model_constants["feature_config"]
    feature_config_json = _serialize_feature_config(feature_config)

    signature = compute_continuous_embedding_signature(
        event_segmentation_job_id=payload.event_segmentation_job_id,
        model_version=payload.model_version,
        hop_seconds=payload.hop_seconds,
        window_size_seconds=window_size_seconds,
        pad_seconds=payload.pad_seconds,
        target_sample_rate=target_sample_rate,
        feature_config=feature_config,
    )

    reusable = await _find_reusable(session, signature)
    if reusable is not None:
        return reusable

    job = ContinuousEmbeddingJob(
        event_segmentation_job_id=payload.event_segmentation_job_id,
        model_version=payload.model_version,
        window_size_seconds=window_size_seconds,
        hop_seconds=payload.hop_seconds,
        pad_seconds=payload.pad_seconds,
        target_sample_rate=target_sample_rate,
        feature_config_json=feature_config_json,
        encoding_signature=signature,
    )
    return await _persist_with_race_recovery(session, job, signature)


async def _create_region_crnn_job(
    session: AsyncSession,
    payload: ContinuousEmbeddingJobCreate,
    model_constants: dict[str, Any],
) -> tuple[ContinuousEmbeddingJob, bool]:
    if payload.region_detection_job_id is None:
        raise ValueError(
            "region_detection_job_id is required for the CRNN region source"
        )
    if payload.event_segmentation_job_id is None:
        raise ValueError(
            "event_segmentation_job_id is required as a Pass-2 disambiguator "
            "for CRNN region jobs"
        )
    if payload.crnn_segmentation_model_id is None:
        raise ValueError("crnn_segmentation_model_id is required for the CRNN source")
    if payload.chunk_size_seconds is None or payload.chunk_size_seconds <= 0:
        raise ValueError("chunk_size_seconds must be > 0 for the CRNN source")
    if payload.chunk_hop_seconds is None or payload.chunk_hop_seconds <= 0:
        raise ValueError("chunk_hop_seconds must be > 0 for the CRNN source")
    if payload.projection_dim is None or payload.projection_dim <= 0:
        raise ValueError("projection_dim must be > 0 for the CRNN source")
    if payload.projection_kind is None:
        raise ValueError("projection_kind is required for the CRNN source")

    region_job = await session.get(RegionDetectionJob, payload.region_detection_job_id)
    if region_job is None:
        raise ValueError(
            f"region_detection_job not found: {payload.region_detection_job_id}"
        )
    if region_job.status != JobStatus.complete.value:
        raise ValueError(
            "continuous embedding requires a completed region_detection_job "
            f"(status={region_job.status!r})"
        )

    seg_job = await session.get(EventSegmentationJob, payload.event_segmentation_job_id)
    if seg_job is None:
        raise ValueError(
            f"event_segmentation_job not found: {payload.event_segmentation_job_id}"
        )
    if seg_job.status != JobStatus.complete.value:
        raise ValueError(
            "continuous embedding requires a completed event_segmentation_job"
        )
    if seg_job.region_detection_job_id != payload.region_detection_job_id:
        raise ValueError(
            "event_segmentation_job parent region does not match "
            "region_detection_job_id submitted for CRNN source"
        )

    seg_model = await session.get(SegmentationModel, payload.crnn_segmentation_model_id)
    if seg_model is None:
        raise ValueError(
            f"segmentation_model not found: {payload.crnn_segmentation_model_id}"
        )
    checkpoint_path = Path(seg_model.model_path)
    if not checkpoint_path.exists():
        raise ValueError(f"segmentation_model checkpoint missing at {checkpoint_path}")
    crnn_sha = compute_checkpoint_sha256(checkpoint_path)

    target_sample_rate = int(model_constants["target_sample_rate"])
    feature_config = model_constants["feature_config"]
    feature_config_json = _serialize_feature_config(feature_config)

    signature = compute_region_crnn_signature(
        region_detection_job_id=payload.region_detection_job_id,
        event_segmentation_job_id=payload.event_segmentation_job_id,
        model_version=payload.model_version,
        crnn_checkpoint_sha256=crnn_sha,
        chunk_size_seconds=payload.chunk_size_seconds,
        chunk_hop_seconds=payload.chunk_hop_seconds,
        projection_kind=payload.projection_kind,
        projection_dim=payload.projection_dim,
        target_sample_rate=target_sample_rate,
        feature_config=feature_config,
    )

    reusable = await _find_reusable(session, signature)
    if reusable is not None:
        return reusable

    job = ContinuousEmbeddingJob(
        event_segmentation_job_id=payload.event_segmentation_job_id,
        region_detection_job_id=payload.region_detection_job_id,
        crnn_segmentation_model_id=payload.crnn_segmentation_model_id,
        crnn_checkpoint_sha256=crnn_sha,
        chunk_size_seconds=payload.chunk_size_seconds,
        chunk_hop_seconds=payload.chunk_hop_seconds,
        projection_kind=payload.projection_kind,
        projection_dim=payload.projection_dim,
        model_version=payload.model_version,
        target_sample_rate=target_sample_rate,
        feature_config_json=feature_config_json,
        encoding_signature=signature,
    )
    return await _persist_with_race_recovery(session, job, signature)


async def _find_reusable(
    session: AsyncSession, signature: str
) -> Optional[tuple[ContinuousEmbeddingJob, bool]]:
    """Resolve a ``signature`` to an existing reusable / resettable row."""
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
    session: AsyncSession, job: ContinuousEmbeddingJob, signature: str
) -> tuple[ContinuousEmbeddingJob, bool]:
    """Insert ``job`` with the unique-constraint race protocol."""
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


async def delete_continuous_embedding_job(
    session: AsyncSession, job_id: str, settings: Settings
) -> bool:
    job = await session.get(ContinuousEmbeddingJob, job_id)
    if job is None:
        return False
    artifact_dir = continuous_embedding_dir(settings.storage_root, job_id)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    await session.delete(job)
    await session.commit()
    return True


async def _get_existing_jobs_by_signature(
    session: AsyncSession, signature: str
) -> list[ContinuousEmbeddingJob]:
    result = await session.execute(
        select(ContinuousEmbeddingJob)
        .where(ContinuousEmbeddingJob.encoding_signature == signature)
        .order_by(ContinuousEmbeddingJob.created_at.desc())
    )
    return list(result.scalars().all())


def _pick_active_or_complete_job(
    jobs: list[ContinuousEmbeddingJob],
) -> Optional[ContinuousEmbeddingJob]:
    for status in (
        JobStatus.complete.value,
        JobStatus.running.value,
        JobStatus.queued.value,
    ):
        for job in jobs:
            if job.status == status:
                return job
    return None


def _pick_resettable_job(
    jobs: list[ContinuousEmbeddingJob],
) -> Optional[ContinuousEmbeddingJob]:
    for job in jobs:
        if job.status in _RESETTABLE_STATUSES:
            return job
    return None


def _reset_job_for_retry(job: ContinuousEmbeddingJob) -> None:
    now = datetime.now(timezone.utc)
    job.status = JobStatus.queued.value
    job.vector_dim = None
    job.total_events = None
    job.merged_spans = None
    job.total_windows = None
    job.total_regions = None
    job.total_chunks = None
    job.parquet_path = None
    job.error_message = None
    job.updated_at = now
