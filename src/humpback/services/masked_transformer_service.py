"""Service layer for masked-transformer sequence jobs (ADR-061).

Mirrors :mod:`humpback.services.hmm_sequence_service` but with
training-signature idempotency and the extend-k-sweep entry point.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import (
    ContinuousEmbeddingJob,
    MaskedTransformerJob,
)
from humpback.services.continuous_embedding_service import (
    SOURCE_KIND_REGION_CRNN,
    source_kind_for,
)
from humpback.storage import (
    masked_transformer_dir,
)

logger = logging.getLogger(__name__)


class CancelTerminalJobError(Exception):
    """Raised when caller attempts to cancel a job in a terminal state."""


class ExtendKSweepError(Exception):
    """Raised when extend-k-sweep is invoked on a non-completed job."""


# ---------------------------------------------------------------------------
# Training signature
# ---------------------------------------------------------------------------


def compute_training_signature(
    *,
    continuous_embedding_job_id: str,
    preset: str,
    mask_fraction: float,
    span_length_min: int,
    span_length_max: int,
    dropout: float,
    mask_weight_bias: bool,
    cosine_loss_weight: float,
    max_epochs: int,
    early_stop_patience: int,
    val_split: float,
    seed: int,
) -> str:
    """Stable signature over training-only config (excludes ``k_values``).

    Excluding ``k_values`` keeps the same trained transformer reusable
    across extend-k-sweep calls — only tokenization is re-run.
    """
    payload = {
        "continuous_embedding_job_id": continuous_embedding_job_id,
        "preset": preset,
        "mask_fraction": float(mask_fraction),
        "span_length_min": int(span_length_min),
        "span_length_max": int(span_length_max),
        "dropout": float(dropout),
        "mask_weight_bias": bool(mask_weight_bias),
        "cosine_loss_weight": float(cosine_loss_weight),
        "max_epochs": int(max_epochs),
        "early_stop_patience": int(early_stop_patience),
        "val_split": float(val_split),
        "seed": int(seed),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_k_values(payload: str | list[int]) -> list[int]:
    if isinstance(payload, list):
        values = list(payload)
    else:
        values = json.loads(payload)
        if not isinstance(values, list):
            raise ValueError("k_values must be a JSON list")
    deduped: list[int] = []
    seen: set[int] = set()
    for raw_v in values:
        v = int(raw_v)
        if v < 2:
            raise ValueError(f"k must be >= 2, got {v}")
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    if not deduped:
        raise ValueError("k_values must be a non-empty list")
    return deduped


def serialize_k_values(values: list[int]) -> str:
    return json.dumps([int(v) for v in values], separators=(",", ":"))


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


async def create_masked_transformer_job(
    session: AsyncSession,
    *,
    continuous_embedding_job_id: str,
    preset: str = "default",
    mask_fraction: float = 0.20,
    span_length_min: int = 2,
    span_length_max: int = 6,
    dropout: float = 0.1,
    mask_weight_bias: bool = True,
    cosine_loss_weight: float = 0.0,
    max_epochs: int = 30,
    early_stop_patience: int = 3,
    val_split: float = 0.1,
    seed: int = 42,
    k_values: list[int] | None = None,
) -> tuple[MaskedTransformerJob, bool]:
    """Create a masked-transformer training job.

    Returns ``(job, created)`` where ``created`` is ``False`` when an
    existing job with the same ``training_signature`` is returned.
    """
    if preset not in {"small", "default", "large"}:
        raise ValueError(f"preset must be one of small/default/large, got {preset!r}")

    cej = await session.get(ContinuousEmbeddingJob, continuous_embedding_job_id)
    if cej is None:
        raise ValueError(
            f"continuous_embedding_job not found: {continuous_embedding_job_id}"
        )
    if cej.status != JobStatus.complete.value:
        raise ValueError(
            "masked-transformer job requires a completed continuous_embedding_job "
            f"(current status: {cej.status!r})"
        )
    if source_kind_for(cej.model_version) != SOURCE_KIND_REGION_CRNN:
        raise ValueError(
            "masked-transformer job requires a CRNN region-based upstream "
            f"continuous_embedding_job (got source_kind={source_kind_for(cej.model_version)!r})"
        )

    k_list = parse_k_values(k_values if k_values is not None else [100])

    signature = compute_training_signature(
        continuous_embedding_job_id=continuous_embedding_job_id,
        preset=preset,
        mask_fraction=mask_fraction,
        span_length_min=span_length_min,
        span_length_max=span_length_max,
        dropout=dropout,
        mask_weight_bias=mask_weight_bias,
        cosine_loss_weight=cosine_loss_weight,
        max_epochs=max_epochs,
        early_stop_patience=early_stop_patience,
        val_split=val_split,
        seed=seed,
    )

    existing = await session.execute(
        select(MaskedTransformerJob).where(
            MaskedTransformerJob.training_signature == signature
        )
    )
    found = existing.scalar_one_or_none()
    if found is not None:
        return found, False

    job = MaskedTransformerJob(
        continuous_embedding_job_id=continuous_embedding_job_id,
        training_signature=signature,
        preset=preset,
        mask_fraction=float(mask_fraction),
        span_length_min=int(span_length_min),
        span_length_max=int(span_length_max),
        dropout=float(dropout),
        mask_weight_bias=bool(mask_weight_bias),
        cosine_loss_weight=float(cosine_loss_weight),
        max_epochs=int(max_epochs),
        early_stop_patience=int(early_stop_patience),
        val_split=float(val_split),
        seed=int(seed),
        k_values=serialize_k_values(k_list),
        status=JobStatus.queued.value,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job, True


async def list_masked_transformer_jobs(
    session: AsyncSession,
    *,
    status: Optional[str] = None,
    continuous_embedding_job_id: Optional[str] = None,
) -> list[MaskedTransformerJob]:
    stmt = select(MaskedTransformerJob).order_by(MaskedTransformerJob.created_at.desc())
    if status is not None:
        stmt = stmt.where(MaskedTransformerJob.status == status)
    if continuous_embedding_job_id is not None:
        stmt = stmt.where(
            MaskedTransformerJob.continuous_embedding_job_id
            == continuous_embedding_job_id
        )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_masked_transformer_job(
    session: AsyncSession, job_id: str
) -> Optional[MaskedTransformerJob]:
    return await session.get(MaskedTransformerJob, job_id)


async def cancel_masked_transformer_job(
    session: AsyncSession, job_id: str
) -> Optional[MaskedTransformerJob]:
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        return None
    if job.status in (JobStatus.queued.value, JobStatus.running.value):
        job.status = JobStatus.canceled.value
        await session.commit()
        return job
    raise CancelTerminalJobError(
        f"masked_transformer_job {job_id} is in terminal state {job.status!r}"
    )


async def delete_masked_transformer_job(
    session: AsyncSession, job_id: str, settings: Settings
) -> bool:
    job = await session.get(MaskedTransformerJob, job_id)
    if job is None:
        return False
    artifact_dir = masked_transformer_dir(settings.storage_root, job_id)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    await session.delete(job)
    await session.commit()
    return True


async def extend_k_sweep_job(
    session: AsyncSession, job_id: str, additional_k: list[int]
) -> MaskedTransformerJob:
    """Extend the k-sweep on a completed job.

    Appends k values not already present in ``k_values`` and requeues
    the job for a follow-up worker pass that runs tokenization +
    interpretation only (the trained transformer + Z are untouched).
    """
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        raise ValueError(f"masked_transformer_job not found: {job_id}")
    if job.status != JobStatus.complete.value:
        raise ExtendKSweepError(
            f"masked_transformer_job {job_id} must be completed to extend k-sweep "
            f"(current status: {job.status!r})"
        )

    additional = parse_k_values(additional_k)
    current = parse_k_values(job.k_values)
    merged: list[int] = list(current)
    seen = set(current)
    for k in additional:
        if k not in seen:
            merged.append(k)
            seen.add(k)

    if merged == current:
        # No new k values — return the job unchanged.
        return job

    job.k_values = serialize_k_values(merged)
    job.status = JobStatus.queued.value
    job.status_reason = "extend_k_sweep"
    await session.commit()
    await session.refresh(job)
    return job


__all__ = [
    "CancelTerminalJobError",
    "ExtendKSweepError",
    "cancel_masked_transformer_job",
    "compute_training_signature",
    "create_masked_transformer_job",
    "delete_masked_transformer_job",
    "extend_k_sweep_job",
    "get_masked_transformer_job",
    "list_masked_transformer_jobs",
    "parse_k_values",
    "serialize_k_values",
]
