"""Hydrophone detection job management."""

from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.classifier import ClassifierModel, DetectionJob


async def create_hydrophone_detection_job(
    session: AsyncSession,
    classifier_model_id: str,
    hydrophone_id: str,
    start_timestamp: float,
    end_timestamp: float,
    confidence_threshold: float = 0.5,
    hop_seconds: float = 1.0,
    high_threshold: float = 0.70,
    low_threshold: float = 0.45,
    local_cache_path: str | None = None,
) -> DetectionJob:
    """Create a hydrophone detection job after validating inputs."""
    from humpback.config import (
        ARCHIVE_SOURCE_IDS,
        ORCASOUND_S3_BUCKET,
        get_archive_source,
    )

    # Validate classifier model exists
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == classifier_model_id)
    )
    cm = result.scalar_one_or_none()
    if cm is None:
        raise ValueError(f"Classifier model not found: {classifier_model_id}")

    # Validate archive source (legacy hydrophone_id field name retained)
    if hydrophone_id not in ARCHIVE_SOURCE_IDS:
        raise ValueError(f"Unknown hydrophone: {hydrophone_id}")

    hydrophone = get_archive_source(hydrophone_id)
    if hydrophone is None:
        raise ValueError(f"Unknown hydrophone: {hydrophone_id}")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")

    if hop_seconds > cm.window_size_seconds:
        raise ValueError(
            f"hop_seconds ({hop_seconds}) must be <= window_size_seconds ({cm.window_size_seconds})"
        )

    # Validate local cache path if provided
    if local_cache_path:
        if hydrophone["provider_kind"] != "orcasound_hls":
            raise ValueError(
                "local_cache_path is only supported for Orcasound HLS sources"
            )
        from pathlib import Path

        cache_dir = Path(local_cache_path) / ORCASOUND_S3_BUCKET / hydrophone_id / "hls"
        if not cache_dir.is_dir():
            raise ValueError(
                f"Local cache path does not contain expected HLS structure: "
                f"{cache_dir} not found"
            )

    job = DetectionJob(
        classifier_model_id=classifier_model_id,
        hydrophone_id=hydrophone_id,
        hydrophone_name=hydrophone["name"],
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        confidence_threshold=confidence_threshold,
        hop_seconds=hop_seconds,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        local_cache_path=local_cache_path,
        detection_mode="windowed",
    )
    session.add(job)
    await session.commit()
    return job


async def list_hydrophone_detection_jobs(session: AsyncSession) -> list[DetectionJob]:
    """List detection jobs that are hydrophone-based."""
    result = await session.execute(
        select(DetectionJob)
        .where(DetectionJob.hydrophone_id.isnot(None))
        .order_by(DetectionJob.created_at.desc())
    )
    return list(result.scalars().all())


async def cancel_hydrophone_detection_job(
    session: AsyncSession, job_id: str
) -> Optional[DetectionJob]:
    """Cancel a running or paused hydrophone detection job. Returns job if found."""
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None
    if job.status not in ("running", "paused", "queued"):
        raise ValueError(f"Job is not running, paused, or queued (status={job.status})")

    from datetime import datetime, timezone

    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(status="canceled", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()
    return job


async def pause_hydrophone_detection_job(
    session: AsyncSession, job_id: str
) -> Optional[DetectionJob]:
    """Pause a running hydrophone detection job. Returns job if found."""
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None
    if job.status != "running":
        raise ValueError(f"Job is not running (status={job.status})")

    from datetime import datetime, timezone

    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(status="paused", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()
    return job


async def resume_hydrophone_detection_job(
    session: AsyncSession, job_id: str
) -> Optional[DetectionJob]:
    """Resume a paused hydrophone detection job. Returns job if found."""
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None
    if job.status != "paused":
        raise ValueError(f"Job is not paused (status={job.status})")

    from datetime import datetime, timezone

    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(status="running", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()
    return job
