"""Detection job management."""

import shutil
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.classifier import ClassifierModel, DetectionJob

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


class DetectionJobDependencyError(Exception):
    """Raised when a detection job cannot be deleted due to downstream deps."""

    def __init__(self, job_id: str, message: str) -> None:
        self.job_id = job_id
        self.message = message
        super().__init__(message)


async def create_detection_job(
    session: AsyncSession,
    classifier_model_id: str,
    audio_folder: str,
    confidence_threshold: float = 0.5,
    hop_seconds: float = 1.0,
    high_threshold: float = 0.70,
    low_threshold: float = 0.45,
) -> DetectionJob:
    """Create a detection job after validating inputs."""
    # Validate classifier model exists
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == classifier_model_id)
    )
    cm = result.scalar_one_or_none()
    if cm is None:
        raise ValueError(f"Classifier model not found: {classifier_model_id}")

    # Validate audio folder
    folder = Path(audio_folder)
    if not folder.is_dir():
        raise ValueError(f"Audio folder not found: {audio_folder}")

    audio_files = [p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_folder}")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")

    if hop_seconds > cm.window_size_seconds:
        raise ValueError(
            f"hop_seconds ({hop_seconds}) must be <= window_size_seconds ({cm.window_size_seconds})"
        )

    job = DetectionJob(
        classifier_model_id=classifier_model_id,
        audio_folder=audio_folder,
        confidence_threshold=confidence_threshold,
        hop_seconds=hop_seconds,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        detection_mode="windowed",
    )
    session.add(job)
    await session.commit()
    return job


async def list_detection_jobs(session: AsyncSession) -> list[DetectionJob]:
    """List local (non-hydrophone) detection jobs."""
    result = await session.execute(
        select(DetectionJob)
        .where(DetectionJob.hydrophone_id.is_(None))
        .order_by(DetectionJob.created_at.desc())
    )
    return list(result.scalars().all())


async def get_detection_job(
    session: AsyncSession, job_id: str
) -> Optional[DetectionJob]:
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def _check_detection_job_dependencies(
    session: AsyncSession, job_id: str
) -> str | None:
    """Return a dependency message if the job cannot be deleted, else None."""
    from sqlalchemy import func

    from humpback.models.labeling import VocalizationLabel
    from humpback.models.training_dataset import TrainingDataset

    # Check vocalization labels
    vl_result = await session.execute(
        select(func.count()).where(VocalizationLabel.detection_job_id == job_id)
    )
    vl_count = vl_result.scalar() or 0

    # Check training datasets referencing this job in source_config JSON
    td_result = await session.execute(select(TrainingDataset))
    td_count = 0
    for td in td_result.scalars().all():
        if job_id in (td.source_config or ""):
            td_count += 1

    parts: list[str] = []
    if vl_count:
        parts.append(f"{vl_count} vocalization label{'s' if vl_count != 1 else ''}")
    if td_count:
        parts.append(f"{td_count} training dataset{'s' if td_count != 1 else ''}")

    if parts:
        return (
            f"Cannot delete detection job: used by {' and '.join(parts)}. "
            "Remove these associations first."
        )
    return None


async def delete_detection_job(
    session: AsyncSession, job_id: str, storage_root: Path
) -> bool:
    """Delete a detection job and its output files.

    Raises DetectionJobDependencyError if the job has downstream dependencies.
    """
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return False

    dep_msg = await _check_detection_job_dependencies(session, job_id)
    if dep_msg:
        raise DetectionJobDependencyError(job_id, dep_msg)

    # Delete detection output directory
    from humpback.storage import detection_dir

    ddir = detection_dir(storage_root, job_id)
    if ddir.is_dir():
        shutil.rmtree(ddir)

    await session.delete(job)
    await session.commit()
    return True


async def bulk_delete_detection_jobs(
    session: AsyncSession, job_ids: list[str], storage_root: Path
) -> tuple[int, list[dict[str, str]]]:
    """Delete multiple detection jobs.

    Returns (deleted_count, blocked_list) where blocked_list contains
    dicts with 'job_id' and 'detail' for jobs that could not be deleted.
    """
    count = 0
    blocked: list[dict[str, str]] = []
    for job_id in job_ids:
        try:
            if await delete_detection_job(session, job_id, storage_root):
                count += 1
        except DetectionJobDependencyError as exc:
            blocked.append({"job_id": exc.job_id, "detail": exc.message})
    return count, blocked
