"""Service layer for label processing jobs."""

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.raven_parser import (
    pair_annotations_with_recordings,
)
from humpback.models.classifier import ClassifierModel
from humpback.models.label_processing import LabelProcessingJob


_VALID_WORKFLOWS = {"score_based", "sample_builder"}


async def create_label_processing_job(
    session: AsyncSession,
    annotation_folder: str,
    audio_folder: str,
    output_root: str,
    classifier_model_id: Optional[str] = None,
    parameters: Optional[dict[str, Any]] = None,
    workflow: str = "score_based",
) -> LabelProcessingJob:
    """Create a label processing job after validating inputs."""
    # Validate workflow
    if workflow not in _VALID_WORKFLOWS:
        raise ValueError(
            f"Invalid workflow '{workflow}'. Must be one of: {_VALID_WORKFLOWS}"
        )

    # Validate classifier model exists (required for score_based, optional for sample_builder)
    if workflow == "score_based" and not classifier_model_id:
        raise ValueError("classifier_model_id is required for score_based workflow")

    if classifier_model_id:
        result = await session.execute(
            select(ClassifierModel).where(ClassifierModel.id == classifier_model_id)
        )
        model = result.scalars().first()
        if model is None:
            raise ValueError(f"Classifier model not found: {classifier_model_id}")

    # Validate folders exist
    ann_path = Path(annotation_folder)
    aud_path = Path(audio_folder)
    if not ann_path.is_dir():
        raise ValueError(f"Annotation folder does not exist: {annotation_folder}")
    if not aud_path.is_dir():
        raise ValueError(f"Audio folder does not exist: {audio_folder}")

    # Validate pairing works (raises ValueError if no pairs)
    pairs = pair_annotations_with_recordings(ann_path, aud_path)
    total_annotations = sum(len(p.annotations) for p in pairs)

    job = LabelProcessingJob(
        workflow=workflow,
        classifier_model_id=classifier_model_id,
        annotation_folder=annotation_folder,
        audio_folder=audio_folder,
        output_root=output_root,
        parameters=json.dumps(parameters) if parameters else None,
        files_total=len(pairs),
        annotations_total=total_annotations,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def list_label_processing_jobs(
    session: AsyncSession,
) -> list[LabelProcessingJob]:
    """List all label processing jobs, newest first."""
    result = await session.execute(
        select(LabelProcessingJob).order_by(LabelProcessingJob.created_at.desc())
    )
    return list(result.scalars().all())


async def get_label_processing_job(
    session: AsyncSession,
    job_id: str,
) -> LabelProcessingJob | None:
    """Get a single label processing job by ID."""
    result = await session.execute(
        select(LabelProcessingJob).where(LabelProcessingJob.id == job_id)
    )
    return result.scalars().first()


async def delete_label_processing_job(
    session: AsyncSession,
    job_id: str,
    storage_root: Path | None = None,
) -> bool:
    """Delete a label processing job and its output artifacts.

    Returns True if job was found and deleted.
    """
    result = await session.execute(
        select(LabelProcessingJob).where(LabelProcessingJob.id == job_id)
    )
    job = result.scalars().first()
    if job is None:
        return False

    # Delete output artifacts
    if job.output_root:
        output_path = Path(job.output_root)
        if output_path.is_dir():
            shutil.rmtree(output_path, ignore_errors=True)

    await session.delete(job)
    await session.commit()
    return True


def preview_annotations(
    annotation_folder: str,
    audio_folder: str,
) -> dict[str, Any]:
    """Preview annotation pairing and call type distribution (dry run)."""
    ann_path = Path(annotation_folder)
    aud_path = Path(audio_folder)

    if not ann_path.is_dir():
        raise ValueError(f"Annotation folder does not exist: {annotation_folder}")
    if not aud_path.is_dir():
        raise ValueError(f"Audio folder does not exist: {audio_folder}")

    pairs = pair_annotations_with_recordings(ann_path, aud_path)

    call_type_counts: Counter[str] = Counter()
    paired_files = []
    for p in pairs:
        for ann in p.annotations:
            call_type_counts[ann.call_type] += 1
        paired_files.append(
            {
                "annotation_file": p.annotation_path.name,
                "audio_file": p.audio_path.name,
                "annotation_count": len(p.annotations),
            }
        )

    return {
        "paired_files": paired_files,
        "total_annotations": sum(len(p.annotations) for p in pairs),
        "call_type_distribution": dict(
            sorted(call_type_counts.items(), key=lambda x: -x[1])
        ),
        "unpaired_annotations": [],
        "unpaired_audio": [],
    }
