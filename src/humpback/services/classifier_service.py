"""Service layer for binary classifier training and detection."""

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.classifier import ClassifierModel, ClassifierTrainingJob, DetectionJob
from humpback.models.processing import EmbeddingSet

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


async def create_training_job(
    session: AsyncSession,
    name: str,
    positive_embedding_set_ids: list[str],
    negative_audio_folder: str,
    parameters: Optional[dict[str, Any]] = None,
) -> ClassifierTrainingJob:
    """Create a classifier training job after validating inputs."""
    if not positive_embedding_set_ids:
        raise ValueError("At least one positive embedding set is required")

    # Load and validate embedding sets
    result = await session.execute(
        select(EmbeddingSet).where(EmbeddingSet.id.in_(positive_embedding_set_ids))
    )
    embedding_sets = list(result.scalars().all())
    if len(embedding_sets) != len(positive_embedding_set_ids):
        found_ids = {es.id for es in embedding_sets}
        missing = set(positive_embedding_set_ids) - found_ids
        raise ValueError(f"Embedding sets not found: {missing}")

    # Validate all share same model_version, vector_dim, etc.
    model_versions = {es.model_version for es in embedding_sets}
    if len(model_versions) > 1:
        raise ValueError(f"Embedding sets use different model versions: {model_versions}")

    vector_dims = {es.vector_dim for es in embedding_sets}
    if len(vector_dims) > 1:
        raise ValueError(f"Embedding sets have different vector dimensions: {vector_dims}")

    # Validate negative audio folder exists and has audio files
    neg_folder = Path(negative_audio_folder)
    if not neg_folder.is_dir():
        raise ValueError(f"Negative audio folder not found: {negative_audio_folder}")

    audio_files = [
        p for p in neg_folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not audio_files:
        raise ValueError(f"No audio files found in {negative_audio_folder}")

    # Use first embedding set's config
    ref = embedding_sets[0]

    job = ClassifierTrainingJob(
        name=name,
        positive_embedding_set_ids=json.dumps(positive_embedding_set_ids),
        negative_audio_folder=negative_audio_folder,
        model_version=ref.model_version,
        window_size_seconds=ref.window_size_seconds,
        target_sample_rate=ref.target_sample_rate,
        feature_config=None,  # inherit from embedding sets
        parameters=json.dumps(parameters) if parameters else None,
    )
    session.add(job)
    await session.commit()
    return job


async def create_detection_job(
    session: AsyncSession,
    classifier_model_id: str,
    audio_folder: str,
    confidence_threshold: float = 0.5,
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

    audio_files = [
        p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_folder}")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")

    job = DetectionJob(
        classifier_model_id=classifier_model_id,
        audio_folder=audio_folder,
        confidence_threshold=confidence_threshold,
    )
    session.add(job)
    await session.commit()
    return job


async def list_training_jobs(session: AsyncSession) -> list[ClassifierTrainingJob]:
    result = await session.execute(
        select(ClassifierTrainingJob).order_by(ClassifierTrainingJob.created_at.desc())
    )
    return list(result.scalars().all())


async def get_training_job(
    session: AsyncSession, job_id: str
) -> Optional[ClassifierTrainingJob]:
    result = await session.execute(
        select(ClassifierTrainingJob).where(ClassifierTrainingJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def list_classifier_models(session: AsyncSession) -> list[ClassifierModel]:
    result = await session.execute(
        select(ClassifierModel).order_by(ClassifierModel.created_at.desc())
    )
    return list(result.scalars().all())


async def get_classifier_model(
    session: AsyncSession, model_id: str
) -> Optional[ClassifierModel]:
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == model_id)
    )
    return result.scalar_one_or_none()


async def delete_classifier_model(
    session: AsyncSession, model_id: str, storage_root: Path
) -> bool:
    """Delete a classifier model and its files. Returns True if found."""
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == model_id)
    )
    cm = result.scalar_one_or_none()
    if cm is None:
        return False

    # Delete files
    from humpback.storage import classifier_dir
    cdir = classifier_dir(storage_root, model_id)
    if cdir.is_dir():
        shutil.rmtree(cdir)

    await session.delete(cm)
    await session.commit()
    return True


async def list_detection_jobs(session: AsyncSession) -> list[DetectionJob]:
    result = await session.execute(
        select(DetectionJob).order_by(DetectionJob.created_at.desc())
    )
    return list(result.scalars().all())


async def get_detection_job(
    session: AsyncSession, job_id: str
) -> Optional[DetectionJob]:
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    return result.scalar_one_or_none()
