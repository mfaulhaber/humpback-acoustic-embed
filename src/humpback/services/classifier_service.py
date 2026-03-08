"""Service layer for binary classifier training and detection."""

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.classifier import ClassifierModel, ClassifierTrainingJob, DetectionJob
from humpback.models.processing import EmbeddingSet

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


async def create_training_job(
    session: AsyncSession,
    name: str,
    positive_embedding_set_ids: list[str],
    negative_embedding_set_ids: list[str],
    parameters: Optional[dict[str, Any]] = None,
) -> ClassifierTrainingJob:
    """Create a classifier training job after validating inputs."""
    if not positive_embedding_set_ids:
        raise ValueError("At least one positive embedding set is required")
    if not negative_embedding_set_ids:
        raise ValueError("At least one negative embedding set is required")

    # Reject overlap between positive and negative sets
    overlap = set(positive_embedding_set_ids) & set(negative_embedding_set_ids)
    if overlap:
        raise ValueError(f"Embedding sets cannot be both positive and negative: {overlap}")

    # Load and validate positive embedding sets
    result = await session.execute(
        select(EmbeddingSet).where(EmbeddingSet.id.in_(positive_embedding_set_ids))
    )
    pos_sets = list(result.scalars().all())
    if len(pos_sets) != len(positive_embedding_set_ids):
        found_ids = {es.id for es in pos_sets}
        missing = set(positive_embedding_set_ids) - found_ids
        raise ValueError(f"Positive embedding sets not found: {missing}")

    # Load and validate negative embedding sets
    result = await session.execute(
        select(EmbeddingSet).where(EmbeddingSet.id.in_(negative_embedding_set_ids))
    )
    neg_sets = list(result.scalars().all())
    if len(neg_sets) != len(negative_embedding_set_ids):
        found_ids = {es.id for es in neg_sets}
        missing = set(negative_embedding_set_ids) - found_ids
        raise ValueError(f"Negative embedding sets not found: {missing}")

    # Validate all sets share same model_version and vector_dim
    all_sets = pos_sets + neg_sets
    model_versions = {es.model_version for es in all_sets}
    if len(model_versions) > 1:
        raise ValueError(f"Embedding sets use different model versions: {model_versions}")

    vector_dims = {es.vector_dim for es in all_sets}
    if len(vector_dims) > 1:
        raise ValueError(f"Embedding sets have different vector dimensions: {vector_dims}")

    # Check encoding signature consistency
    encoding_sigs = {es.encoding_signature for es in all_sets if es.encoding_signature}
    if len(encoding_sigs) > 1:
        if parameters is None:
            parameters = {}
        parameters["_config_mismatch_warning"] = (
            f"Embedding sets use {len(encoding_sigs)} different encoding signatures. "
            "Results may be unreliable when mixing different processing configurations."
        )

    # Use first positive embedding set's config
    ref = pos_sets[0]

    job = ClassifierTrainingJob(
        name=name,
        positive_embedding_set_ids=json.dumps(positive_embedding_set_ids),
        negative_embedding_set_ids=json.dumps(negative_embedding_set_ids),
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

    audio_files = [
        p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    ]
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
    )
    session.add(job)
    await session.commit()
    return job


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
) -> DetectionJob:
    """Create a hydrophone detection job after validating inputs."""
    from humpback.config import HYDROPHONE_IDS, ORCASOUND_HYDROPHONES

    # Validate classifier model exists
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == classifier_model_id)
    )
    cm = result.scalar_one_or_none()
    if cm is None:
        raise ValueError(f"Classifier model not found: {classifier_model_id}")

    # Validate hydrophone
    if hydrophone_id not in HYDROPHONE_IDS:
        raise ValueError(f"Unknown hydrophone: {hydrophone_id}")

    hydrophone = next(h for h in ORCASOUND_HYDROPHONES if h["id"] == hydrophone_id)

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")

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
    """Cancel a running hydrophone detection job. Returns job if found."""
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
        .values(status="canceled", updated_at=datetime.now(timezone.utc))
    )
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


async def delete_training_job(
    session: AsyncSession, job_id: str, storage_root: Path
) -> bool:
    """Delete a training job. If it produced a model, cascade-delete the model too."""
    result = await session.execute(
        select(ClassifierTrainingJob).where(ClassifierTrainingJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return False

    # Cascade-delete the associated classifier model if any
    if job.classifier_model_id:
        await delete_classifier_model(session, job.classifier_model_id, storage_root)

    await session.delete(job)
    await session.commit()
    return True


async def bulk_delete_training_jobs(
    session: AsyncSession, job_ids: list[str], storage_root: Path
) -> int:
    """Delete multiple training jobs. Returns count of deleted jobs."""
    count = 0
    for job_id in job_ids:
        if await delete_training_job(session, job_id, storage_root):
            count += 1
    return count


async def delete_detection_job(
    session: AsyncSession, job_id: str, storage_root: Path
) -> bool:
    """Delete a detection job and its output files."""
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return False

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
) -> int:
    """Delete multiple detection jobs. Returns count of deleted jobs."""
    count = 0
    for job_id in job_ids:
        if await delete_detection_job(session, job_id, storage_root):
            count += 1
    return count


async def bulk_delete_classifier_models(
    session: AsyncSession, model_ids: list[str], storage_root: Path
) -> int:
    """Delete multiple classifier models. Returns count of deleted models."""
    count = 0
    for model_id in model_ids:
        if await delete_classifier_model(session, model_id, storage_root):
            count += 1
    return count


async def get_training_data_summary(
    session: AsyncSession, model_id: str
) -> Optional[dict[str, Any]]:
    """Build training data provenance summary for a classifier model."""
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == model_id)
    )
    cm = result.scalar_one_or_none()
    if cm is None:
        return None

    # Find the training job
    if not cm.training_job_id:
        return None
    result = await session.execute(
        select(ClassifierTrainingJob).where(ClassifierTrainingJob.id == cm.training_job_id)
    )
    tj = result.scalar_one_or_none()
    if tj is None:
        return None

    pos_ids = json.loads(tj.positive_embedding_set_ids)
    neg_ids = json.loads(tj.negative_embedding_set_ids)

    async def _resolve_sources(es_ids: list[str]) -> tuple[list[dict], int]:
        if not es_ids:
            return [], 0
        result = await session.execute(
            select(EmbeddingSet).where(EmbeddingSet.id.in_(es_ids))
        )
        sets = list(result.scalars().all())
        sources = []
        total = 0
        for es in sets:
            n_vectors = 0
            try:
                meta = pq.read_metadata(es.parquet_path)
                n_vectors = meta.num_rows
            except Exception:
                pass
            total += n_vectors
            duration = n_vectors * cm.window_size_seconds if n_vectors else None
            sources.append({
                "embedding_set_id": es.id,
                "audio_file_id": es.audio_file_id,
                "n_vectors": n_vectors,
                "duration_represented_sec": duration,
            })
        return sources, total

    pos_sources, total_pos = await _resolve_sources(pos_ids)
    neg_sources, total_neg = await _resolve_sources(neg_ids)

    balance = total_pos / total_neg if total_neg > 0 else float("inf")

    return {
        "model_id": cm.id,
        "model_name": cm.name,
        "positive_sources": pos_sources,
        "negative_sources": neg_sources,
        "total_positive": total_pos,
        "total_negative": total_neg,
        "balance_ratio": balance,
        "window_size_seconds": cm.window_size_seconds,
        "positive_duration_sec": total_pos * cm.window_size_seconds if total_pos else None,
        "negative_duration_sec": total_neg * cm.window_size_seconds if total_neg else None,
    }
