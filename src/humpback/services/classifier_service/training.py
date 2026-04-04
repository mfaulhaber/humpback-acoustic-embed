"""Training job management, retrain workflows, and training data summary."""

import json
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.audio import AudioFile
from humpback.models.classifier import (
    ClassifierModel,
    ClassifierTrainingJob,
)
from humpback.models.processing import EmbeddingSet
from humpback.models.retrain import RetrainWorkflow
from humpback.services.classifier_service.models import (
    delete_classifier_model,
    get_classifier_model,
)


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
        raise ValueError(
            f"Embedding sets cannot be both positive and negative: {overlap}"
        )

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
        raise ValueError(
            f"Embedding sets use different model versions: {model_versions}"
        )

    vector_dims = {es.vector_dim for es in all_sets}
    if len(vector_dims) > 1:
        raise ValueError(
            f"Embedding sets have different vector dimensions: {vector_dims}"
        )

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


# ---- Training Data Summary ----


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
        select(ClassifierTrainingJob).where(
            ClassifierTrainingJob.id == cm.training_job_id
        )
    )
    tj = result.scalar_one_or_none()
    if tj is None:
        return None

    if tj.source_mode == "autoresearch_candidate":
        training_summary = (
            json.loads(cm.training_summary) if cm.training_summary else {}
        )
        total_pos = int(training_summary.get("n_positive") or 0)
        total_neg = int(training_summary.get("n_negative") or 0)
        balance = total_pos / total_neg if total_neg > 0 else float("inf")
        return {
            "model_id": cm.id,
            "model_name": cm.name,
            "positive_sources": [],
            "negative_sources": [],
            "total_positive": total_pos,
            "total_negative": total_neg,
            "balance_ratio": balance,
            "window_size_seconds": cm.window_size_seconds,
            "positive_duration_sec": total_pos * cm.window_size_seconds
            if total_pos
            else None,
            "negative_duration_sec": total_neg * cm.window_size_seconds
            if total_neg
            else None,
        }

    pos_ids = json.loads(tj.positive_embedding_set_ids)
    neg_ids = json.loads(tj.negative_embedding_set_ids)

    async def _resolve_sources(es_ids: list[str]) -> tuple[list[dict], int]:
        if not es_ids:
            return [], 0
        result = await session.execute(
            select(EmbeddingSet).where(EmbeddingSet.id.in_(es_ids))
        )
        sets = list(result.scalars().all())

        # Batch-load audio files for folder_path + filename
        audio_ids = [es.audio_file_id for es in sets if es.audio_file_id]
        audio_map: dict[str, AudioFile] = {}
        if audio_ids:
            af_result = await session.execute(
                select(AudioFile).where(AudioFile.id.in_(audio_ids))
            )
            audio_map = {af.id: af for af in af_result.scalars().all()}

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
            af = audio_map.get(es.audio_file_id) if es.audio_file_id else None
            sources.append(
                {
                    "embedding_set_id": es.id,
                    "audio_file_id": es.audio_file_id,
                    "filename": af.filename if af else None,
                    "folder_path": af.folder_path if af else None,
                    "n_vectors": n_vectors,
                    "duration_represented_sec": duration,
                }
            )
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
        "positive_duration_sec": total_pos * cm.window_size_seconds
        if total_pos
        else None,
        "negative_duration_sec": total_neg * cm.window_size_seconds
        if total_neg
        else None,
    }


# ---- Retrain Workflows ----


async def trace_folder_roots(
    session: AsyncSession, training_job: ClassifierTrainingJob
) -> dict[str, list[str]]:
    """Trace back from training job's embedding sets to import folder roots."""
    pos_ids = json.loads(training_job.positive_embedding_set_ids)
    neg_ids = json.loads(training_job.negative_embedding_set_ids)

    async def _resolve_roots(es_ids: list[str]) -> list[str]:
        if not es_ids:
            return []
        result = await session.execute(
            select(EmbeddingSet.audio_file_id).where(EmbeddingSet.id.in_(es_ids))
        )
        audio_file_ids = list(set(result.scalars().all()))
        if not audio_file_ids:
            return []

        result = await session.execute(
            select(AudioFile.source_folder, AudioFile.folder_path).where(
                AudioFile.id.in_(audio_file_ids),
                AudioFile.source_folder.isnot(None),
            )
        )
        rows = result.all()

        roots = set()
        for source_folder, folder_path in rows:
            parts = folder_path.split("/")
            import_root = Path(source_folder)
            for _ in range(len(parts) - 1):
                import_root = import_root.parent
            roots.add(str(import_root))
        return sorted(roots)

    return {
        "positive_folder_roots": await _resolve_roots(pos_ids),
        "negative_folder_roots": await _resolve_roots(neg_ids),
    }


async def collect_embedding_sets_for_folders(
    session: AsyncSession,
    folder_roots: list[str],
    model_version: str,
) -> list[str]:
    """Find all embedding set IDs for audio files under the given import roots."""
    all_ids = []
    for root in folder_roots:
        base_name = Path(root).name
        result = await session.execute(
            select(AudioFile.id).where(
                AudioFile.source_folder.isnot(None),
                (AudioFile.folder_path == base_name)
                | AudioFile.folder_path.startswith(f"{base_name}/"),
            )
        )
        audio_ids = list(result.scalars().all())
        if not audio_ids:
            continue

        result = await session.execute(
            select(EmbeddingSet.id).where(
                EmbeddingSet.audio_file_id.in_(audio_ids),
                EmbeddingSet.model_version == model_version,
            )
        )
        all_ids.extend(result.scalars().all())
    return sorted(set(all_ids))


async def get_retrain_info(
    session: AsyncSession, model_id: str
) -> Optional[dict[str, Any]]:
    """Pre-flight info for retrain: folder roots and parameters."""
    cm = await get_classifier_model(session, model_id)
    if cm is None:
        return None

    if not cm.training_job_id:
        return None

    tj = await get_training_job(session, cm.training_job_id)
    if tj is None:
        return None
    if tj.source_mode == "autoresearch_candidate":
        return None

    roots = await trace_folder_roots(session, tj)

    parameters = json.loads(tj.parameters) if tj.parameters else {}
    parameters.pop("_config_mismatch_warning", None)

    return {
        "model_id": cm.id,
        "model_name": cm.name,
        "model_version": cm.model_version,
        "window_size_seconds": cm.window_size_seconds,
        "target_sample_rate": cm.target_sample_rate,
        "feature_config": json.loads(cm.feature_config) if cm.feature_config else None,
        "positive_folder_roots": roots["positive_folder_roots"],
        "negative_folder_roots": roots["negative_folder_roots"],
        "parameters": parameters,
    }


async def create_retrain_workflow(
    session: AsyncSession,
    source_model_id: str,
    new_model_name: str,
    parameter_overrides: Optional[dict[str, Any]] = None,
) -> RetrainWorkflow:
    """Create a retrain workflow from an existing classifier model."""
    cm = await get_classifier_model(session, source_model_id)
    if cm is None:
        raise ValueError(f"Source classifier model not found: {source_model_id}")

    if not cm.training_job_id:
        raise ValueError("Source model has no associated training job")

    tj = await get_training_job(session, cm.training_job_id)
    if tj is None:
        raise ValueError("Source model's training job not found")
    if tj.source_mode == "autoresearch_candidate":
        raise ValueError("Candidate-backed models do not support folder-root retrain")

    roots = await trace_folder_roots(session, tj)
    if not roots["positive_folder_roots"]:
        raise ValueError("Cannot trace positive folder roots from training data")
    if not roots["negative_folder_roots"]:
        raise ValueError("Cannot trace negative folder roots from training data")

    base_params = json.loads(tj.parameters) if tj.parameters else {}
    base_params.pop("_config_mismatch_warning", None)
    if parameter_overrides:
        base_params.update(parameter_overrides)

    workflow = RetrainWorkflow(
        source_model_id=source_model_id,
        new_model_name=new_model_name,
        model_version=cm.model_version,
        window_size_seconds=cm.window_size_seconds,
        target_sample_rate=cm.target_sample_rate,
        feature_config=cm.feature_config,
        parameters=json.dumps(base_params) if base_params else None,
        positive_folder_roots=json.dumps(roots["positive_folder_roots"]),
        negative_folder_roots=json.dumps(roots["negative_folder_roots"]),
    )
    session.add(workflow)
    await session.commit()
    return workflow


async def list_retrain_workflows(session: AsyncSession) -> list[RetrainWorkflow]:
    result = await session.execute(
        select(RetrainWorkflow).order_by(RetrainWorkflow.created_at.desc())
    )
    return list(result.scalars().all())


async def get_retrain_workflow(
    session: AsyncSession, workflow_id: str
) -> Optional[RetrainWorkflow]:
    result = await session.execute(
        select(RetrainWorkflow).where(RetrainWorkflow.id == workflow_id)
    )
    return result.scalar_one_or_none()
