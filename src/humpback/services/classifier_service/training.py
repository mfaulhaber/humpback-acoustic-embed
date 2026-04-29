"""Training job management, retrain workflows, and training data summary."""

import json
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.classifier import (
    ClassifierModel,
    ClassifierTrainingJob,
    DetectionJob,
)
from humpback.models.retrain import RetrainWorkflow
from humpback.services.classifier_service.models import (
    delete_classifier_model,
)


async def create_training_job(
    session: AsyncSession,
    name: str,
    positive_embedding_set_ids: list[str],
    negative_embedding_set_ids: list[str],
    parameters: Optional[dict[str, Any]] = None,
) -> ClassifierTrainingJob:
    """Create a classifier training job after validating inputs."""
    del (
        session,
        name,
        positive_embedding_set_ids,
        negative_embedding_set_ids,
        parameters,
    )
    raise ValueError(
        "Embedding-set classifier training is retired; create training jobs "
        "from detection_job_ids and embedding_model_version"
    )


def _promote_legacy_embeddings(
    detection_job_id: str,
    cm: "ClassifierModel | None",
    embedding_model_version: str,
    storage_root: "Path",
    dest_path: "Path",
) -> bool:
    """Copy legacy-path embeddings to the model-versioned path if the source
    classifier model matches.  Returns True on success."""
    import shutil

    from humpback.storage import detection_dir

    if cm is None or cm.model_version != embedding_model_version:
        return False
    legacy_path = (
        detection_dir(storage_root, detection_job_id) / "detection_embeddings.parquet"
    )
    if not legacy_path.exists():
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(legacy_path), str(dest_path))
    return True


async def create_training_job_from_detection_manifest(
    session: AsyncSession,
    name: str,
    detection_job_ids: list[str],
    embedding_model_version: str,
    storage_root: Path,
    parameters: Optional[dict[str, Any]] = None,
) -> ClassifierTrainingJob:
    """Create a training job that sources embeddings from labeled detection jobs.

    Validates that each detection job exists, has its embeddings available for
    the requested ``embedding_model_version``, and contains at least one
    positive and one negative binary-labeled row across the selection. Persists
    ``source_mode="detection_manifest"`` and the detection job IDs on the job
    row.
    """
    from humpback.classifier.detection_rows import read_detection_row_store
    from humpback.models.classifier import DetectionJob
    from humpback.models.model_registry import ModelConfig
    from humpback.storage import (
        detection_embeddings_path,
        detection_row_store_path,
    )

    if not detection_job_ids:
        raise ValueError("At least one detection_job_id is required")
    if not embedding_model_version:
        raise ValueError("embedding_model_version is required")

    # Resolve the embedding model for its window params (falls back to the
    # detection jobs' source classifier if the registry row doesn't carry them).
    mc_result = await session.execute(
        select(ModelConfig).where(ModelConfig.name == embedding_model_version)
    )
    mc = mc_result.scalar_one_or_none()
    if mc is None:
        raise ValueError(
            f"Embedding model {embedding_model_version!r} is not registered"
        )

    # Validate each detection job: exists, complete, and has embeddings at the
    # requested model-versioned path.
    dj_result = await session.execute(
        select(DetectionJob).where(DetectionJob.id.in_(detection_job_ids))
    )
    dj_rows = list(dj_result.scalars().all())
    if len(dj_rows) != len(set(detection_job_ids)):
        found = {dj.id for dj in dj_rows}
        missing = set(detection_job_ids) - found
        raise ValueError(f"Detection jobs not found: {sorted(missing)}")

    window_sizes: set[float] = set()
    sample_rates: set[int] = set()
    total_pos = 0
    total_neg = 0
    for dj in dj_rows:
        # Resolve source classifier first — needed for window params and
        # legacy embedding path fallback.
        cm_result = await session.execute(
            select(ClassifierModel).where(ClassifierModel.id == dj.classifier_model_id)
        )
        cm = cm_result.scalar_one_or_none()
        if cm is not None:
            window_sizes.add(cm.window_size_seconds)
            sample_rates.add(cm.target_sample_rate)

        emb_path = detection_embeddings_path(
            storage_root, dj.id, embedding_model_version
        )
        if not emb_path.exists():
            if not _promote_legacy_embeddings(
                dj.id, cm, embedding_model_version, storage_root, emb_path
            ):
                raise ValueError(
                    f"Detection job {dj.id} has no embeddings for "
                    f"{embedding_model_version!r} — re-embed first"
                )
        rs_path = detection_row_store_path(storage_root, dj.id)
        if not rs_path.exists():
            raise ValueError(f"Detection job {dj.id} has no row store")
        _, rows = read_detection_row_store(rs_path)
        for row in rows:
            if row.get("humpback") == "1" or row.get("orca") == "1":
                total_pos += 1
            elif row.get("ship") == "1" or row.get("background") == "1":
                total_neg += 1

    if total_pos < 1 or total_neg < 1:
        raise ValueError(
            "Selected detection jobs must include at least one positive and "
            f"one negative binary label (pos={total_pos}, neg={total_neg})"
        )

    if not window_sizes or not sample_rates:
        raise ValueError(
            "Could not resolve window_size/target_sample_rate from detection jobs"
        )
    window_size = next(iter(window_sizes))
    sample_rate = next(iter(sample_rates))

    job = ClassifierTrainingJob(
        name=name,
        model_version=embedding_model_version,
        window_size_seconds=window_size,
        target_sample_rate=sample_rate,
        feature_config=None,
        parameters=json.dumps(parameters) if parameters else None,
        source_mode="detection_manifest",
        source_detection_job_ids=json.dumps(list(detection_job_ids)),
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

    if tj.source_mode == "detection_manifest":
        training_summary = (
            json.loads(cm.training_summary) if cm.training_summary else {}
        )
        total_pos = int(training_summary.get("n_positive") or 0)
        total_neg = int(training_summary.get("n_negative") or 0)
        balance = total_pos / total_neg if total_neg > 0 else float("inf")

        det_job_ids = training_summary.get("detection_job_ids", [])
        detection_sources: list[dict[str, Any]] = []

        if det_job_ids:
            det_result = await session.execute(
                select(DetectionJob).where(DetectionJob.id.in_(det_job_ids))
            )
            det_jobs = {dj.id: dj for dj in det_result.scalars().all()}

            tds = training_summary.get("training_data_source", {})
            per_job_counts = {
                entry["detection_job_id"]: entry
                for entry in tds.get("per_job_counts", [])
            }

            for job_id in det_job_ids:
                dj = det_jobs.get(job_id)
                counts = per_job_counts.get(job_id, {})
                detection_sources.append(
                    {
                        "detection_job_id": job_id,
                        "hydrophone_name": dj.hydrophone_name if dj else None,
                        "start_timestamp": dj.start_timestamp if dj else None,
                        "end_timestamp": dj.end_timestamp if dj else None,
                        "positive_count": counts.get("positive_count"),
                        "negative_count": counts.get("negative_count"),
                    }
                )

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
            "detection_sources": detection_sources,
        }

    training_summary = json.loads(cm.training_summary) if cm.training_summary else {}
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


# ---- Retrain Workflows ----


async def trace_folder_roots(
    session: AsyncSession, training_job: ClassifierTrainingJob
) -> dict[str, list[str]]:
    del session, training_job
    return {"positive_folder_roots": [], "negative_folder_roots": []}


async def collect_embedding_sets_for_folders(
    session: AsyncSession,
    folder_roots: list[str],
    model_version: str,
) -> list[str]:
    del session, folder_roots, model_version
    raise ValueError(
        "Classifier retrain is retired because it depended on the legacy "
        "audio/processing workflow"
    )


async def get_retrain_info(
    session: AsyncSession, model_id: str
) -> Optional[dict[str, Any]]:
    del session, model_id
    return None


async def create_retrain_workflow(
    session: AsyncSession,
    source_model_id: str,
    new_model_name: str,
    parameter_overrides: Optional[dict[str, Any]] = None,
) -> RetrainWorkflow:
    del session, source_model_id, new_model_name, parameter_overrides
    raise ValueError(
        "Classifier retrain is retired because it depended on the legacy "
        "audio/processing workflow"
    )


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
