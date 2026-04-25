import json
import shutil
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.classifier import DetectionJob
from humpback.models.clustering import Cluster, ClusterAssignment, ClusteringJob
from humpback.models.detection_embedding_job import DetectionEmbeddingJob
from humpback.models.processing import EmbeddingSet
from humpback.models.vocalization import (
    VocalizationClassifierModel,
    VocalizationInferenceJob,
)
from humpback.schemas.clustering import ClusteringEligibleDetectionJobOut
from humpback.storage import cluster_dir, detection_embeddings_path


async def create_clustering_job(
    session: AsyncSession,
    embedding_set_ids: list[str],
    parameters: Optional[dict[str, Any]] = None,
    refined_from_job_id: Optional[str] = None,
    storage_root: Optional[Path] = None,
) -> ClusteringJob:
    if not embedding_set_ids:
        raise ValueError("At least one embedding set is required")

    # Validate that all embedding sets share the same vector_dim
    result = await session.execute(
        select(EmbeddingSet).where(EmbeddingSet.id.in_(embedding_set_ids))
    )
    sets = list(result.scalars().all())
    if len(sets) != len(embedding_set_ids):
        found_ids = {s.id for s in sets}
        missing = [eid for eid in embedding_set_ids if eid not in found_ids]
        raise ValueError(f"Embedding sets not found: {missing}")
    dims = {s.vector_dim for s in sets}
    if len(dims) > 1:
        raise ValueError(
            f"Cannot cluster embedding sets with different vector dimensions: {dims}"
        )
    model_versions = {s.model_version for s in sets}
    if len(model_versions) > 1:
        raise ValueError(
            f"Cannot cluster embedding sets from different models: {model_versions}"
        )

    # Validate refined_from_job_id if provided
    if refined_from_job_id is not None:
        source_result = await session.execute(
            select(ClusteringJob).where(ClusteringJob.id == refined_from_job_id)
        )
        source_job = source_result.scalar_one_or_none()
        if source_job is None:
            raise ValueError(f"Source clustering job not found: {refined_from_job_id}")
        if source_job.status != "complete":
            raise ValueError(
                f"Source clustering job is not complete (status={source_job.status})"
            )
        if storage_root is not None:
            refined_path = (
                cluster_dir(storage_root, refined_from_job_id)
                / "refined_embeddings.parquet"
            )
            if not refined_path.exists():
                raise ValueError(
                    f"Source job {refined_from_job_id} has no refined embeddings"
                )

    job = ClusteringJob(
        embedding_set_ids=json.dumps(embedding_set_ids),
        parameters=json.dumps(parameters) if parameters else None,
        refined_from_job_id=refined_from_job_id,
    )
    session.add(job)
    await session.commit()
    return job


async def list_clustering_jobs(session: AsyncSession) -> list[ClusteringJob]:
    result = await session.execute(
        select(ClusteringJob).order_by(ClusteringJob.created_at.desc())
    )
    return list(result.scalars().all())


async def get_clustering_job(
    session: AsyncSession, job_id: str
) -> Optional[ClusteringJob]:
    result = await session.execute(
        select(ClusteringJob).where(ClusteringJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def list_clusters(session: AsyncSession, job_id: str) -> list[Cluster]:
    result = await session.execute(
        select(Cluster)
        .where(Cluster.clustering_job_id == job_id)
        .order_by(Cluster.cluster_label)
    )
    return list(result.scalars().all())


async def get_cluster_assignments(
    session: AsyncSession, cluster_id: str
) -> list[ClusterAssignment]:
    result = await session.execute(
        select(ClusterAssignment).where(ClusterAssignment.cluster_id == cluster_id)
    )
    return list(result.scalars().all())


async def delete_clustering_job(
    session: AsyncSession, job_id: str, storage_root: Path
) -> None:
    job = await get_clustering_job(session, job_id)
    if job is None:
        raise ValueError(f"Clustering job not found: {job_id}")

    await session.delete(job)
    await session.commit()

    output_dir = cluster_dir(storage_root, job_id)
    if output_dir.exists():
        shutil.rmtree(output_dir)


async def create_vocalization_clustering_job(
    session: AsyncSession,
    detection_job_ids: list[str],
    parameters: Optional[dict[str, Any]] = None,
    storage_root: Optional[Path] = None,
) -> ClusteringJob:
    if not detection_job_ids:
        raise ValueError("At least one detection job is required")

    active_model_result = await session.execute(
        select(VocalizationClassifierModel).where(
            VocalizationClassifierModel.is_active.is_(True)
        )
    )
    active_model = active_model_result.scalar_one_or_none()
    if active_model is None:
        raise ValueError("No active vocalization model")

    for djid in detection_job_ids:
        dj_result = await session.execute(
            select(DetectionJob).where(DetectionJob.id == djid)
        )
        dj = dj_result.scalar_one_or_none()
        if dj is None:
            raise ValueError(f"Detection job not found: {djid}")

        inf_result = await session.execute(
            select(VocalizationInferenceJob).where(
                VocalizationInferenceJob.source_type == "detection_job",
                VocalizationInferenceJob.source_id == djid,
                VocalizationInferenceJob.vocalization_model_id == active_model.id,
                VocalizationInferenceJob.status == "complete",
            )
        )
        inf_job = inf_result.scalar_one_or_none()
        if inf_job is None:
            raise ValueError(
                f"Detection job {djid} has no completed inference from the active model"
            )

        emb_result = await session.execute(
            select(DetectionEmbeddingJob)
            .where(
                DetectionEmbeddingJob.detection_job_id == djid,
                DetectionEmbeddingJob.status == "complete",
            )
            .order_by(DetectionEmbeddingJob.created_at.desc())
            .limit(1)
        )
        emb_job = emb_result.scalar_one_or_none()
        if emb_job is None:
            raise ValueError(f"Detection job {djid} has no completed embedding job")

        if storage_root is not None:
            parquet_path = detection_embeddings_path(
                storage_root, djid, emb_job.model_version
            )
            if not parquet_path.exists():
                raise ValueError(
                    f"Embeddings parquet not found for detection job {djid}"
                )

    job = ClusteringJob(
        embedding_set_ids=json.dumps([]),
        detection_job_ids=json.dumps(detection_job_ids),
        parameters=json.dumps(parameters) if parameters else None,
    )
    session.add(job)
    await session.commit()
    return job


async def list_clustering_eligible_detection_jobs(
    session: AsyncSession,
) -> list[ClusteringEligibleDetectionJobOut]:
    active_result = await session.execute(
        select(VocalizationClassifierModel).where(
            VocalizationClassifierModel.is_active.is_(True)
        )
    )
    active_model = active_result.scalar_one_or_none()
    if active_model is None:
        return []

    stmt = (
        select(DetectionJob)
        .join(
            VocalizationInferenceJob,
            (VocalizationInferenceJob.source_id == DetectionJob.id)
            & (VocalizationInferenceJob.source_type == "detection_job"),
        )
        .join(
            DetectionEmbeddingJob,
            DetectionEmbeddingJob.detection_job_id == DetectionJob.id,
        )
        .where(
            VocalizationInferenceJob.vocalization_model_id == active_model.id,
            VocalizationInferenceJob.status == "complete",
            DetectionEmbeddingJob.status == "complete",
        )
        .order_by(DetectionJob.created_at.desc())
    )
    result = await session.execute(stmt)
    jobs = list(result.scalars().unique().all())

    out = []
    for dj in jobs:
        detection_count = None
        if dj.result_summary:
            try:
                summary = json.loads(dj.result_summary)
                detection_count = summary.get("total_detections")
            except (json.JSONDecodeError, AttributeError):
                pass
        out.append(
            ClusteringEligibleDetectionJobOut(
                id=dj.id,
                hydrophone_name=dj.hydrophone_name,
                start_timestamp=dj.start_timestamp,
                end_timestamp=dj.end_timestamp,
                detection_count=detection_count,
            )
        )
    return out


async def list_vocalization_clustering_jobs(
    session: AsyncSession,
) -> list[ClusteringJob]:
    result = await session.execute(
        select(ClusteringJob)
        .where(ClusteringJob.detection_job_ids.isnot(None))
        .order_by(ClusteringJob.created_at.desc())
    )
    return list(result.scalars().all())
