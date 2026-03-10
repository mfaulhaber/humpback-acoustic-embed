import json
import shutil
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.clustering import Cluster, ClusterAssignment, ClusteringJob
from humpback.models.processing import EmbeddingSet
from humpback.storage import cluster_dir


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
