import json
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from humpback.models.clustering import Cluster, ClusterAssignment, ClusteringJob


async def create_clustering_job(
    session: AsyncSession,
    embedding_set_ids: list[str],
    parameters: Optional[dict[str, Any]] = None,
) -> ClusteringJob:
    job = ClusteringJob(
        embedding_set_ids=json.dumps(embedding_set_ids),
        parameters=json.dumps(parameters) if parameters else None,
    )
    session.add(job)
    await session.commit()
    return job


async def get_clustering_job(
    session: AsyncSession, job_id: str
) -> Optional[ClusteringJob]:
    result = await session.execute(
        select(ClusteringJob).where(ClusteringJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def list_clusters(
    session: AsyncSession, job_id: str
) -> list[Cluster]:
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
