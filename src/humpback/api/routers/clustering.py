import json

from fastapi import APIRouter, HTTPException

from humpback.api.deps import SessionDep
from humpback.schemas.clustering import (
    ClusterAssignmentOut,
    ClusteringJobCreate,
    ClusteringJobOut,
    ClusterOut,
)
from humpback.services import clustering_service

router = APIRouter(prefix="/clustering", tags=["clustering"])


def _job_to_out(job) -> ClusteringJobOut:
    return ClusteringJobOut(
        id=job.id,
        status=job.status,
        embedding_set_ids=json.loads(job.embedding_set_ids),
        parameters=json.loads(job.parameters) if job.parameters else None,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _cluster_to_out(c) -> ClusterOut:
    return ClusterOut(
        id=c.id,
        clustering_job_id=c.clustering_job_id,
        cluster_label=c.cluster_label,
        size=c.size,
        metadata_summary=json.loads(c.metadata_summary) if c.metadata_summary else None,
    )


@router.get("/jobs")
async def list_jobs(session: SessionDep) -> list[ClusteringJobOut]:
    jobs = await clustering_service.list_clustering_jobs(session)
    return [_job_to_out(j) for j in jobs]


@router.post("/jobs", status_code=201)
async def create_job(body: ClusteringJobCreate, session: SessionDep) -> ClusteringJobOut:
    try:
        job = await clustering_service.create_clustering_job(
            session, body.embedding_set_ids, body.parameters
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _job_to_out(job)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, session: SessionDep) -> ClusteringJobOut:
    job = await clustering_service.get_clustering_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Clustering job not found")
    return _job_to_out(job)


@router.get("/jobs/{job_id}/clusters")
async def list_clusters(job_id: str, session: SessionDep) -> list[ClusterOut]:
    clusters = await clustering_service.list_clusters(session, job_id)
    return [_cluster_to_out(c) for c in clusters]


@router.get("/clusters/{cluster_id}/assignments")
async def get_assignments(cluster_id: str, session: SessionDep) -> list[ClusterAssignmentOut]:
    assignments = await clustering_service.get_cluster_assignments(session, cluster_id)
    return [ClusterAssignmentOut.model_validate(a) for a in assignments]
