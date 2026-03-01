import json
from pathlib import Path

import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from humpback.api.deps import SessionDep, SettingsDep
from humpback.clustering.metrics import extract_category_from_folder_path
from humpback.models.audio import AudioFile
from humpback.models.processing import EmbeddingSet
from humpback.schemas.clustering import (
    ClusterAssignmentOut,
    ClusteringJobCreate,
    ClusteringJobOut,
    ClusterOut,
)
from humpback.services import clustering_service
from humpback.storage import cluster_dir

router = APIRouter(prefix="/clustering", tags=["clustering"])


def _job_to_out(job) -> ClusteringJobOut:
    return ClusteringJobOut(
        id=job.id,
        status=job.status,
        embedding_set_ids=json.loads(job.embedding_set_ids),
        parameters=json.loads(job.parameters) if job.parameters else None,
        error_message=job.error_message,
        metrics=json.loads(job.metrics_json) if job.metrics_json else None,
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


@router.get("/jobs/{job_id}/visualization")
async def get_visualization(job_id: str, session: SessionDep, settings: SettingsDep):
    job = await clustering_service.get_clustering_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Clustering job not found")
    if job.status != "complete":
        raise HTTPException(400, "Clustering job is not complete")

    umap_path = cluster_dir(Path(settings.storage_root), job_id) / "umap_coords.parquet"
    if not umap_path.exists():
        raise HTTPException(404, "UMAP coordinates not available for this job")

    table = pq.read_table(str(umap_path))
    es_ids = table.column("embedding_set_id").to_pylist()

    # Resolve embedding_set_id → audio filename
    unique_es_ids = list(set(es_ids))
    stmt = (
        select(
            EmbeddingSet.id,
            EmbeddingSet.audio_file_id,
            EmbeddingSet.window_size_seconds,
            AudioFile.filename,
            AudioFile.folder_path,
        )
        .join(AudioFile, EmbeddingSet.audio_file_id == AudioFile.id)
        .where(EmbeddingSet.id.in_(unique_es_ids))
    )
    rows = (await session.execute(stmt)).all()
    es_to_filename = {r[0]: r[3] for r in rows}
    es_to_folder_path = {r[0]: r[4] for r in rows}
    es_to_audio_file_id = {r[0]: r[1] for r in rows}
    es_to_window_size = {r[0]: r[2] for r in rows}

    audio_filenames = [es_to_filename.get(es_id, es_id) for es_id in es_ids]
    audio_file_ids = [es_to_audio_file_id.get(es_id, "") for es_id in es_ids]
    window_sizes = [es_to_window_size.get(es_id, 5.0) for es_id in es_ids]
    categories = []
    for es_id in es_ids:
        folder_path = es_to_folder_path.get(es_id)
        cat = extract_category_from_folder_path(folder_path) if folder_path else None
        categories.append(cat or "Unknown")

    return {
        "x": table.column("x").to_pylist(),
        "y": table.column("y").to_pylist(),
        "cluster_label": table.column("cluster_label").to_pylist(),
        "embedding_set_id": es_ids,
        "embedding_row_index": table.column("embedding_row_index").to_pylist(),
        "audio_filename": audio_filenames,
        "audio_file_id": audio_file_ids,
        "window_size_seconds": window_sizes,
        "category": categories,
    }


@router.get("/jobs/{job_id}/metrics")
async def get_metrics(job_id: str, session: SessionDep):
    """Return parsed metrics_json for a clustering job."""
    job = await clustering_service.get_clustering_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Clustering job not found")
    if job.status != "complete":
        raise HTTPException(400, "Clustering job is not complete")
    if not job.metrics_json:
        return {}
    return json.loads(job.metrics_json)


@router.get("/jobs/{job_id}/parameter-sweep")
async def get_parameter_sweep(job_id: str, session: SessionDep, settings: SettingsDep):
    """Return parameter_sweep.json from the cluster output directory."""
    job = await clustering_service.get_clustering_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Clustering job not found")
    if job.status != "complete":
        raise HTTPException(400, "Clustering job is not complete")

    sweep_path = cluster_dir(Path(settings.storage_root), job_id) / "parameter_sweep.json"
    if not sweep_path.exists():
        raise HTTPException(404, "Parameter sweep data not available for this job")

    return json.loads(sweep_path.read_text())


@router.get("/clusters/{cluster_id}/assignments")
async def get_assignments(cluster_id: str, session: SessionDep) -> list[ClusterAssignmentOut]:
    assignments = await clustering_service.get_cluster_assignments(session, cluster_id)
    return [ClusterAssignmentOut.model_validate(a) for a in assignments]
