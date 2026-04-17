"""Detection embedding endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from humpback.api.deps import SessionDep, SettingsDep
from humpback.models.detection_embedding_job import DetectionEmbeddingJob
from humpback.schemas.classifier import (
    DetectionEmbeddingJobOut,
    DetectionEmbeddingJobStatus,
    EmbeddingJobListItem,
    EmbeddingStatusResponse,
)
from humpback.services import classifier_service
from humpback.storage import detection_embeddings_path, detection_row_store_path

router = APIRouter()


@router.get("/detection-jobs/{job_id}/embedding")
async def get_detection_embedding(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    row_id: str = Query(...),
):
    """Return the stored embedding vector for a detection row."""
    from humpback.classifier.detector import read_detection_embedding

    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")

    # Resolve model_version from the classifier model
    from sqlalchemy import select as sa_select

    from humpback.models.classifier import ClassifierModel

    cm_result = await session.execute(
        sa_select(ClassifierModel).where(ClassifierModel.id == job.classifier_model_id)
    )
    cm = cm_result.scalar_one_or_none()
    model_version = cm.model_version if cm else "unknown"

    emb_path = detection_embeddings_path(settings.storage_root, job.id, model_version)
    if not emb_path.exists():
        raise HTTPException(404, "No stored embeddings for this detection job")

    embedding = read_detection_embedding(emb_path, row_id)
    if embedding is None:
        raise HTTPException(404, "Embedding not found for specified detection row")

    return {
        "vector": embedding,
        "model_version": model_version,
        "vector_dim": len(embedding),
    }


# ---- Detection Embedding Status / Generation ----


@router.get(
    "/detection-jobs/{job_id}/embedding-status",
    response_model=EmbeddingStatusResponse,
)
async def get_embedding_status(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
):
    """Check whether detection embeddings exist for a job."""
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")

    from humpback.services.classifier_service.models import (
        resolve_detection_job_model_version,
    )

    model_version = await resolve_detection_job_model_version(session, job.id)
    emb_path = detection_embeddings_path(settings.storage_root, job.id, model_version)
    if not emb_path.exists():
        return EmbeddingStatusResponse(has_embeddings=False)

    import pyarrow.parquet as pq

    table = pq.read_table(str(emb_path))
    count = table.num_rows

    # Check if row store and embeddings are in sync.
    sync_needed: bool | None = None
    rs_path = detection_row_store_path(settings.storage_root, job.id)
    if rs_path.exists():
        from humpback.classifier.detector import diff_row_store_vs_embeddings

        try:
            diff = diff_row_store_vs_embeddings(rs_path, emb_path)
            sync_needed = bool(diff.missing or diff.orphaned_indices)
        except Exception:
            sync_needed = None

    return EmbeddingStatusResponse(
        has_embeddings=True, count=count, sync_needed=sync_needed
    )


@router.post(
    "/detection-jobs/{job_id}/generate-embeddings",
    status_code=202,
    response_model=DetectionEmbeddingJobOut,
)
async def generate_embeddings(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    mode: str = Query("full", pattern="^(full|sync)$"),
):
    """Queue post-hoc embedding generation for a detection job.

    mode=full: re-run detection to collect embeddings (rejects if embeddings exist).
    mode=sync: diff row store vs embeddings, generate missing, remove orphans
               (requires embeddings to already exist).
    """
    from sqlalchemy import select as sa_select

    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    if job.status != "complete":
        raise HTTPException(400, "Detection job is not complete")

    from humpback.services.classifier_service.models import (
        resolve_detection_job_model_version,
    )

    model_version = await resolve_detection_job_model_version(session, job.id)
    emb_path = detection_embeddings_path(settings.storage_root, job.id, model_version)

    if mode == "full":
        if emb_path.exists():
            raise HTTPException(409, "Embeddings already exist for this detection job")
    elif mode == "sync":
        if not emb_path.exists():
            raise HTTPException(400, "No embeddings exist — run full generation first")

    # Check if generation already in progress
    existing = await session.execute(
        sa_select(DetectionEmbeddingJob).where(
            DetectionEmbeddingJob.detection_job_id == job_id,
            DetectionEmbeddingJob.model_version == model_version,
            DetectionEmbeddingJob.status.in_(["queued", "running"]),
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(409, "Embedding generation already in progress")

    from humpback.services.detection_embedding_service import create_reembedding_job

    return await create_reembedding_job(session, job_id, model_version, mode=mode)


@router.get(
    "/detection-jobs/{job_id}/embedding-generation-status",
    response_model=DetectionEmbeddingJobOut | None,
)
async def get_embedding_generation_status(
    job_id: str,
    session: SessionDep,
):
    """Get the most recent embedding generation job for a detection job."""
    from sqlalchemy import select as sa_select

    result = await session.execute(
        sa_select(DetectionEmbeddingJob)
        .where(DetectionEmbeddingJob.detection_job_id == job_id)
        .order_by(DetectionEmbeddingJob.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


@router.get(
    "/detection-embedding-jobs",
    response_model=list[DetectionEmbeddingJobStatus],
)
async def list_detection_embedding_jobs(
    session: SessionDep,
    detection_job_ids: list[str] = Query(..., min_length=1),
    model_version: str = Query(..., min_length=1),
):
    """Return status rows for each ``(detection_job_id, model_version)`` pair.

    Pairs with no existing row are returned with ``status="not_started"`` so the
    caller always gets one entry per requested detection job id.
    """
    from humpback.services.detection_embedding_service import list_reembedding_jobs

    jobs = await list_reembedding_jobs(session, detection_job_ids, model_version)
    out: list[DetectionEmbeddingJobStatus] = []
    for det_job_id in detection_job_ids:
        j = jobs.get(det_job_id)
        if j is None:
            out.append(
                DetectionEmbeddingJobStatus(
                    detection_job_id=det_job_id,
                    model_version=model_version,
                    status="not_started",
                )
            )
        else:
            out.append(
                DetectionEmbeddingJobStatus(
                    detection_job_id=j.detection_job_id,
                    model_version=j.model_version,
                    status=j.status,
                    rows_processed=j.rows_processed,
                    rows_total=j.rows_total,
                    error_message=j.error_message,
                    created_at=j.created_at,
                    updated_at=j.updated_at,
                )
            )
    return out


@router.get(
    "/embedding-jobs",
    response_model=list[EmbeddingJobListItem],
)
async def list_embedding_jobs(
    session: SessionDep,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """List all detection embedding jobs, newest first, with detection job context."""
    from sqlalchemy import select as sa_select

    from humpback.models.classifier import DetectionJob

    result = await session.execute(
        sa_select(DetectionEmbeddingJob)
        .order_by(DetectionEmbeddingJob.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    jobs = result.scalars().all()

    # Batch-load detection job context
    det_job_ids = {j.detection_job_id for j in jobs}
    det_jobs_map: dict[str, DetectionJob] = {}
    if det_job_ids:
        det_result = await session.execute(
            sa_select(DetectionJob).where(DetectionJob.id.in_(det_job_ids))
        )
        for dj in det_result.scalars().all():
            det_jobs_map[dj.id] = dj

    items: list[EmbeddingJobListItem] = []
    for j in jobs:
        dj = det_jobs_map.get(j.detection_job_id)
        audio_folder_basename = None
        if dj and dj.audio_folder:
            audio_folder_basename = Path(dj.audio_folder).name
        items.append(
            EmbeddingJobListItem(
                id=j.id,
                status=j.status,
                detection_job_id=j.detection_job_id,
                model_version=j.model_version,
                mode=j.mode,
                progress_current=j.progress_current,
                progress_total=j.progress_total,
                rows_processed=j.rows_processed,
                rows_total=j.rows_total,
                error_message=j.error_message,
                result_summary=j.result_summary,
                created_at=j.created_at,
                updated_at=j.updated_at,
                hydrophone_name=dj.hydrophone_name if dj else None,
                audio_folder=audio_folder_basename,
            )
        )
    return items
