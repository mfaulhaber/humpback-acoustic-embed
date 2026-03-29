import json
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from humpback.api.deps import SessionDep
from humpback.schemas.processing import (
    EmbeddingSetOut,
    ProcessingJobCreate,
    ProcessingJobOut,
)
from humpback.services import processing_service


class BulkDeleteRequest(BaseModel):
    ids: List[str]


class FolderEmbeddingSetRequest(BaseModel):
    folder_path: str


class FolderEmbeddingSetResponse(BaseModel):
    folder_path: str
    embedding_set_ids: list[str]
    total_files: int
    processed_files: int
    pending_files: int
    status: str  # "ready", "processing", "queued"


router = APIRouter(prefix="/processing", tags=["processing"])


def _job_to_out(job, skipped: bool = False) -> ProcessingJobOut:
    return ProcessingJobOut(
        id=job.id,
        audio_file_id=job.audio_file_id,
        status=job.status,
        encoding_signature=job.encoding_signature,
        model_version=job.model_version,
        window_size_seconds=job.window_size_seconds,
        target_sample_rate=job.target_sample_rate,
        feature_config=json.loads(job.feature_config) if job.feature_config else None,
        error_message=job.error_message,
        warning_message=job.warning_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        skipped=skipped,
    )


@router.post("/jobs", status_code=201)
async def create_job(
    body: ProcessingJobCreate, session: SessionDep
) -> ProcessingJobOut:
    try:
        job, skipped = await processing_service.create_processing_job(
            session,
            body.audio_file_id,
            body.model_version,
            body.window_size_seconds,
            body.target_sample_rate,
            body.feature_config,
        )
    except ValueError as e:
        message = str(e)
        status = 404 if message.startswith("Audio file not found:") else 400
        raise HTTPException(status, message)
    return _job_to_out(job, skipped=skipped)


@router.get("/jobs")
async def list_jobs(session: SessionDep) -> list[ProcessingJobOut]:
    jobs = await processing_service.list_processing_jobs(session)
    return [_job_to_out(j) for j in jobs]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, session: SessionDep) -> ProcessingJobOut:
    job = await processing_service.get_processing_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Processing job not found")
    return _job_to_out(job)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, session: SessionDep) -> ProcessingJobOut:
    job = await processing_service.cancel_processing_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Processing job not found")
    return _job_to_out(job)


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, session: SessionDep) -> dict:
    try:
        deleted = await processing_service.delete_processing_job(session, job_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not deleted:
        raise HTTPException(404, "Processing job not found")
    return {"status": "deleted"}


@router.post("/jobs/bulk-delete")
async def bulk_delete_jobs(body: BulkDeleteRequest, session: SessionDep) -> dict:
    count = await processing_service.bulk_delete_processing_jobs(session, body.ids)
    return {"status": "deleted", "count": count}


@router.get("/embedding-sets")
async def list_embedding_sets(session: SessionDep) -> list[EmbeddingSetOut]:
    sets = await processing_service.list_embedding_sets(session)
    return [EmbeddingSetOut.model_validate(s) for s in sets]


@router.get("/embedding-sets/{es_id}")
async def get_embedding_set(es_id: str, session: SessionDep) -> EmbeddingSetOut:
    es = await processing_service.get_embedding_set(session, es_id)
    if es is None:
        raise HTTPException(404, "Embedding set not found")
    return EmbeddingSetOut.model_validate(es)


@router.post("/folder-embedding-set")
async def folder_embedding_set(
    body: FolderEmbeddingSetRequest, session: SessionDep
) -> FolderEmbeddingSetResponse:
    """Import folder and find or create embedding sets for its audio files.

    Returns current status: 'ready' if all files have embedding sets,
    'processing'/'queued' if work is still in progress.
    """
    # Import audio files from folder (idempotent)
    from humpback.services import audio_service

    try:
        await audio_service.import_folder(session, body.folder_path)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Find audio files belonging to this folder
    audio_files = await processing_service.find_audio_files_for_folder(
        session, body.folder_path
    )
    if not audio_files:
        raise HTTPException(404, "No audio files found in folder")

    total_files = len(audio_files)
    embedding_set_ids: list[str] = []
    pending = 0

    for af in audio_files:
        es = await processing_service.find_embedding_set_for_audio(session, af.id)
        if es is not None:
            embedding_set_ids.append(es.id)
        else:
            pending += 1
            # Queue processing job if none exists yet
            await processing_service.ensure_processing_job(session, af.id)

    processed_files = len(embedding_set_ids)
    if pending == 0:
        status = "ready"
    elif processed_files > 0:
        status = "processing"
    else:
        status = "queued"

    return FolderEmbeddingSetResponse(
        folder_path=body.folder_path,
        embedding_set_ids=embedding_set_ids,
        total_files=total_files,
        processed_files=processed_files,
        pending_files=pending,
        status=status,
    )
