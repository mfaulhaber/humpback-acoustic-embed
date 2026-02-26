import json

from fastapi import APIRouter, HTTPException

from humpback.api.deps import SessionDep
from humpback.schemas.processing import EmbeddingSetOut, ProcessingJobCreate, ProcessingJobOut
from humpback.services import processing_service

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
        created_at=job.created_at,
        updated_at=job.updated_at,
        skipped=skipped,
    )


@router.post("/jobs", status_code=201)
async def create_job(body: ProcessingJobCreate, session: SessionDep) -> ProcessingJobOut:
    job, skipped = await processing_service.create_processing_job(
        session,
        body.audio_file_id,
        body.model_version,
        body.window_size_seconds,
        body.target_sample_rate,
        body.feature_config,
    )
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
