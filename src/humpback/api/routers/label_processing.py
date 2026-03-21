"""API router for label processing jobs."""

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.api.deps import get_session
from humpback.schemas.label_processing import (
    LabelProcessingJobCreate,
    LabelProcessingJobOut,
    LabelProcessingPreview,
)
from humpback.services.label_processing_service import (
    create_label_processing_job,
    delete_label_processing_job,
    get_label_processing_job,
    list_label_processing_jobs,
    preview_annotations,
)

router = APIRouter(prefix="/label-processing", tags=["label-processing"])


def _job_to_out(job) -> LabelProcessingJobOut:
    """Convert a LabelProcessingJob ORM instance to its response schema."""
    return LabelProcessingJobOut(
        id=job.id,
        status=job.status,
        workflow=job.workflow or "score_based",
        classifier_model_id=job.classifier_model_id,
        annotation_folder=job.annotation_folder,
        audio_folder=job.audio_folder,
        output_root=job.output_root,
        parameters=json.loads(job.parameters) if job.parameters else None,
        files_processed=job.files_processed,
        files_total=job.files_total,
        annotations_total=job.annotations_total,
        result_summary=json.loads(job.result_summary) if job.result_summary else None,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.post("/jobs", response_model=LabelProcessingJobOut)
async def create_job(
    body: LabelProcessingJobCreate,
    session: AsyncSession = Depends(get_session),
):
    try:
        job = await create_label_processing_job(
            session,
            annotation_folder=body.annotation_folder,
            audio_folder=body.audio_folder,
            output_root=body.output_root,
            classifier_model_id=body.classifier_model_id,
            parameters=body.parameters,
            workflow=body.workflow,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _job_to_out(job)


@router.get("/jobs", response_model=list[LabelProcessingJobOut])
async def list_jobs(session: AsyncSession = Depends(get_session)):
    jobs = await list_label_processing_jobs(session)
    return [_job_to_out(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=LabelProcessingJobOut)
async def get_job(job_id: str, session: AsyncSession = Depends(get_session)):
    job = await get_label_processing_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_out(job)


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, session: AsyncSession = Depends(get_session)):
    deleted = await delete_label_processing_job(session, job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted"}


@router.get("/preview", response_model=LabelProcessingPreview)
async def preview(
    annotation_folder: str = Query(...),
    audio_folder: str = Query(...),
):
    try:
        result = preview_annotations(annotation_folder, audio_folder)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
