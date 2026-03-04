"""API router for binary classifier training and detection."""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from humpback.api.deps import SessionDep, SettingsDep
from humpback.schemas.classifier import (
    ClassifierModelOut,
    ClassifierTrainingJobCreate,
    ClassifierTrainingJobOut,
    DetectionJobCreate,
    DetectionJobOut,
)
from humpback.services import classifier_service

router = APIRouter(prefix="/classifier", tags=["classifier"])


def _training_job_to_out(job) -> ClassifierTrainingJobOut:
    return ClassifierTrainingJobOut(
        id=job.id,
        status=job.status,
        name=job.name,
        positive_embedding_set_ids=json.loads(job.positive_embedding_set_ids),
        negative_audio_folder=job.negative_audio_folder,
        model_version=job.model_version,
        window_size_seconds=job.window_size_seconds,
        target_sample_rate=job.target_sample_rate,
        feature_config=json.loads(job.feature_config) if job.feature_config else None,
        parameters=json.loads(job.parameters) if job.parameters else None,
        classifier_model_id=job.classifier_model_id,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _model_to_out(m) -> ClassifierModelOut:
    return ClassifierModelOut(
        id=m.id,
        name=m.name,
        model_path=m.model_path,
        model_version=m.model_version,
        vector_dim=m.vector_dim,
        window_size_seconds=m.window_size_seconds,
        target_sample_rate=m.target_sample_rate,
        feature_config=json.loads(m.feature_config) if m.feature_config else None,
        training_summary=json.loads(m.training_summary) if m.training_summary else None,
        training_job_id=m.training_job_id,
        created_at=m.created_at,
        updated_at=m.updated_at,
    )


def _detection_job_to_out(job) -> DetectionJobOut:
    return DetectionJobOut(
        id=job.id,
        status=job.status,
        classifier_model_id=job.classifier_model_id,
        audio_folder=job.audio_folder,
        confidence_threshold=job.confidence_threshold,
        output_tsv_path=job.output_tsv_path,
        result_summary=json.loads(job.result_summary) if job.result_summary else None,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


# ---- Training Jobs ----

@router.post("/training-jobs", status_code=201)
async def create_training_job(
    body: ClassifierTrainingJobCreate, session: SessionDep
) -> ClassifierTrainingJobOut:
    try:
        job = await classifier_service.create_training_job(
            session,
            body.name,
            body.positive_embedding_set_ids,
            body.negative_audio_folder,
            body.parameters,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _training_job_to_out(job)


@router.get("/training-jobs")
async def list_training_jobs(session: SessionDep) -> list[ClassifierTrainingJobOut]:
    jobs = await classifier_service.list_training_jobs(session)
    return [_training_job_to_out(j) for j in jobs]


@router.get("/training-jobs/{job_id}")
async def get_training_job(
    job_id: str, session: SessionDep
) -> ClassifierTrainingJobOut:
    job = await classifier_service.get_training_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Training job not found")
    return _training_job_to_out(job)


# ---- Classifier Models ----

@router.get("/models")
async def list_models(session: SessionDep) -> list[ClassifierModelOut]:
    models = await classifier_service.list_classifier_models(session)
    return [_model_to_out(m) for m in models]


@router.get("/models/{model_id}")
async def get_model(model_id: str, session: SessionDep) -> ClassifierModelOut:
    m = await classifier_service.get_classifier_model(session, model_id)
    if m is None:
        raise HTTPException(404, "Classifier model not found")
    return _model_to_out(m)


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str, session: SessionDep, settings: SettingsDep
) -> dict:
    deleted = await classifier_service.delete_classifier_model(
        session, model_id, settings.storage_root
    )
    if not deleted:
        raise HTTPException(404, "Classifier model not found")
    return {"status": "deleted"}


# ---- Detection Jobs ----

@router.post("/detection-jobs", status_code=201)
async def create_detection_job(
    body: DetectionJobCreate, session: SessionDep
) -> DetectionJobOut:
    try:
        job = await classifier_service.create_detection_job(
            session,
            body.classifier_model_id,
            body.audio_folder,
            body.confidence_threshold,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _detection_job_to_out(job)


@router.get("/detection-jobs")
async def list_detection_jobs(session: SessionDep) -> list[DetectionJobOut]:
    jobs = await classifier_service.list_detection_jobs(session)
    return [_detection_job_to_out(j) for j in jobs]


@router.get("/detection-jobs/{job_id}")
async def get_detection_job(
    job_id: str, session: SessionDep
) -> DetectionJobOut:
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    return _detection_job_to_out(job)


@router.get("/detection-jobs/{job_id}/download")
async def download_detections(
    job_id: str, session: SessionDep
) -> FileResponse:
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    if job.status != "complete" or not job.output_tsv_path:
        raise HTTPException(400, "Detection job not complete or no output available")
    from pathlib import Path
    tsv_path = Path(job.output_tsv_path)
    if not tsv_path.is_file():
        raise HTTPException(404, "TSV file not found on disk")
    return FileResponse(
        tsv_path,
        media_type="text/tab-separated-values",
        filename=f"detections_{job_id}.tsv",
    )
