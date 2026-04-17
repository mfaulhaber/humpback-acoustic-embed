"""Training job and retrain workflow endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from humpback.api.deps import SessionDep, SettingsDep
from humpback.schemas.classifier import (
    ClassifierTrainingJobCreate,
    ClassifierTrainingJobOut,
    RetrainFolderInfo,
    RetrainWorkflowCreate,
    RetrainWorkflowOut,
    TrainingDataSummaryResponse,
    TrainingSourceInfo,
)
from humpback.schemas.converters import (
    classifier_training_job_to_out as _training_job_to_out,
    retrain_workflow_to_out as _retrain_workflow_to_out,
)
from humpback.services import classifier_service

router = APIRouter()


class BulkDeleteRequest(BaseModel):
    ids: list[str]


# ---- Training Jobs ----


@router.post("/training-jobs", status_code=201)
async def create_training_job(
    body: ClassifierTrainingJobCreate, session: SessionDep, settings: SettingsDep
) -> ClassifierTrainingJobOut:
    try:
        if body.detection_job_ids:
            if body.embedding_model_version is None:
                raise HTTPException(
                    400, "embedding_model_version is required for detection-job sources"
                )
            job = await classifier_service.create_training_job_from_detection_manifest(
                session,
                body.name,
                body.detection_job_ids,
                body.embedding_model_version,
                settings.storage_root,
                body.parameters,
            )
        else:
            job = await classifier_service.create_training_job(
                session,
                body.name,
                body.positive_embedding_set_ids,
                body.negative_embedding_set_ids,
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


@router.delete("/training-jobs/{job_id}")
async def delete_training_job(
    job_id: str, session: SessionDep, settings: SettingsDep
) -> dict:
    deleted = await classifier_service.delete_training_job(
        session, job_id, settings.storage_root
    )
    if not deleted:
        raise HTTPException(404, "Training job not found")
    return {"status": "deleted"}


@router.post("/training-jobs/bulk-delete")
async def bulk_delete_training_jobs(
    body: BulkDeleteRequest, session: SessionDep, settings: SettingsDep
) -> dict:
    count = await classifier_service.bulk_delete_training_jobs(
        session, body.ids, settings.storage_root
    )
    return {"status": "deleted", "count": count}


# ---- Training Data Summary ----


@router.get("/models/{model_id}/training-summary")
async def get_training_summary(
    model_id: str,
    session: SessionDep,
) -> TrainingDataSummaryResponse:
    """Return training data provenance for a classifier model."""
    summary = await classifier_service.get_training_data_summary(session, model_id)
    if summary is None:
        raise HTTPException(404, "Model or training job not found")

    return TrainingDataSummaryResponse(
        model_id=summary["model_id"],
        model_name=summary["model_name"],
        positive_sources=[TrainingSourceInfo(**s) for s in summary["positive_sources"]],
        negative_sources=[TrainingSourceInfo(**s) for s in summary["negative_sources"]],
        total_positive=summary["total_positive"],
        total_negative=summary["total_negative"],
        balance_ratio=summary["balance_ratio"],
        window_size_seconds=summary["window_size_seconds"],
        positive_duration_sec=summary["positive_duration_sec"],
        negative_duration_sec=summary["negative_duration_sec"],
    )


# ---- Retrain Workflows ----


@router.get("/models/{model_id}/retrain-info")
async def get_retrain_info(model_id: str, session: SessionDep) -> RetrainFolderInfo:
    info = await classifier_service.get_retrain_info(session, model_id)
    if info is None:
        raise HTTPException(404, "Model or training job not found")
    return RetrainFolderInfo(**info)


@router.post("/retrain", status_code=201)
async def create_retrain_workflow(
    body: RetrainWorkflowCreate, session: SessionDep
) -> RetrainWorkflowOut:
    try:
        wf = await classifier_service.create_retrain_workflow(
            session, body.source_model_id, body.new_model_name, body.parameters
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _retrain_workflow_to_out(wf)


@router.get("/retrain-workflows")
async def list_retrain_workflows(
    session: SessionDep,
) -> list[RetrainWorkflowOut]:
    wfs = await classifier_service.list_retrain_workflows(session)
    return [_retrain_workflow_to_out(wf) for wf in wfs]


@router.get("/retrain-workflows/{workflow_id}")
async def get_retrain_workflow(
    workflow_id: str, session: SessionDep
) -> RetrainWorkflowOut:
    wf = await classifier_service.get_retrain_workflow(session, workflow_id)
    if wf is None:
        raise HTTPException(404, "Retrain workflow not found")
    return _retrain_workflow_to_out(wf)
