"""Hydrophone detection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from humpback.api.deps import SessionDep
from humpback.schemas.classifier import (
    DetectionJobOut,
    HydrophoneDetectionJobCreate,
    HydrophoneInfo,
)
from humpback.schemas.converters import detection_job_to_out as _detection_job_to_out
from humpback.services import classifier_service

router = APIRouter()


@router.get("/hydrophones")
async def list_hydrophones() -> list[HydrophoneInfo]:
    """List configured archive sources on the legacy hydrophone endpoint."""
    from humpback.config import HYDROPHONE_UI_SOURCES

    return [
        HydrophoneInfo(
            id=source["id"],
            name=source["name"],
            location=source["location"],
            provider_kind=source["provider_kind"],
        )
        for source in HYDROPHONE_UI_SOURCES
    ]


@router.post("/hydrophone-detection-jobs", status_code=201)
async def create_hydrophone_detection_job(
    body: HydrophoneDetectionJobCreate, session: SessionDep
) -> DetectionJobOut:
    try:
        job = await classifier_service.create_hydrophone_detection_job(
            session,
            body.classifier_model_id,
            body.hydrophone_id,
            body.start_timestamp,
            body.end_timestamp,
            body.confidence_threshold,
            body.hop_seconds,
            body.high_threshold,
            body.low_threshold,
            body.local_cache_path,
            body.window_selection,
            body.min_prominence,
            body.max_logit_drop,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _detection_job_to_out(job)


@router.get("/hydrophone-detection-jobs")
async def list_hydrophone_detection_jobs(
    session: SessionDep,
) -> list[DetectionJobOut]:
    jobs = await classifier_service.list_hydrophone_detection_jobs(session)
    return [_detection_job_to_out(j) for j in jobs]


@router.post("/hydrophone-detection-jobs/{job_id}/cancel")
async def cancel_hydrophone_detection_job(job_id: str, session: SessionDep) -> dict:
    try:
        job = await classifier_service.cancel_hydrophone_detection_job(session, job_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if job is None:
        raise HTTPException(404, "Hydrophone detection job not found")
    return {"status": "canceled"}


@router.post("/hydrophone-detection-jobs/{job_id}/pause")
async def pause_hydrophone_detection_job(job_id: str, session: SessionDep) -> dict:
    try:
        job = await classifier_service.pause_hydrophone_detection_job(session, job_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if job is None:
        raise HTTPException(404, "Hydrophone detection job not found")
    return {"status": "paused"}


@router.post("/hydrophone-detection-jobs/{job_id}/resume")
async def resume_hydrophone_detection_job(job_id: str, session: SessionDep) -> dict:
    try:
        job = await classifier_service.resume_hydrophone_detection_job(session, job_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if job is None:
        raise HTTPException(404, "Hydrophone detection job not found")
    return {"status": "running"}
