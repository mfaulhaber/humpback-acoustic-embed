"""Phase 0 API router for the call parsing pipeline.

Phase 0 ships the parent-run CRUD, pure DB list/get/delete for each
pass job table, and 501-Not-Implemented shells for every endpoint whose
body requires pass logic (creating pass jobs directly, exporting
sequences, streaming per-pass parquet artifacts). The 501 detail
messages name the owning Pass so the frontend surfaces a clear
placeholder until that Pass lands.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from humpback.api.deps import SessionDep, SettingsDep
from humpback.schemas.call_parsing import (
    CallParsingRunCreate,
    CallParsingRunResponse,
    EventClassificationJobSummary,
    EventSegmentationJobSummary,
    RegionDetectionJobSummary,
)
from humpback.services import call_parsing as service

router = APIRouter(prefix="/call-parsing", tags=["call-parsing"])


def _run_to_response(run, rd, es, ec) -> CallParsingRunResponse:
    return CallParsingRunResponse(
        id=run.id,
        audio_source_id=run.audio_source_id,
        status=run.status,
        config_snapshot=run.config_snapshot,
        error_message=run.error_message,
        created_at=run.created_at,
        updated_at=run.updated_at,
        completed_at=run.completed_at,
        region_detection_job=(
            RegionDetectionJobSummary.model_validate(rd) if rd is not None else None
        ),
        event_segmentation_job=(
            EventSegmentationJobSummary.model_validate(es) if es is not None else None
        ),
        event_classification_job=(
            EventClassificationJobSummary.model_validate(ec) if ec is not None else None
        ),
    )


# ---- Parent runs --------------------------------------------------------


@router.post("/runs", status_code=201, response_model=CallParsingRunResponse)
async def create_run(body: CallParsingRunCreate, session: SessionDep):
    run = await service.create_parent_run(
        session,
        audio_source_id=body.audio_source_id,
        config_snapshot=body.config_snapshot,
    )
    loaded = await service.load_run_with_children(session, run.id)
    assert loaded is not None
    return _run_to_response(*loaded)


@router.get("/runs", response_model=list[CallParsingRunResponse])
async def list_runs(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    runs = await service.list_parent_runs(session, limit=limit, offset=offset)
    responses: list[CallParsingRunResponse] = []
    for run in runs:
        rd, es, ec = await service._load_child_jobs(session, run)
        responses.append(_run_to_response(run, rd, es, ec))
    return responses


@router.get("/runs/{run_id}", response_model=CallParsingRunResponse)
async def get_run(run_id: str, session: SessionDep):
    loaded = await service.load_run_with_children(session, run_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_response(*loaded)


@router.delete("/runs/{run_id}", status_code=204)
async def delete_run(run_id: str, session: SessionDep, settings: SettingsDep):
    deleted = await service.delete_parent_run(session, run_id, settings)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return None


@router.get("/runs/{run_id}/sequence")
async def get_run_sequence(run_id: str):
    return JSONResponse(
        status_code=501,
        content={
            "detail": "Pass 4 (sequence export) not yet implemented",
        },
    )


# ---- Pass 1: region detection jobs --------------------------------------


@router.post("/region-jobs")
async def create_region_job():
    return JSONResponse(
        status_code=501,
        content={
            "detail": "Pass 1 (region detection) creation not yet implemented",
        },
    )


@router.get("/region-jobs", response_model=list[RegionDetectionJobSummary])
async def list_region_jobs(session: SessionDep):
    jobs = await service.list_region_detection_jobs(session)
    return [RegionDetectionJobSummary.model_validate(j) for j in jobs]


@router.get("/region-jobs/{job_id}", response_model=RegionDetectionJobSummary)
async def get_region_job(job_id: str, session: SessionDep):
    job = await service.get_region_detection_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Region detection job not found")
    return RegionDetectionJobSummary.model_validate(job)


@router.delete("/region-jobs/{job_id}", status_code=204)
async def delete_region_job(job_id: str, session: SessionDep, settings: SettingsDep):
    deleted = await service.delete_region_detection_job(session, job_id, settings)
    if not deleted:
        raise HTTPException(status_code=404, detail="Region detection job not found")
    return None


@router.get("/region-jobs/{job_id}/trace")
async def get_region_trace(job_id: str):
    return JSONResponse(
        status_code=501,
        content={
            "detail": "Pass 1 (region detection) trace access not yet implemented"
        },
    )


@router.get("/region-jobs/{job_id}/regions")
async def get_region_regions(job_id: str):
    return JSONResponse(
        status_code=501,
        content={
            "detail": "Pass 1 (region detection) region access not yet implemented"
        },
    )


# ---- Pass 2: segmentation jobs ------------------------------------------


@router.post("/segmentation-jobs")
async def create_segmentation_job():
    return JSONResponse(
        status_code=501,
        content={
            "detail": "Pass 2 (event segmentation) creation not yet implemented",
        },
    )


@router.get("/segmentation-jobs", response_model=list[EventSegmentationJobSummary])
async def list_segmentation_jobs(session: SessionDep):
    jobs = await service.list_event_segmentation_jobs(session)
    return [EventSegmentationJobSummary.model_validate(j) for j in jobs]


@router.get("/segmentation-jobs/{job_id}", response_model=EventSegmentationJobSummary)
async def get_segmentation_job(job_id: str, session: SessionDep):
    job = await service.get_event_segmentation_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Event segmentation job not found")
    return EventSegmentationJobSummary.model_validate(job)


@router.delete("/segmentation-jobs/{job_id}", status_code=204)
async def delete_segmentation_job(
    job_id: str, session: SessionDep, settings: SettingsDep
):
    deleted = await service.delete_event_segmentation_job(session, job_id, settings)
    if not deleted:
        raise HTTPException(status_code=404, detail="Event segmentation job not found")
    return None


@router.get("/segmentation-jobs/{job_id}/events")
async def get_segmentation_events(job_id: str):
    return JSONResponse(
        status_code=501,
        content={
            "detail": "Pass 2 (event segmentation) event access not yet implemented"
        },
    )


# ---- Pass 3: classification jobs ----------------------------------------


@router.post("/classification-jobs")
async def create_classification_job():
    return JSONResponse(
        status_code=501,
        content={
            "detail": "Pass 3 (event classification) creation not yet implemented",
        },
    )


@router.get("/classification-jobs", response_model=list[EventClassificationJobSummary])
async def list_classification_jobs(session: SessionDep):
    jobs = await service.list_event_classification_jobs(session)
    return [EventClassificationJobSummary.model_validate(j) for j in jobs]


@router.get(
    "/classification-jobs/{job_id}", response_model=EventClassificationJobSummary
)
async def get_classification_job(job_id: str, session: SessionDep):
    job = await service.get_event_classification_job(session, job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="Event classification job not found"
        )
    return EventClassificationJobSummary.model_validate(job)


@router.delete("/classification-jobs/{job_id}", status_code=204)
async def delete_classification_job(
    job_id: str, session: SessionDep, settings: SettingsDep
):
    deleted = await service.delete_event_classification_job(session, job_id, settings)
    if not deleted:
        raise HTTPException(
            status_code=404, detail="Event classification job not found"
        )
    return None


@router.get("/classification-jobs/{job_id}/typed-events")
async def get_classification_typed_events(job_id: str):
    return JSONResponse(
        status_code=501,
        content={
            "detail": (
                "Pass 3 (event classification) typed-event access not yet implemented"
            )
        },
    )
