"""FastAPI router for retained Sequence Models / Continuous Embedding APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response

from humpback.api.deps import SessionDep, SettingsDep
from humpback.schemas.sequence_models import (
    ContinuousEmbeddingJobCreate,
    ContinuousEmbeddingJobDetail,
    ContinuousEmbeddingJobManifest,
    ContinuousEmbeddingJobOut,
    EventEncoderJobCreate,
    EventEncoderJobDetail,
    EventEncoderJobOut,
)
from humpback.services.continuous_embedding_service import (
    CancelTerminalJobError,
    cancel_continuous_embedding_job,
    create_continuous_embedding_job,
    delete_continuous_embedding_job,
    get_continuous_embedding_job,
    list_continuous_embedding_jobs,
)
from humpback.services.event_encoder_service import (
    CancelEventEncoderTerminalJobError,
    cancel_event_encoder_job,
    create_event_encoder_job,
    delete_event_encoder_job,
    get_event_encoder_job,
    list_event_encoder_jobs,
)

router = APIRouter(prefix="/sequence-models", tags=["sequence-models"])


def _to_out(job) -> ContinuousEmbeddingJobOut:
    return ContinuousEmbeddingJobOut.model_validate(job)


def _to_event_encoder_out(job) -> EventEncoderJobOut:
    return EventEncoderJobOut.model_validate(job)


@router.post("/continuous-embeddings")
async def create_continuous_embedding(
    body: ContinuousEmbeddingJobCreate,
    session: SessionDep,
    response: Response,
) -> ContinuousEmbeddingJobOut:
    try:
        job, created = await create_continuous_embedding_job(session, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    response.status_code = 201 if created else 200
    return _to_out(job)


@router.get("/continuous-embeddings")
async def list_continuous_embeddings(
    session: SessionDep,
    status: Optional[str] = Query(default=None),
) -> list[ContinuousEmbeddingJobOut]:
    jobs = await list_continuous_embedding_jobs(session, status=status)
    return [_to_out(j) for j in jobs]


@router.get("/continuous-embeddings/{job_id}")
async def get_continuous_embedding(
    job_id: str,
    session: SessionDep,
) -> ContinuousEmbeddingJobDetail:
    job = await get_continuous_embedding_job(session, job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="continuous embedding job not found"
        )

    manifest: Optional[ContinuousEmbeddingJobManifest] = None
    if job.parquet_path:
        manifest_path = Path(job.parquet_path).with_name("manifest.json")
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = ContinuousEmbeddingJobManifest.model_validate(payload)
            except Exception:
                manifest = None

    return ContinuousEmbeddingJobDetail(job=_to_out(job), manifest=manifest)


@router.post("/continuous-embeddings/{job_id}/cancel")
async def cancel_continuous_embedding(
    job_id: str,
    session: SessionDep,
) -> ContinuousEmbeddingJobOut:
    try:
        job = await cancel_continuous_embedding_job(session, job_id)
    except CancelTerminalJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if job is None:
        raise HTTPException(
            status_code=404, detail="continuous embedding job not found"
        )
    return _to_out(job)


@router.delete("/continuous-embeddings/{job_id}", status_code=204)
async def delete_continuous_embedding(
    job_id: str, session: SessionDep, settings: SettingsDep
):
    deleted = await delete_continuous_embedding_job(session, job_id, settings)
    if not deleted:
        raise HTTPException(
            status_code=404, detail="continuous embedding job not found"
        )
    return None


@router.post("/event-encoders")
async def create_event_encoder(
    body: EventEncoderJobCreate,
    session: SessionDep,
    response: Response,
) -> EventEncoderJobOut:
    try:
        job, created = await create_event_encoder_job(session, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    response.status_code = 201 if created else 200
    return _to_event_encoder_out(job)


@router.get("/event-encoders")
async def list_event_encoders(
    session: SessionDep,
    status: Optional[str] = Query(default=None),
) -> list[EventEncoderJobOut]:
    jobs = await list_event_encoder_jobs(session, status=status)
    return [_to_event_encoder_out(j) for j in jobs]


@router.get("/event-encoders/{job_id}")
async def get_event_encoder(job_id: str, session: SessionDep) -> EventEncoderJobDetail:
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")

    manifest = _load_json_sidecar(job.manifest_path)
    report = _load_json_sidecar(job.report_path)
    return EventEncoderJobDetail(
        job=_to_event_encoder_out(job),
        manifest=manifest,
        report=report,
    )


@router.post("/event-encoders/{job_id}/cancel")
async def cancel_event_encoder(
    job_id: str,
    session: SessionDep,
) -> EventEncoderJobOut:
    try:
        job = await cancel_event_encoder_job(session, job_id)
    except CancelEventEncoderTerminalJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")
    return _to_event_encoder_out(job)


@router.delete("/event-encoders/{job_id}", status_code=204)
async def delete_event_encoder(job_id: str, session: SessionDep, settings: SettingsDep):
    deleted = await delete_event_encoder_job(session, job_id, settings)
    if not deleted:
        raise HTTPException(status_code=404, detail="event encoder job not found")
    return None


def _load_json_sidecar(path_value: Optional[str]) -> Optional[dict]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None
