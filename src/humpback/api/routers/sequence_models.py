"""FastAPI router for the Sequence Models track.

Mounts under ``/sequence-models/`` and exposes endpoints for both the
continuous-embedding producer (PR 1) and HMM sequence jobs (PR 2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Query, Response

from humpback.api.deps import SessionDep, SettingsDep
from humpback.config import Settings
from humpback.schemas.sequence_models import (
    ContinuousEmbeddingJobCreate,
    ContinuousEmbeddingJobDetail,
    ContinuousEmbeddingJobManifest,
    ContinuousEmbeddingJobOut,
    DwellHistogramResponse,
    ExemplarRecord,
    ExemplarsResponse,
    HMMSequenceJobCreate,
    HMMSequenceJobDetail,
    HMMSequenceJobOut,
    HMMStateSummary,
    LabelDistributionResponse,
    OverlayPoint,
    OverlayResponse,
    TransitionMatrixResponse,
)
from humpback.services.continuous_embedding_service import (
    CancelTerminalJobError,
    cancel_continuous_embedding_job,
    create_continuous_embedding_job,
    delete_continuous_embedding_job,
    get_continuous_embedding_job,
    list_continuous_embedding_jobs,
)
from humpback.services.hmm_sequence_service import (
    CancelTerminalJobError as HMMCancelTerminalJobError,
    cancel_hmm_sequence_job,
    create_hmm_sequence_job,
    delete_hmm_sequence_job,
    generate_interpretations,
    generate_label_distribution,
    get_hmm_sequence_job,
    list_hmm_sequence_jobs,
)
from humpback.storage import (
    hmm_sequence_exemplars_path,
    hmm_sequence_label_distribution_path,
    hmm_sequence_overlay_path,
    hmm_sequence_states_path,
    hmm_sequence_summary_path,
    hmm_sequence_transition_matrix_path,
)

router = APIRouter(prefix="/sequence-models", tags=["sequence-models"])


def _to_out(job) -> ContinuousEmbeddingJobOut:
    return ContinuousEmbeddingJobOut.model_validate(job)


@router.post("/continuous-embeddings")
async def create_continuous_embedding(
    body: ContinuousEmbeddingJobCreate,
    session: SessionDep,
    response: Response,
) -> ContinuousEmbeddingJobOut:
    try:
        job, created = await create_continuous_embedding_job(session, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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


# ---------------------------------------------------------------------------
# HMM Sequence Jobs (PR 2)
# ---------------------------------------------------------------------------


def _hmm_to_out(job) -> HMMSequenceJobOut:
    return HMMSequenceJobOut.model_validate(job)


def _load_summary(settings: Settings, job_id: str) -> list[HMMStateSummary] | None:
    summary_path = hmm_sequence_summary_path(settings.storage_root, job_id)
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return [HMMStateSummary.model_validate(s) for s in payload.get("states", [])]
    except Exception:
        return None


@router.post("/hmm-sequences", status_code=201)
async def create_hmm_sequence(
    body: HMMSequenceJobCreate,
    session: SessionDep,
) -> HMMSequenceJobOut:
    try:
        job = await create_hmm_sequence_job(session, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _hmm_to_out(job)


@router.get("/hmm-sequences")
async def list_hmm_sequences(
    session: SessionDep,
    status: Optional[str] = Query(default=None),
    continuous_embedding_job_id: Optional[str] = Query(default=None),
) -> list[HMMSequenceJobOut]:
    jobs = await list_hmm_sequence_jobs(
        session,
        status=status,
        continuous_embedding_job_id=continuous_embedding_job_id,
    )
    return [_hmm_to_out(j) for j in jobs]


@router.get("/hmm-sequences/{job_id}")
async def get_hmm_sequence(
    job_id: str,
    session: SessionDep,
) -> HMMSequenceJobDetail:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    settings = Settings.from_repo_env()
    summary = _load_summary(settings, job_id)
    return HMMSequenceJobDetail(job=_hmm_to_out(job), summary=summary)


@router.get("/hmm-sequences/{job_id}/states")
async def get_hmm_states(
    job_id: str,
    session: SessionDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    settings = Settings.from_repo_env()
    states_path = hmm_sequence_states_path(settings.storage_root, job_id)
    if not states_path.exists():
        raise HTTPException(status_code=404, detail="states.parquet not found")
    table = pq.read_table(states_path)
    total = table.num_rows
    sliced = table.slice(offset, limit)
    rows = sliced.to_pydict()
    items = []
    for i in range(sliced.num_rows):
        items.append({col: rows[col][i] for col in rows})
    return {"total": total, "offset": offset, "limit": limit, "items": items}


@router.get("/hmm-sequences/{job_id}/transitions")
async def get_hmm_transitions(
    job_id: str,
    session: SessionDep,
) -> TransitionMatrixResponse:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    settings = Settings.from_repo_env()
    tm_path = hmm_sequence_transition_matrix_path(settings.storage_root, job_id)
    if not tm_path.exists():
        raise HTTPException(status_code=404, detail="transition matrix not found")
    matrix = np.load(tm_path)
    return TransitionMatrixResponse(
        n_states=matrix.shape[0],
        matrix=matrix.tolist(),
    )


@router.get("/hmm-sequences/{job_id}/dwell")
async def get_hmm_dwell(
    job_id: str,
    session: SessionDep,
) -> DwellHistogramResponse:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    settings = Settings.from_repo_env()
    summary = _load_summary(settings, job_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="state summary not found")
    histograms = {str(s.state): s.dwell_histogram for s in summary}
    return DwellHistogramResponse(n_states=len(summary), histograms=histograms)


# ---------------------------------------------------------------------------
# Interpretation visualizations (PR 3)
# ---------------------------------------------------------------------------


@router.get("/hmm-sequences/{job_id}/overlay")
async def get_hmm_overlay(
    job_id: str,
    session: SessionDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=5000, ge=1, le=50000),
) -> OverlayResponse:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    settings = Settings.from_repo_env()
    overlay_path = hmm_sequence_overlay_path(settings.storage_root, job_id)
    if not overlay_path.exists():
        raise HTTPException(status_code=404, detail="overlay not found")
    table = pq.read_table(overlay_path)
    total = table.num_rows
    sliced = table.slice(offset, limit)
    rows = sliced.to_pydict()
    items = [
        OverlayPoint(**{col: rows[col][i] for col in rows})
        for i in range(sliced.num_rows)
    ]
    return OverlayResponse(total=total, items=items)


@router.get("/hmm-sequences/{job_id}/label-distribution")
async def get_hmm_label_distribution(
    job_id: str,
    session: SessionDep,
) -> LabelDistributionResponse:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    if job.status != "complete":
        raise HTTPException(status_code=400, detail="job not complete")
    settings = Settings.from_repo_env()
    dist_path = hmm_sequence_label_distribution_path(settings.storage_root, job_id)
    if dist_path.exists():
        payload = json.loads(dist_path.read_text(encoding="utf-8"))
        return LabelDistributionResponse.model_validate(payload)
    dist = await generate_label_distribution(session, settings.storage_root, job)
    return LabelDistributionResponse.model_validate(dist)


@router.get("/hmm-sequences/{job_id}/exemplars")
async def get_hmm_exemplars(
    job_id: str,
    session: SessionDep,
) -> ExemplarsResponse:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    settings = Settings.from_repo_env()
    exemplars_path = hmm_sequence_exemplars_path(settings.storage_root, job_id)
    if not exemplars_path.exists():
        raise HTTPException(status_code=404, detail="exemplars not found")
    payload = json.loads(exemplars_path.read_text(encoding="utf-8"))
    return ExemplarsResponse(
        n_states=payload["n_states"],
        states={
            k: [ExemplarRecord.model_validate(r) for r in v]
            for k, v in payload["states"].items()
        },
    )


@router.post("/hmm-sequences/{job_id}/generate-interpretations")
async def regenerate_interpretations(
    job_id: str,
    session: SessionDep,
) -> dict:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    if job.status != "complete":
        raise HTTPException(status_code=400, detail="job not complete")
    settings = Settings.from_repo_env()

    from humpback.models.sequence_models import ContinuousEmbeddingJob

    cej = await session.get(ContinuousEmbeddingJob, job.continuous_embedding_job_id)
    if cej is None:
        raise HTTPException(
            status_code=400, detail="source continuous embedding job not found"
        )

    generate_interpretations(settings.storage_root, job, cej)
    await generate_label_distribution(session, settings.storage_root, job)

    return {"status": "ok", "job_id": job_id}


@router.post("/hmm-sequences/{job_id}/cancel")
async def cancel_hmm_sequence(
    job_id: str,
    session: SessionDep,
) -> HMMSequenceJobOut:
    try:
        job = await cancel_hmm_sequence_job(session, job_id)
    except HMMCancelTerminalJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    return _hmm_to_out(job)


@router.delete("/hmm-sequences/{job_id}", status_code=204)
async def delete_hmm_sequence(job_id: str, session: SessionDep, settings: SettingsDep):
    deleted = await delete_hmm_sequence_job(session, job_id, settings)
    if not deleted:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    return None
