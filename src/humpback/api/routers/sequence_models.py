"""FastAPI router for the Sequence Models track.

Mounts under ``/sequence-models/`` and exposes endpoints for both the
continuous-embedding producer (PR 1) and HMM sequence jobs (PR 2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Query, Response

from humpback.api.deps import SessionDep, SettingsDep
from humpback.config import Settings
from humpback.models.processing import JobStatus
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
    MotifExtractionJobCreate,
    MotifExtractionJobDetail,
    MotifExtractionJobOut,
    MotifExtractionManifest,
    MotifOccurrence,
    MotifOccurrencesResponse,
    MotifsResponse,
    MotifSummary,
    OverlayPoint,
    OverlayResponse,
    StateTierComposition,
    TransitionMatrixResponse,
)
from humpback.services.continuous_embedding_service import (
    SOURCE_KIND_REGION_CRNN,
    source_kind_for,
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
from humpback.services.motif_extraction_service import (
    CancelTerminalJobError as MotifCancelTerminalJobError,
    cancel_motif_extraction_job,
    create_motif_extraction_job,
    delete_motif_extraction_job,
    get_motif_extraction_job,
    list_motif_extraction_jobs,
)
from humpback.storage import (
    hmm_sequence_exemplars_path,
    hmm_sequence_label_distribution_path,
    hmm_sequence_overlay_path,
    hmm_sequence_states_path,
    hmm_sequence_summary_path,
    hmm_sequence_transition_matrix_path,
    motif_extraction_manifest_path,
    motif_extraction_motifs_path,
    motif_extraction_occurrences_path,
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


# ---------------------------------------------------------------------------
# HMM Sequence Jobs (PR 2)
# ---------------------------------------------------------------------------


def _hmm_to_out(job) -> HMMSequenceJobOut:
    return HMMSequenceJobOut.model_validate(job)


def _motif_to_out(job) -> MotifExtractionJobOut:
    return MotifExtractionJobOut.model_validate(job)


def _load_summary(settings: Settings, job_id: str) -> list[HMMStateSummary] | None:
    summary_path = hmm_sequence_summary_path(settings.storage_root, job_id)
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return [HMMStateSummary.model_validate(s) for s in payload.get("states", [])]
    except Exception:
        return None


def _load_tier_composition(
    settings: Settings, job_id: str
) -> list[StateTierComposition] | None:
    """Return per-state tier composition from ``state_summary.json``."""
    summary_path = hmm_sequence_summary_path(settings.storage_root, job_id)
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        raw = payload.get("tier_composition")
        if not isinstance(raw, list):
            return None
        return [StateTierComposition.model_validate(entry) for entry in raw]
    except Exception:
        return None


def _require_columns(table: pa.Table, path: Path, columns: set[str]) -> None:
    missing = sorted(columns.difference(table.column_names))
    if missing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{path.name} is missing canonical timestamp columns: "
                f"{', '.join(missing)}"
            ),
        )


@router.post("/hmm-sequences", status_code=201)
async def create_hmm_sequence(
    body: HMMSequenceJobCreate,
    session: SessionDep,
) -> HMMSequenceJobOut:
    try:
        job = await create_hmm_sequence_job(session, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
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
    from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
    from humpback.models.sequence_models import ContinuousEmbeddingJob

    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")

    cej = await session.get(ContinuousEmbeddingJob, job.continuous_embedding_job_id)
    region_detection_job_id = ""
    rdj = None
    source_kind = "surfperch"
    if cej:
        source_kind = source_kind_for(cej.model_version)
        if source_kind == SOURCE_KIND_REGION_CRNN and cej.region_detection_job_id:
            region_detection_job_id = cej.region_detection_job_id
            rdj = await session.get(RegionDetectionJob, region_detection_job_id)
        elif cej.event_segmentation_job_id:
            seg_job = await session.get(
                EventSegmentationJob, cej.event_segmentation_job_id
            )
            if seg_job:
                region_detection_job_id = seg_job.region_detection_job_id
                rdj = await session.get(RegionDetectionJob, region_detection_job_id)

    settings = Settings.from_repo_env()
    summary = _load_summary(settings, job_id)
    tier_composition = (
        _load_tier_composition(settings, job_id)
        if source_kind == SOURCE_KIND_REGION_CRNN
        else None
    )
    return HMMSequenceJobDetail(
        job=_hmm_to_out(job),
        region_detection_job_id=region_detection_job_id,
        region_start_timestamp=rdj.start_timestamp if rdj else None,
        region_end_timestamp=rdj.end_timestamp if rdj else None,
        summary=summary,
        tier_composition=tier_composition,
        source_kind=source_kind,
    )


@router.get("/hmm-sequences/{job_id}/states")
async def get_hmm_states(
    job_id: str,
    session: SessionDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=50000),
) -> dict:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    settings = Settings.from_repo_env()
    states_path = hmm_sequence_states_path(settings.storage_root, job_id)
    if not states_path.exists():
        raise HTTPException(status_code=404, detail="states.parquet not found")
    table = pq.read_table(states_path)
    _require_columns(table, states_path, {"start_timestamp", "end_timestamp"})
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


def _project_legacy_overlay_row(row: dict) -> dict:
    """Read-time adapter for pre-ADR-059 overlay parquets (transitional).

    Pre-PR SurfPerch overlay parquets carry ``merged_span_id`` (int) and
    ``window_index_in_span`` (int); the unified shape uses
    ``sequence_id`` (string) and ``position_in_sequence`` (int). New
    artifacts already match the unified shape so this adapter is a no-op
    for them. Disk files are never rewritten.
    """
    if "sequence_id" in row and "position_in_sequence" in row:
        return row
    projected = {k: v for k, v in row.items()}
    if "merged_span_id" in projected:
        projected["sequence_id"] = str(projected.pop("merged_span_id"))
    if "window_index_in_span" in projected:
        projected["position_in_sequence"] = projected.pop("window_index_in_span")
    return projected


@router.get("/hmm-sequences/{job_id}/overlay")
async def get_hmm_overlay(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=5000, ge=1, le=50000),
) -> OverlayResponse:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    overlay_path = hmm_sequence_overlay_path(settings.storage_root, job_id)
    if not overlay_path.exists():
        raise HTTPException(status_code=404, detail="overlay not found")
    table = pq.read_table(overlay_path)
    _require_columns(table, overlay_path, {"start_timestamp", "end_timestamp"})
    total = table.num_rows
    sliced = table.slice(offset, limit)
    rows = sliced.to_pydict()
    items = [
        OverlayPoint(**_project_legacy_overlay_row({col: rows[col][i] for col in rows}))
        for i in range(sliced.num_rows)
    ]
    return OverlayResponse(total=total, items=items)


def _project_legacy_label_distribution(payload: dict) -> dict:
    """Read-time adapter for pre-ADR-060 flat label-distribution JSON (transitional).

    Pre-PR SurfPerch label-distribution files have the shape
    ``states[state] = {label: count}``; the unified shape (ADR-060) is
    ``states[state] = {tier: {label: count}}``. New artifacts already match
    the unified shape so this adapter is a no-op for them. Disk files are
    never rewritten by this code path; the Refresh button rewrites them in
    unified format on demand.
    """
    states = payload.get("states")
    if not isinstance(states, dict):
        return payload
    projected_states: dict[str, dict[str, dict[str, int]]] = {}
    for state_key, inner in states.items():
        if not isinstance(inner, dict) or not inner:
            projected_states[state_key] = inner if isinstance(inner, dict) else {}
            continue
        # Detection rule: legacy flat shape's first inner value is int;
        # unified nested shape's first inner value is dict.
        first_value = next(iter(inner.values()))
        if isinstance(first_value, dict):
            projected_states[state_key] = inner
        else:
            projected_states[state_key] = {"all": inner}
    return {**payload, "states": projected_states}


@router.get("/hmm-sequences/{job_id}/label-distribution")
async def get_hmm_label_distribution(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> LabelDistributionResponse:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    if job.status != "complete":
        raise HTTPException(status_code=400, detail="job not complete")
    dist_path = hmm_sequence_label_distribution_path(settings.storage_root, job_id)
    if dist_path.exists():
        payload = json.loads(dist_path.read_text(encoding="utf-8"))
        return LabelDistributionResponse.model_validate(
            _project_legacy_label_distribution(payload)
        )
    dist = await generate_label_distribution(session, settings.storage_root, job)
    return LabelDistributionResponse.model_validate(dist)


def _project_legacy_exemplar_row(row: dict) -> dict:
    """Read-time adapter for pre-ADR-059 exemplar JSON records (transitional).

    Pre-PR SurfPerch records carry ``merged_span_id`` (int) and
    ``window_index_in_span`` (int); the unified shape uses
    ``sequence_id`` (string) and ``position_in_sequence`` (int) plus an
    ``extras`` dict. New artifacts already match the unified shape so
    this adapter is a no-op for them. Disk files are never rewritten.
    """
    if "sequence_id" in row and "position_in_sequence" in row:
        if "extras" not in row:
            projected = {k: v for k, v in row.items()}
            projected["extras"] = {}
            return projected
        return row
    projected = {k: v for k, v in row.items()}
    if "merged_span_id" in projected:
        projected["sequence_id"] = str(projected.pop("merged_span_id"))
    if "window_index_in_span" in projected:
        projected["position_in_sequence"] = projected.pop("window_index_in_span")
    if "extras" not in projected:
        projected["extras"] = {}
    return projected


@router.get("/hmm-sequences/{job_id}/exemplars")
async def get_hmm_exemplars(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> ExemplarsResponse:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    exemplars_path = hmm_sequence_exemplars_path(settings.storage_root, job_id)
    if not exemplars_path.exists():
        raise HTTPException(status_code=404, detail="exemplars not found")
    payload = json.loads(exemplars_path.read_text(encoding="utf-8"))
    return ExemplarsResponse(
        n_states=payload["n_states"],
        states={
            k: [
                ExemplarRecord.model_validate(_project_legacy_exemplar_row(r))
                for r in v
            ]
            for k, v in payload["states"].items()
        },
    )


@router.post("/hmm-sequences/{job_id}/generate-interpretations")
async def regenerate_interpretations(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> dict:
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    if job.status != "complete":
        raise HTTPException(status_code=400, detail="job not complete")

    from humpback.models.sequence_models import ContinuousEmbeddingJob

    cej = await session.get(ContinuousEmbeddingJob, job.continuous_embedding_job_id)
    if cej is None:
        raise HTTPException(
            status_code=400, detail="source continuous embedding job not found"
        )

    generate_interpretations(settings.storage_root, job, cej)
    await generate_label_distribution(session, settings.storage_root, job)

    return {
        "status": "ok",
        "job_id": job_id,
        "label_distribution_generated": True,
    }


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


# ---------------------------------------------------------------------------
# Motif Extraction Jobs
# ---------------------------------------------------------------------------


@router.post("/motif-extractions")
async def create_motif_extraction(
    body: MotifExtractionJobCreate,
    session: SessionDep,
    response: Response,
) -> MotifExtractionJobOut:
    try:
        job, created = await create_motif_extraction_job(session, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    response.status_code = 201 if created else 200
    return _motif_to_out(job)


@router.get("/motif-extractions")
async def list_motif_extractions(
    session: SessionDep,
    status: Optional[str] = Query(default=None),
    hmm_sequence_job_id: Optional[str] = Query(default=None),
) -> list[MotifExtractionJobOut]:
    jobs = await list_motif_extraction_jobs(
        session,
        status=status,
        hmm_sequence_job_id=hmm_sequence_job_id,
    )
    return [_motif_to_out(j) for j in jobs]


@router.get("/motif-extractions/{job_id}")
async def get_motif_extraction(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> MotifExtractionJobDetail:
    job = await get_motif_extraction_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="motif extraction job not found")
    manifest = None
    manifest_path = motif_extraction_manifest_path(settings.storage_root, job_id)
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = MotifExtractionManifest.model_validate(payload)
        except Exception:
            manifest = None
    return MotifExtractionJobDetail(job=_motif_to_out(job), manifest=manifest)


@router.post("/motif-extractions/{job_id}/cancel")
async def cancel_motif_extraction(
    job_id: str,
    session: SessionDep,
) -> MotifExtractionJobOut:
    try:
        job = await cancel_motif_extraction_job(session, job_id)
    except MotifCancelTerminalJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if job is None:
        raise HTTPException(status_code=404, detail="motif extraction job not found")
    return _motif_to_out(job)


@router.delete("/motif-extractions/{job_id}", status_code=204)
async def delete_motif_extraction(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
):
    deleted = await delete_motif_extraction_job(session, job_id, settings)
    if not deleted:
        raise HTTPException(status_code=404, detail="motif extraction job not found")
    return None


def _parquet_items(path: Path, offset: int, limit: int) -> tuple[int, list[dict]]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found")
    table = pq.read_table(path)
    total = table.num_rows
    sliced = table.slice(offset, limit)
    rows = sliced.to_pydict()
    return total, [{col: rows[col][i] for col in rows} for i in range(sliced.num_rows)]


@router.get("/motif-extractions/{job_id}/motifs")
async def get_motifs(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=5000),
) -> MotifsResponse:
    job = await get_motif_extraction_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="motif extraction job not found")
    if job.status != JobStatus.complete.value:
        raise HTTPException(status_code=400, detail="job not complete")
    total, items = _parquet_items(
        motif_extraction_motifs_path(settings.storage_root, job_id), offset, limit
    )
    return MotifsResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=[MotifSummary.model_validate(item) for item in items],
    )


@router.get("/motif-extractions/{job_id}/motifs/{motif_key}/occurrences")
async def get_motif_occurrences(
    job_id: str,
    motif_key: str,
    session: SessionDep,
    settings: SettingsDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=5000),
) -> MotifOccurrencesResponse:
    job = await get_motif_extraction_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="motif extraction job not found")
    if job.status != JobStatus.complete.value:
        raise HTTPException(status_code=400, detail="job not complete")
    path = motif_extraction_occurrences_path(settings.storage_root, job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="occurrences.parquet not found")
    table = pq.read_table(path)
    data = table.to_pylist()
    filtered = [row for row in data if row.get("motif_key") == motif_key]
    total = len(filtered)
    sliced = filtered[offset : offset + limit]
    return MotifOccurrencesResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=[MotifOccurrence.model_validate(item) for item in sliced],
    )
