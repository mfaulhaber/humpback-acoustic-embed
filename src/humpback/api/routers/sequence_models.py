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
    ExtendKSweepRequest,
    GenerateInterpretationsRequest,
    HMMSequenceJobCreate,
    HMMSequenceJobDetail,
    HMMSequenceJobOut,
    HMMStateSummary,
    LabelDistributionResponse,
    LossCurveResponse,
    MaskedTransformerNearestNeighborReportRequest,
    MaskedTransformerNearestNeighborReportResponse,
    MaskedTransformerJobCreate,
    MaskedTransformerJobDetail,
    MaskedTransformerJobOut,
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
    ReconstructionErrorResponse,
    ReconstructionErrorRow,
    RegenerateLabelDistributionRequest,
    RunLengthsResponse,
    StateTierComposition,
    TokenRow,
    TokensResponse,
    TransitionMatrixResponse,
)
from humpback.sequence_models import retrieval_diagnostics
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
    get_hmm_sequence_job,
    list_hmm_sequence_jobs,
    regenerate_label_distribution as hmm_regenerate_label_distribution,
)
from humpback.services.masked_transformer_service import (
    CancelTerminalJobError as MTCancelTerminalJobError,
    ExtendKSweepError,
    cancel_masked_transformer_job,
    create_masked_transformer_job,
    delete_masked_transformer_job,
    extend_k_sweep_job,
    generate_interpretations as mt_generate_interpretations,
    get_masked_transformer_job,
    list_masked_transformer_jobs,
    parse_k_values,
    regenerate_label_distribution as mt_regenerate_label_distribution,
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
    masked_transformer_k_decoded_path,
    masked_transformer_k_exemplars_path,
    masked_transformer_k_label_distribution_path,
    masked_transformer_k_overlay_path,
    masked_transformer_k_run_lengths_path,
    masked_transformer_loss_curve_path,
    masked_transformer_reconstruction_error_path,
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
        raise HTTPException(status_code=404, detail="decoded.parquet not found")
    table = pq.read_table(states_path)
    _require_columns(table, states_path, {"start_timestamp", "end_timestamp"})
    total = table.num_rows
    sliced = table.slice(offset, limit)
    rows = sliced.to_pydict()
    items = []
    # API contract preserved: surface the canonical decoded label as
    # ``viterbi_state`` regardless of whether the on-disk column is the
    # new ``label`` (post-ADR-061) or the legacy ``viterbi_state`` name.
    for i in range(sliced.num_rows):
        item: dict[str, object] = {}
        for col, values in rows.items():
            out_col = "viterbi_state" if col == "label" else col
            item[out_col] = values[i]
        items.append(item)
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
        return LabelDistributionResponse.model_validate(payload)
    from humpback.models.sequence_models import ContinuousEmbeddingJob

    cej = await session.get(ContinuousEmbeddingJob, job.continuous_embedding_job_id)
    if cej is None:
        raise HTTPException(
            status_code=400, detail="source continuous embedding job not found"
        )
    dist = await generate_interpretations(session, settings.storage_root, job, cej)
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
    """Legacy endpoint: regenerate against the currently bound Classify FK.

    Equivalent to ``regenerate-label-distribution`` with no body.
    Kept for compatibility with worker-triggered runs.
    """
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    if job.status != "complete":
        raise HTTPException(status_code=400, detail="job not complete")

    try:
        await hmm_regenerate_label_distribution(
            session, settings.storage_root, job, requested_classify_id=None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "ok",
        "job_id": job_id,
        "label_distribution_generated": True,
    }


@router.post("/hmm-sequences/{job_id}/regenerate-label-distribution")
async def regenerate_hmm_label_distribution(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    body: RegenerateLabelDistributionRequest = RegenerateLabelDistributionRequest(),
) -> dict:
    """Rebuild the label-distribution and exemplar artifacts (optional re-bind).

    Body is optional; when absent the current FK is used. When
    ``event_classification_job_id`` is provided, validation runs first,
    artifacts are written via temp-then-rename, and the FK update commits
    only after the artifact write succeeds (spec §6.7).
    """
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="hmm sequence job not found")
    if job.status != "complete":
        raise HTTPException(status_code=400, detail="job not complete")

    try:
        dist = await hmm_regenerate_label_distribution(
            session,
            settings.storage_root,
            job,
            requested_classify_id=body.event_classification_job_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "ok",
        "job_id": job_id,
        "event_classification_job_id": job.event_classification_job_id,
        "label_distribution": dist,
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
    masked_transformer_job_id: Optional[str] = Query(default=None),
    parent_kind: Optional[str] = Query(default=None),
    k: Optional[int] = Query(default=None, ge=2),
) -> list[MotifExtractionJobOut]:
    jobs = await list_motif_extraction_jobs(
        session,
        status=status,
        hmm_sequence_job_id=hmm_sequence_job_id,
        masked_transformer_job_id=masked_transformer_job_id,
        parent_kind=parent_kind,
        k=k,
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


# ---------------------------------------------------------------------------
# Masked Transformer Jobs (ADR-061)
# ---------------------------------------------------------------------------


def _mt_to_out(job) -> MaskedTransformerJobOut:
    return MaskedTransformerJobOut.model_validate(job)


def _resolve_k(job, k_query: Optional[int]) -> int:
    """Resolve the k query parameter, defaulting to first k_values entry."""
    try:
        configured = parse_k_values(job.k_values)
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=409, detail="invalid k_values payload")
    if not configured:
        raise HTTPException(status_code=409, detail="job has no configured k_values")
    if k_query is None:
        return configured[0]
    if int(k_query) not in configured:
        raise HTTPException(
            status_code=404, detail=f"k={k_query} is not in this job's k_values"
        )
    return int(k_query)


@router.post("/masked-transformers", status_code=201)
async def create_masked_transformer(
    body: MaskedTransformerJobCreate,
    session: SessionDep,
    response: Response,
) -> MaskedTransformerJobOut:
    try:
        job, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=body.continuous_embedding_job_id,
            event_classification_job_id=body.event_classification_job_id,
            preset=body.preset,
            mask_fraction=body.mask_fraction,
            span_length_min=body.span_length_min,
            span_length_max=body.span_length_max,
            dropout=body.dropout,
            mask_weight_bias=body.mask_weight_bias,
            cosine_loss_weight=body.cosine_loss_weight,
            batch_size=body.batch_size,
            retrieval_head_enabled=body.retrieval_head_enabled,
            retrieval_dim=body.retrieval_dim,
            retrieval_hidden_dim=body.retrieval_hidden_dim,
            retrieval_l2_normalize=body.retrieval_l2_normalize,
            sequence_construction_mode=body.sequence_construction_mode,
            event_centered_fraction=body.event_centered_fraction,
            pre_event_context_sec=body.pre_event_context_sec,
            post_event_context_sec=body.post_event_context_sec,
            contrastive_loss_weight=body.contrastive_loss_weight,
            contrastive_temperature=body.contrastive_temperature,
            contrastive_label_source=body.contrastive_label_source,
            contrastive_min_events_per_label=body.contrastive_min_events_per_label,
            contrastive_min_regions_per_label=body.contrastive_min_regions_per_label,
            require_cross_region_positive=body.require_cross_region_positive,
            related_label_policy_json=body.related_label_policy_json,
            max_epochs=body.max_epochs,
            early_stop_patience=body.early_stop_patience,
            val_split=body.val_split,
            seed=body.seed,
            k_values=body.k_values,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    response.status_code = 201 if created else 200
    return _mt_to_out(job)


@router.get("/masked-transformers")
async def list_masked_transformers(
    session: SessionDep,
    status: Optional[str] = Query(default=None),
    continuous_embedding_job_id: Optional[str] = Query(default=None),
) -> list[MaskedTransformerJobOut]:
    jobs = await list_masked_transformer_jobs(
        session,
        status=status,
        continuous_embedding_job_id=continuous_embedding_job_id,
    )
    return [_mt_to_out(j) for j in jobs]


@router.get("/masked-transformers/{job_id}")
async def get_masked_transformer(
    job_id: str,
    session: SessionDep,
) -> MaskedTransformerJobDetail:
    from humpback.models.call_parsing import RegionDetectionJob
    from humpback.models.sequence_models import ContinuousEmbeddingJob

    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="masked transformer job not found")

    cej = await session.get(ContinuousEmbeddingJob, job.continuous_embedding_job_id)
    region_detection_job_id: Optional[str] = None
    region_start: Optional[float] = None
    region_end: Optional[float] = None
    if cej is not None and cej.region_detection_job_id:
        region_detection_job_id = cej.region_detection_job_id
        rdj = await session.get(RegionDetectionJob, region_detection_job_id)
        if rdj is not None:
            region_start = rdj.start_timestamp
            region_end = rdj.end_timestamp

    return MaskedTransformerJobDetail(
        job=_mt_to_out(job),
        region_detection_job_id=region_detection_job_id,
        region_start_timestamp=region_start,
        region_end_timestamp=region_end,
        tier_composition=None,
        source_kind=SOURCE_KIND_REGION_CRNN,
    )


@router.post("/masked-transformers/{job_id}/cancel")
async def cancel_masked_transformer(
    job_id: str,
    session: SessionDep,
) -> MaskedTransformerJobOut:
    try:
        job = await cancel_masked_transformer_job(session, job_id)
    except MTCancelTerminalJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if job is None:
        raise HTTPException(status_code=404, detail="masked transformer job not found")
    return _mt_to_out(job)


@router.delete("/masked-transformers/{job_id}", status_code=204)
async def delete_masked_transformer(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
):
    deleted = await delete_masked_transformer_job(session, job_id, settings)
    if not deleted:
        raise HTTPException(status_code=404, detail="masked transformer job not found")
    return None


@router.post("/masked-transformers/{job_id}/extend-k-sweep")
async def extend_k_sweep(
    job_id: str,
    body: ExtendKSweepRequest,
    session: SessionDep,
) -> MaskedTransformerJobOut:
    try:
        job = await extend_k_sweep_job(session, job_id, body.additional_k)
    except ExtendKSweepError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _mt_to_out(job)


@router.post("/masked-transformers/{job_id}/generate-interpretations")
async def generate_masked_transformer_interpretations(
    job_id: str,
    body: GenerateInterpretationsRequest,
    session: SessionDep,
    settings: SettingsDep,
) -> dict:
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="masked transformer job not found")
    if job.status != JobStatus.complete.value:
        raise HTTPException(status_code=400, detail="job not complete")

    configured = parse_k_values(job.k_values)
    requested = body.k_values or configured
    invalid = [k for k in requested if k not in configured]
    if invalid:
        raise HTTPException(
            status_code=422, detail=f"k values not in job's k_values: {invalid}"
        )

    for k in requested:
        await mt_generate_interpretations(session, settings.storage_root, job, int(k))

    return {"status": "ok", "job_id": job_id, "k_values": requested}


@router.post("/masked-transformers/{job_id}/regenerate-label-distribution")
async def regenerate_mt_label_distribution(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    k: Optional[int] = Query(default=None, ge=2),
    body: RegenerateLabelDistributionRequest = RegenerateLabelDistributionRequest(),
) -> dict:
    """Rebuild every per-k label-distribution artifact (optional re-bind).

    The MT regenerate always rebuilds **all** ``k<N>/label_distribution.json``
    files in one shot so the per-k caches stay coherent. The ``k`` query
    parameter selects which payload is returned in the response; when
    omitted, the first configured k is used.
    """
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="masked transformer job not found")
    if job.status != JobStatus.complete.value:
        raise HTTPException(status_code=400, detail="job not complete")

    resolved_k = _resolve_k(job, k)

    try:
        per_k = await mt_regenerate_label_distribution(
            session,
            settings.storage_root,
            job,
            requested_classify_id=body.event_classification_job_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "status": "ok",
        "job_id": job_id,
        "k": resolved_k,
        "event_classification_job_id": job.event_classification_job_id,
        "label_distribution": per_k.get(resolved_k),
    }


@router.post("/masked-transformers/{job_id}/nearest-neighbor-report")
async def get_mt_nearest_neighbor_report(
    job_id: str,
    body: MaskedTransformerNearestNeighborReportRequest,
    session: SessionDep,
    settings: SettingsDep,
) -> MaskedTransformerNearestNeighborReportResponse:
    options = retrieval_diagnostics.RetrievalReportOptions(
        k=body.k,
        embedding_space=body.embedding_space,
        samples=body.samples,
        topn=body.topn,
        seed=body.seed,
        retrieval_modes=tuple(body.retrieval_modes),
        embedding_variants=tuple(body.embedding_variants),
        include_query_rows=body.include_query_rows,
        include_neighbor_rows=body.include_neighbor_rows,
        include_event_level=body.include_event_level,
    )
    try:
        payload = await retrieval_diagnostics.build_nearest_neighbor_report(
            session,
            storage_root=settings.storage_root,
            job_id=job_id,
            options=options,
        )
    except retrieval_diagnostics.RetrievalDiagnosticsNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except retrieval_diagnostics.RetrievalDiagnosticsConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except retrieval_diagnostics.RetrievalDiagnosticsInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return MaskedTransformerNearestNeighborReportResponse.model_validate(payload)


@router.get("/masked-transformers/{job_id}/loss-curve")
async def get_loss_curve(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> LossCurveResponse:
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="masked transformer job not found")
    path = masked_transformer_loss_curve_path(settings.storage_root, job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="loss_curve.json not found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LossCurveResponse.model_validate(payload)


@router.get("/masked-transformers/{job_id}/reconstruction-error")
async def get_reconstruction_error(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=5000, ge=1, le=50000),
) -> ReconstructionErrorResponse:
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="masked transformer job not found")
    path = masked_transformer_reconstruction_error_path(settings.storage_root, job_id)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail="reconstruction_error.parquet not found"
        )
    table = pq.read_table(path)
    total = table.num_rows
    sliced = table.slice(offset, limit)
    rows = sliced.to_pydict()
    items = [
        ReconstructionErrorRow(**{col: rows[col][i] for col in rows})
        for i in range(sliced.num_rows)
    ]
    return ReconstructionErrorResponse(
        total=total, offset=offset, limit=limit, items=items
    )


@router.get("/masked-transformers/{job_id}/tokens")
async def get_tokens(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    k: Optional[int] = Query(default=None, ge=2),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=5000, ge=1, le=50000),
) -> TokensResponse:
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="masked transformer job not found")
    resolved_k = _resolve_k(job, k)
    path = masked_transformer_k_decoded_path(settings.storage_root, job_id, resolved_k)
    if not path.exists():
        raise HTTPException(status_code=404, detail="decoded.parquet not found")
    table = pq.read_table(path)
    total = table.num_rows
    sliced = table.slice(offset, limit)
    rows = sliced.to_pydict()
    items = [
        TokenRow(**{col: rows[col][i] for col in rows if col in TokenRow.model_fields})
        for i in range(sliced.num_rows)
    ]
    return TokensResponse(total=total, offset=offset, limit=limit, items=items)


@router.get("/masked-transformers/{job_id}/overlay")
async def get_mt_overlay(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    k: Optional[int] = Query(default=None, ge=2),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=5000, ge=1, le=50000),
) -> OverlayResponse:
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="masked transformer job not found")
    resolved_k = _resolve_k(job, k)
    path = masked_transformer_k_overlay_path(settings.storage_root, job_id, resolved_k)
    if not path.exists():
        raise HTTPException(status_code=404, detail="overlay.parquet not found")
    table = pq.read_table(path)
    _require_columns(table, path, {"start_timestamp", "end_timestamp"})
    total = table.num_rows
    sliced = table.slice(offset, limit)
    rows = sliced.to_pydict()
    items = [
        OverlayPoint(**_project_legacy_overlay_row({col: rows[col][i] for col in rows}))
        for i in range(sliced.num_rows)
    ]
    return OverlayResponse(total=total, items=items)


@router.get("/masked-transformers/{job_id}/exemplars")
async def get_mt_exemplars(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    k: Optional[int] = Query(default=None, ge=2),
) -> ExemplarsResponse:
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="masked transformer job not found")
    resolved_k = _resolve_k(job, k)
    path = masked_transformer_k_exemplars_path(
        settings.storage_root, job_id, resolved_k
    )
    if not path.exists():
        raise HTTPException(status_code=404, detail="exemplars.json not found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExemplarsResponse(
        n_states=payload["n_states"],
        states={
            key: [
                ExemplarRecord.model_validate(_project_legacy_exemplar_row(r))
                for r in records
            ]
            for key, records in payload["states"].items()
        },
    )


@router.get("/masked-transformers/{job_id}/label-distribution")
async def get_mt_label_distribution(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    k: Optional[int] = Query(default=None, ge=2),
) -> LabelDistributionResponse:
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="masked transformer job not found")
    resolved_k = _resolve_k(job, k)
    path = masked_transformer_k_label_distribution_path(
        settings.storage_root, job_id, resolved_k
    )
    if not path.exists():
        raise HTTPException(status_code=404, detail="label_distribution.json not found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LabelDistributionResponse.model_validate(payload)


@router.get("/masked-transformers/{job_id}/run-lengths")
async def get_run_lengths(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    k: Optional[int] = Query(default=None, ge=2),
) -> RunLengthsResponse:
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="masked transformer job not found")
    resolved_k = _resolve_k(job, k)
    path = masked_transformer_k_run_lengths_path(
        settings.storage_root, job_id, resolved_k
    )
    if not path.exists():
        raise HTTPException(status_code=404, detail="run_lengths.json not found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunLengthsResponse.model_validate(payload)
