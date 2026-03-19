"""API router for binary classifier training and detection."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from humpback.api.deps import SessionDep, SettingsDep
from humpback.classifier.detection_rows import (
    DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE,
    DEFAULT_POSITIVE_SELECTION_MIN_SCORE,
    DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW,
    ROW_STORE_FIELDNAMES,
    apply_effective_positive_selection,
    ensure_detection_row_store,
    iter_detection_rows_as_tsv,
    normalize_detection_row,
    read_detection_row_store,
    read_tsv_rows,
    safe_float,
    resolve_clip_bounds,
    sync_detection_tsv,
    write_detection_row_store,
)
from humpback.classifier.detector import read_window_diagnostics_table
from humpback.classifier.archive import ArchiveProvider
from humpback.classifier.providers import build_archive_playback_provider
from humpback.schemas.classifier import (
    ClassifierModelOut,
    ClassifierTrainingJobCreate,
    ClassifierTrainingJobOut,
    DetectionJobCreate,
    DetectionJobOut,
    DiagnosticsResponse,
    DiagnosticsSummaryResponse,
    HydrophoneDetectionJobCreate,
    HydrophoneInfo,
    PerFileDiagnosticSummary,
    RetrainFolderInfo,
    RetrainWorkflowCreate,
    RetrainWorkflowOut,
    TrainingDataSummaryResponse,
    TrainingSourceInfo,
    WindowDiagnosticRecord,
)
from humpback.services import classifier_service
from humpback.storage import (  # noqa: E402
    detection_diagnostics_path,
    detection_embeddings_path,
    detection_row_store_path,
    detection_tsv_path,
)

if TYPE_CHECKING:
    from humpback.processing.spectrogram_cache import SpectrogramCache

router = APIRouter(prefix="/classifier", tags=["classifier"])


def _training_job_to_out(job) -> ClassifierTrainingJobOut:
    return ClassifierTrainingJobOut(
        id=job.id,
        status=job.status,
        name=job.name,
        positive_embedding_set_ids=json.loads(job.positive_embedding_set_ids),
        negative_embedding_set_ids=json.loads(job.negative_embedding_set_ids),
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
        hop_seconds=job.hop_seconds,
        high_threshold=job.high_threshold,
        low_threshold=job.low_threshold,
        output_tsv_path=job.output_tsv_path,
        output_row_store_path=job.output_row_store_path,
        result_summary=json.loads(job.result_summary) if job.result_summary else None,
        error_message=job.error_message,
        files_processed=job.files_processed,
        files_total=job.files_total,
        extract_status=job.extract_status,
        extract_error=job.extract_error,
        extract_summary=json.loads(job.extract_summary)
        if job.extract_summary
        else None,
        hydrophone_id=job.hydrophone_id,
        hydrophone_name=job.hydrophone_name,
        start_timestamp=job.start_timestamp,
        end_timestamp=job.end_timestamp,
        segments_processed=job.segments_processed,
        segments_total=job.segments_total,
        time_covered_sec=job.time_covered_sec,
        alerts=json.loads(job.alerts) if job.alerts else None,
        local_cache_path=job.local_cache_path,
        has_positive_labels=job.has_positive_labels,
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
            body.hop_seconds,
            body.high_threshold,
            body.low_threshold,
            detection_mode=body.detection_mode,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _detection_job_to_out(job)


@router.get("/detection-jobs")
async def list_detection_jobs(session: SessionDep) -> list[DetectionJobOut]:
    jobs = await classifier_service.list_detection_jobs(session)
    return [_detection_job_to_out(j) for j in jobs]


@router.get("/detection-jobs/{job_id}")
async def get_detection_job(job_id: str, session: SessionDep) -> DetectionJobOut:
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    return _detection_job_to_out(job)


@router.get("/detection-jobs/{job_id}/download")
async def download_detections(
    job_id: str, session: SessionDep, settings: SettingsDep
) -> Response:
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    if job.status not in ("paused", "complete", "canceled"):
        raise HTTPException(400, "Detection job not complete or no output available")

    window_size_seconds = await _get_classifier_window_size(
        session, job.classifier_model_id
    )
    fieldnames, rows = await _ensure_detection_row_store_for_job(
        session,
        job,
        settings=settings,
        window_size_seconds=window_size_seconds,
    )

    return StreamingResponse(
        iter_detection_rows_as_tsv(rows, fieldnames=fieldnames),
        media_type="text/tab-separated-values",
        headers={
            "Content-Disposition": f'attachment; filename="detections_{job_id}.tsv"'
        },
    )


# ---- Hydrophone Detection ----


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
            detection_mode=body.detection_mode,
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


# ---- Diagnostics ----


@router.get("/detection-jobs/{job_id}/diagnostics")
async def get_detection_diagnostics(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    filename: Optional[str] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
) -> DiagnosticsResponse:
    """Return paginated per-window diagnostic records from a detection job."""
    import pyarrow.compute as _pc

    pc: Any = _pc

    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")

    diag_path = detection_diagnostics_path(settings.storage_root, job_id)
    if not diag_path.exists():
        raise HTTPException(404, "Diagnostics not available for this job")

    table = read_window_diagnostics_table(diag_path)

    filenames = sorted(pc.unique(table.column("filename")).to_pylist())

    if filename:
        table = read_window_diagnostics_table(diag_path, filename=filename)

    total = table.num_rows
    page = table.slice(offset, limit)

    has_end_sec = "end_sec" in page.column_names

    records = [
        WindowDiagnosticRecord(
            filename=page.column("filename")[i].as_py(),
            window_index=page.column("window_index")[i].as_py(),
            offset_sec=page.column("offset_sec")[i].as_py(),
            end_sec=page.column("end_sec")[i].as_py()
            if has_end_sec
            else page.column("offset_sec")[i].as_py() + 5.0,
            confidence=page.column("confidence")[i].as_py(),
            is_overlapped=page.column("is_overlapped")[i].as_py(),
            overlap_sec=page.column("overlap_sec")[i].as_py(),
        )
        for i in range(page.num_rows)
    ]

    return DiagnosticsResponse(records=records, total=total, filenames=filenames)


@router.get("/detection-jobs/{job_id}/diagnostics/summary")
async def get_detection_diagnostics_summary(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> DiagnosticsSummaryResponse:
    """Return aggregate diagnostic statistics for a detection job."""
    import pyarrow.compute as _pc

    pc: Any = _pc

    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")

    diag_path = detection_diagnostics_path(settings.storage_root, job_id)
    if not diag_path.exists():
        raise HTTPException(404, "Diagnostics not available for this job")

    table = read_window_diagnostics_table(diag_path)
    total_windows = table.num_rows

    is_overlapped = table.column("is_overlapped")
    confidence = table.column("confidence")
    filenames_col = table.column("filename")

    overlapped_mask = pc.equal(is_overlapped, True)
    total_overlapped = pc.sum(overlapped_mask.cast("int32")).as_py()
    overlapped_ratio = total_overlapped / total_windows if total_windows > 0 else 0.0

    # Overlapped vs normal mean confidence
    overlapped_conf = pc.filter(confidence, overlapped_mask)
    normal_mask = pc.invert(overlapped_mask)
    normal_conf = pc.filter(confidence, normal_mask)

    overlapped_mean = (
        float(pc.mean(overlapped_conf).as_py()) if len(overlapped_conf) > 0 else None
    )
    normal_mean = float(pc.mean(normal_conf).as_py()) if len(normal_conf) > 0 else None

    # Confidence histogram (10 bins from 0 to 1)
    conf_np = confidence.to_pylist()
    conf_arr = np.array(conf_np, dtype=np.float32)
    is_overlapped_arr = np.array(is_overlapped.to_pylist(), dtype=bool)
    bin_edges = np.linspace(0, 1, 11)
    hist_all, _ = np.histogram(conf_arr, bins=bin_edges)
    histogram = []
    for i in range(len(bin_edges) - 1):
        if i == len(bin_edges) - 2:
            bin_mask = (conf_arr >= bin_edges[i]) & (conf_arr <= bin_edges[i + 1])
        else:
            bin_mask = (conf_arr >= bin_edges[i]) & (conf_arr < bin_edges[i + 1])
        n_overlapped_in_bin = int(is_overlapped_arr[bin_mask].sum())
        histogram.append(
            {
                "bin_start": float(bin_edges[i]),
                "bin_end": float(bin_edges[i + 1]),
                "count": int(hist_all[i]),
                "count_overlapped": n_overlapped_in_bin,
            }
        )

    # Per-file summaries
    unique_files = sorted(pc.unique(filenames_col).to_pylist())
    per_file = []
    for fname in unique_files:
        file_mask = pc.equal(filenames_col, fname)
        file_table = table.filter(file_mask)
        file_conf = file_table.column("confidence")
        file_overlapped = file_table.column("is_overlapped")
        file_overlapped_mask = pc.equal(file_overlapped, True)
        file_normal_mask = pc.invert(file_overlapped_mask)

        n_overlapped = pc.sum(file_overlapped_mask.cast("int32")).as_py()
        overlapped_c = pc.filter(file_conf, file_overlapped_mask)
        normal_c = pc.filter(file_conf, file_normal_mask)

        pf_summary = PerFileDiagnosticSummary(
            filename=fname,
            n_windows=file_table.num_rows,
            n_overlapped=n_overlapped,
            mean_confidence=float(pc.mean(file_conf).as_py()),
            mean_confidence_overlapped=float(pc.mean(overlapped_c).as_py())
            if len(overlapped_c) > 0
            else None,
            mean_confidence_normal=float(pc.mean(normal_c).as_py())
            if len(normal_c) > 0
            else None,
        )
        per_file.append(pf_summary)

    return DiagnosticsSummaryResponse(
        total_windows=total_windows,
        total_overlapped=total_overlapped,
        overlapped_ratio=overlapped_ratio,
        confidence_histogram=histogram,
        overlapped_mean_confidence=overlapped_mean,
        normal_mean_confidence=normal_mean,
        per_file=per_file,
    )


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


def _retrain_workflow_to_out(wf) -> RetrainWorkflowOut:
    return RetrainWorkflowOut(
        id=wf.id,
        status=wf.status,
        source_model_id=wf.source_model_id,
        new_model_name=wf.new_model_name,
        model_version=wf.model_version,
        window_size_seconds=wf.window_size_seconds,
        target_sample_rate=wf.target_sample_rate,
        feature_config=json.loads(wf.feature_config) if wf.feature_config else None,
        parameters=json.loads(wf.parameters) if wf.parameters else None,
        positive_folder_roots=json.loads(wf.positive_folder_roots),
        negative_folder_roots=json.loads(wf.negative_folder_roots),
        import_summary=json.loads(wf.import_summary) if wf.import_summary else None,
        processing_job_ids=json.loads(wf.processing_job_ids)
        if wf.processing_job_ids
        else None,
        processing_total=wf.processing_total,
        processing_complete=wf.processing_complete,
        training_job_id=wf.training_job_id,
        new_model_id=wf.new_model_id,
        error_message=wf.error_message,
        created_at=wf.created_at,
        updated_at=wf.updated_at,
    )


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


# ---- Extraction ----


class ExtractRequest(BaseModel):
    job_ids: list[str]
    positive_output_path: Optional[str] = None
    negative_output_path: Optional[str] = None
    positive_selection_smoothing_window: int = Field(
        default=DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW
    )
    positive_selection_min_score: float = Field(
        default=DEFAULT_POSITIVE_SELECTION_MIN_SCORE,
        ge=0.0,
        le=1.0,
    )
    positive_selection_extend_min_score: float = Field(
        default=DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE,
        ge=0.0,
        le=1.0,
    )

    @field_validator("positive_selection_smoothing_window")
    @classmethod
    def validate_positive_selection_smoothing_window(cls, value: int) -> int:
        if value < 1 or value % 2 == 0:
            raise ValueError(
                "positive_selection_smoothing_window must be an odd integer >= 1"
            )
        return value


@router.post("/detection-jobs/extract")
async def extract_labeled_samples(
    body: ExtractRequest, session: SessionDep, settings: SettingsDep
) -> dict:
    """Queue extraction of labeled samples from completed detection jobs."""
    from datetime import datetime, timezone

    from sqlalchemy import select, update

    from humpback.models.classifier import DetectionJob

    # Validate jobs exist and are complete
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id.in_(body.job_ids))
    )
    jobs = list(result.scalars().all())
    if len(jobs) != len(body.job_ids):
        found_ids = {j.id for j in jobs}
        missing = [jid for jid in body.job_ids if jid not in found_ids]
        raise HTTPException(404, f"Detection jobs not found: {missing}")

    for j in jobs:
        if j.status not in ("paused", "complete", "canceled"):
            raise HTTPException(
                400, f"Detection job {j.id} is not complete (status={j.status})"
            )

    pos_path = body.positive_output_path or settings.positive_sample_path
    neg_path = body.negative_output_path or settings.negative_sample_path
    config = json.dumps(
        {
            "positive_output_path": pos_path,
            "negative_output_path": neg_path,
            "positive_selection_smoothing_window": (
                body.positive_selection_smoothing_window
            ),
            "positive_selection_min_score": body.positive_selection_min_score,
            "positive_selection_extend_min_score": (
                body.positive_selection_extend_min_score
            ),
        }
    )

    for j in jobs:
        await session.execute(
            update(DetectionJob)
            .where(DetectionJob.id == j.id)
            .values(
                extract_status="queued",
                extract_error=None,
                extract_summary=None,
                extract_config=config,
                updated_at=datetime.now(timezone.utc),
            )
        )
    await session.commit()

    return {"status": "queued", "count": len(jobs)}


@router.get("/extraction-settings")
async def get_extraction_settings(settings: SettingsDep) -> dict:
    """Return default extraction output paths from config."""
    return {
        "positive_output_path": settings.positive_sample_path,
        "negative_output_path": settings.negative_sample_path,
        "positive_selection_smoothing_window": (
            DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW
        ),
        "positive_selection_min_score": DEFAULT_POSITIVE_SELECTION_MIN_SCORE,
        "positive_selection_extend_min_score": (
            DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE
        ),
    }


# ---- Browse Directories ----


@router.get("/browse-directories")
async def browse_directories(root: str = Query("/")):
    """List subdirectories of a server path for folder selection."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise HTTPException(400, f"Not a directory: {root}")
    try:
        subdirs = sorted(
            [
                {"name": entry.name, "path": str(entry)}
                for entry in root_path.iterdir()
                if entry.is_dir() and not entry.name.startswith(".")
            ],
            key=lambda d: d["name"],
        )
    except PermissionError:
        raise HTTPException(403, f"Permission denied: {root}")
    return {"path": str(root_path), "subdirectories": subdirs}


# ---- Delete Training Jobs ----


class BulkDeleteRequest(BaseModel):
    ids: list[str]


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


# ---- Delete Detection Jobs ----


@router.delete("/detection-jobs/{job_id}")
async def delete_detection_job(
    job_id: str, session: SessionDep, settings: SettingsDep
) -> dict:
    deleted = await classifier_service.delete_detection_job(
        session, job_id, settings.storage_root
    )
    if not deleted:
        raise HTTPException(404, "Detection job not found")
    return {"status": "deleted"}


@router.post("/detection-jobs/bulk-delete")
async def bulk_delete_detection_jobs(
    body: BulkDeleteRequest, session: SessionDep, settings: SettingsDep
) -> dict:
    count = await classifier_service.bulk_delete_detection_jobs(
        session, body.ids, settings.storage_root
    )
    return {"status": "deleted", "count": count}


# ---- Bulk Delete Classifier Models ----


@router.post("/models/bulk-delete")
async def bulk_delete_models(
    body: BulkDeleteRequest, session: SessionDep, settings: SettingsDep
) -> dict:
    count = await classifier_service.bulk_delete_classifier_models(
        session, body.ids, settings.storage_root
    )
    return {"status": "deleted", "count": count}


# ---- Detection Content ----


async def _get_classifier_window_size(
    session: SessionDep, classifier_model_id: str
) -> float:
    """Resolve window size from classifier model; default to 5.0 when missing."""
    from sqlalchemy import select

    from humpback.models.classifier import ClassifierModel

    result = await session.execute(
        select(ClassifierModel.window_size_seconds).where(
            ClassifierModel.id == classifier_model_id
        )
    )
    ws = result.scalar_one_or_none()
    return float(ws) if ws is not None else 5.0


async def _ensure_detection_row_store_for_job(
    session: SessionDep,
    job,
    *,
    settings: SettingsDep,
    window_size_seconds: float,
) -> tuple[list[str], list[dict[str, str]]]:
    rs_path = detection_row_store_path(settings.storage_root, job.id)
    tsv = detection_tsv_path(settings.storage_root, job.id)

    if rs_path.is_file():
        fieldnames, rows = read_detection_row_store(rs_path)
    else:
        if not tsv.is_file():
            raise HTTPException(404, "Detection output not found on disk")
        diag_path = detection_diagnostics_path(settings.storage_root, job.id)
        fieldnames, rows = ensure_detection_row_store(
            row_store_path=rs_path,
            tsv_path=tsv,
            diagnostics_path=diag_path if diag_path.exists() else None,
            is_hydrophone=job.hydrophone_id is not None,
            window_size_seconds=window_size_seconds,
            detection_mode=job.detection_mode,
        )

    return fieldnames, rows


@router.get("/detection-jobs/{job_id}/content")
async def get_detection_content(
    job_id: str, session: SessionDep, settings: SettingsDep
) -> list[dict]:
    """Return normalized detection rows from the active artifact source."""
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    if job.status not in ("running", "paused", "complete", "canceled"):
        raise HTTPException(400, "Detection job not ready or no output available")

    window_size_seconds = await _get_classifier_window_size(
        session, job.classifier_model_id
    )
    is_hydrophone = job.hydrophone_id is not None

    tsv = detection_tsv_path(settings.storage_root, job.id)
    if job.status == "running":
        if not tsv.is_file():
            raise HTTPException(404, "TSV file not found on disk")
        _fieldnames, raw_rows = read_tsv_rows(tsv)
    else:
        _fieldnames, raw_rows = await _ensure_detection_row_store_for_job(
            session,
            job,
            settings=settings,
            window_size_seconds=window_size_seconds,
        )

    return [
        normalize_detection_row(
            row,
            is_hydrophone=is_hydrophone,
            window_size_seconds=window_size_seconds,
        )
        for row in raw_rows
    ]


# ---- Detection Labels ----


class DetectionLabelRow(BaseModel):
    filename: str
    start_sec: float
    end_sec: float
    humpback: Optional[Literal[0, 1]] = None
    orca: Optional[Literal[0, 1]] = None
    ship: Optional[Literal[0, 1]] = None
    background: Optional[Literal[0, 1]] = None


class DetectionRowStateUpdate(BaseModel):
    row_id: str
    humpback: Optional[Literal[0, 1]] = None
    orca: Optional[Literal[0, 1]] = None
    ship: Optional[Literal[0, 1]] = None
    background: Optional[Literal[0, 1]] = None
    manual_positive_selection_start_sec: float | None = None
    manual_positive_selection_end_sec: float | None = None


def _serialize_label(value: int | None) -> str:
    """Serialize label value for TSV: None → '', 0 → '0', 1 → '1'."""
    if value is None:
        return ""
    return str(value)


@router.put("/detection-jobs/{job_id}/labels")
async def save_detection_labels(
    job_id: str,
    body: list[DetectionLabelRow],
    session: SessionDep,
    settings: SettingsDep,
) -> dict:
    """Merge label annotations into the canonical detection row store."""
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    if job.status not in ("paused", "complete", "canceled"):
        raise HTTPException(400, "Detection job not complete or no output available")

    rs_path = detection_row_store_path(settings.storage_root, job.id)
    tsv = detection_tsv_path(settings.storage_root, job.id)
    window_size_seconds = await _get_classifier_window_size(
        session, job.classifier_model_id
    )

    label_map: dict[tuple[str, float, float], DetectionLabelRow] = {}
    for row in body:
        label_map[(row.filename, row.start_sec, row.end_sec)] = row

    fieldnames, existing_rows = await _ensure_detection_row_store_for_job(
        session,
        job,
        settings=settings,
        window_size_seconds=window_size_seconds,
    )

    updated_rows: list[dict[str, str]] = []
    for row in existing_rows:
        key = (
            row.get("filename", ""),
            safe_float(row.get("start_sec"), 0.0),
            safe_float(row.get("end_sec"), 0.0),
        )
        update = label_map.get(key)
        out_row = {field: row.get(field, "") for field in ROW_STORE_FIELDNAMES}
        if update:
            out_row["humpback"] = _serialize_label(update.humpback)
            out_row["orca"] = _serialize_label(update.orca)
            out_row["ship"] = _serialize_label(update.ship)
            out_row["background"] = _serialize_label(update.background)
        apply_effective_positive_selection(
            out_row,
            window_size_seconds=window_size_seconds,
        )
        updated_rows.append(out_row)

    write_detection_row_store(rs_path, updated_rows)
    sync_detection_tsv(tsv, updated_rows, fieldnames)

    has_positive = any(
        row.get("humpback") == "1" or row.get("orca") == "1" for row in updated_rows
    )
    job.has_positive_labels = has_positive
    await session.commit()

    return {"status": "ok", "updated": len(label_map)}


@router.put("/detection-jobs/{job_id}/row-state")
async def save_detection_row_state(
    job_id: str,
    body: DetectionRowStateUpdate,
    session: SessionDep,
    settings: SettingsDep,
) -> dict:
    """Persist one row's labels and manual positive-selection bounds."""
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    if job.status not in ("paused", "complete", "canceled"):
        raise HTTPException(400, "Detection job not complete or no output available")

    rs_path = detection_row_store_path(settings.storage_root, job.id)
    tsv = detection_tsv_path(settings.storage_root, job.id)

    window_size_seconds = await _get_classifier_window_size(
        session, job.classifier_model_id
    )
    fieldnames, existing_rows = await _ensure_detection_row_store_for_job(
        session,
        job,
        settings=settings,
        window_size_seconds=window_size_seconds,
    )

    matched_row: dict[str, str] | None = None
    for row in existing_rows:
        if row.get("row_id") != body.row_id:
            continue
        row["humpback"] = _serialize_label(body.humpback)
        row["orca"] = _serialize_label(body.orca)
        row["ship"] = _serialize_label(body.ship)
        row["background"] = _serialize_label(body.background)

        if (
            body.manual_positive_selection_start_sec is None
            and body.manual_positive_selection_end_sec is None
        ):
            row["manual_positive_selection_start_sec"] = ""
            row["manual_positive_selection_end_sec"] = ""
        else:
            if (
                body.manual_positive_selection_start_sec is None
                or body.manual_positive_selection_end_sec is None
            ):
                raise HTTPException(400, "Both manual bounds are required")
            normalized = normalize_detection_row(
                row,
                is_hydrophone=job.hydrophone_id is not None,
                window_size_seconds=window_size_seconds,
            )
            clip_start, clip_end = resolve_clip_bounds(
                normalized,
                window_size_seconds=window_size_seconds,
            )
            manual_start = body.manual_positive_selection_start_sec
            manual_end = body.manual_positive_selection_end_sec
            if manual_end <= manual_start:
                raise HTTPException(400, "Manual bounds must have positive duration")
            if manual_start + 1e-6 < clip_start or manual_end - 1e-6 > clip_end:
                raise HTTPException(400, "Manual bounds must stay within the clip")
            manual_duration = manual_end - manual_start
            if manual_duration + 1e-6 < window_size_seconds:
                raise HTTPException(
                    400,
                    "Manual bounds must be at least one classifier window long",
                )
            duration_delta = manual_duration / window_size_seconds
            if abs(round(duration_delta) - duration_delta) > 1e-6:
                raise HTTPException(
                    400,
                    "Manual bounds must move in whole window-size steps",
                )
            row["manual_positive_selection_start_sec"] = f"{manual_start:.6f}"
            row["manual_positive_selection_end_sec"] = f"{manual_end:.6f}"

        apply_effective_positive_selection(
            row,
            window_size_seconds=window_size_seconds,
        )
        matched_row = row
        break

    if matched_row is None:
        raise HTTPException(404, "Detection row not found")

    write_detection_row_store(rs_path, existing_rows)
    sync_detection_tsv(tsv, existing_rows, fieldnames)

    has_positive = any(
        row.get("humpback") == "1" or row.get("orca") == "1" for row in existing_rows
    )
    job.has_positive_labels = has_positive
    await session.commit()

    return {
        "status": "ok",
        "row": normalize_detection_row(
            matched_row,
            is_hydrophone=job.hydrophone_id is not None,
            window_size_seconds=window_size_seconds,
        ),
    }


# ---- Audio Slice Streaming ----


def _encode_wav_response(segment: "np.ndarray", sr: int, normalize: bool) -> Response:  # noqa: F821
    """Encode audio segment as WAV response."""
    if normalize:
        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment / peak

    pcm = (segment * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    n_samples = len(pcm)
    data_size = n_samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())

    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"Content-Length": str(buf.tell())},
    )


class _DecodedAudioCache:
    """Thread-safe LRU cache for decoded detection audio arrays."""

    def __init__(self, max_entries: int = 64) -> None:
        import collections
        import threading

        self._max = max_entries
        self._cache: collections.OrderedDict[
            tuple[str, str, float, float], tuple[np.ndarray, int]
        ] = collections.OrderedDict()
        self._lock = threading.Lock()

    def get(
        self, job_id: str, filename: str, start_sec: float, duration_sec: float
    ) -> tuple[np.ndarray, int] | None:
        key = (job_id, filename, start_sec, duration_sec)
        with self._lock:
            val = self._cache.get(key)
            if val is not None:
                self._cache.move_to_end(key)
            return val

    def put(
        self,
        job_id: str,
        filename: str,
        start_sec: float,
        duration_sec: float,
        audio: np.ndarray,
        sr: int,
    ) -> None:
        key = (job_id, filename, start_sec, duration_sec)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                self._cache[key] = (audio, sr)
                while len(self._cache) > self._max:
                    self._cache.popitem(last=False)


_decoded_audio_cache = _DecodedAudioCache()


class _NoaaPlaybackProviderRegistry:
    """Process-level cache of NOAA playback provider instances.

    CachingNoaaGCSProvider._load_prefix populates _files_by_prefix lazily
    on first build_timeline() call. Reusing instances across HTTP requests
    means subsequent audio-slice/spectrogram calls hit the warm in-memory
    dict instead of re-reading manifest JSON from disk (~500ms–2s per prefix).

    Only NOAA sources are cached (Orcasound/HLS providers are already fast).
    Thread safety: registry insertion is Lock-protected; _load_prefix double-
    reads are benign (idempotent, GIL-protected dict assignment).
    """

    def __init__(self) -> None:
        import threading

        self._registry: dict[tuple[str, str], ArchiveProvider] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        source_id: str,
        cache_path: str | None,
        noaa_cache_path: str,
    ) -> ArchiveProvider:
        key = (source_id, noaa_cache_path)
        with self._lock:
            if key not in self._registry:
                self._registry[key] = build_archive_playback_provider(
                    source_id,
                    cache_path=cache_path,
                    noaa_cache_path=noaa_cache_path,
                )
            return self._registry[key]


_noaa_provider_registry = _NoaaPlaybackProviderRegistry()


async def _resolve_detection_audio(
    job, settings, filename: str, start_sec: float, duration_sec: float
) -> tuple:
    """Decode audio for a detection row. Returns (audio_array, sample_rate).

    Results are cached in an in-memory LRU so the second request (e.g.
    spectrogram after audio-slice) returns instantly.
    """
    cached = _decoded_audio_cache.get(job.id, filename, start_sec, duration_sec)
    if cached is not None:
        return cached

    import asyncio

    import numpy as np

    if job.hydrophone_id:
        from humpback.classifier.s3_stream import resolve_audio_slice

        if job.start_timestamp is None or job.end_timestamp is None:
            raise HTTPException(400, "Hydrophone job missing start/end timestamps")

        cache_path = job.local_cache_path or settings.s3_cache_path
        try:
            from humpback.config import get_archive_source as _get_archive_source

            _src = _get_archive_source(job.hydrophone_id)
            if (
                _src is not None
                and _src.get("provider_kind") == "noaa_gcs"
                and settings.noaa_cache_path is not None
            ):
                local_provider = _noaa_provider_registry.get_or_create(
                    job.hydrophone_id,
                    cache_path,
                    settings.noaa_cache_path,
                )
            else:
                local_provider = build_archive_playback_provider(
                    job.hydrophone_id,
                    cache_path=cache_path,
                    noaa_cache_path=settings.noaa_cache_path,
                )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        target_sr = 32000

        try:
            segment = await asyncio.to_thread(
                resolve_audio_slice,
                local_provider,
                job.start_timestamp,
                job.end_timestamp,
                filename,
                start_sec,
                duration_sec,
                target_sr,
                job.start_timestamp,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))

        result = (segment, target_sr)
        _decoded_audio_cache.put(
            job.id, filename, start_sec, duration_sec, segment, target_sr
        )
        return result

    # Local audio folder path
    from humpback.processing.audio_io import decode_audio

    audio_folder = Path(job.audio_folder)
    file_path = audio_folder / filename

    try:
        file_path.resolve().relative_to(audio_folder.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid filename (path traversal)")

    if not file_path.is_file():
        raise HTTPException(404, f"Audio file not found: {filename}")

    audio, sr = await asyncio.to_thread(decode_audio, file_path)

    total_duration = len(audio) / sr
    end_sec_requested = start_sec + duration_sec
    if end_sec_requested > total_duration:
        start_sec = max(0.0, total_duration - duration_sec)

    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    start_sample = min(start_sample, len(audio))
    end_sample = min(end_sample, len(audio))
    segment = audio[start_sample:end_sample]

    result_arr = np.asarray(segment)
    _decoded_audio_cache.put(job.id, filename, start_sec, duration_sec, result_arr, sr)
    return result_arr, sr


@router.get("/detection-jobs/{job_id}/audio-slice")
async def get_detection_audio_slice(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    filename: str = Query(...),
    start_sec: float = Query(..., ge=0),
    duration_sec: float = Query(..., gt=0),
    normalize: bool = Query(True),
):
    """Stream a WAV slice from a detection job's audio folder or S3."""
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")

    segment, sr = await _resolve_detection_audio(
        job, settings, filename, start_sec, duration_sec
    )
    return _encode_wav_response(segment, sr, normalize)


# ---- Spectrogram ----


def _get_spectrogram_cache(settings) -> SpectrogramCache:
    from humpback.processing.spectrogram_cache import SpectrogramCache

    cache_dir = settings.storage_root / "spectrogram_cache"
    return SpectrogramCache(cache_dir, settings.spectrogram_cache_max_items)


@router.get("/detection-jobs/{job_id}/spectrogram")
async def get_detection_spectrogram(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    filename: str = Query(...),
    start_sec: float = Query(..., ge=0),
    duration_sec: float = Query(..., gt=0),
):
    """Return a PNG spectrogram for a detection clip."""
    import asyncio

    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")

    from humpback.processing.spectrogram_cache import SpectrogramCache

    cache = _get_spectrogram_cache(settings)
    cache_key = SpectrogramCache._make_key(
        job_id,
        filename,
        start_sec,
        duration_sec,
        settings.spectrogram_hop_length,
        settings.spectrogram_dynamic_range_db,
        2048,
        settings.spectrogram_width_px,
        settings.spectrogram_height_px,
    )

    cached = cache.get(cache_key)
    if cached is not None:
        return Response(content=cached, media_type="image/png")

    segment, sr = await _resolve_detection_audio(
        job, settings, filename, start_sec, duration_sec
    )

    from humpback.processing.spectrogram import generate_spectrogram_png

    png_bytes = await asyncio.to_thread(
        generate_spectrogram_png,
        segment,
        sr,
        hop_length=settings.spectrogram_hop_length,
        dynamic_range_db=settings.spectrogram_dynamic_range_db,
        width_px=settings.spectrogram_width_px,
        height_px=settings.spectrogram_height_px,
    )

    cache.put(cache_key, png_bytes)
    return Response(content=png_bytes, media_type="image/png")


# ---- Detection Embeddings ----


@router.get("/detection-jobs/{job_id}/embedding")
async def get_detection_embedding(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    filename: str = Query(...),
    start_sec: float = Query(..., ge=0),
    end_sec: float = Query(..., gt=0),
):
    """Return the stored embedding vector for a detection row."""
    from humpback.classifier.detector import read_detection_embedding

    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")

    emb_path = detection_embeddings_path(settings.storage_root, job.id)
    if not emb_path.exists():
        raise HTTPException(404, "No stored embeddings for this detection job")

    embedding = read_detection_embedding(emb_path, filename, start_sec, end_sec)
    if embedding is None:
        raise HTTPException(404, "Embedding not found for specified detection row")

    # Resolve model_version from the classifier model
    from sqlalchemy import select as sa_select

    from humpback.models.classifier import ClassifierModel

    cm_result = await session.execute(
        sa_select(ClassifierModel).where(ClassifierModel.id == job.classifier_model_id)
    )
    cm = cm_result.scalar_one_or_none()
    model_version = cm.model_version if cm else "unknown"

    return {
        "vector": embedding,
        "model_version": model_version,
        "vector_dim": len(embedding),
    }
