"""API router for the call parsing pipeline.

Parent-run CRUD + pass-job list/get/delete are functional for every
pass. Pass 1 exposes creation, trace, and regions endpoints backed by
the region detection worker. Pass 2 exposes segmentation-job creation,
events retrieval, and full CRUD for segmentation training jobs and
models. Pass 3 exposes classification-job creation, typed-events
retrieval, and model-family validation.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from humpback.api.deps import SessionDep, SettingsDep
from humpback.call_parsing.storage import (
    classification_job_dir,
    read_events,
    read_regions,
    read_trace,
    read_typed_events,
    region_job_dir,
    segmentation_job_dir,
)
from humpback.schemas.call_parsing import (
    BoundaryCorrectionRequest,
    BoundaryCorrectionResponse,
    CallParsingRunCreate,
    CallParsingRunResponse,
    ClassifierModelResponse,
    ClassifierTrainingJobResponse,
    CreateClassifierTrainingJobRequest,
    CreateDatasetFromCorrectionsRequest,
    CreateDatasetFromCorrectionsResponse,
    CreateEventClassificationJobRequest,
    CreateRegionJobRequest,
    CreateSegmentationJobRequest,
    QuickRetrainRequest,
    QuickRetrainResponse,
    EventClassificationJobSummary,
    EventSegmentationJobSummary,
    RegionDetectionJobSummary,
    SegmentationJobWithCorrectionCount,
    SegmentationModelResponse,
    SegmentationTrainingDatasetSummary,
    TypeCorrectionRequest,
    TypeCorrectionResponse,
)
from humpback.services import call_parsing as service

router = APIRouter(prefix="/call-parsing", tags=["call-parsing"])


def _run_to_response(run, rd, es, ec) -> CallParsingRunResponse:
    return CallParsingRunResponse(
        id=run.id,
        audio_file_id=run.audio_file_id,
        hydrophone_id=run.hydrophone_id,
        start_timestamp=run.start_timestamp,
        end_timestamp=run.end_timestamp,
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
    try:
        run = await service.create_parent_run(session, body)
    except service.CallParsingFKError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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


@router.post("/region-jobs", status_code=201, response_model=RegionDetectionJobSummary)
async def create_region_job(body: CreateRegionJobRequest, session: SessionDep):
    try:
        job = await service.create_region_job(session, body)
    except service.CallParsingFKError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    await session.commit()
    await session.refresh(job)
    return RegionDetectionJobSummary.model_validate(job)


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
async def get_region_trace(job_id: str, session: SessionDep, settings: SettingsDep):
    job = await service.get_region_detection_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Region detection job not found")
    if job.status != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Region detection job status is {job.status!r}, not 'complete'",
        )
    trace_path = region_job_dir(settings.storage_root, job_id) / "trace.parquet"
    if not trace_path.exists():
        raise HTTPException(status_code=404, detail="trace.parquet not found")
    rows = read_trace(trace_path)
    return [{"time_sec": row.time_sec, "score": row.score} for row in rows]


@router.get("/region-jobs/{job_id}/regions")
async def get_region_regions(job_id: str, session: SessionDep, settings: SettingsDep):
    job = await service.get_region_detection_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Region detection job not found")
    if job.status != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Region detection job status is {job.status!r}, not 'complete'",
        )
    regions_path = region_job_dir(settings.storage_root, job_id) / "regions.parquet"
    if not regions_path.exists():
        raise HTTPException(status_code=404, detail="regions.parquet not found")
    rows = sorted(read_regions(regions_path), key=lambda r: r.start_sec)
    return [
        {
            "region_id": r.region_id,
            "start_sec": r.start_sec,
            "end_sec": r.end_sec,
            "padded_start_sec": r.padded_start_sec,
            "padded_end_sec": r.padded_end_sec,
            "max_score": r.max_score,
            "mean_score": r.mean_score,
            "n_windows": r.n_windows,
        }
        for r in rows
    ]


@router.get("/region-jobs/{job_id}/tile")
async def get_region_tile(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    zoom_level: str = Query(
        ..., description="Zoom level (e.g. 24h, 6h, 1h, 15m, 5m, 1m)"
    ),
    tile_index: int = Query(..., ge=0, description="Tile index within the zoom level"),
) -> Response:
    """Return a PCEN spectrogram PNG tile for a region detection job."""
    from humpback.processing.timeline_tiles import ZOOM_LEVELS, tile_count

    job = await service.get_region_detection_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Region detection job not found")
    if job.status != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Region detection job status is {job.status!r}, not 'complete'",
        )

    if zoom_level not in ZOOM_LEVELS:
        raise HTTPException(
            400, f"Invalid zoom level: {zoom_level}. Must be one of {ZOOM_LEVELS}"
        )

    job_start = job.start_timestamp or 0.0
    job_end = job.end_timestamp or job_start
    duration = max(0.0, job_end - job_start)
    max_tiles = tile_count(zoom_level, job_duration_sec=duration)
    if tile_index >= max_tiles:
        raise HTTPException(
            400,
            f"Tile index {tile_index} out of range (max {max_tiles - 1} for {zoom_level})",
        )

    tile_bytes = await asyncio.to_thread(
        _render_region_tile_sync,
        job=job,
        zoom_level=zoom_level,
        tile_index=tile_index,
        settings=settings,
    )
    return Response(content=tile_bytes, media_type="image/png")


def _render_region_tile_sync(
    *,
    job,
    zoom_level: str,
    tile_index: int,
    settings,
) -> bytes:
    """Render a spectrogram tile for a region detection job (CPU-bound)."""
    from humpback.processing.pcen_rendering import PcenParams
    from humpback.processing.timeline_audio import resolve_timeline_audio
    from humpback.processing.timeline_tiles import (
        generate_timeline_tile,
        tile_time_range,
    )

    job_start = job.start_timestamp or 0.0

    start_epoch, end_epoch = tile_time_range(
        zoom_level, tile_index=tile_index, job_start_timestamp=job_start
    )
    duration_sec = end_epoch - start_epoch

    # Compute appropriate sample rate for tile width
    width_px = settings.timeline_tile_width_px
    desired_samples = width_px * 4 * 256
    sr = max(
        200,
        min(int(desired_samples / duration_sec) if duration_sec > 0 else 32000, 32000),
    )

    # PCEN warm-up: borrow lead-in audio so the filter can settle
    warmup_sec = max(0.0, min(float(settings.pcen_warmup_sec), start_epoch - job_start))
    fetch_start = start_epoch - warmup_sec
    fetch_duration = duration_sec + warmup_sec

    audio = resolve_timeline_audio(
        hydrophone_id=job.hydrophone_id or "",
        local_cache_path=settings.s3_cache_path or "",
        job_start_timestamp=job.start_timestamp,
        job_end_timestamp=job.end_timestamp,
        start_sec=fetch_start,
        duration_sec=fetch_duration,
        target_sr=sr,
        noaa_cache_path=settings.noaa_cache_path,
    )

    warmup_samples = int(round(warmup_sec * sr))

    n_fft = min(2048, len(audio))
    if n_fft < 16:
        n_fft = 16
    hop_length = max(1, n_fft // 8)

    pcen_params = PcenParams(
        time_constant=settings.pcen_time_constant_sec,
        gain=settings.pcen_gain,
        bias=settings.pcen_bias,
        power=settings.pcen_power,
        eps=settings.pcen_eps,
    )

    return generate_timeline_tile(
        audio,
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        warmup_samples=warmup_samples,
        pcen_params=pcen_params,
        vmin=settings.pcen_vmin,
        vmax=settings.pcen_vmax,
        width_px=settings.timeline_tile_width_px,
        height_px=settings.timeline_tile_height_px,
    )


@router.get("/region-jobs/{job_id}/audio-slice")
async def get_region_audio_slice(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    start_sec: float = Query(..., description="Start time in seconds (job-relative)"),
    duration_sec: float = Query(
        ..., gt=0, le=30, description="Duration in seconds (max 30)"
    ),
) -> Response:
    """Return a WAV audio slice for a region detection job."""
    job = await service.get_region_detection_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Region detection job not found")
    if job.status != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Region detection job status is {job.status!r}, not 'complete'",
        )

    wav_bytes = await asyncio.to_thread(
        _render_region_audio_slice_sync,
        job=job,
        start_sec=start_sec,
        duration_sec=duration_sec,
        settings=settings,
    )
    return Response(content=wav_bytes, media_type="audio/wav")


def _render_region_audio_slice_sync(
    *,
    job,
    start_sec: float,
    duration_sec: float,
    settings,
) -> bytes:
    """Fetch and encode an audio slice as WAV (CPU-bound)."""
    import io
    import struct

    import numpy as np

    from humpback.processing.timeline_audio import resolve_timeline_audio

    job_start = job.start_timestamp or 0.0
    start_epoch = job_start + start_sec
    sr = 16000

    audio = resolve_timeline_audio(
        hydrophone_id=job.hydrophone_id or "",
        local_cache_path=settings.s3_cache_path or "",
        job_start_timestamp=job.start_timestamp,
        job_end_timestamp=job.end_timestamp,
        start_sec=start_epoch,
        duration_sec=duration_sec,
        target_sr=sr,
        noaa_cache_path=settings.noaa_cache_path,
    )

    # Normalize for playback
    from humpback.processing.audio_encoding import normalize_for_playback

    audio = normalize_for_playback(
        audio, target_rms_dbfs=settings.playback_target_rms_dbfs
    )

    # Encode as 16-bit PCM WAV
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    data_size = len(pcm) * 2
    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


# ---- Pass 2: segmentation jobs ------------------------------------------


@router.post(
    "/segmentation-jobs",
    status_code=201,
    response_model=EventSegmentationJobSummary,
)
async def create_segmentation_job(
    body: CreateSegmentationJobRequest, session: SessionDep
):
    try:
        job = await service.create_segmentation_job(session, body)
    except service.CallParsingFKError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except service.CallParsingStateError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    await session.commit()
    await session.refresh(job)
    return EventSegmentationJobSummary.model_validate(job)


@router.get("/segmentation-jobs", response_model=list[EventSegmentationJobSummary])
async def list_segmentation_jobs(session: SessionDep):
    jobs = await service.list_event_segmentation_jobs(session)
    return [EventSegmentationJobSummary.model_validate(j) for j in jobs]


@router.get(
    "/segmentation-jobs/with-correction-counts",
    response_model=list[SegmentationJobWithCorrectionCount],
)
async def list_segmentation_jobs_with_correction_counts(session: SessionDep):
    rows = await service.list_segmentation_jobs_with_correction_counts(session)
    return [SegmentationJobWithCorrectionCount(**row) for row in rows]


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
async def get_segmentation_events(
    job_id: str, session: SessionDep, settings: SettingsDep
):
    job = await service.get_event_segmentation_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Event segmentation job not found")
    if job.status != "complete":
        raise HTTPException(
            status_code=409,
            detail=(f"Event segmentation job status is {job.status!r}, not 'complete'"),
        )
    events_path = segmentation_job_dir(settings.storage_root, job_id) / "events.parquet"
    if not events_path.exists():
        raise HTTPException(status_code=404, detail="events.parquet not found")
    events = read_events(events_path)
    return [
        {
            "event_id": e.event_id,
            "region_id": e.region_id,
            "start_sec": e.start_sec,
            "end_sec": e.end_sec,
            "center_sec": e.center_sec,
            "segmentation_confidence": e.segmentation_confidence,
        }
        for e in events
    ]


# ---- Pass 2: segmentation training datasets / jobs ----------------------


@router.get(
    "/segmentation-training-datasets",
    response_model=list[SegmentationTrainingDatasetSummary],
)
async def list_segmentation_training_datasets(session: SessionDep):
    rows = await service.list_segmentation_training_datasets(session)
    return [SegmentationTrainingDatasetSummary(**row) for row in rows]


@router.post(
    "/segmentation-training-datasets/from-corrections",
    response_model=CreateDatasetFromCorrectionsResponse,
    status_code=201,
)
async def create_dataset_from_corrections(
    request: CreateDatasetFromCorrectionsRequest,
    session: SessionDep,
    settings: SettingsDep,
):
    try:
        dataset, sample_count = await service.create_dataset_from_corrections(
            session,
            segmentation_job_ids=request.segmentation_job_ids,
            settings=settings,
            name=request.name,
            description=request.description,
        )
    except ValueError as exc:
        msg = str(exc)
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg) from exc
        raise HTTPException(status_code=400, detail=msg) from exc
    return CreateDatasetFromCorrectionsResponse(
        id=dataset.id,
        name=dataset.name,
        sample_count=sample_count,
        created_at=dataset.created_at,
    )


@router.post(
    "/segmentation-training/quick-retrain",
    response_model=QuickRetrainResponse,
    status_code=201,
)
async def quick_retrain(
    request: QuickRetrainRequest,
    session: SessionDep,
    settings: SettingsDep,
):
    try:
        (
            dataset_id,
            training_job_id,
            sample_count,
        ) = await service.create_dataset_and_train(
            session,
            segmentation_job_id=request.segmentation_job_id,
            settings=settings,
        )
    except ValueError as exc:
        msg = str(exc)
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg) from exc
        raise HTTPException(status_code=400, detail=msg) from exc
    return QuickRetrainResponse(
        dataset_id=dataset_id,
        training_job_id=training_job_id,
        sample_count=sample_count,
    )


# ---- Pass 2: segmentation models ----------------------------------------


@router.get(
    "/segmentation-models",
    response_model=list[SegmentationModelResponse],
)
async def list_segmentation_models(session: SessionDep):
    models = await service.list_segmentation_models(session)
    return [SegmentationModelResponse.model_validate(m) for m in models]


@router.get(
    "/segmentation-models/{model_id}",
    response_model=SegmentationModelResponse,
)
async def get_segmentation_model(model_id: str, session: SessionDep):
    model = await service.get_segmentation_model(session, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Segmentation model not found")
    return SegmentationModelResponse.model_validate(model)


@router.delete("/segmentation-models/{model_id}", status_code=204)
async def delete_segmentation_model(
    model_id: str, session: SessionDep, settings: SettingsDep
):
    try:
        deleted = await service.delete_segmentation_model(session, model_id, settings)
    except service.CallParsingStateError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Segmentation model not found")
    return None


# ---- Pass 3: classification jobs ----------------------------------------


@router.post(
    "/classification-jobs",
    status_code=201,
    response_model=EventClassificationJobSummary,
)
async def create_classification_job(
    body: CreateEventClassificationJobRequest, session: SessionDep
):
    try:
        job = await service.create_event_classification_job(session, body)
    except service.CallParsingFKError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except service.CallParsingStateError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    except service.CallParsingValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc
    await session.commit()
    await session.refresh(job)
    return EventClassificationJobSummary.model_validate(job)


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
async def get_classification_typed_events(
    job_id: str, session: SessionDep, settings: SettingsDep
):
    job = await service.get_event_classification_job(session, job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="Event classification job not found"
        )
    if job.status != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Event classification job status is {job.status!r}, not 'complete'",
        )
    typed_path = (
        classification_job_dir(settings.storage_root, job_id) / "typed_events.parquet"
    )
    if not typed_path.exists():
        raise HTTPException(status_code=404, detail="typed_events.parquet not found")
    typed_events = read_typed_events(typed_path)
    return sorted(
        [
            {
                "event_id": te.event_id,
                "start_sec": te.start_sec,
                "end_sec": te.end_sec,
                "type_name": te.type_name,
                "score": te.score,
                "above_threshold": te.above_threshold,
            }
            for te in typed_events
        ],
        key=lambda r: (r["start_sec"], r["type_name"]),
    )


# ---- Boundary corrections (Pass 2) ---------------------------------------


@router.post("/segmentation-jobs/{job_id}/corrections")
async def upsert_boundary_corrections(
    job_id: str, body: BoundaryCorrectionRequest, session: SessionDep
):
    try:
        count = await service.upsert_boundary_corrections(
            session, job_id, body.corrections
        )
    except service.CallParsingFKError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except service.CallParsingStateError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    return {"count": count}


@router.get(
    "/segmentation-jobs/{job_id}/corrections",
    response_model=list[BoundaryCorrectionResponse],
)
async def list_boundary_corrections(job_id: str, session: SessionDep):
    rows = await service.list_boundary_corrections(session, job_id)
    return [BoundaryCorrectionResponse.model_validate(r) for r in rows]


@router.delete("/segmentation-jobs/{job_id}/corrections", status_code=204)
async def clear_boundary_corrections(job_id: str, session: SessionDep):
    await service.clear_boundary_corrections(session, job_id)
    return None


# ---- Type corrections (Pass 3) -------------------------------------------


@router.post("/classification-jobs/{job_id}/corrections")
async def upsert_type_corrections(
    job_id: str, body: TypeCorrectionRequest, session: SessionDep
):
    try:
        count = await service.upsert_type_corrections(session, job_id, body.corrections)
    except service.CallParsingFKError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except service.CallParsingStateError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    return {"count": count}


@router.get(
    "/classification-jobs/{job_id}/corrections",
    response_model=list[TypeCorrectionResponse],
)
async def list_type_corrections(job_id: str, session: SessionDep):
    rows = await service.list_type_corrections(session, job_id)
    return [TypeCorrectionResponse.model_validate(r) for r in rows]


@router.delete("/classification-jobs/{job_id}/corrections", status_code=204)
async def clear_type_corrections(job_id: str, session: SessionDep):
    await service.clear_type_corrections(session, job_id)
    return None


# ---- Classifier feedback training jobs (Pass 3) ---------------------------


@router.post(
    "/classifier-training-jobs",
    status_code=201,
    response_model=ClassifierTrainingJobResponse,
)
async def create_classifier_training_job(
    body: CreateClassifierTrainingJobRequest, session: SessionDep
):
    try:
        job = await service.create_classifier_training_job(session, body)
    except service.CallParsingFKError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except service.CallParsingStateError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    await session.commit()
    return ClassifierTrainingJobResponse.model_validate(job)


@router.get(
    "/classifier-training-jobs",
    response_model=list[ClassifierTrainingJobResponse],
)
async def list_classifier_training_jobs(session: SessionDep):
    jobs = await service.list_classifier_training_jobs(session)
    return [ClassifierTrainingJobResponse.model_validate(j) for j in jobs]


@router.get(
    "/classifier-training-jobs/{job_id}",
    response_model=ClassifierTrainingJobResponse,
)
async def get_classifier_training_job(job_id: str, session: SessionDep):
    job = await service.get_classifier_training_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Classifier training job not found")
    return ClassifierTrainingJobResponse.model_validate(job)


@router.delete("/classifier-training-jobs/{job_id}", status_code=204)
async def delete_classifier_training_job(job_id: str, session: SessionDep):
    deleted = await service.delete_classifier_training_job(session, job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Classifier training job not found")
    return None


# ---- Classifier model management (Pass 3) ---------------------------------


@router.get(
    "/classifier-models",
    response_model=list[ClassifierModelResponse],
)
async def list_classifier_models(session: SessionDep):
    models = await service.list_classifier_models(session)
    return [ClassifierModelResponse.model_validate(m) for m in models]


@router.delete("/classifier-models/{model_id}", status_code=204)
async def delete_classifier_model(
    model_id: str, session: SessionDep, settings: SettingsDep
):
    try:
        deleted = await service.delete_classifier_model(session, model_id, settings)
    except service.CallParsingStateError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Classifier model not found")
    return None
