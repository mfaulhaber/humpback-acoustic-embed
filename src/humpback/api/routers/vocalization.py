"""API router for vocalization type classification."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select

from humpback.api.deps import SessionDep, SettingsDep
from humpback.models.training_dataset import TrainingDataset, TrainingDatasetLabel
from humpback.models.vocalization import (
    VocalizationClassifierModel,
    VocalizationInferenceJob,
    VocalizationTrainingJob,
)
from humpback.schemas.vocalization import (
    TrainingDatasetExtendRequest,
    TrainingDatasetLabelCreate,
    TrainingDatasetLabelOut,
    TrainingDatasetRowLabelOut,
    TrainingDatasetRowOut,
    VocalizationInferenceJobCreate,
    VocalizationInferenceJobOut,
    VocalizationModelOut,
    VocalizationPredictionRow,
    VocalizationTrainingJobCreate,
    VocalizationTrainingJobOut,
    VocalizationTrainingSourceOut,
    VocalizationTypeCreate,
    VocalizationTypeImportRequest,
    VocalizationTypeImportResponse,
    VocalizationTypeOut,
    VocalizationTypeUpdate,
)
from humpback.services.vocalization_service import (
    activate_model,
    create_type,
    delete_type,
    get_inference_job,
    get_model,
    get_training_job,
    import_types_from_embedding_sets,
    list_types,
    update_type,
)

from humpback.schemas.clustering import VocalizationClusteringJobCreate
from humpback.schemas.converters import (
    cluster_to_out as _cluster_to_out,
    clustering_job_to_out as _clustering_job_to_out,
    training_dataset_to_out as _dataset_to_out,
    vocalization_inference_job_to_out as _inference_job_to_out,
    vocalization_model_to_out as _model_to_out,
    vocalization_training_job_to_out as _training_job_to_out,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vocalization", tags=["vocalization"])


# ---- Vocabulary ----


@router.get("/types", response_model=list[VocalizationTypeOut])
async def list_vocalization_types(session: SessionDep):
    types = await list_types(session)
    return [VocalizationTypeOut.model_validate(t) for t in types]


@router.post("/types", response_model=VocalizationTypeOut, status_code=201)
async def create_vocalization_type(body: VocalizationTypeCreate, session: SessionDep):
    if body.name.strip().lower() == "(negative)":
        raise HTTPException(
            400, '"(Negative)" is reserved and cannot be used as a type name'
        )
    try:
        vt = await create_type(session, body.name, body.description)
    except Exception as e:
        if "UNIQUE" in str(e).upper():
            raise HTTPException(409, f"Type '{body.name}' already exists") from e
        raise
    return VocalizationTypeOut.model_validate(vt)


@router.put("/types/{type_id}", response_model=VocalizationTypeOut)
async def update_vocalization_type(
    type_id: str, body: VocalizationTypeUpdate, session: SessionDep
):
    if body.name is not None and body.name.strip().lower() == "(negative)":
        raise HTTPException(
            400, '"(Negative)" is reserved and cannot be used as a type name'
        )
    vt = await update_type(session, type_id, body.name, body.description)
    if vt is None:
        raise HTTPException(404, "Vocalization type not found")
    return VocalizationTypeOut.model_validate(vt)


@router.delete("/types/{type_id}", status_code=204)
async def delete_vocalization_type(type_id: str, session: SessionDep):
    try:
        deleted = await delete_type(session, type_id)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    if not deleted:
        raise HTTPException(404, "Vocalization type not found")


@router.post("/types/import", response_model=VocalizationTypeImportResponse)
async def import_vocalization_types(
    body: VocalizationTypeImportRequest, session: SessionDep
):
    added, skipped = await import_types_from_embedding_sets(
        session, body.embedding_set_ids
    )
    return VocalizationTypeImportResponse(added=added, skipped=skipped)


# ---- Models ----


@router.get("/models", response_model=list[VocalizationModelOut])
async def list_vocalization_models(session: SessionDep):
    result = await session.execute(
        select(VocalizationClassifierModel).order_by(
            VocalizationClassifierModel.created_at.desc()
        )
    )
    return [_model_to_out(m) for m in result.scalars().all()]


@router.get("/models/{model_id}", response_model=VocalizationModelOut)
async def get_vocalization_model(model_id: str, session: SessionDep):
    model = await get_model(session, model_id)
    if model is None:
        raise HTTPException(404, "Vocalization model not found")
    return _model_to_out(model)


@router.put("/models/{model_id}/activate", response_model=VocalizationModelOut)
async def activate_vocalization_model(model_id: str, session: SessionDep):
    model = await activate_model(session, model_id)
    if model is None:
        raise HTTPException(404, "Vocalization model not found")
    return _model_to_out(model)


@router.get(
    "/models/{model_id}/training-source",
    response_model=VocalizationTrainingSourceOut,
)
async def get_model_training_source(model_id: str, session: SessionDep):
    """Return the source_config and parameters from the training job that
    produced this model."""
    model = await get_model(session, model_id)
    if model is None:
        raise HTTPException(404, "Vocalization model not found")

    # Find the training job that produced this model
    result = await session.execute(
        select(VocalizationTrainingJob).where(
            VocalizationTrainingJob.vocalization_model_id == model_id
        )
    )
    training_job = result.scalar_one_or_none()
    if training_job is None:
        return VocalizationTrainingSourceOut()

    return VocalizationTrainingSourceOut(
        source_config=json.loads(training_job.source_config),
        parameters=json.loads(training_job.parameters)
        if training_job.parameters
        else None,
    )


# ---- Training Jobs ----


@router.post(
    "/training-jobs", response_model=VocalizationTrainingJobOut, status_code=201
)
async def create_vocalization_training_job(
    body: VocalizationTrainingJobCreate, session: SessionDep
):
    if body.source_config and body.training_dataset_id:
        raise HTTPException(
            400, "Provide source_config or training_dataset_id, not both"
        )
    if not body.source_config and not body.training_dataset_id:
        raise HTTPException(400, "Provide source_config or training_dataset_id")

    source_config_json = (
        json.dumps(body.source_config.model_dump())
        if body.source_config
        else json.dumps({})
    )
    job = VocalizationTrainingJob(
        source_config=source_config_json,
        training_dataset_id=body.training_dataset_id,
        parameters=json.dumps(body.parameters) if body.parameters else None,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return _training_job_to_out(job)


@router.get("/training-jobs", response_model=list[VocalizationTrainingJobOut])
async def list_vocalization_training_jobs(session: SessionDep):
    result = await session.execute(
        select(VocalizationTrainingJob).order_by(
            VocalizationTrainingJob.created_at.desc()
        )
    )
    return [_training_job_to_out(j) for j in result.scalars().all()]


@router.get("/training-jobs/{job_id}", response_model=VocalizationTrainingJobOut)
async def get_vocalization_training_job_detail(job_id: str, session: SessionDep):
    job = await get_training_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Vocalization training job not found")
    return _training_job_to_out(job)


# ---- Inference Jobs ----


@router.post(
    "/inference-jobs", response_model=VocalizationInferenceJobOut, status_code=201
)
async def create_vocalization_inference_job(
    body: VocalizationInferenceJobCreate, session: SessionDep
):
    # Validate model exists
    model = await get_model(session, body.vocalization_model_id)
    if model is None:
        raise HTTPException(404, "Vocalization model not found")

    job = VocalizationInferenceJob(
        vocalization_model_id=body.vocalization_model_id,
        source_type=body.source_type,
        source_id=body.source_id,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return _inference_job_to_out(job)


@router.get("/inference-jobs", response_model=list[VocalizationInferenceJobOut])
async def list_vocalization_inference_jobs(session: SessionDep):
    result = await session.execute(
        select(VocalizationInferenceJob).order_by(
            VocalizationInferenceJob.created_at.desc()
        )
    )
    return [_inference_job_to_out(j) for j in result.scalars().all()]


@router.get("/inference-jobs/{job_id}", response_model=VocalizationInferenceJobOut)
async def get_vocalization_inference_job_detail(job_id: str, session: SessionDep):
    job = await get_inference_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Vocalization inference job not found")
    return _inference_job_to_out(job)


@router.get(
    "/inference-jobs/{job_id}/results",
    response_model=list[VocalizationPredictionRow],
)
async def get_vocalization_inference_results(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    thresholds: str | None = Query(None),
    sort: str | None = Query(None),
):
    """Paginated results with optional threshold overrides.

    thresholds query param: JSON string like {"whup": 0.3, "moan": 0.6}
    """
    from pathlib import Path

    from humpback.classifier.vocalization_inference import read_predictions

    job = await get_inference_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Vocalization inference job not found")
    if job.status != "complete":
        raise HTTPException(400, f"Job is {job.status}, not complete")
    if not job.output_path:
        raise HTTPException(404, "No output for this job")

    model = await get_model(session, job.vocalization_model_id)
    if model is None:
        raise HTTPException(404, "Model not found")

    vocabulary: list[str] = json.loads(model.vocabulary_snapshot)
    stored_thresholds: dict[str, float] = json.loads(model.per_class_thresholds)

    threshold_overrides: dict[str, float] | None = None
    if thresholds:
        try:
            threshold_overrides = json.loads(thresholds)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid thresholds JSON")

    rows = read_predictions(
        Path(job.output_path), vocabulary, stored_thresholds, threshold_overrides
    )

    # Resolve start_utc/end_utc from row store for row_id-keyed predictions
    if rows and "row_id" in rows[0] and "start_utc" not in rows[0]:
        if job.source_type == "detection_job":
            from humpback.classifier.detection_rows import read_detection_row_store
            from humpback.storage import detection_row_store_path

            rs_path = detection_row_store_path(settings.storage_root, job.source_id)
            if rs_path.exists():
                _fields, rs_rows = read_detection_row_store(rs_path)
                utc_by_rid: dict[str, tuple[float, float]] = {}
                for rr in rs_rows:
                    rid = rr.get("row_id", "")
                    if rid:
                        utc_by_rid[rid] = (
                            float(rr.get("start_utc", "0")),
                            float(rr.get("end_utc", "0")),
                        )
                for r in rows:
                    rid = r.get("row_id")
                    if rid and rid in utc_by_rid:
                        r["start_utc"] = utc_by_rid[rid][0]
                        r["end_utc"] = utc_by_rid[rid][1]

    # Server-side sort (must happen before pagination so limit applies to sorted order)
    if sort == "confidence_desc":
        rows.sort(key=lambda r: r.get("confidence") or -1.0, reverse=True)
    elif sort == "score_desc":
        rows.sort(
            key=lambda r: max(r["scores"].values()) if r["scores"] else 0.0,
            reverse=True,
        )
    elif sort == "uncertainty":
        avg_threshold = (
            sum(stored_thresholds.values()) / len(stored_thresholds)
            if stored_thresholds
            else 0.5
        )
        if threshold_overrides:
            merged = {**stored_thresholds, **threshold_overrides}
            avg_threshold = sum(merged.values()) / len(merged) if merged else 0.5
        rows.sort(
            key=lambda r: abs(
                (max(r["scores"].values()) if r["scores"] else 0.0) - avg_threshold
            )
        )
    elif sort == "chronological":
        rows.sort(key=lambda r: r.get("start_utc") or r.get("start_sec") or 0.0)

    # Paginate
    page = rows[offset : offset + limit]
    return [
        VocalizationPredictionRow(
            row_id=r.get("row_id"),
            filename=r.get("filename"),
            start_sec=r.get("start_sec"),
            end_sec=r.get("end_sec"),
            start_utc=r.get("start_utc"),
            end_utc=r.get("end_utc"),
            confidence=r.get("confidence"),
            scores=r["scores"],
            tags=r["tags"],
        )
        for r in page
    ]


@router.get("/inference-jobs/{job_id}/export")
async def export_vocalization_inference(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    thresholds: str | None = Query(None),
):
    """Export results as TSV with applied thresholds."""
    import io
    from pathlib import Path

    from humpback.classifier.vocalization_inference import read_predictions

    job = await get_inference_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Vocalization inference job not found")
    if job.status != "complete":
        raise HTTPException(400, f"Job is {job.status}, not complete")
    if not job.output_path:
        raise HTTPException(404, "No output for this job")

    model = await get_model(session, job.vocalization_model_id)
    if model is None:
        raise HTTPException(404, "Model not found")

    vocabulary: list[str] = json.loads(model.vocabulary_snapshot)
    stored_thresholds: dict[str, float] = json.loads(model.per_class_thresholds)

    threshold_overrides: dict[str, float] | None = None
    if thresholds:
        try:
            threshold_overrides = json.loads(thresholds)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid thresholds JSON")

    rows = read_predictions(
        Path(job.output_path), vocabulary, stored_thresholds, threshold_overrides
    )

    # Build TSV
    buf = io.StringIO()
    header_cols: list[str] = []
    has_row_id = rows and "row_id" in rows[0]
    has_filename = rows and "filename" in rows[0]
    has_utc = rows and "start_utc" in rows[0]
    if has_row_id:
        header_cols.append("row_id")
    if has_filename:
        header_cols.extend(["filename", "start_sec", "end_sec"])
    if has_utc:
        header_cols.extend(["start_utc", "end_utc"])
    header_cols.extend(vocabulary)
    header_cols.append("tags")
    buf.write("\t".join(header_cols) + "\n")

    for r in rows:
        vals: list[str] = []
        if has_row_id:
            vals.append(r.get("row_id", ""))
        if has_filename:
            vals.extend(
                [
                    r.get("filename", ""),
                    str(r.get("start_sec", "")),
                    str(r.get("end_sec", "")),
                ]
            )
        if has_utc:
            vals.extend([str(r.get("start_utc", "")), str(r.get("end_utc", ""))])
        for t in vocabulary:
            vals.append(str(r["scores"].get(t, 0.0)))
        vals.append(",".join(r["tags"]))
        buf.write("\t".join(vals) + "\n")

    content = buf.getvalue()

    return StreamingResponse(
        iter([content]),
        media_type="text/tab-separated-values",
        headers={
            "Content-Disposition": f'attachment; filename="vocalization_{job_id}.tsv"'
        },
    )


# ---- Training Datasets ----


async def _get_dataset_or_404(session: SessionDep, dataset_id: str) -> TrainingDataset:
    result = await session.execute(
        select(TrainingDataset).where(TrainingDataset.id == dataset_id)
    )
    ds = result.scalar_one_or_none()
    if ds is None:
        raise HTTPException(404, "Training dataset not found")
    return ds


@router.get("/training-datasets")
async def list_training_datasets(session: SessionDep):
    result = await session.execute(
        select(TrainingDataset).order_by(TrainingDataset.created_at.desc())
    )
    return [_dataset_to_out(d) for d in result.scalars().all()]


@router.get("/training-datasets/{dataset_id}")
async def get_training_dataset(dataset_id: str, session: SessionDep):
    ds = await _get_dataset_or_404(session, dataset_id)
    return _dataset_to_out(ds)


@router.get("/training-datasets/{dataset_id}/rows")
async def get_training_dataset_rows(
    dataset_id: str,
    session: SessionDep,
    type: str | None = Query(None),
    group: str | None = Query(None),
    source_type: str | None = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
):
    """Paginated rows with labels, filterable by type, group, and source_type."""

    import pyarrow.parquet as pq

    ds = await _get_dataset_or_404(session, dataset_id)

    # Load labels grouped by row_index
    label_result = await session.execute(
        select(TrainingDatasetLabel).where(
            TrainingDatasetLabel.training_dataset_id == dataset_id
        )
    )
    all_labels = label_result.scalars().all()
    labels_by_row: dict[int, list[str]] = {}
    label_objs_by_row: dict[int, list] = {}
    for lbl in all_labels:
        labels_by_row.setdefault(lbl.row_index, []).append(lbl.label)
        label_objs_by_row.setdefault(lbl.row_index, []).append(lbl)

    # Read parquet metadata columns (not embeddings)
    table = pq.read_table(
        ds.parquet_path,
        columns=[
            "row_index",
            "filename",
            "start_sec",
            "end_sec",
            "source_type",
            "source_id",
            "confidence",
        ],
    )

    # Build row dicts
    rows = []
    for i in range(table.num_rows):
        # Filter by source_type (before label filters)
        if source_type is not None:
            if table.column("source_type")[i].as_py() != source_type:
                continue

        row_idx = int(table.column("row_index")[i].as_py())
        row_label_names = labels_by_row.get(row_idx, [])

        # Filter by type and group
        if type is not None and group is not None:
            has_type = type in row_label_names
            if group == "positive" and not has_type:
                continue
            if group == "negative" and has_type:
                continue

        conf_val = table.column("confidence")[i].as_py()
        row_label_objs = [
            TrainingDatasetRowLabelOut(id=lbl.id, label=lbl.label)
            for lbl in label_objs_by_row.get(row_idx, [])
        ]
        rows.append(
            TrainingDatasetRowOut(
                row_index=row_idx,
                filename=table.column("filename")[i].as_py(),
                start_sec=float(table.column("start_sec")[i].as_py()),
                end_sec=float(table.column("end_sec")[i].as_py()),
                source_type=table.column("source_type")[i].as_py(),
                source_id=table.column("source_id")[i].as_py(),
                confidence=float(conf_val) if conf_val is not None else None,
                labels=row_label_objs,
            )
        )

    # Paginate
    total = len(rows)
    page = rows[offset : offset + limit]
    return {"total": total, "rows": page}


@router.get("/training-datasets/{dataset_id}/spectrogram")
async def get_training_dataset_spectrogram(
    dataset_id: str,
    session: SessionDep,
    settings: SettingsDep,
    row_index: int = Query(..., ge=0),
):
    """Spectrogram PNG for a training dataset row, delegated to source audio."""
    import asyncio

    from humpback.processing.spectrogram import generate_spectrogram_png

    ds = await _get_dataset_or_404(session, dataset_id)
    audio, sr = await _resolve_training_row_audio(ds, row_index, session, settings)

    png_bytes = await asyncio.to_thread(
        generate_spectrogram_png,
        audio,
        sr,
        hop_length=settings.spectrogram_hop_length,
        dynamic_range_db=settings.spectrogram_dynamic_range_db,
        width_px=settings.spectrogram_width_px,
        height_px=settings.spectrogram_height_px,
    )
    return Response(content=png_bytes, media_type="image/png")


@router.get("/training-datasets/{dataset_id}/audio-slice")
async def get_training_dataset_audio_slice(
    dataset_id: str,
    session: SessionDep,
    settings: SettingsDep,
    row_index: int = Query(..., ge=0),
    normalize: bool = Query(True),
):
    """Audio WAV for a training dataset row, delegated to source audio."""
    ds = await _get_dataset_or_404(session, dataset_id)
    audio, sr = await _resolve_training_row_audio(ds, row_index, session, settings)

    # Reuse the WAV encoder from classifier router
    import io
    import struct

    import numpy as np

    if normalize:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    data_size = len(pcm) * 2
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


async def _resolve_training_row_audio(ds, row_index, session, settings):
    """Read source audio for a training dataset row."""
    import asyncio
    from pathlib import Path

    import numpy as np
    import pyarrow.parquet as pq

    from humpback.classifier.detection_rows import parse_recording_timestamp
    from humpback.processing.audio_io import decode_audio

    table = pq.read_table(
        ds.parquet_path,
        columns=[
            "row_index",
            "source_type",
            "source_id",
            "filename",
            "start_sec",
            "end_sec",
        ],
    )
    if row_index >= table.num_rows:
        raise HTTPException(404, f"Row {row_index} not found")

    source_type = table.column("source_type")[row_index].as_py()
    source_id = table.column("source_id")[row_index].as_py()
    filename = table.column("filename")[row_index].as_py()
    start_sec = float(table.column("start_sec")[row_index].as_py())
    end_sec = float(table.column("end_sec")[row_index].as_py())
    duration_sec = end_sec - start_sec

    if source_type == "detection_job":
        # Delegate to detection job audio resolution
        from humpback.services import classifier_service

        job = await classifier_service.get_detection_job(session, source_id)
        if job is None:
            raise HTTPException(404, f"Detection job {source_id} not found")

        ts = parse_recording_timestamp(filename)
        start_utc = (ts.timestamp() if ts else 0.0) + start_sec

        if job.hydrophone_id:
            from humpback.classifier.s3_stream import resolve_audio_slice
            from humpback.config import get_archive_source

            cache_path = job.local_cache_path or settings.s3_cache_path
            src = get_archive_source(job.hydrophone_id)
            if (
                src is not None
                and src.get("provider_kind") == "noaa_gcs"
                and settings.noaa_cache_path is not None
            ):
                from humpback.api.routers.classifier import (
                    _noaa_provider_registry,
                )

                local_provider = _noaa_provider_registry.get_or_create(
                    job.hydrophone_id,
                    cache_path,
                    settings.noaa_cache_path,
                )
            else:
                from humpback.api.routers.classifier import (
                    build_archive_playback_provider,
                )

                local_provider = build_archive_playback_provider(
                    job.hydrophone_id,
                    cache_path=cache_path,
                    noaa_cache_path=settings.noaa_cache_path,
                )
            target_sr = 32000
            if job.start_timestamp is None or job.end_timestamp is None:
                raise HTTPException(400, "Hydrophone job missing timestamps")
            segment = await asyncio.to_thread(
                resolve_audio_slice,
                local_provider,
                job.start_timestamp,
                job.end_timestamp,
                start_utc,
                duration_sec,
                target_sr,
            )
            return np.asarray(segment), target_sr

        # Local audio
        from humpback.classifier.extractor import (
            _build_local_audio_index,
            _resolve_local_audio_for_row,
        )

        if not job.audio_folder:
            raise HTTPException(400, "Detection job has no audio folder")
        audio_folder = Path(job.audio_folder)
        audio_index = _build_local_audio_index(audio_folder)
        resolved = _resolve_local_audio_for_row(start_utc, audio_index)
        if resolved is None:
            raise HTTPException(404, f"No audio for start_utc={start_utc}")
        file_path, _base_epoch, offset_sec = resolved

        raw_audio, sr = await asyncio.to_thread(decode_audio, file_path)
        start_sample = int(offset_sec * sr)
        end_sample = int((offset_sec + duration_sec) * sr)
        return np.asarray(raw_audio[start_sample:end_sample]), sr

    elif source_type == "embedding_set":
        from humpback.models.audio import AudioFile
        from humpback.models.processing import EmbeddingSet
        from humpback.storage import resolve_audio_path

        es_result = await session.execute(
            select(EmbeddingSet).where(EmbeddingSet.id == source_id)
        )
        es = es_result.scalar_one_or_none()
        if es is None:
            raise HTTPException(404, f"Embedding set {source_id} not found")

        af_result = await session.execute(
            select(AudioFile).where(AudioFile.id == es.audio_file_id)
        )
        af = af_result.scalar_one_or_none()
        if af is None:
            raise HTTPException(404, "Audio file not found")

        audio_path = resolve_audio_path(af, settings.storage_root)
        raw_audio, sr = await asyncio.to_thread(decode_audio, audio_path)
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        return np.asarray(raw_audio[start_sample:end_sample]), sr

    else:
        raise HTTPException(400, f"Unknown source type: {source_type}")


@router.post("/training-datasets/{dataset_id}/extend")
async def extend_training_dataset_endpoint(
    dataset_id: str,
    body: "TrainingDatasetExtendRequest",
    session: SessionDep,
    settings: SettingsDep,
):
    from humpback.services.training_dataset import extend_training_dataset

    ds = await _get_dataset_or_404(session, dataset_id)
    updated = await extend_training_dataset(
        session,
        ds,
        {
            "embedding_set_ids": body.embedding_set_ids,
            "detection_job_ids": body.detection_job_ids,
        },
        settings.storage_root,
    )
    await session.commit()
    return _dataset_to_out(updated)


@router.post("/training-datasets/{dataset_id}/labels", status_code=201)
async def create_training_dataset_label(
    dataset_id: str,
    body: "TrainingDatasetLabelCreate",
    session: SessionDep,
):

    await _get_dataset_or_404(session, dataset_id)

    # Enforce "(Negative)" mutual exclusivity
    existing_result = await session.execute(
        select(TrainingDatasetLabel).where(
            TrainingDatasetLabel.training_dataset_id == dataset_id,
            TrainingDatasetLabel.row_index == body.row_index,
        )
    )
    existing = existing_result.scalars().all()

    # Skip if this exact label already exists on this row
    if any(lbl.label == body.label for lbl in existing):
        dup = next(lbl for lbl in existing if lbl.label == body.label)
        return TrainingDatasetLabelOut.model_validate(dup)

    if body.label == "(Negative)":
        # Remove all type labels for this row
        for lbl in existing:
            if lbl.label != "(Negative)":
                await session.delete(lbl)
    else:
        # Remove any "(Negative)" label for this row
        for lbl in existing:
            if lbl.label == "(Negative)":
                await session.delete(lbl)

    label = TrainingDatasetLabel(
        training_dataset_id=dataset_id,
        row_index=body.row_index,
        label=body.label,
        source="manual",
    )
    session.add(label)
    await session.commit()
    await session.refresh(label)
    return TrainingDatasetLabelOut.model_validate(label)


@router.delete("/training-datasets/{dataset_id}/labels/{label_id}", status_code=204)
async def delete_training_dataset_label(
    dataset_id: str,
    label_id: str,
    session: SessionDep,
):

    result = await session.execute(
        select(TrainingDatasetLabel).where(
            TrainingDatasetLabel.id == label_id,
            TrainingDatasetLabel.training_dataset_id == dataset_id,
        )
    )
    label = result.scalar_one_or_none()
    if label is None:
        raise HTTPException(404, "Label not found")

    await session.delete(label)
    await session.commit()


# ---- Vocalization Clustering ----


@router.get("/clustering-eligible-jobs")
async def list_clustering_eligible_jobs(session: SessionDep):
    from humpback.services.clustering_service import (
        list_clustering_eligible_detection_jobs,
    )

    return await list_clustering_eligible_detection_jobs(session)


@router.post("/clustering-jobs", status_code=201)
async def create_vocalization_clustering_job_endpoint(
    body: "VocalizationClusteringJobCreate",
    session: SessionDep,
    settings: SettingsDep,
):
    from pathlib import Path

    from humpback.services.clustering_service import (
        create_vocalization_clustering_job,
    )

    try:
        job = await create_vocalization_clustering_job(
            session,
            body.detection_job_ids,
            body.parameters,
            storage_root=Path(settings.storage_root),
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return _clustering_job_to_out(job)


@router.get("/clustering-jobs")
async def list_vocalization_clustering_jobs_endpoint(session: SessionDep):
    from humpback.services.clustering_service import (
        list_vocalization_clustering_jobs,
    )

    jobs = await list_vocalization_clustering_jobs(session)
    return [_clustering_job_to_out(j) for j in jobs]


@router.get("/clustering-jobs/{job_id}")
async def get_vocalization_clustering_job(job_id: str, session: SessionDep):
    from humpback.services.clustering_service import get_clustering_job

    job = await get_clustering_job(session, job_id)
    if job is None or job.detection_job_ids is None:
        raise HTTPException(404, "Vocalization clustering job not found")
    return _clustering_job_to_out(job)


@router.delete("/clustering-jobs/{job_id}")
async def delete_vocalization_clustering_job(
    job_id: str, session: SessionDep, settings: SettingsDep
):
    from humpback.services.clustering_service import (
        delete_clustering_job,
        get_clustering_job,
    )

    job = await get_clustering_job(session, job_id)
    if job is None or job.detection_job_ids is None:
        raise HTTPException(404, "Vocalization clustering job not found")
    from pathlib import Path

    await delete_clustering_job(session, job_id, Path(settings.storage_root))
    return {"status": "deleted"}


@router.get("/clustering-jobs/{job_id}/clusters")
async def get_vocalization_clustering_clusters(job_id: str, session: SessionDep):
    from humpback.services.clustering_service import get_clustering_job, list_clusters

    job = await get_clustering_job(session, job_id)
    if job is None or job.detection_job_ids is None:
        raise HTTPException(404, "Vocalization clustering job not found")
    clusters = await list_clusters(session, job_id)
    return [_cluster_to_out(c) for c in clusters]


@router.get("/clustering-jobs/{job_id}/visualization")
async def get_vocalization_clustering_visualization(
    job_id: str, session: SessionDep, settings: SettingsDep
):
    from pathlib import Path

    import pyarrow.parquet as pq

    from humpback.models.detection_embedding_job import DetectionEmbeddingJob
    from humpback.services.clustering_service import get_clustering_job
    from humpback.storage import (
        cluster_dir,
        detection_embeddings_path,
        detection_row_store_path,
    )

    job = await get_clustering_job(session, job_id)
    if job is None or job.detection_job_ids is None:
        raise HTTPException(404, "Vocalization clustering job not found")
    if job.status != "complete":
        raise HTTPException(400, "Clustering job is not complete")

    umap_path = cluster_dir(Path(settings.storage_root), job_id) / "umap_coords.parquet"
    if not umap_path.exists():
        raise HTTPException(404, "UMAP coordinates not available for this job")

    table = pq.read_table(str(umap_path))
    es_ids = table.column("embedding_set_id").to_pylist()
    row_indices = table.column("embedding_row_index").to_pylist()

    from humpback.models.classifier import DetectionJob

    unique_dj_ids = list(set(es_ids))
    dj_result = await session.execute(
        select(DetectionJob).where(DetectionJob.id.in_(unique_dj_ids))
    )
    dj_map = {dj.id: dj.hydrophone_name or dj.id for dj in dj_result.scalars().all()}

    # Build per-detection-job row_id order and start_utc from row store
    ordered_row_ids_by_dj: dict[str, list[str]] = {}
    start_utc_map: dict[tuple[str, int], float | None] = {}

    for dj_id in unique_dj_ids:
        emb_result = await session.execute(
            select(DetectionEmbeddingJob)
            .where(
                DetectionEmbeddingJob.detection_job_id == dj_id,
                DetectionEmbeddingJob.status == "complete",
            )
            .order_by(DetectionEmbeddingJob.created_at.desc())
            .limit(1)
        )
        emb_job = emb_result.scalar_one_or_none()
        if emb_job is None:
            continue

        emb_parquet = detection_embeddings_path(
            Path(settings.storage_root), dj_id, emb_job.model_version
        )
        if not emb_parquet.exists():
            continue

        emb_table = pq.read_table(str(emb_parquet), columns=["row_id"])
        ordered_row_ids = emb_table.column("row_id").to_pylist()
        ordered_row_ids_by_dj[dj_id] = ordered_row_ids

        row_store = detection_row_store_path(Path(settings.storage_root), dj_id)
        if row_store.exists():
            rs_table = pq.read_table(str(row_store), columns=["row_id", "start_utc"])
            rs_row_ids = rs_table.column("row_id").to_pylist()
            rs_start_utcs = rs_table.column("start_utc").to_pylist()
            rid_to_utc: dict[str, float] = {}
            for ri, su in zip(rs_row_ids, rs_start_utcs):
                if su:
                    try:
                        rid_to_utc[ri] = float(su)
                    except (ValueError, TypeError):
                        pass
            for idx, rid in enumerate(ordered_row_ids):
                start_utc_map[(dj_id, idx)] = rid_to_utc.get(rid)

    # Build (detection_job_id, row_index) → vocalization label mapping
    voc_label_map: dict[tuple[str, int], str] = {}

    active_model_result = await session.execute(
        select(VocalizationClassifierModel).where(
            VocalizationClassifierModel.is_active.is_(True)
        )
    )
    active_model = active_model_result.scalar_one_or_none()

    if active_model is not None:
        vocabulary: list[str] = json.loads(active_model.vocabulary_snapshot)
        thresholds: dict[str, float] = json.loads(active_model.per_class_thresholds)

        for dj_id in unique_dj_ids:
            ordered_row_ids = ordered_row_ids_by_dj.get(dj_id)
            if ordered_row_ids is None:
                continue

            inf_result = await session.execute(
                select(VocalizationInferenceJob).where(
                    VocalizationInferenceJob.source_type == "detection_job",
                    VocalizationInferenceJob.source_id == dj_id,
                    VocalizationInferenceJob.vocalization_model_id == active_model.id,
                    VocalizationInferenceJob.status == "complete",
                )
            )
            inf_job = inf_result.scalar_one_or_none()
            if inf_job is None or not inf_job.output_path:
                continue

            pred_path = Path(inf_job.output_path)
            if not pred_path.exists():
                continue

            pred_table = pq.read_table(str(pred_path))
            pred_cols = set(pred_table.column_names)
            pred_row_ids = (
                pred_table.column("row_id").to_pylist() if "row_id" in pred_cols else []
            )

            row_id_to_label: dict[str, str] = {}
            for pi in range(pred_table.num_rows):
                rid = pred_row_ids[pi] if pred_row_ids else str(pi)
                best_type = ""
                best_score = -1.0
                for type_name in vocabulary:
                    if type_name not in pred_cols:
                        continue
                    score = float(pred_table.column(type_name)[pi].as_py())
                    t = thresholds.get(type_name, 0.5)
                    if score >= t and score > best_score:
                        best_score = score
                        best_type = type_name
                row_id_to_label[rid] = best_type or "unlabeled"

            for idx, rid in enumerate(ordered_row_ids):
                label = row_id_to_label.get(rid, "unlabeled")
                voc_label_map[(dj_id, idx)] = label

    hydrophone_names = [dj_map.get(es_id, es_id) for es_id in es_ids]
    categories = [
        voc_label_map.get((es_ids[i], row_indices[i]), "unlabeled")
        for i in range(len(es_ids))
    ]
    start_utcs = [
        start_utc_map.get((es_ids[i], row_indices[i])) for i in range(len(es_ids))
    ]

    return {
        "x": table.column("x").to_pylist(),
        "y": table.column("y").to_pylist(),
        "cluster_label": table.column("cluster_label").to_pylist(),
        "embedding_set_id": es_ids,
        "embedding_row_index": row_indices,
        "audio_filename": hydrophone_names,
        "audio_file_id": [""] * len(es_ids),
        "window_size_seconds": [5.0] * len(es_ids),
        "category": categories,
        "start_utc": start_utcs,
    }


@router.get("/clustering-jobs/{job_id}/metrics")
async def get_vocalization_clustering_metrics(job_id: str, session: SessionDep):
    from humpback.services.clustering_service import get_clustering_job

    job = await get_clustering_job(session, job_id)
    if job is None or job.detection_job_ids is None:
        raise HTTPException(404, "Vocalization clustering job not found")
    if not job.metrics_json:
        return {}
    return json.loads(job.metrics_json)


@router.get("/clustering-jobs/{job_id}/stability")
async def get_vocalization_clustering_stability(
    job_id: str, session: SessionDep, settings: SettingsDep
):
    from pathlib import Path

    from humpback.services.clustering_service import get_clustering_job
    from humpback.storage import cluster_dir

    job = await get_clustering_job(session, job_id)
    if job is None or job.detection_job_ids is None:
        raise HTTPException(404, "Vocalization clustering job not found")
    if job.status != "complete":
        raise HTTPException(400, "Clustering job is not complete")

    stability_path = (
        cluster_dir(Path(settings.storage_root), job_id) / "stability_summary.json"
    )
    if not stability_path.exists():
        raise HTTPException(404, "Stability data not available")

    return json.loads(stability_path.read_text())
