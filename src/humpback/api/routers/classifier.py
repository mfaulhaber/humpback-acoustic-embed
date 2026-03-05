"""API router for binary classifier training and detection."""

import csv
import io
import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from humpback.api.deps import SessionDep, SettingsDep
from humpback.schemas.classifier import (
    ClassifierModelOut,
    ClassifierTrainingJobCreate,
    ClassifierTrainingJobOut,
    DetectionJobCreate,
    DetectionJobOut,
)
from humpback.services import classifier_service

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
        output_tsv_path=job.output_tsv_path,
        result_summary=json.loads(job.result_summary) if job.result_summary else None,
        error_message=job.error_message,
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
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _detection_job_to_out(job)


@router.get("/detection-jobs")
async def list_detection_jobs(session: SessionDep) -> list[DetectionJobOut]:
    jobs = await classifier_service.list_detection_jobs(session)
    return [_detection_job_to_out(j) for j in jobs]


@router.get("/detection-jobs/{job_id}")
async def get_detection_job(
    job_id: str, session: SessionDep
) -> DetectionJobOut:
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    return _detection_job_to_out(job)


@router.get("/detection-jobs/{job_id}/download")
async def download_detections(
    job_id: str, session: SessionDep
) -> FileResponse:
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    if job.status != "complete" or not job.output_tsv_path:
        raise HTTPException(400, "Detection job not complete or no output available")
    from pathlib import Path
    tsv_path = Path(job.output_tsv_path)
    if not tsv_path.is_file():
        raise HTTPException(404, "TSV file not found on disk")
    return FileResponse(
        tsv_path,
        media_type="text/tab-separated-values",
        filename=f"detections_{job_id}.tsv",
    )


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


def _parse_label(value: str | None) -> int | None:
    """Parse a label column value: '0' → 0, '1' → 1, else → None."""
    if value is not None:
        value = value.strip()
    if value == "0":
        return 0
    if value == "1":
        return 1
    return None


@router.get("/detection-jobs/{job_id}/content")
async def get_detection_content(job_id: str, session: SessionDep) -> list[dict]:
    """Parse detection TSV and return rows as JSON."""
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    if job.status != "complete" or not job.output_tsv_path:
        raise HTTPException(400, "Detection job not complete or no output available")
    tsv_path = Path(job.output_tsv_path)
    if not tsv_path.is_file():
        raise HTTPException(404, "TSV file not found on disk")

    rows = []
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "filename": row.get("filename", ""),
                    "start_sec": float(row.get("start_sec", 0)),
                    "end_sec": float(row.get("end_sec", 0)),
                    "avg_confidence": float(row.get("avg_confidence", 0)),
                    "peak_confidence": float(row.get("peak_confidence", 0)),
                    "humpback": _parse_label(row.get("humpback")),
                    "ship": _parse_label(row.get("ship")),
                    "background": _parse_label(row.get("background")),
                }
            )
    return rows


# ---- Detection Labels ----


class DetectionLabelRow(BaseModel):
    filename: str
    start_sec: float
    end_sec: float
    humpback: Optional[int] = None
    ship: Optional[int] = None
    background: Optional[int] = None


def _serialize_label(value: int | None) -> str:
    """Serialize label value for TSV: None → '', 0 → '0', 1 → '1'."""
    if value is None:
        return ""
    return str(value)


@router.put("/detection-jobs/{job_id}/labels")
async def save_detection_labels(
    job_id: str, body: list[DetectionLabelRow], session: SessionDep
) -> dict:
    """Merge label annotations into the detection TSV file."""
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    if job.status != "complete" or not job.output_tsv_path:
        raise HTTPException(400, "Detection job not complete or no output available")
    tsv_path = Path(job.output_tsv_path)
    if not tsv_path.is_file():
        raise HTTPException(404, "TSV file not found on disk")

    # Build lookup of label updates keyed by (filename, start_sec, end_sec)
    label_map: dict[tuple[str, float, float], DetectionLabelRow] = {}
    for row in body:
        label_map[(row.filename, row.start_sec, row.end_sec)] = row

    # Read existing TSV
    existing_rows = []
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            existing_rows.append(row)

    # Merge labels
    fieldnames = [
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "humpback",
        "ship",
        "background",
    ]
    updated_rows = []
    for row in existing_rows:
        key = (
            row.get("filename", ""),
            float(row.get("start_sec", 0)),
            float(row.get("end_sec", 0)),
        )
        update = label_map.get(key)
        out_row = {
            "filename": row.get("filename", ""),
            "start_sec": row.get("start_sec", "0"),
            "end_sec": row.get("end_sec", "0"),
            "avg_confidence": row.get("avg_confidence", "0"),
            "peak_confidence": row.get("peak_confidence", "0"),
            "humpback": _serialize_label(update.humpback) if update else row.get("humpback", ""),
            "ship": _serialize_label(update.ship) if update else row.get("ship", ""),
            "background": _serialize_label(update.background) if update else row.get("background", ""),
        }
        updated_rows.append(out_row)

    # Write atomically via temp file
    tsv_dir = tsv_path.parent
    fd, tmp_path = tempfile.mkstemp(dir=str(tsv_dir), suffix=".tsv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(updated_rows)
        os.replace(tmp_path, str(tsv_path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return {"status": "ok", "updated": len(label_map)}


# ---- Audio Slice Streaming ----


@router.get("/detection-jobs/{job_id}/audio-slice")
async def get_detection_audio_slice(
    job_id: str,
    session: SessionDep,
    filename: str = Query(...),
    start_sec: float = Query(..., ge=0),
    duration_sec: float = Query(..., gt=0),
    normalize: bool = Query(True),
):
    """Stream a WAV slice from a detection job's audio folder."""
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")

    audio_folder = Path(job.audio_folder)
    file_path = audio_folder / filename

    # Path traversal check
    try:
        file_path.resolve().relative_to(audio_folder.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid filename (path traversal)")

    if not file_path.is_file():
        raise HTTPException(404, f"Audio file not found: {filename}")

    import asyncio

    import numpy as np

    from humpback.processing.audio_io import decode_audio

    audio, sr = await asyncio.to_thread(decode_audio, file_path)

    total_duration = len(audio) / sr
    # Clamp start so we always have audio to return: if the requested
    # range extends past the end of the file (common for the last detection
    # window which was zero-padded), shift start backwards.
    end_sec_requested = start_sec + duration_sec
    if end_sec_requested > total_duration:
        start_sec = max(0.0, total_duration - duration_sec)

    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    start_sample = min(start_sample, len(audio))
    end_sample = min(end_sample, len(audio))
    segment = audio[start_sample:end_sample]

    # Peak-normalize so every clip plays at consistent loudness
    if normalize:
        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment / peak

    # Encode as 16-bit PCM WAV in memory
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
