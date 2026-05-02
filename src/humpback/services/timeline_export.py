"""Export a detection job's timeline as a self-contained static bundle."""

from __future__ import annotations

import json
import logging
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.detection_rows import (
    normalize_detection_row,
    read_detection_row_store,
)
from humpback.config import Settings
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.labeling import VocalizationLabel
from humpback.models.vocalization import (
    VocalizationClassifierModel,
    VocalizationInferenceJob,
    VocalizationType,
)
from humpback.processing.timeline_cache import TimelineTileCache
from humpback.processing.timeline_renderers import DEFAULT_TIMELINE_RENDERER
from humpback.processing.timeline_repository import TimelineTileRequest
from humpback.processing.timeline_tiles import (
    ZOOM_LEVELS,
    tile_count,
    tile_duration_sec,
)
from humpback.services.timeline_tile_service import (
    pcm_cache_bytes_limit,
    repository_from_settings,
    source_ref_from_job,
)
from humpback.storage import detection_diagnostics_path, detection_row_store_path

logger = logging.getLogger(__name__)

_AUDIO_CHUNK_SEC = 300
_AUDIO_SAMPLE_RATE = 32000
_TILE_SIZE = (512, 256)
_MANIFEST_VERSION = 1

# Labels in priority order for flattening (first non-null wins).
_LABEL_COLUMNS = ("humpback", "orca", "ship", "background")


@dataclass
class ExportResult:
    job_id: str
    output_path: str
    tile_count: int
    audio_chunk_count: int
    manifest_size_bytes: int


class ExportError(Exception):
    """Raised when export preconditions are not met."""

    def __init__(self, message: str, *, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


async def export_timeline(
    job_id: str,
    output_dir: Path,
    db: AsyncSession,
    settings: Settings,
    *,
    progress_callback: Any | None = None,
) -> ExportResult:
    """Export a detection job timeline to a self-contained directory.

    Args:
        job_id: Detection job UUID.
        output_dir: Parent directory; output goes to ``output_dir / job_id``.
        db: Async database session.
        settings: Application settings.
        progress_callback: Optional ``(stage, current, total)`` callable for
            progress reporting (e.g. CLI stderr output).
    """
    # ---- 1. Validate ----
    result = await db.execute(select(DetectionJob).where(DetectionJob.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise ExportError("Detection job not found", status_code=404)
    if job.status != "complete":
        raise ExportError("Detection job must be complete before export")
    if not job.hydrophone_id:
        raise ExportError("Only hydrophone detection jobs can be exported")

    job_start = job.start_timestamp or 0.0
    job_end = job.end_timestamp or 0.0
    job_duration = job_end - job_start
    if job_duration <= 0:
        raise ExportError("Detection job has no valid time range")

    cache = TimelineTileCache(
        settings.storage_root / "timeline_cache",
        max_jobs=settings.timeline_cache_max_jobs,
        memory_cache_max_items=0,  # No memory cache needed for export
    )
    repository = repository_from_settings(settings)
    source_ref = source_ref_from_job(job, settings)
    renderer = DEFAULT_TIMELINE_RENDERER

    # Check all tiles are rendered; prepare if needed
    expected_tiles = 0
    needs_prepare = False
    for zoom in ZOOM_LEVELS:
        expected = tile_count(zoom, job_duration_sec=job_duration)
        actual = repository.tile_count_for_zoom(
            source_ref,
            renderer.renderer_id,
            renderer.version,
            zoom,
            freq_min=0,
            freq_max=3000,
            width_px=settings.timeline_tile_width_px,
            height_px=settings.timeline_tile_height_px,
        )
        if actual < expected:
            needs_prepare = True
        expected_tiles += expected

    if needs_prepare:
        if progress_callback:
            progress_callback("prepare", 0, expected_tiles)

        import asyncio

        from humpback.api.routers.timeline import _prepare_tiles_sync

        await asyncio.to_thread(
            _prepare_tiles_sync,
            job=job,
            settings=settings,
            cache=cache,
        )

        # Verify all tiles now exist
        for zoom in ZOOM_LEVELS:
            expected = tile_count(zoom, job_duration_sec=job_duration)
            actual = repository.tile_count_for_zoom(
                source_ref,
                renderer.renderer_id,
                renderer.version,
                zoom,
                freq_min=0,
                freq_max=3000,
                width_px=settings.timeline_tile_width_px,
                height_px=settings.timeline_tile_height_px,
            )
            if actual < expected:
                raise ExportError(
                    f"Tile preparation incomplete at {zoom}: {actual}/{expected}"
                )

        if progress_callback:
            progress_callback("prepare", expected_tiles, expected_tiles)

    # ---- 2. Create output directory ----
    job_dir = output_dir / job_id
    tiles_dir = job_dir / "tiles"
    audio_dir = job_dir / "audio"
    for zoom in ZOOM_LEVELS:
        (tiles_dir / zoom).mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # ---- 3. Copy tiles ----
    total_copied = 0
    for zoom in ZOOM_LEVELS:
        n_tiles = tile_count(zoom, job_duration_sec=job_duration)
        dst_dir = tiles_dir / zoom
        for i in range(n_tiles):
            src = repository.tile_path(
                source_ref,
                renderer.renderer_id,
                renderer.version,
                TimelineTileRequest(
                    zoom_level=zoom,
                    tile_index=i,
                    freq_min=0,
                    freq_max=3000,
                    width_px=settings.timeline_tile_width_px,
                    height_px=settings.timeline_tile_height_px,
                ),
            )
            dst = dst_dir / f"tile_{i:04d}.png"
            shutil.copy2(src, dst)
            total_copied += 1
        if progress_callback:
            progress_callback("tiles", total_copied, expected_tiles)

    # ---- 4. Generate audio chunks ----
    from humpback.processing.audio_encoding import encode_mp3, normalize_for_playback
    from humpback.processing.timeline_audio import resolve_timeline_audio

    n_chunks = math.ceil(job_duration / _AUDIO_CHUNK_SEC)
    for i in range(n_chunks):
        chunk_start = job_start + i * _AUDIO_CHUNK_SEC
        chunk_duration = min(_AUDIO_CHUNK_SEC, job_end - chunk_start)

        audio = resolve_timeline_audio(
            hydrophone_id=job.hydrophone_id,
            local_cache_path=job.local_cache_path or settings.s3_cache_path or "",
            job_start_timestamp=job_start,
            job_end_timestamp=job_end,
            start_sec=chunk_start,
            duration_sec=chunk_duration,
            target_sr=_AUDIO_SAMPLE_RATE,
            noaa_cache_path=settings.noaa_cache_path,
            timeline_cache=repository,
            job_id=source_ref.span_key,
            manifest_cache_items=settings.timeline_manifest_memory_cache_items,
            pcm_cache_max_bytes=pcm_cache_bytes_limit(settings),
        )
        audio = normalize_for_playback(
            audio,
            target_rms_dbfs=settings.playback_target_rms_dbfs,
            ceiling=settings.playback_ceiling,
        )

        mp3_bytes = encode_mp3(audio, _AUDIO_SAMPLE_RATE)
        chunk_path = audio_dir / f"chunk_{i:04d}.mp3"
        chunk_path.write_bytes(mp3_bytes)

        if progress_callback:
            progress_callback("audio", i + 1, n_chunks)

    # ---- 5. Build manifest ----
    manifest = await _build_manifest(
        job=job,
        job_duration=job_duration,
        db=db,
        settings=settings,
    )
    manifest_bytes = json.dumps(manifest, indent=2).encode()
    (job_dir / "manifest.json").write_bytes(manifest_bytes)

    return ExportResult(
        job_id=job_id,
        output_path=str(job_dir),
        tile_count=expected_tiles,
        audio_chunk_count=n_chunks,
        manifest_size_bytes=len(manifest_bytes),
    )


def _flatten_label(row: dict[str, Any]) -> str | None:
    """Extract the single active label from a normalized detection row."""
    for col in _LABEL_COLUMNS:
        val = row.get(col)
        if val is not None and val != 0:
            return col
    return None


async def _build_manifest(
    *,
    job: DetectionJob,
    job_duration: float,
    db: AsyncSession,
    settings: Settings,
) -> dict[str, Any]:
    """Assemble the full manifest matching the consumer contract schema."""
    job_start = job.start_timestamp or 0.0
    job_end = job.end_timestamp or 0.0

    # ---- Classifier model info ----
    model_result = await db.execute(
        select(ClassifierModel).where(ClassifierModel.id == job.classifier_model_id)
    )
    model = model_result.scalar_one_or_none()
    model_name = model.name if model else "unknown"
    model_version = model.model_version if model else "unknown"
    window_sec = model.window_size_seconds if model else 5.0

    # ---- Tiles ----
    tile_durations: dict[str, float] = {}
    tile_counts: dict[str, int] = {}
    for zoom in ZOOM_LEVELS:
        tile_durations[zoom] = tile_duration_sec(zoom)
        tile_counts[zoom] = tile_count(zoom, job_duration_sec=job_duration)

    # ---- Audio ----
    n_chunks = math.ceil(job_duration / _AUDIO_CHUNK_SEC)

    # ---- Confidence scores ----
    confidence_scores = _read_confidence_scores(
        job_id=job.id,
        job_start=job_start,
        job_end=job_end,
        window_sec=window_sec,
        settings=settings,
    )

    # ---- Detection rows ----
    detections = _read_detections(job_id=job.id, settings=settings)

    # ---- Vocalization labels ----
    voc_labels = await _read_vocalization_labels(
        job_id=job.id, db=db, settings=settings
    )

    # ---- Vocalization types ----
    voc_types = await _read_vocalization_types(db)

    return {
        "version": _MANIFEST_VERSION,
        "job": {
            "id": job.id,
            "hydrophone_name": job.hydrophone_name or "",
            "hydrophone_id": job.hydrophone_id or "",
            "start_timestamp": job_start,
            "end_timestamp": job_end,
            "species": (model_name.split("_")[0] if model else "unknown"),
            "window_selection": job.window_selection or "nms",
            "model_name": model_name,
            "model_version": model_version,
        },
        "tiles": {
            "zoom_levels": list(ZOOM_LEVELS),
            "tile_size": list(_TILE_SIZE),
            "tile_durations": tile_durations,
            "tile_counts": tile_counts,
        },
        "audio": {
            "chunk_duration_sec": _AUDIO_CHUNK_SEC,
            "chunk_count": n_chunks,
            "format": "mp3",
            "sample_rate": _AUDIO_SAMPLE_RATE,
        },
        "confidence": {
            "window_sec": window_sec,
            "scores": confidence_scores,
        },
        "detections": detections,
        "vocalization_labels": voc_labels,
        "vocalization_types": voc_types,
    }


def _read_confidence_scores(
    *,
    job_id: str,
    job_start: float,
    job_end: float,
    window_sec: float,
    settings: Settings,
) -> list[float | None]:
    """Read and bucket confidence scores from diagnostics parquet."""
    diag_path = detection_diagnostics_path(settings.storage_root, job_id)
    if not diag_path.exists():
        return []

    job_duration = job_end - job_start
    table = pq.read_table(str(diag_path))

    offset_col = table.column("offset_sec").to_pylist()
    score_col = table.column("confidence").to_pylist()
    filename_col = (
        table.column("filename").to_pylist()
        if "filename" in table.column_names
        else [None] * len(offset_col)
    )

    # Compute timeline-absolute offset for each row
    absolute_offsets: list[float] = []
    for offset, filename in zip(offset_col, filename_col):
        if filename is not None:
            file_epoch = _parse_filename_epoch(filename)
            if file_epoch is not None:
                absolute_offsets.append(file_epoch - job_start + offset)
                continue
        absolute_offsets.append(offset)

    n_buckets = max(1, int(job_duration / window_sec))
    bucket_sums: list[float] = [0.0] * n_buckets
    bucket_counts: list[int] = [0] * n_buckets

    for abs_offset, score in zip(absolute_offsets, score_col):
        idx = int(abs_offset / window_sec)
        if 0 <= idx < n_buckets:
            bucket_sums[idx] += score
            bucket_counts[idx] += 1

    return [
        bucket_sums[i] / bucket_counts[i] if bucket_counts[i] > 0 else None
        for i in range(n_buckets)
    ]


def _parse_filename_epoch(filename: str) -> float | None:
    """Parse a compact UTC timestamp filename to epoch seconds."""
    import re
    from datetime import datetime, timezone

    m = re.match(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z", filename)
    if not m:
        return None
    dt = datetime(
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        int(m.group(4)),
        int(m.group(5)),
        int(m.group(6)),
        tzinfo=timezone.utc,
    )
    return dt.timestamp()


def _read_detections(
    *,
    job_id: str,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Read detection rows and normalize for the export manifest."""
    rs_path = detection_row_store_path(settings.storage_root, job_id)
    if not rs_path.exists():
        return []

    _fieldnames, raw_rows = read_detection_row_store(rs_path)
    result: list[dict[str, Any]] = []
    for row in raw_rows:
        normalized = normalize_detection_row(row)
        label = _flatten_label(normalized)
        result.append(
            {
                "row_id": normalized["row_id"],
                "start_utc": normalized["start_utc"],
                "end_utc": normalized["end_utc"],
                "avg_confidence": normalized["avg_confidence"],
                "peak_confidence": normalized["peak_confidence"],
                "label": label,
            }
        )
    return result


async def _read_vocalization_labels(
    *,
    job_id: str,
    db: AsyncSession,
    settings: Settings,
) -> list[dict[str, Any]]:
    """Read vocalization labels (manual + inference) for the manifest."""
    from humpback.classifier.vocalization_inference import read_predictions

    rs_path = detection_row_store_path(settings.storage_root, job_id)
    utc_by_row_id: dict[str, tuple[float, float]] = {}
    if rs_path.exists():
        _fields, rs_rows = read_detection_row_store(rs_path)
        for r in rs_rows:
            rid = r.get("row_id", "")
            if rid:
                utc_by_row_id[rid] = (
                    float(r.get("start_utc", "0")),
                    float(r.get("end_utc", "0")),
                )

    # Manual labels from DB
    manual_result = await db.execute(
        select(VocalizationLabel)
        .where(VocalizationLabel.detection_job_id == job_id)
        .order_by(VocalizationLabel.created_at)
    )
    out: list[dict[str, Any]] = []
    manual_keys: set[tuple[str, str]] = set()
    for r in manual_result.scalars().all():
        utc = utc_by_row_id.get(r.row_id)
        if utc is None:
            continue
        out.append(
            {
                "start_utc": utc[0],
                "end_utc": utc[1],
                "type": r.label,
                "confidence": r.confidence,
                "source": r.source,
            }
        )
        manual_keys.add((r.row_id, r.label))

    # Inference predictions from most recent completed inference job
    inf_result = await db.execute(
        select(VocalizationInferenceJob)
        .where(VocalizationInferenceJob.source_type == "detection_job")
        .where(VocalizationInferenceJob.source_id == job_id)
        .where(VocalizationInferenceJob.status == "complete")
        .order_by(VocalizationInferenceJob.created_at.desc())
    )
    inf_jobs = inf_result.scalars().all()

    if inf_jobs:
        inf_job = inf_jobs[0]
        if inf_job.output_path and Path(inf_job.output_path).exists():
            model_result = await db.execute(
                select(VocalizationClassifierModel).where(
                    VocalizationClassifierModel.id == inf_job.vocalization_model_id
                )
            )
            model = model_result.scalar_one_or_none()
            if model:
                import json as _json

                vocabulary: list[str] = _json.loads(model.vocabulary_snapshot)
                thresholds: dict[str, float] = _json.loads(model.per_class_thresholds)
                predictions = read_predictions(
                    Path(inf_job.output_path), vocabulary, thresholds
                )
                for pred in predictions:
                    rid = pred.get("row_id")
                    if rid is None:
                        continue
                    utc = utc_by_row_id.get(rid)
                    if utc is None:
                        continue
                    for tag in pred["tags"]:
                        if (rid, tag) in manual_keys:
                            continue
                        score = pred["scores"].get(tag)
                        out.append(
                            {
                                "start_utc": utc[0],
                                "end_utc": utc[1],
                                "type": tag,
                                "confidence": score,
                                "source": "inference",
                            }
                        )

    return out


async def _read_vocalization_types(db: AsyncSession) -> list[dict[str, Any]]:
    """Read vocalization type vocabulary."""
    result = await db.execute(select(VocalizationType).order_by(VocalizationType.name))
    return [{"id": r.id, "name": r.name} for r in result.scalars().all()]


def _print_progress(stage: str, current: int, total: int) -> None:
    """Default progress reporter for CLI usage."""
    sys.stderr.write(f"\r{stage}: {current}/{total}")
    if current == total:
        sys.stderr.write("\n")
    sys.stderr.flush()
