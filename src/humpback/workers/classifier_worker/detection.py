"""Local detection and extraction job execution."""

import asyncio
import json
import logging
import sys  # noqa: F401 — used by _pkg() for monkeypatch-compatible lookups
from pathlib import Path
from typing import Any, cast

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.detection_rows import (
    ROW_STORE_FIELDNAMES,
    append_detection_row_store,
    ensure_detection_row_store,
    ensure_row_ids,
    format_optional_float,
    format_optional_int,
    normalize_detection_row,
    read_detection_row_store,
)
from humpback.classifier.detector import (
    AUDIO_EXTENSIONS,
    run_detection,
    write_window_diagnostics,
)
from humpback.classifier.extractor import (
    DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE,
    DEFAULT_POSITIVE_SELECTION_MIN_SCORE,
    DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW,
)
from humpback.config import Settings
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.storage import (
    detection_dir,
    detection_row_store_path,
    ensure_dir,
)
from humpback.workers.queue import (
    complete_detection_job,
    complete_extraction_job,
    fail_detection_job,
    fail_extraction_job,
)

logger = logging.getLogger(__name__)

# Late-bound parent package reference for monkeypatch compatibility.
# Tests patch names like ``humpback.workers.classifier_worker.joblib``;
# sub-modules must look them up through the package at call time.
_PKG = "humpback.workers.classifier_worker"


def _pkg():
    return sys.modules[_PKG]


def _detection_dicts_to_store_rows(
    detections: list[dict],
) -> list[dict[str, str]]:
    """Convert raw detection dicts (from the detector) to row-store format.

    Each detection dict has UTC fields (start_utc, end_utc, etc.) produced by
    the detector.  We normalize through the row-store pipeline to produce dicts
    keyed by ROW_STORE_FIELDNAMES with string values.
    """
    store_rows: list[dict[str, str]] = []
    for det in detections:
        str_det: dict[str, str] = {k: str(v) for k, v in det.items()}
        normalized = normalize_detection_row(str_det)
        out_row: dict[str, str] = {field: "" for field in ROW_STORE_FIELDNAMES}
        out_row["start_utc"] = format_optional_float(normalized["start_utc"])
        out_row["end_utc"] = format_optional_float(normalized["end_utc"])
        out_row["avg_confidence"] = format_optional_float(normalized["avg_confidence"])
        out_row["peak_confidence"] = format_optional_float(
            normalized["peak_confidence"]
        )
        out_row["n_windows"] = format_optional_int(normalized["n_windows"])
        out_row["raw_start_utc"] = format_optional_float(normalized["raw_start_utc"])
        out_row["raw_end_utc"] = format_optional_float(normalized["raw_end_utc"])
        out_row["merged_event_count"] = format_optional_int(
            normalized["merged_event_count"]
        )
        out_row["hydrophone_name"] = normalized["hydrophone_name"] or ""
        for label in ("humpback", "orca", "ship", "background"):
            value = normalized[label]
            out_row[label] = "" if value is None else str(value)
        store_rows.append(out_row)
    ensure_row_ids(store_rows)
    return store_rows


async def run_detection_job(
    session: AsyncSession,
    job: DetectionJob,
    settings: Settings,
    session_factory=None,
) -> None:
    """Execute a detection job end-to-end."""
    try:
        from sqlalchemy import select

        # Load classifier model
        result = await session.execute(
            select(ClassifierModel).where(ClassifierModel.id == job.classifier_model_id)
        )
        cm = result.scalar_one()

        # Load sklearn pipeline
        pipeline = _pkg().joblib.load(cm.model_path)

        # Load embedding model
        model, input_format = await _pkg().get_model_by_version(
            session, cm.model_version, settings
        )

        feature_config = json.loads(cm.feature_config) if cm.feature_config else None

        # Set up output directory and row store path for incremental writes
        ddir = ensure_dir(detection_dir(settings.storage_root, job.id))
        rs_path = detection_row_store_path(settings.storage_root, job.id)
        if not job.audio_folder:
            raise ValueError("Detection job missing audio_folder")
        audio_folder = Path(job.audio_folder)

        # Count audio files and set initial progress in DB
        audio_files = sorted(
            p for p in audio_folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
        )
        files_total = len(audio_files)

        # Remove stale row store from prior run to prevent duplicate appends
        if rs_path.exists():
            rs_path.unlink()

        await session.execute(
            update(DetectionJob)
            .where(DetectionJob.id == job.id)
            .values(
                files_total=files_total,
                files_processed=0,
            )
        )
        await session.commit()

        # Build incremental callback for per-file progress
        loop = asyncio.get_event_loop()

        def on_file_complete(file_detections: list[dict], files_done: int, total: int):
            # Append detections to Parquet row store (synchronous file I/O, safe from thread)
            if file_detections:
                store_rows = _detection_dicts_to_store_rows(file_detections)
                append_detection_row_store(rs_path, store_rows)

            # Schedule async DB progress update on the event loop
            if session_factory is not None:

                async def _update_progress():
                    try:
                        async with cast(Any, session_factory)() as progress_session:
                            await progress_session.execute(
                                update(DetectionJob)
                                .where(DetectionJob.id == job.id)
                                .values(files_processed=files_done)
                            )
                            await progress_session.commit()
                    except Exception:
                        logger.debug(
                            "Failed to update detection progress", exc_info=True
                        )

                loop.call_soon_threadsafe(asyncio.ensure_future, _update_progress())

        # Run detection (CPU-bound) with diagnostics and incremental callback
        (
            detections,
            summary,
            diagnostics,
            detection_embeddings,
        ) = await asyncio.to_thread(
            run_detection,
            audio_folder,
            pipeline,
            model,
            cm.window_size_seconds,
            cm.target_sample_rate,
            job.confidence_threshold,
            input_format,
            feature_config,
            True,  # emit_diagnostics
            job.hop_seconds,
            job.high_threshold,
            job.low_threshold,
            on_file_complete,
            job.detection_mode,
            True,  # emit_embeddings
            job.window_selection,
            job.min_prominence,
        )

        # Write window diagnostics
        if diagnostics:
            diag_path = ddir / "window_diagnostics.parquet"
            write_window_diagnostics(diagnostics, diag_path)
            summary["has_diagnostics"] = True
        else:
            diag_path = None

        ensure_detection_row_store(
            row_store_path=rs_path,
            diagnostics_path=diag_path,
            window_size_seconds=cm.window_size_seconds,
            refresh_existing=True,
            detection_mode=job.detection_mode,
        )

        # Write detection embeddings (after row store so row_ids are available)
        if detection_embeddings:
            from humpback.classifier.detector import (
                match_embedding_records_to_row_store,
                write_detection_embeddings,
            )

            _, rs_rows = read_detection_row_store(rs_path)
            detection_embeddings = match_embedding_records_to_row_store(
                detection_embeddings, rs_rows
            )
            emb_path = ddir / "detection_embeddings.parquet"
            write_detection_embeddings(detection_embeddings, emb_path)
            summary["has_detection_embeddings"] = True

        summary_path = ddir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        # Update job (use explicit SQL since job may be detached)
        await session.execute(
            update(DetectionJob)
            .where(DetectionJob.id == job.id)
            .values(
                result_summary=json.dumps(summary),
                files_processed=files_total,
            )
        )
        await complete_detection_job(session, job.id)

    except Exception as e:
        logger.exception("Detection job %s failed", job.id)
        try:
            await session.rollback()
        except Exception:
            pass
        try:
            await fail_detection_job(session, job.id, str(e))
        except Exception:
            logger.exception("Failed to mark detection job as failed")


async def run_extraction_job(
    session: AsyncSession,
    job: DetectionJob,
    settings: Settings,
) -> None:
    """Execute a labeled sample extraction job."""
    try:
        if job.detection_mode != "windowed":
            raise ValueError(
                "Legacy merged-mode extraction is no longer supported; rerun the detection job in windowed mode"
            )
        config = json.loads(job.extract_config) if job.extract_config else {}
        pos_path = config.get("positive_output_path", settings.positive_sample_path)
        neg_path = config.get("negative_output_path", settings.negative_sample_path)
        smoothing_window = int(
            config.get(
                "positive_selection_smoothing_window",
                DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW,
            )
        )
        min_score = float(
            config.get(
                "positive_selection_min_score",
                DEFAULT_POSITIVE_SELECTION_MIN_SCORE,
            )
        )
        extend_min_score = float(
            config.get(
                "positive_selection_extend_min_score",
                DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE,
            )
        )

        from humpback.storage import (
            detection_diagnostics_path as _det_diag_path,
            detection_row_store_path as _det_rs_path,
            detection_tsv_path as _det_tsv_path,
        )

        tsv_path = _det_tsv_path(settings.storage_root, job.id)
        diagnostics_path = _det_diag_path(settings.storage_root, job.id)
        row_store_path = _det_rs_path(settings.storage_root, job.id)

        # Look up window_size_seconds from the classifier model
        from sqlalchemy import select as sa_select

        cm_result = await session.execute(
            sa_select(ClassifierModel).where(
                ClassifierModel.id == job.classifier_model_id
            )
        )
        cm = cm_result.scalar_one_or_none()
        ws = cm.window_size_seconds if cm else 5.0
        target_sample_rate = cm.target_sample_rate if cm else 32000
        feature_config = (
            json.loads(cm.feature_config) if cm and cm.feature_config else None
        )
        pipeline = _pkg().joblib.load(cm.model_path) if cm is not None else None
        model = None
        input_format = "spectrogram"
        if cm is not None:
            model, input_format = await _pkg().get_model_by_version(
                session, cm.model_version, settings
            )

        ensure_detection_row_store(
            row_store_path=row_store_path,
            diagnostics_path=diagnostics_path if diagnostics_path.exists() else None,
            window_size_seconds=ws,
            detection_mode=job.detection_mode,
            tsv_path=tsv_path,
        )

        if job.hydrophone_id:
            from humpback.classifier.extractor import extract_hydrophone_labeled_samples

            cache_path = job.local_cache_path or settings.s3_cache_path
            extract_provider = _pkg().build_archive_playback_provider(
                job.hydrophone_id,
                cache_path=cache_path,
                noaa_cache_path=settings.noaa_cache_path,
            )

            summary = await asyncio.to_thread(
                extract_hydrophone_labeled_samples,
                tsv_path=str(tsv_path),
                provider=extract_provider,
                positive_output_path=pos_path,
                negative_output_path=neg_path,
                target_sample_rate=target_sample_rate,
                window_size_seconds=ws,
                stream_start_timestamp=job.start_timestamp,
                stream_end_timestamp=job.end_timestamp,
                window_diagnostics_path=diagnostics_path,
                positive_selection_smoothing_window=smoothing_window,
                positive_selection_min_score=min_score,
                positive_selection_extend_min_score=extend_min_score,
                fallback_pipeline=pipeline,
                fallback_model=model,
                fallback_input_format=input_format,
                fallback_feature_config=feature_config,
                row_store_path=row_store_path,
                spectrogram_hop_length=settings.spectrogram_hop_length,
                spectrogram_dynamic_range_db=settings.spectrogram_dynamic_range_db,
                spectrogram_width_px=settings.spectrogram_width_px,
                spectrogram_height_px=settings.spectrogram_height_px,
            )
        else:
            if job.audio_folder is None:
                raise ValueError("Local extraction job missing audio_folder")
            summary = await asyncio.to_thread(
                _pkg().extract_labeled_samples,
                tsv_path=str(tsv_path),
                audio_folder=job.audio_folder,
                positive_output_path=pos_path,
                negative_output_path=neg_path,
                window_size_seconds=ws,
                window_diagnostics_path=diagnostics_path,
                positive_selection_smoothing_window=smoothing_window,
                positive_selection_min_score=min_score,
                positive_selection_extend_min_score=extend_min_score,
                fallback_pipeline=pipeline,
                fallback_model=model,
                fallback_target_sample_rate=target_sample_rate,
                fallback_input_format=input_format,
                fallback_feature_config=feature_config,
                row_store_path=row_store_path,
                spectrogram_hop_length=settings.spectrogram_hop_length,
                spectrogram_dynamic_range_db=settings.spectrogram_dynamic_range_db,
                spectrogram_width_px=settings.spectrogram_width_px,
                spectrogram_height_px=settings.spectrogram_height_px,
            )

        await session.execute(
            update(DetectionJob)
            .where(DetectionJob.id == job.id)
            .values(extract_summary=json.dumps(summary))
        )
        await complete_extraction_job(session, job.id)

    except Exception as e:
        logger.exception("Extraction job %s failed", job.id)
        try:
            await session.rollback()
        except Exception:
            pass
        try:
            await fail_extraction_job(session, job.id, str(e))
        except Exception:
            logger.exception("Failed to mark extraction job as failed")
