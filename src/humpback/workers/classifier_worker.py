"""Worker for classifier training and detection jobs."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import joblib
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.detector import (
    AUDIO_EXTENSIONS,
    append_detections_tsv,
    read_detections_tsv,
    run_detection,
    write_detections_tsv,
    write_window_diagnostics,
)
from humpback.classifier.extractor import extract_labeled_samples
from humpback.classifier.providers import (
    build_archive_detection_provider,
    build_archive_playback_provider,
)
from humpback.classifier.trainer import train_binary_classifier
from humpback.config import Settings
from humpback.models.classifier import (
    ClassifierModel,
    ClassifierTrainingJob,
    DetectionJob,
)
from humpback.processing.embeddings import read_embeddings
from humpback.storage import classifier_dir, detection_dir, ensure_dir
from humpback.workers.model_cache import get_model_by_version
from humpback.workers.queue import (
    complete_detection_job,
    complete_extraction_job,
    complete_training_job,
    fail_detection_job,
    fail_extraction_job,
    fail_training_job,
)

logger = logging.getLogger(__name__)

HYDROPHONE_TSV_FIELDNAMES = [
    "filename",
    "start_sec",
    "end_sec",
    "avg_confidence",
    "peak_confidence",
    "n_windows",
    "raw_start_sec",
    "raw_end_sec",
    "merged_event_count",
    "detection_filename",
    "extract_filename",
    "hydrophone_name",
]


async def run_training_job(
    session: AsyncSession,
    job: ClassifierTrainingJob,
    settings: Settings,
) -> None:
    """Execute a classifier training job end-to-end."""
    try:
        # Load positive embeddings from parquet files
        import numpy as np
        from sqlalchemy import select

        from humpback.models.processing import EmbeddingSet

        es_ids = json.loads(job.positive_embedding_set_ids)
        result = await session.execute(
            select(EmbeddingSet).where(EmbeddingSet.id.in_(es_ids))
        )
        embedding_sets = list(result.scalars().all())

        positive_parts: list[np.ndarray] = []
        for es in embedding_sets:
            _, vectors = read_embeddings(Path(es.parquet_path))
            positive_parts.append(vectors)
        positive_embeddings = np.vstack(positive_parts)

        # Load negative embeddings from parquet files
        neg_ids = json.loads(job.negative_embedding_set_ids)
        neg_result = await session.execute(
            select(EmbeddingSet).where(EmbeddingSet.id.in_(neg_ids))
        )
        neg_embedding_sets = list(neg_result.scalars().all())

        negative_parts: list[np.ndarray] = []
        for es in neg_embedding_sets:
            _, vectors = read_embeddings(Path(es.parquet_path))
            negative_parts.append(vectors)
        negative_embeddings = np.vstack(negative_parts)

        # Parse parameters
        parameters = json.loads(job.parameters) if job.parameters else None

        # Train classifier (CPU-bound)
        pipeline, summary = await asyncio.to_thread(
            train_binary_classifier,
            positive_embeddings,
            negative_embeddings,
            parameters,
        )

        # Save model atomically
        cdir = ensure_dir(classifier_dir(settings.storage_root, job.id))
        tmp_path = cdir / "model.tmp.joblib"
        final_path = cdir / "model.joblib"
        joblib.dump(pipeline, tmp_path)
        tmp_path.rename(final_path)

        # Save training summary
        summary_path = cdir / "training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        # Create ClassifierModel record
        cm = ClassifierModel(
            name=job.name,
            model_path=str(final_path),
            model_version=job.model_version,
            vector_dim=embedding_sets[0].vector_dim,
            window_size_seconds=job.window_size_seconds,
            target_sample_rate=job.target_sample_rate,
            feature_config=job.feature_config,
            training_summary=json.dumps(summary),
            training_job_id=job.id,
        )
        session.add(cm)
        await session.flush()

        # Update job with model reference (use explicit SQL since job may be detached)
        await session.execute(
            update(ClassifierTrainingJob)
            .where(ClassifierTrainingJob.id == job.id)
            .values(classifier_model_id=cm.id)
        )
        await complete_training_job(session, job.id)

    except Exception as e:
        logger.exception("Training job %s failed", job.id)
        try:
            await session.rollback()
        except Exception:
            pass
        try:
            await fail_training_job(session, job.id, str(e))
        except Exception:
            logger.exception("Failed to mark training job as failed")


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
        pipeline = joblib.load(cm.model_path)

        # Load embedding model
        model, input_format = await get_model_by_version(
            session, cm.model_version, settings
        )

        feature_config = json.loads(cm.feature_config) if cm.feature_config else None

        # Set up output directory and TSV path early for incremental writes
        ddir = ensure_dir(detection_dir(settings.storage_root, job.id))
        tsv_path = ddir / "detections.tsv"
        if not job.audio_folder:
            raise ValueError("Detection job missing audio_folder")
        audio_folder = Path(job.audio_folder)

        # Count audio files and set initial progress in DB
        audio_files = sorted(
            p for p in audio_folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
        )
        files_total = len(audio_files)

        # Remove stale TSV from prior run to prevent duplicate appends
        if tsv_path.exists():
            tsv_path.unlink()

        await session.execute(
            update(DetectionJob)
            .where(DetectionJob.id == job.id)
            .values(
                output_tsv_path=str(tsv_path),
                files_total=files_total,
                files_processed=0,
            )
        )
        await session.commit()

        # Build incremental callback for per-file progress
        loop = asyncio.get_event_loop()

        def on_file_complete(file_detections: list[dict], files_done: int, total: int):
            # Append detections to TSV (synchronous file I/O, safe from thread)
            if file_detections:
                append_detections_tsv(file_detections, tsv_path)

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
        detections, summary, diagnostics = await asyncio.to_thread(
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
        )

        # Overwrite TSV with final authoritative version
        write_detections_tsv(detections, tsv_path)

        # Write window diagnostics
        if diagnostics:
            diag_path = ddir / "window_diagnostics.parquet"
            write_window_diagnostics(diagnostics, diag_path)
            summary["has_diagnostics"] = True

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


async def run_hydrophone_detection_job(
    session: AsyncSession,
    job: DetectionJob,
    settings: Settings,
    session_factory=None,
) -> None:
    """Execute a hydrophone detection job end-to-end."""
    import threading

    try:
        from sqlalchemy import select

        # Load classifier model
        result = await session.execute(
            select(ClassifierModel).where(ClassifierModel.id == job.classifier_model_id)
        )
        cm = result.scalar_one()

        # Load sklearn pipeline
        pipeline = joblib.load(cm.model_path)

        # Load embedding model
        model, input_format = await get_model_by_version(
            session, cm.model_version, settings
        )

        feature_config = json.loads(cm.feature_config) if cm.feature_config else None

        # Set up output directory and TSV path
        ddir = ensure_dir(detection_dir(settings.storage_root, job.id))
        tsv_path = ddir / "detections.tsv"

        await session.execute(
            update(DetectionJob)
            .where(DetectionJob.id == job.id)
            .values(output_tsv_path=str(tsv_path))
        )
        await session.commit()

        # Resume support: if segments were already processed, read prior detections
        skip_segments = 0
        prior_detections: list[dict] = []
        if job.segments_processed and job.segments_processed > 0 and tsv_path.exists():
            prior_detections = read_detections_tsv(
                tsv_path, fieldnames=HYDROPHONE_TSV_FIELDNAMES
            )
            skip_segments = job.segments_processed
            # Rewrite TSV cleanly with prior detections only (removes partial appends)
            write_detections_tsv(
                prior_detections, tsv_path, fieldnames=HYDROPHONE_TSV_FIELDNAMES
            )
            logger.info(
                "Resuming hydrophone job %s from segment %d with %d prior detections",
                job.id,
                skip_segments,
                len(prior_detections),
            )

        # Cancel and pause support
        cancel_event = threading.Event()
        pause_gate = threading.Event()
        pause_gate.set()  # Initially not paused (set = open gate)
        loop = asyncio.get_event_loop()

        def _fmt_utc(ts: float | None) -> str:
            if ts is None:
                return "unknown"
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        if (
            job.hydrophone_id is None
            or job.start_timestamp is None
            or job.end_timestamp is None
        ):
            raise ValueError(
                "Hydrophone detection job missing hydrophone_id or time bounds"
            )
        hydrophone_id = job.hydrophone_id
        start_timestamp = job.start_timestamp
        end_timestamp = job.end_timestamp
        hydrophone_provider = build_archive_detection_provider(
            hydrophone_id,
            local_cache_path=job.local_cache_path,
            s3_cache_path=settings.s3_cache_path,
            noaa_cache_path=settings.noaa_cache_path,
        )
        hydrophone_short_name = hydrophone_provider.source_id

        # Progress callback
        def on_chunk_complete(
            chunk_detections: list[dict],
            segments_done: int,
            segments_total: int,
            time_covered_sec: float,
        ):
            if chunk_detections:
                for det in chunk_detections:
                    det["hydrophone_name"] = hydrophone_short_name
                append_detections_tsv(
                    chunk_detections,
                    tsv_path,
                    fieldnames=HYDROPHONE_TSV_FIELDNAMES,
                )

            if session_factory is not None:

                async def _update_progress():
                    try:
                        async with cast(Any, session_factory)() as progress_session:
                            await progress_session.execute(
                                update(DetectionJob)
                                .where(DetectionJob.id == job.id)
                                .values(
                                    segments_processed=segments_done,
                                    segments_total=segments_total,
                                    time_covered_sec=time_covered_sec,
                                )
                            )
                            await progress_session.commit()
                    except Exception:
                        logger.debug(
                            "Failed to update hydrophone progress", exc_info=True
                        )

                loop.call_soon_threadsafe(asyncio.ensure_future, _update_progress())

        # Alert callback
        alerts_list: list[dict] = []

        def on_alert(alert: dict):
            alerts_list.append(alert)
            if session_factory is not None:

                async def _update_alerts():
                    try:
                        async with cast(Any, session_factory)() as alert_session:
                            await alert_session.execute(
                                update(DetectionJob)
                                .where(DetectionJob.id == job.id)
                                .values(alerts=json.dumps(alerts_list))
                            )
                            await alert_session.commit()
                    except Exception:
                        logger.debug("Failed to update alerts", exc_info=True)

                loop.call_soon_threadsafe(asyncio.ensure_future, _update_alerts())

        # Poll for cancellation/pause in background
        async def _poll_cancel():
            while not cancel_event.is_set():
                await asyncio.sleep(2)
                try:
                    if session_factory is not None:
                        async with cast(Any, session_factory)() as poll_session:
                            result = await poll_session.execute(
                                select(DetectionJob.status).where(
                                    DetectionJob.id == job.id
                                )
                            )
                            status = result.scalar_one_or_none()
                            if status == "canceled":
                                cancel_event.set()
                                pause_gate.set()  # Unblock if paused so thread can exit
                                return
                            elif status == "paused":
                                pause_gate.clear()  # Block the detection thread
                            elif status == "running":
                                pause_gate.set()  # Unblock the detection thread
                except Exception:
                    pass

        cancel_task = asyncio.ensure_future(_poll_cancel())
        try:
            from humpback.classifier.hydrophone_detector import run_hydrophone_detection

            detections, summary = await asyncio.to_thread(
                run_hydrophone_detection,
                hydrophone_provider,
                start_timestamp,
                end_timestamp,
                pipeline,
                model,
                cm.window_size_seconds,
                cm.target_sample_rate,
                job.confidence_threshold,
                input_format,
                feature_config,
                job.hop_seconds,
                job.high_threshold,
                job.low_threshold,
                on_chunk_complete,
                on_alert,
                cancel_event.is_set,
                pause_gate,
                skip_segments,
                prior_detections,
                settings.hydrophone_prefetch_enabled,
                settings.hydrophone_prefetch_workers,
                settings.hydrophone_prefetch_inflight_segments,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "No hydrophone audio segments found for hydrophone "
                f"'{hydrophone_id}' in requested UTC range "
                f"[{_fmt_utc(start_timestamp)}, {_fmt_utc(end_timestamp)}]"
            ) from exc
        finally:
            cancel_task.cancel()

        if cancel_event.is_set():
            # Write what we have
            for det in detections:
                det.setdefault("hydrophone_name", hydrophone_short_name)
            write_detections_tsv(
                detections,
                tsv_path,
                fieldnames=HYDROPHONE_TSV_FIELDNAMES,
            )
            await session.execute(
                update(DetectionJob)
                .where(DetectionJob.id == job.id)
                .values(
                    status="canceled",
                    result_summary=json.dumps(summary),
                    alerts=json.dumps(alerts_list) if alerts_list else None,
                    updated_at=__import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ),
                )
            )
            await session.commit()
            return

        # Write final TSV
        for det in detections:
            det.setdefault("hydrophone_name", hydrophone_short_name)
        write_detections_tsv(
            detections,
            tsv_path,
            fieldnames=HYDROPHONE_TSV_FIELDNAMES,
        )

        summary_path = ddir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        await session.execute(
            update(DetectionJob)
            .where(DetectionJob.id == job.id)
            .values(
                result_summary=json.dumps(summary),
                alerts=json.dumps(alerts_list) if alerts_list else None,
            )
        )
        await complete_detection_job(session, job.id)

    except Exception as e:
        logger.exception("Hydrophone detection job %s failed", job.id)
        try:
            await session.rollback()
        except Exception:
            pass
        try:
            await fail_detection_job(session, job.id, str(e))
        except Exception:
            logger.exception("Failed to mark hydrophone detection job as failed")


async def run_extraction_job(
    session: AsyncSession,
    job: DetectionJob,
    settings: Settings,
) -> None:
    """Execute a labeled sample extraction job."""
    try:
        config = json.loads(job.extract_config) if job.extract_config else {}
        pos_path = config.get("positive_output_path", settings.positive_sample_path)
        neg_path = config.get("negative_output_path", settings.negative_sample_path)

        if not job.output_tsv_path:
            raise ValueError("Detection job has no output TSV path")

        # Look up window_size_seconds from the classifier model
        from sqlalchemy import select as sa_select

        cm_result = await session.execute(
            sa_select(ClassifierModel).where(
                ClassifierModel.id == job.classifier_model_id
            )
        )
        cm = cm_result.scalar_one_or_none()
        ws = cm.window_size_seconds if cm else 5.0

        if job.hydrophone_id:
            from humpback.classifier.extractor import extract_hydrophone_labeled_samples

            cache_path = job.local_cache_path or settings.s3_cache_path
            extract_provider = build_archive_playback_provider(
                job.hydrophone_id,
                cache_path=cache_path,
                noaa_cache_path=settings.noaa_cache_path,
            )

            summary = await asyncio.to_thread(
                extract_hydrophone_labeled_samples,
                job.output_tsv_path,
                extract_provider,
                pos_path,
                neg_path,
                cm.target_sample_rate if cm else 32000,
                ws,
                job.start_timestamp,
                job.end_timestamp,
            )
        else:
            if job.audio_folder is None:
                raise ValueError("Local extraction job missing audio_folder")
            summary = await asyncio.to_thread(
                extract_labeled_samples,
                job.output_tsv_path,
                job.audio_folder,
                pos_path,
                neg_path,
                ws,
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
