"""Worker for post-hoc detection embedding generation."""

import asyncio
import json
import logging
from pathlib import Path

import joblib
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.detector import run_detection, write_detection_embeddings
from humpback.config import Settings
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.detection_embedding_job import DetectionEmbeddingJob
from humpback.storage import detection_dir, ensure_dir
from humpback.workers.model_cache import get_model_by_version
from humpback.workers.queue import (
    complete_detection_embedding_job,
    fail_detection_embedding_job,
)

logger = logging.getLogger(__name__)


async def run_detection_embedding_job(
    session: AsyncSession,
    job: DetectionEmbeddingJob,
    settings: Settings,
) -> None:
    """Generate detection embeddings for an existing detection job."""
    try:
        from sqlalchemy import select

        # Load the detection job
        result = await session.execute(
            select(DetectionJob).where(DetectionJob.id == job.detection_job_id)
        )
        det_job = result.scalar_one_or_none()
        if det_job is None:
            raise ValueError(f"Detection job {job.detection_job_id} not found")
        if det_job.status != "complete":
            raise ValueError(
                f"Detection job {job.detection_job_id} status is {det_job.status}, "
                "expected complete"
            )
        if not det_job.audio_folder:
            raise ValueError(
                f"Detection job {job.detection_job_id} has no audio_folder "
                "(hydrophone jobs already generate embeddings during detection)"
            )

        # Load classifier model and embedding model
        cm_result = await session.execute(
            select(ClassifierModel).where(
                ClassifierModel.id == det_job.classifier_model_id
            )
        )
        cm = cm_result.scalar_one()
        pipeline = joblib.load(cm.model_path)
        model, input_format = await get_model_by_version(
            session, cm.model_version, settings
        )
        feature_config = json.loads(cm.feature_config) if cm.feature_config else None

        audio_folder = Path(det_job.audio_folder)

        # Count audio files for progress tracking
        audio_extensions = {".wav", ".flac", ".mp3", ".aif", ".aiff"}
        audio_files = sorted(
            p for p in audio_folder.rglob("*") if p.suffix.lower() in audio_extensions
        )
        files_total = len(audio_files)

        await session.execute(
            update(DetectionEmbeddingJob)
            .where(DetectionEmbeddingJob.id == job.id)
            .values(progress_total=files_total, progress_current=0)
        )
        await session.commit()

        def on_file_complete(
            _detections: list[dict], files_done: int, total: int
        ) -> None:
            pass  # Progress updated after run_detection completes

        # Re-run detection to collect embeddings (reuses existing infrastructure)
        (
            _detections,
            _summary,
            _diagnostics,
            detection_embeddings,
        ) = await asyncio.to_thread(
            run_detection,
            audio_folder,
            pipeline,
            model,
            cm.window_size_seconds,
            cm.target_sample_rate,
            det_job.confidence_threshold,
            input_format,
            feature_config,
            False,  # emit_diagnostics — not needed
            det_job.hop_seconds,
            det_job.high_threshold,
            det_job.low_threshold,
            on_file_complete,
            det_job.detection_mode,
            True,  # emit_embeddings
        )

        # Write embeddings
        ddir = ensure_dir(detection_dir(settings.storage_root, det_job.id))
        emb_path = ddir / "detection_embeddings.parquet"
        if detection_embeddings:
            write_detection_embeddings(detection_embeddings, emb_path)

        count = len(detection_embeddings) if detection_embeddings else 0
        await session.execute(
            update(DetectionEmbeddingJob)
            .where(DetectionEmbeddingJob.id == job.id)
            .values(progress_current=files_total)
        )
        await session.commit()

        logger.info(
            "Detection embedding job %s complete: %d embeddings for detection %s",
            job.id,
            count,
            job.detection_job_id,
        )
        await complete_detection_embedding_job(session, job.id)

    except Exception as e:
        logger.exception("Detection embedding job %s failed", job.id)
        try:
            await session.rollback()
        except Exception:
            pass
        try:
            await fail_detection_embedding_job(session, job.id, str(e))
        except Exception:
            logger.exception("Failed to mark detection embedding job as failed")
