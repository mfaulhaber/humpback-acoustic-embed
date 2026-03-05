"""Worker for classifier training and detection jobs."""

import asyncio
import json
import logging
from pathlib import Path

import joblib
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.detector import run_detection, write_detections_tsv
from humpback.classifier.trainer import train_binary_classifier
from humpback.config import Settings
from humpback.models.classifier import ClassifierModel, ClassifierTrainingJob, DetectionJob
from humpback.processing.embeddings import read_embeddings
from humpback.storage import classifier_dir, detection_dir, ensure_dir
from humpback.workers.model_cache import get_model_by_version
from humpback.workers.queue import (
    complete_detection_job,
    complete_training_job,
    fail_detection_job,
    fail_training_job,
)

logger = logging.getLogger(__name__)


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

        # Run detection (CPU-bound)
        detections, summary = await asyncio.to_thread(
            run_detection,
            Path(job.audio_folder),
            pipeline,
            model,
            cm.window_size_seconds,
            cm.target_sample_rate,
            job.confidence_threshold,
            input_format,
            feature_config,
        )

        # Write outputs
        ddir = ensure_dir(detection_dir(settings.storage_root, job.id))
        tsv_path = ddir / "detections.tsv"
        write_detections_tsv(detections, tsv_path)

        summary_path = ddir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        # Update job (use explicit SQL since job may be detached)
        await session.execute(
            update(DetectionJob)
            .where(DetectionJob.id == job.id)
            .values(
                output_tsv_path=str(tsv_path),
                result_summary=json.dumps(summary),
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
