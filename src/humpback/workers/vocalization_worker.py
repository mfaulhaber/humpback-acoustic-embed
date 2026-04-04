"""Worker functions for vocalization training and inference jobs."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.training_dataset import TrainingDataset, TrainingDatasetLabel
from humpback.models.vocalization import (
    VocalizationClassifierModel,
    VocalizationInferenceJob,
    VocalizationTrainingJob,
)

logger = logging.getLogger(__name__)


async def run_vocalization_training_job(
    session: AsyncSession,
    job: VocalizationTrainingJob,
    settings: Settings,
) -> None:
    """Execute a multi-label vocalization training job."""
    try:
        from humpback.classifier.vocalization_trainer import (
            save_model_artifacts,
            train_multilabel_classifiers,
        )
        from humpback.services.training_dataset import (
            create_training_dataset_snapshot,
        )

        parameters = json.loads(job.parameters) if job.parameters else {}

        # Mode A: source_config → create new training dataset
        # Mode B: training_dataset_id → reuse existing dataset
        if job.training_dataset_id:
            ds_result = await session.execute(
                select(TrainingDataset).where(
                    TrainingDataset.id == job.training_dataset_id
                )
            )
            dataset = ds_result.scalar_one_or_none()
            if dataset is None:
                raise ValueError(
                    f"Training dataset {job.training_dataset_id} not found"
                )
        else:
            source_config = json.loads(job.source_config)
            dataset = await create_training_dataset_snapshot(
                session, source_config, settings.storage_root
            )
            # Persist the dataset link on the job
            job.training_dataset_id = dataset.id
            await session.flush()

        # Read embeddings from the training dataset parquet
        X, all_label_sets = await _load_training_dataset(session, dataset)

        if len(X) == 0:
            raise ValueError("No embeddings in training dataset")

        logger.info("Collected %d embeddings for vocalization training", len(X))

        pipelines, thresholds, per_class_metrics = await asyncio.to_thread(
            train_multilabel_classifiers, X, all_label_sets, parameters
        )

        # Save model artifacts
        model_dir = settings.storage_root / "vocalization_models" / job.id
        await asyncio.to_thread(
            save_model_artifacts,
            model_dir,
            pipelines,
            thresholds,
            per_class_metrics,
            parameters,
        )

        vocabulary = sorted(pipelines.keys())

        # Create model record
        model = VocalizationClassifierModel(
            name=f"vocalization-{job.id[:8]}",
            model_dir_path=str(model_dir),
            vocabulary_snapshot=json.dumps(vocabulary),
            per_class_thresholds=json.dumps(thresholds),
            per_class_metrics=json.dumps(per_class_metrics),
            training_dataset_id=dataset.id,
            training_summary=json.dumps(
                {
                    "n_embeddings": len(X),
                    "n_types_trained": len(vocabulary),
                    "n_types_filtered": len(all_label_sets[0]) if all_label_sets else 0,
                    "vocabulary": vocabulary,
                    "per_class_metrics": per_class_metrics,
                }
            ),
        )
        session.add(model)
        await session.flush()

        result_summary = {
            "n_embeddings": len(X),
            "types_trained": vocabulary,
            "per_class_counts": {
                t: m["n_positive"] for t, m in per_class_metrics.items()
            },
        }

        await session.execute(
            update(VocalizationTrainingJob)
            .where(VocalizationTrainingJob.id == job.id)
            .values(
                status="complete",
                vocalization_model_id=model.id,
                training_dataset_id=dataset.id,
                result_summary=json.dumps(result_summary),
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
        logger.info("Vocalization training job %s complete", job.id)

    except Exception as e:
        logger.exception("Vocalization training job %s failed", job.id)
        await session.execute(
            update(VocalizationTrainingJob)
            .where(VocalizationTrainingJob.id == job.id)
            .values(
                status="failed",
                error_message=str(e),
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()


async def _load_training_dataset(
    session: AsyncSession,
    dataset: TrainingDataset,
) -> tuple[np.ndarray, list[set[str]]]:
    """Load embeddings and multi-hot label sets from a training dataset.

    Returns (X, all_label_sets) where X is (N, D) and all_label_sets[i] is
    the set of type labels for row i. "(Negative)" labels produce an empty set.
    """
    from humpback.processing.embeddings import read_embedding_vectors

    X = read_embedding_vectors(dataset.parquet_path)
    n_rows = X.shape[0]
    if n_rows == 0:
        return X, []

    # Build label index from training_dataset_labels grouped by row_index
    result = await session.execute(
        select(TrainingDatasetLabel).where(
            TrainingDatasetLabel.training_dataset_id == dataset.id
        )
    )
    all_labels = result.scalars().all()

    labels_by_row: dict[int, set[str]] = {}
    for lbl in all_labels:
        if lbl.row_index not in labels_by_row:
            labels_by_row[lbl.row_index] = set()
        labels_by_row[lbl.row_index].add(lbl.label)

    all_label_sets: list[set[str]] = []
    for i in range(n_rows):
        label_set = labels_by_row.get(i, set())
        # "(Negative)" means confirmed negative for all types
        if "(Negative)" in label_set:
            label_set = set()
        all_label_sets.append(label_set)

    return X, all_label_sets


async def run_vocalization_inference_job(
    session: AsyncSession,
    job: VocalizationInferenceJob,
    settings: Settings,
) -> None:
    """Execute a vocalization inference job."""
    try:
        from humpback.classifier.vocalization_inference import run_inference

        # Load model
        model_result = await session.execute(
            select(VocalizationClassifierModel).where(
                VocalizationClassifierModel.id == job.vocalization_model_id
            )
        )
        model = model_result.scalar_one_or_none()
        if model is None:
            raise ValueError(f"Model {job.vocalization_model_id} not found")

        model_dir = Path(model.model_dir_path)

        # Load embeddings based on source type
        (
            embeddings,
            row_ids,
            filenames,
            start_secs,
            end_secs,
            _start_utcs,
            _end_utcs,
            confidences,
        ) = await _load_source_embeddings(session, job, settings)

        # Run inference
        output_dir = settings.storage_root / "vocalization_inference" / job.id
        output_path = output_dir / "predictions.parquet"

        result_summary = await asyncio.to_thread(
            run_inference,
            model_dir,
            embeddings,
            output_path,
            row_ids=row_ids if row_ids else None,
            filenames=filenames if filenames else None,
            start_secs=start_secs if start_secs else None,
            end_secs=end_secs if end_secs else None,
            confidences=confidences,
        )

        await session.execute(
            update(VocalizationInferenceJob)
            .where(VocalizationInferenceJob.id == job.id)
            .values(
                status="complete",
                output_path=str(output_path),
                result_summary=json.dumps(result_summary),
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
        logger.info("Vocalization inference job %s complete", job.id)

    except Exception as e:
        logger.exception("Vocalization inference job %s failed", job.id)
        await session.execute(
            update(VocalizationInferenceJob)
            .where(VocalizationInferenceJob.id == job.id)
            .values(
                status="failed",
                error_message=str(e),
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()


async def _load_source_embeddings(
    session: AsyncSession,
    job: VocalizationInferenceJob,
    settings: Settings,
) -> tuple[
    np.ndarray,
    list[str],
    list[str],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float] | None,
]:
    """Load embeddings from the job's source.

    Returns (embeddings, row_ids, filenames, start_secs, end_secs, start_utcs,
    end_utcs, confidences).  row_ids populated for detection_job sources with
    new schema; filenames/start_secs/end_secs populated for embedding_set sources.
    """
    if job.source_type == "detection_job":
        from humpback.storage import detection_embeddings_path

        emb_path = detection_embeddings_path(settings.storage_root, job.source_id)
        if not emb_path.exists():
            raise FileNotFoundError(f"No embeddings for detection job {job.source_id}")

        table = pq.read_table(str(emb_path))
        col_names = set(table.column_names)
        embeddings = np.array(
            [row.as_py() for row in table.column("embedding")],
            dtype=np.float32,
        )

        confidences: list[float] | None = None
        if "confidence" in col_names:
            raw = table.column("confidence").to_pylist()
            confidences = [float(c) if c is not None else 0.0 for c in raw]

        # New schema: row_id-based embeddings.
        if "row_id" in col_names:
            row_ids = table.column("row_id").to_pylist()
            return (
                embeddings,
                row_ids,
                [],
                [],
                [],
                [],
                [],
                confidences,
            )

        # Legacy schema: filename/start_sec/end_sec.
        from humpback.classifier.detection_rows import parse_recording_timestamp

        filenames = table.column("filename").to_pylist()
        start_secs = [float(s) for s in table.column("start_sec").to_pylist()]
        end_secs = [float(s) for s in table.column("end_sec").to_pylist()]
        start_utcs: list[float] = []
        end_utcs: list[float] = []
        for i, fname in enumerate(filenames):
            ts = parse_recording_timestamp(fname)
            base = ts.timestamp() if ts else 0.0
            start_utcs.append(base + start_secs[i])
            end_utcs.append(base + end_secs[i])

        return (
            embeddings,
            [],
            filenames,
            start_secs,
            end_secs,
            start_utcs,
            end_utcs,
            confidences,
        )

    elif job.source_type == "embedding_set":
        from humpback.models.processing import EmbeddingSet

        es_result = await session.execute(
            select(EmbeddingSet).where(EmbeddingSet.id == job.source_id)
        )
        es = es_result.scalar_one_or_none()
        if es is None:
            raise ValueError(f"Embedding set {job.source_id} not found")

        parquet_path = Path(es.parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        table = pq.read_table(str(parquet_path))
        filenames = table.column("filename").to_pylist()
        start_secs = [float(s) for s in table.column("start_sec").to_pylist()]
        end_secs = [float(s) for s in table.column("end_sec").to_pylist()]
        embeddings = np.array(
            [row.as_py() for row in table.column("embedding")],
            dtype=np.float32,
        )
        return embeddings, [], filenames, start_secs, end_secs, [], [], None

    elif job.source_type == "rescore":
        # Re-score from a previous inference job's output
        prev_job_result = await session.execute(
            select(VocalizationInferenceJob).where(
                VocalizationInferenceJob.id == job.source_id
            )
        )
        prev_job = prev_job_result.scalar_one_or_none()
        if prev_job is None:
            raise ValueError(f"Previous inference job {job.source_id} not found")
        if not prev_job.output_path:
            raise ValueError("Previous job has no output")

        # Load the previous job's source embeddings recursively
        return await _load_source_embeddings(
            session,
            VocalizationInferenceJob(
                source_type=prev_job.source_type,
                source_id=prev_job.source_id,
                vocalization_model_id=job.vocalization_model_id,
            ),
            settings,
        )

    else:
        raise ValueError(f"Unknown source type: {job.source_type}")
