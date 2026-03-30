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
from humpback.models.audio import AudioFile
from humpback.models.labeling import VocalizationLabel
from humpback.models.processing import EmbeddingSet
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

        source_config = json.loads(job.source_config)
        parameters = json.loads(job.parameters) if job.parameters else {}
        embedding_set_ids: list[str] = source_config.get("embedding_set_ids", [])
        detection_job_ids: list[str] = source_config.get("detection_job_ids", [])

        # Collect embeddings and multi-hot labels
        all_embeddings: list[np.ndarray] = []
        all_label_sets: list[set[str]] = []
        seen_keys: set[str] = set()

        # Source 1: Embedding sets with call-type folder structure
        for es_id in embedding_set_ids:
            es_result = await session.execute(
                select(EmbeddingSet).where(EmbeddingSet.id == es_id)
            )
            es = es_result.scalar_one_or_none()
            if es is None:
                logger.warning("Embedding set %s not found, skipping", es_id)
                continue

            af_result = await session.execute(
                select(AudioFile).where(AudioFile.id == es.audio_file_id)
            )
            af = af_result.scalar_one_or_none()
            if af is None:
                continue

            # Infer type from folder path leaf
            type_name = None
            if af.folder_path:
                parts = Path(af.folder_path).parts
                if parts:
                    type_name = parts[-1].strip().title()

            if not type_name:
                logger.warning(
                    "No folder-based type for embedding set %s, skipping", es_id
                )
                continue

            parquet_path = Path(es.parquet_path)
            if not parquet_path.exists():
                logger.warning("Parquet %s not found, skipping", parquet_path)
                continue

            table = pq.read_table(str(parquet_path))
            embeddings_col = table.column("embedding")

            for i in range(table.num_rows):
                # Dedup by (embedding_set_id, row_index)
                key = f"{es_id}:{i}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                vec = np.array(embeddings_col[i].as_py(), dtype=np.float32)
                all_embeddings.append(vec)
                all_label_sets.append({type_name})

        # Source 2: Detection job vocalization labels
        for det_job_id in detection_job_ids:
            from humpback.storage import detection_embeddings_path

            emb_path = detection_embeddings_path(settings.storage_root, det_job_id)
            if not emb_path.exists():
                logger.warning(
                    "No embeddings for detection job %s, skipping", det_job_id
                )
                continue

            # Get vocalization labels for this detection job
            result = await session.execute(
                select(VocalizationLabel).where(
                    VocalizationLabel.detection_job_id == det_job_id
                )
            )
            voc_labels = result.scalars().all()

            # Build multi-hot label index by (start_utc, end_utc)
            labels_by_utc: dict[tuple[float, float], set[str]] = {}
            for vl in voc_labels:
                key = (vl.start_utc, vl.end_utc)
                if key not in labels_by_utc:
                    labels_by_utc[key] = set()
                labels_by_utc[key].add(vl.label)

            table = pq.read_table(str(emb_path))
            filenames = table.column("filename").to_pylist()
            start_secs = table.column("start_sec").to_pylist()
            end_secs = table.column("end_sec").to_pylist()
            embeddings_col = table.column("embedding")

            from humpback.classifier.detection_rows import (
                parse_recording_timestamp,
            )

            for i in range(table.num_rows):
                fname = filenames[i]
                ts = parse_recording_timestamp(fname)
                base_epoch = ts.timestamp() if ts else 0.0
                row_start = base_epoch + float(start_secs[i])
                row_end = base_epoch + float(end_secs[i])
                utc_key = (row_start, row_end)

                dedup_key = f"{fname}:{start_secs[i]}:{end_secs[i]}"
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                vec = np.array(embeddings_col[i].as_py(), dtype=np.float32)
                all_embeddings.append(vec)
                all_label_sets.append(labels_by_utc.get(utc_key, set()))

        if not all_embeddings:
            raise ValueError("No embeddings collected from source config")

        X = np.vstack(all_embeddings)
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
            filenames,
            start_secs,
            end_secs,
            start_utcs,
            end_utcs,
            confidences,
        ) = await _load_source_embeddings(session, job, settings)

        # Run inference
        output_dir = settings.storage_root / "vocalization_inference" / job.id
        output_path = output_dir / "predictions.parquet"

        result_summary = await asyncio.to_thread(
            run_inference,
            model_dir,
            embeddings,
            filenames,
            start_secs,
            end_secs,
            output_path,
            start_utcs=start_utcs if start_utcs else None,
            end_utcs=end_utcs if end_utcs else None,
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
    list[float],
    list[float],
    list[float],
    list[float],
    list[float] | None,
]:
    """Load embeddings from the job's source.

    Returns (embeddings, filenames, start_secs, end_secs, start_utcs, end_utcs,
    confidences).  confidences is None when unavailable (embedding sets, old jobs).
    """
    if job.source_type == "detection_job":
        from humpback.classifier.detection_rows import parse_recording_timestamp
        from humpback.storage import detection_embeddings_path

        emb_path = detection_embeddings_path(settings.storage_root, job.source_id)
        if not emb_path.exists():
            raise FileNotFoundError(f"No embeddings for detection job {job.source_id}")

        table = pq.read_table(str(emb_path))
        filenames = table.column("filename").to_pylist()
        start_secs = [float(s) for s in table.column("start_sec").to_pylist()]
        end_secs = [float(s) for s in table.column("end_sec").to_pylist()]
        embeddings = np.array(
            [row.as_py() for row in table.column("embedding")],
            dtype=np.float32,
        )

        confidences: list[float] | None = None
        if "confidence" in table.schema.names:
            raw = table.column("confidence").to_pylist()
            confidences = [float(c) if c is not None else 0.0 for c in raw]

        start_utcs: list[float] = []
        end_utcs: list[float] = []
        for i, fname in enumerate(filenames):
            ts = parse_recording_timestamp(fname)
            base = ts.timestamp() if ts else 0.0
            start_utcs.append(base + start_secs[i])
            end_utcs.append(base + end_secs[i])

        return (
            embeddings,
            filenames,
            start_secs,
            end_secs,
            start_utcs,
            end_utcs,
            confidences,
        )

    elif job.source_type == "embedding_set":
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
        return embeddings, filenames, start_secs, end_secs, [], [], None

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
