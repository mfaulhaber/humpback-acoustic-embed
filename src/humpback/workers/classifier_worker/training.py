"""Classifier training job execution."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import joblib
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.classifier import (
    AutoresearchCandidate,
    ClassifierModel,
    ClassifierTrainingJob,
)
from humpback.storage import classifier_dir, ensure_dir

logger = logging.getLogger(__name__)


def _merge_candidate_standard_metrics(
    summary: dict[str, Any],
    promotion_provenance: dict[str, Any] | None,
) -> None:
    """Merge standard metric fields into an autoresearch-candidate summary.

    Reads from training_data_source (sample counts), split_metrics (test-split
    precision/recall/confusion), and trainer_parameters (classifier config).
    """
    tds = summary.get("training_data_source", {})
    n_pos = int(tds.get("positive_count") or 0)
    n_neg = int(tds.get("negative_count") or 0)
    summary["n_positive"] = n_pos
    summary["n_negative"] = n_neg
    summary["balance_ratio"] = round(n_pos / n_neg, 4) if n_neg > 0 else 0.0

    if promotion_provenance is None:
        return

    split_metrics = promotion_provenance.get("split_metrics") or {}
    ar_metrics: dict[str, Any] = {}
    for _split_name, split_data in split_metrics.items():
        if isinstance(split_data, dict) and "autoresearch" in split_data:
            ar_metrics = split_data["autoresearch"]
            break

    if ar_metrics:
        precision = float(ar_metrics.get("precision", 0))
        recall = float(ar_metrics.get("recall", 0))
        summary["cv_precision"] = precision
        summary["cv_recall"] = recall
        denom = precision + recall
        summary["cv_f1"] = (
            round(2 * precision * recall / denom, 6) if denom > 0 else 0.0
        )

        tp = int(ar_metrics.get("tp", 0))
        fp = int(ar_metrics.get("fp", 0))
        fn = int(ar_metrics.get("fn", 0))
        tn = int(ar_metrics.get("tn", 0))
        total = tp + fp + fn + tn
        summary["cv_accuracy"] = round((tp + tn) / total, 6) if total > 0 else 0.0
        summary["train_confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    trainer_params = promotion_provenance.get("trainer_parameters") or {}
    if "classifier_type" in trainer_params:
        summary["classifier_type"] = trainer_params["classifier_type"]
    class_weight = trainer_params.get("class_weight")
    if class_weight:
        summary["effective_class_weights"] = class_weight
        summary["class_weight_strategy"] = "custom"


async def run_training_job(
    session: AsyncSession,
    job: ClassifierTrainingJob,
    settings: Settings,
) -> None:
    """Execute a classifier training job end-to-end."""
    try:
        # Parse parameters
        parameters = json.loads(job.parameters) if job.parameters else None
        promotion_provenance = (
            json.loads(job.source_comparison_context)
            if job.source_comparison_context
            else None
        )

        if job.source_mode == "autoresearch_candidate":
            if not job.manifest_path:
                raise ValueError("Candidate-backed training job missing manifest_path")

            from humpback.classifier.replay import (
                apply_context_pooling,
                build_embedding_lookup,
                build_replay_pipeline,
                verify_replay,  # noqa: F811
            )
            from humpback.classifier.trainer import load_manifest_split_data

            promoted_config = (
                json.loads(job.promoted_config) if job.promoted_config else {}
            )

            split_data = await asyncio.to_thread(
                load_manifest_split_data,
                Path(job.manifest_path),
                split=job.training_split_name or "train",
            )
            source_summary = split_data.source_summary
            vector_dim = int(source_summary["vector_dim"])

            # Context pooling on raw embeddings (data-level, before pipeline)
            embedding_lookup = build_embedding_lookup(
                split_data.manifest, split_data.parquet_cache
            )
            pooling_mode = promoted_config.get("context_pooling", "center")
            pooled_lookup, pooling_report = apply_context_pooling(
                split_data.manifest,
                embedding_lookup,
                split_data.parquet_cache,
                pooling_mode,
            )

            # Re-collect train split arrays from pooled embeddings
            from humpback.classifier.replay import collect_split_arrays

            _ids, y_train, X_train, _neg_groups = collect_split_arrays(
                split_data.manifest,
                pooled_lookup,
                split_data.source_summary["split"],
            )

            # Build and fit replay pipeline
            pipeline, effective_config = await asyncio.to_thread(
                build_replay_pipeline,
                promoted_config,
                X_train,
                y_train,
            )

            effective_config.context_pooling = pooling_mode
            effective_config.context_pooling_applied_count = (
                pooling_report.applied_count
            )
            effective_config.context_pooling_fallback_count = (
                pooling_report.fallback_count
            )

            summary: dict[str, Any] = {
                "training_source_mode": job.source_mode,
                "training_data_source": source_summary,
                "replay_effective_config": effective_config.to_dict(),
                "replay_pooling_report": {
                    "applied_count": pooling_report.applied_count,
                    "fallback_count": pooling_report.fallback_count,
                },
                "promoted_config": promoted_config,
            }
            if promotion_provenance:
                summary["promotion_provenance"] = promotion_provenance

            # Replay verification — compare against imported candidate metrics
            candidate_split_metrics = (
                promotion_provenance.get("split_metrics", {})
                if promotion_provenance
                else {}
            )
            threshold = float(promoted_config.get("threshold", 0.5))
            if candidate_split_metrics:
                replay_verification = await asyncio.to_thread(
                    verify_replay,
                    pipeline,
                    split_data.manifest,
                    split_data.parquet_cache,
                    promoted_config,
                    candidate_split_metrics,
                    threshold,
                    settings.replay_metric_tolerance,
                    effective_config,
                )
                summary["replay_verification"] = replay_verification
                # Also store in source_comparison_context for API access
                if promotion_provenance is not None:
                    promotion_provenance["replay_verification"] = replay_verification

            _merge_candidate_standard_metrics(summary, promotion_provenance)
        elif job.source_mode == "detection_manifest":
            from humpback.classifier.trainer import (
                load_manifest_split_embeddings,
                train_binary_classifier,
            )
            from humpback.services.hyperparameter_service.manifest import (
                generate_manifest,
            )

            det_job_ids = json.loads(job.source_detection_job_ids or "[]")
            if not det_job_ids:
                raise ValueError(
                    "detection_manifest job has no source_detection_job_ids"
                )

            # Build and persist the manifest.
            manifest = await asyncio.to_thread(
                generate_manifest,
                detection_job_ids=det_job_ids,
                embedding_model_version=job.model_version,
            )
            cdir_manifest = ensure_dir(classifier_dir(settings.storage_root, job.id))
            manifest_path = cdir_manifest / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))

            # Update the job row with the manifest path.
            await session.execute(
                update(ClassifierTrainingJob)
                .where(ClassifierTrainingJob.id == job.id)
                .values(manifest_path=str(manifest_path))
            )

            # Load training split embeddings.
            (
                positive_embeddings,
                negative_embeddings,
                source_summary,
            ) = await asyncio.to_thread(
                load_manifest_split_embeddings,
                manifest_path,
                split="train",
            )
            vector_dim = int(source_summary["vector_dim"])

            # Train classifier (CPU-bound).
            pipeline, train_summary = await asyncio.to_thread(
                train_binary_classifier,
                positive_embeddings,
                negative_embeddings,
                parameters,
            )
            summary = cast(dict[str, Any], train_summary)
            summary["training_source_mode"] = job.source_mode
            summary["training_data_source"] = source_summary
            summary["detection_job_ids"] = det_job_ids
            summary["manifest_path"] = str(manifest_path)
        else:
            raise ValueError(
                "Embedding-set classifier training jobs are retired; create a "
                "new training job from labeled detection jobs"
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
            vector_dim=vector_dim,
            window_size_seconds=job.window_size_seconds,
            target_sample_rate=job.target_sample_rate,
            feature_config=job.feature_config,
            training_summary=json.dumps(summary),
            training_job_id=job.id,
            training_source_mode=job.source_mode,
            source_candidate_id=job.source_candidate_id,
            source_model_id=job.source_model_id,
            promotion_provenance=json.dumps(promotion_provenance)
            if promotion_provenance
            else None,
        )
        session.add(cm)
        await session.flush()

        completion_time = datetime.now(timezone.utc)
        job_update_values: dict[str, Any] = {
            "classifier_model_id": cm.id,
            "status": "complete",
            "updated_at": completion_time,
        }
        if promotion_provenance is not None:
            job_update_values["source_comparison_context"] = json.dumps(
                promotion_provenance
            )
        await session.execute(
            update(ClassifierTrainingJob)
            .where(ClassifierTrainingJob.id == job.id)
            .values(**job_update_values)
        )
        if job.source_candidate_id:
            await session.execute(
                update(AutoresearchCandidate)
                .where(AutoresearchCandidate.id == job.source_candidate_id)
                .values(
                    status="complete",
                    training_job_id=job.id,
                    new_model_id=cm.id,
                    error_message=None,
                    updated_at=completion_time,
                )
            )
        await session.commit()

    except Exception as e:
        logger.exception("Training job %s failed", job.id)
        try:
            await session.rollback()
        except Exception:
            pass
        try:
            failure_time = datetime.now(timezone.utc)
            await session.execute(
                update(ClassifierTrainingJob)
                .where(ClassifierTrainingJob.id == job.id)
                .values(
                    status="failed",
                    error_message=str(e),
                    updated_at=failure_time,
                )
            )
            if job.source_candidate_id:
                await session.execute(
                    update(AutoresearchCandidate)
                    .where(AutoresearchCandidate.id == job.source_candidate_id)
                    .values(
                        status="failed",
                        error_message=str(e),
                        updated_at=failure_time,
                    )
                )
            await session.commit()
        except Exception:
            logger.exception("Failed to mark training job as failed")
