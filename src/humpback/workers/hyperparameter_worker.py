"""Worker functions for hyperparameter manifest generation and search jobs."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.hyperparameter import (
    HyperparameterManifest,
    HyperparameterSearchJob,
)

logger = logging.getLogger(__name__)


async def run_manifest_job(
    session: AsyncSession,
    job: HyperparameterManifest,
    settings: Settings,
) -> None:
    """Execute a hyperparameter manifest generation job."""
    try:
        from humpback.services.hyperparameter_service.manifest import generate_manifest
        from humpback.storage import hyperparameter_manifest_path

        training_job_ids = (
            json.loads(job.training_job_ids) if job.training_job_ids else []
        )
        detection_job_ids = (
            json.loads(job.detection_job_ids) if job.detection_job_ids else []
        )
        split_ratio_list = (
            json.loads(job.split_ratio) if job.split_ratio else [70, 15, 15]
        )
        split_ratio = (split_ratio_list[0], split_ratio_list[1], split_ratio_list[2])

        manifest = generate_manifest(
            training_job_ids=training_job_ids or None,
            detection_job_ids=detection_job_ids or None,
            split_ratio=split_ratio,
            seed=job.seed,
        )

        # Write manifest JSON to disk
        manifest_path = hyperparameter_manifest_path(settings.storage_root, job.id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Compute summary
        examples = manifest["examples"]
        example_count = len(examples)
        split_summary: dict[str, dict[str, int]] = {}
        for split_name in ["train", "val", "test"]:
            split_examples = [ex for ex in examples if ex.get("split") == split_name]
            if split_examples:
                split_summary[split_name] = {
                    "total": len(split_examples),
                    "positive": sum(
                        1 for ex in split_examples if int(ex["label"]) == 1
                    ),
                    "negative": sum(
                        1 for ex in split_examples if int(ex["label"]) == 0
                    ),
                }

        detection_job_summaries = manifest.get("metadata", {}).get(
            "detection_job_summaries", {}
        )

        await session.execute(
            update(HyperparameterManifest)
            .where(HyperparameterManifest.id == job.id)
            .values(
                status="complete",
                manifest_path=str(manifest_path),
                example_count=example_count,
                split_summary=json.dumps(split_summary),
                detection_job_summaries=json.dumps(detection_job_summaries),
                completed_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
        logger.info(f"Manifest job {job.id} complete: {example_count} examples")

    except Exception:
        logger.exception(f"Manifest job {job.id} failed")
        import traceback

        error_msg = traceback.format_exc()
        await session.execute(
            update(HyperparameterManifest)
            .where(HyperparameterManifest.id == job.id)
            .values(
                status="failed",
                error_message=error_msg[-2000:],
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()


async def run_hyperparameter_search_job(
    session: AsyncSession,
    job: HyperparameterSearchJob,
    settings: Settings,
) -> None:
    """Execute a hyperparameter search job."""
    try:
        from humpback.services.hyperparameter_service.search import run_search
        from humpback.services.hyperparameter_service.train_eval import load_manifest
        from humpback.storage import hyperparameter_search_results_dir

        # Load the manifest from the referenced manifest job
        manifest_result = await session.execute(
            HyperparameterManifest.__table__.select().where(
                HyperparameterManifest.id == job.manifest_id
            )
        )
        manifest_row = manifest_result.mappings().first()
        if manifest_row is None:
            raise ValueError(f"Manifest {job.manifest_id} not found")
        if manifest_row["status"] != "complete":
            raise ValueError(
                f"Manifest {job.manifest_id} is not complete "
                f"(status: {manifest_row['status']})"
            )
        manifest_path = manifest_row["manifest_path"]
        if not manifest_path:
            raise ValueError(f"Manifest {job.manifest_id} has no manifest_path")

        manifest = load_manifest(manifest_path)
        search_space = json.loads(job.search_space)
        results_dir = hyperparameter_search_results_dir(settings.storage_root, job.id)

        # Store results_dir on the job
        await session.execute(
            update(HyperparameterSearchJob)
            .where(HyperparameterSearchJob.id == job.id)
            .values(
                results_dir=str(results_dir),
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

        # Progress callback: update DB periodically
        job_id = job.id

        async def _update_progress(
            trials_completed: int,
            best_objective: float | None,
            best_config: dict | None,
            best_metrics: dict | None,
        ) -> None:
            await session.execute(
                update(HyperparameterSearchJob)
                .where(HyperparameterSearchJob.id == job_id)
                .values(
                    trials_completed=trials_completed,
                    best_objective=best_objective,
                    best_config=json.dumps(best_config) if best_config else None,
                    best_metrics=json.dumps(best_metrics) if best_metrics else None,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()

        _pending_progress_task: asyncio.Task | None = None  # type: ignore[type-arg]

        def sync_progress_callback(
            trials_completed: int,
            best_objective: float | None,
            best_config: dict | None,
            best_metrics: dict | None,
        ) -> None:
            nonlocal _pending_progress_task

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Cancel any still-pending progress update to avoid
                    # concurrent session.commit() calls.
                    if _pending_progress_task and not _pending_progress_task.done():
                        _pending_progress_task.cancel()
                    _pending_progress_task = loop.create_task(
                        _update_progress(
                            trials_completed, best_objective, best_config, best_metrics
                        )
                    )
                else:
                    loop.run_until_complete(
                        _update_progress(
                            trials_completed, best_objective, best_config, best_metrics
                        )
                    )
            except Exception:
                logger.warning("Failed to update search progress", exc_info=True)

        # Run the search (CPU-bound, runs synchronously)
        summary = run_search(
            manifest=manifest,
            search_space=search_space,
            n_trials=job.n_trials,
            seed=job.seed,
            results_dir=results_dir,
            progress_callback=sync_progress_callback,
        )

        # Drain any in-flight progress task before touching the session again
        if _pending_progress_task is not None:
            if not _pending_progress_task.done():
                _pending_progress_task.cancel()
            try:
                await _pending_progress_task
            except (asyncio.CancelledError, Exception):
                pass

        # Run comparison if a production model was specified
        comparison_result = None
        if job.comparison_model_id:
            from humpback.services.hyperparameter_service.comparison import (
                compare_classifiers,
                resolve_production_classifier,
            )

            production_classifier = resolve_production_classifier(
                settings,
                classifier_id=job.comparison_model_id,
            )

            best_run_path = results_dir / "best_run.json"
            if best_run_path.exists():
                with open(best_run_path) as f:
                    best_run = json.load(f)

                kwargs: dict[str, Any] = {}
                if job.comparison_threshold is not None:
                    kwargs["production_threshold"] = job.comparison_threshold
                comparison_result = compare_classifiers(
                    manifest,
                    best_run,
                    production_classifier,
                    **kwargs,
                )

        if comparison_result is not None:
            comparison_file = results_dir / "comparison.json"
            comparison_file.write_text(json.dumps(comparison_result, indent=2))

        await session.execute(
            update(HyperparameterSearchJob)
            .where(HyperparameterSearchJob.id == job.id)
            .values(
                status="complete",
                trials_completed=summary["total_trials"],
                best_objective=summary["best_objective"],
                best_config=json.dumps(summary["best_config"])
                if summary["best_config"]
                else None,
                best_metrics=json.dumps(summary["best_metrics"])
                if summary["best_metrics"]
                else None,
                comparison_result=json.dumps(comparison_result)
                if comparison_result
                else None,
                completed_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
        logger.info(
            f"Search job {job.id} complete: {summary['total_trials']} trials, "
            f"best objective: {summary['best_objective']}"
        )

    except Exception:
        logger.exception(f"Search job {job.id} failed")
        import traceback

        error_msg = traceback.format_exc()
        await session.execute(
            update(HyperparameterSearchJob)
            .where(HyperparameterSearchJob.id == job.id)
            .values(
                status="failed",
                error_message=error_msg[-2000:],
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
