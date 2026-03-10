"""Worker orchestration for retrain workflows.

Each call to poll_retrain_workflows:
1. Claims any queued workflow (queued → importing)
2. Loads all active (non-terminal) workflows
3. Advances each one step
4. Returns True if any work was done
"""

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.classifier import ClassifierTrainingJob
from humpback.models.processing import ProcessingJob
from humpback.models.retrain import RetrainWorkflow
from humpback.services import classifier_service
from humpback.services.audio_service import import_folder
from humpback.services.processing_service import create_processing_job
from humpback.workers.queue import claim_retrain_workflow

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("queued", "importing", "processing", "training")
TERMINAL_STATUSES = ("complete", "failed")


async def poll_retrain_workflows(
    session: AsyncSession,
    settings: Settings,
    session_factory,
) -> bool:
    """Poll and advance retrain workflows. Returns True if any work done."""
    did_work = False

    # 1. Claim any queued workflow
    async with session_factory() as claim_session:
        claimed = await claim_retrain_workflow(claim_session)
    if claimed:
        logger.info(f"Claimed retrain workflow {claimed.id}")
        did_work = True

    # 2. Load all active workflows
    result = await session.execute(
        select(RetrainWorkflow).where(RetrainWorkflow.status.in_(ACTIVE_STATUSES))
    )
    active = list(result.scalars().all())

    # 3. Advance each one step
    for wf in active:
        try:
            advanced = await _advance_workflow(wf, settings, session_factory)
            if advanced:
                did_work = True
        except Exception as e:
            logger.exception(f"Retrain workflow {wf.id} failed")
            async with session_factory() as err_session:
                await err_session.execute(
                    update(RetrainWorkflow)
                    .where(RetrainWorkflow.id == wf.id)
                    .values(
                        status="failed",
                        error_message=str(e),
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                await err_session.commit()
            did_work = True

    return did_work


async def _advance_workflow(
    wf: RetrainWorkflow,
    settings: Settings,
    session_factory,
) -> bool:
    """Advance a single workflow one step. Returns True if work done."""
    if wf.status == "importing":
        return await _step_importing(wf, session_factory)
    elif wf.status == "processing":
        return await _step_processing(wf, settings, session_factory)
    elif wf.status == "training":
        return await _step_training(wf, session_factory)
    return False


async def _step_importing(wf: RetrainWorkflow, session_factory) -> bool:
    """Import all folder roots and transition to processing."""
    pos_roots = json.loads(wf.positive_folder_roots)
    neg_roots = json.loads(wf.negative_folder_roots)
    all_roots = list(set(pos_roots + neg_roots))

    summary = {}
    for root in all_roots:
        async with session_factory() as session:
            try:
                result = await import_folder(session, root)
                summary[root] = {
                    "imported": result.imported,
                    "skipped": result.skipped,
                    "errors": result.errors,
                }
            except ValueError as e:
                summary[root] = {"error": str(e)}

    async with session_factory() as session:
        await session.execute(
            update(RetrainWorkflow)
            .where(RetrainWorkflow.id == wf.id)
            .values(
                status="processing",
                import_summary=json.dumps(summary),
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    logger.info(f"Retrain {wf.id}: import complete, transitioning to processing")
    return True


async def _step_processing(
    wf: RetrainWorkflow, settings: Settings, session_factory
) -> bool:
    """Create processing jobs if needed, then poll for completion."""
    if wf.processing_job_ids is None:
        # First time in processing state: create jobs for unprocessed audio
        return await _create_processing_jobs(wf, settings, session_factory)

    # Poll processing jobs for completion
    return await _poll_processing_jobs(wf, session_factory)


async def _create_processing_jobs(
    wf: RetrainWorkflow, settings: Settings, session_factory
) -> bool:
    """Find audio files in folder roots that need processing, create jobs."""
    from humpback.models.audio import AudioFile
    from pathlib import Path

    pos_roots = json.loads(wf.positive_folder_roots)
    neg_roots = json.loads(wf.negative_folder_roots)
    all_roots = list(set(pos_roots + neg_roots))

    feature_config = json.loads(wf.feature_config) if wf.feature_config else None

    job_ids = []
    skipped = 0

    async with session_factory() as session:
        for root in all_roots:
            base_name = Path(root).name
            result = await session.execute(
                select(AudioFile.id).where(
                    AudioFile.source_folder.isnot(None),
                    (AudioFile.folder_path == base_name)
                    | AudioFile.folder_path.startswith(f"{base_name}/"),
                )
            )
            audio_ids = list(result.scalars().all())

            for audio_id in audio_ids:
                pjob, was_skipped = await create_processing_job(
                    session,
                    audio_id,
                    wf.model_version,
                    wf.window_size_seconds,
                    wf.target_sample_rate,
                    feature_config,
                )
                if was_skipped:
                    skipped += 1
                else:
                    job_ids.append(pjob.id)

    total = len(job_ids)
    logger.info(
        f"Retrain {wf.id}: created {total} processing jobs, {skipped} already complete"
    )

    async with session_factory() as session:
        values = {
            "processing_job_ids": json.dumps(job_ids),
            "processing_total": total,
            "processing_complete": 0,
            "updated_at": datetime.now(timezone.utc),
        }
        # If no new jobs needed, skip straight to training
        if total == 0:
            values["status"] = "training"
            logger.info(
                f"Retrain {wf.id}: all audio already processed, skipping to training"
            )

        await session.execute(
            update(RetrainWorkflow).where(RetrainWorkflow.id == wf.id).values(**values)
        )
        await session.commit()

    return True


async def _poll_processing_jobs(wf: RetrainWorkflow, session_factory) -> bool:
    """Check processing job status. Transition when all complete."""
    job_ids = json.loads(wf.processing_job_ids)
    if not job_ids:
        # Edge case: empty list, move to training
        async with session_factory() as session:
            await session.execute(
                update(RetrainWorkflow)
                .where(RetrainWorkflow.id == wf.id)
                .values(
                    status="training",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
        return True

    async with session_factory() as session:
        result = await session.execute(
            select(ProcessingJob.status).where(ProcessingJob.id.in_(job_ids))
        )
        statuses = list(result.scalars().all())

    complete = sum(1 for s in statuses if s == "complete")
    failed = sum(1 for s in statuses if s == "failed")

    async with session_factory() as session:
        if failed > 0:
            await session.execute(
                update(RetrainWorkflow)
                .where(RetrainWorkflow.id == wf.id)
                .values(
                    status="failed",
                    processing_complete=complete,
                    error_message=f"{failed} processing job(s) failed",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
            return True

        if complete == len(job_ids):
            await session.execute(
                update(RetrainWorkflow)
                .where(RetrainWorkflow.id == wf.id)
                .values(
                    status="training",
                    processing_complete=complete,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
            logger.info(
                f"Retrain {wf.id}: processing complete, transitioning to training"
            )
            return True

        # Still in progress — update count
        if complete != (wf.processing_complete or 0):
            await session.execute(
                update(RetrainWorkflow)
                .where(RetrainWorkflow.id == wf.id)
                .values(
                    processing_complete=complete,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
            return True

    return False


async def _step_training(wf: RetrainWorkflow, session_factory) -> bool:
    """Create training job if needed, then poll for completion."""
    if wf.training_job_id is None:
        return await _create_training_job(wf, session_factory)
    return await _poll_training_job(wf, session_factory)


async def _create_training_job(wf: RetrainWorkflow, session_factory) -> bool:
    """Collect embedding sets and create a training job."""
    # Guard: re-check from DB in case a prior attempt committed the training job
    # but crashed before updating the workflow (crash between two commits).
    async with session_factory() as session:
        result = await session.execute(
            select(RetrainWorkflow.training_job_id).where(RetrainWorkflow.id == wf.id)
        )
        existing_tj_id = result.scalar_one_or_none()
    if existing_tj_id:
        # Prior attempt succeeded; update in-memory state and proceed to polling
        wf.training_job_id = existing_tj_id
        return await _poll_training_job(wf, session_factory)

    pos_roots = json.loads(wf.positive_folder_roots)
    neg_roots = json.loads(wf.negative_folder_roots)

    async with session_factory() as session:
        pos_ids = await classifier_service.collect_embedding_sets_for_folders(
            session, pos_roots, wf.model_version
        )
        neg_ids = await classifier_service.collect_embedding_sets_for_folders(
            session, neg_roots, wf.model_version
        )

    if not pos_ids:
        async with session_factory() as session:
            await session.execute(
                update(RetrainWorkflow)
                .where(RetrainWorkflow.id == wf.id)
                .values(
                    status="failed",
                    error_message="No positive embedding sets found in folder roots",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
        return True

    if not neg_ids:
        async with session_factory() as session:
            await session.execute(
                update(RetrainWorkflow)
                .where(RetrainWorkflow.id == wf.id)
                .values(
                    status="failed",
                    error_message="No negative embedding sets found in folder roots",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
        return True

    parameters = json.loads(wf.parameters) if wf.parameters else None

    async with session_factory() as session:
        try:
            tj = await classifier_service.create_training_job(
                session,
                wf.new_model_name,
                pos_ids,
                neg_ids,
                parameters,
            )
        except ValueError as e:
            await session.rollback()
            await session.execute(
                update(RetrainWorkflow)
                .where(RetrainWorkflow.id == wf.id)
                .values(
                    status="failed",
                    error_message=f"Training job creation failed: {e}",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
            return True

        await session.execute(
            update(RetrainWorkflow)
            .where(RetrainWorkflow.id == wf.id)
            .values(
                training_job_id=tj.id,
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    logger.info(f"Retrain {wf.id}: created training job {tj.id}")
    return True


async def _poll_training_job(wf: RetrainWorkflow, session_factory) -> bool:
    """Check training job status. Finalize when complete."""
    async with session_factory() as session:
        result = await session.execute(
            select(ClassifierTrainingJob).where(
                ClassifierTrainingJob.id == wf.training_job_id
            )
        )
        tj = result.scalar_one_or_none()

    if tj is None:
        async with session_factory() as session:
            await session.execute(
                update(RetrainWorkflow)
                .where(RetrainWorkflow.id == wf.id)
                .values(
                    status="failed",
                    error_message="Training job not found",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
        return True

    if tj.status == "failed":
        async with session_factory() as session:
            await session.execute(
                update(RetrainWorkflow)
                .where(RetrainWorkflow.id == wf.id)
                .values(
                    status="failed",
                    error_message=f"Training failed: {tj.error_message}",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
        return True

    if tj.status == "complete":
        async with session_factory() as session:
            await session.execute(
                update(RetrainWorkflow)
                .where(RetrainWorkflow.id == wf.id)
                .values(
                    status="complete",
                    new_model_id=tj.classifier_model_id,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
        logger.info(f"Retrain {wf.id}: complete, new model {tj.classifier_model_id}")
        return True

    return False
