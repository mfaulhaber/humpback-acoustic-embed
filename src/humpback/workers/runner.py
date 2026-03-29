"""Worker runner: polls for jobs and executes them."""

import asyncio
import logging
import signal
import time

from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory
from humpback.services.model_registry_service import seed_default_model
from humpback.workers.classifier_worker import (
    run_detection_job,
    run_extraction_job,
    run_hydrophone_detection_job,
    run_training_job,
)
from humpback.workers.clustering_worker import run_clustering_job
from humpback.workers.processing_worker import run_processing_job
from humpback.workers.queue import (
    claim_clustering_job,
    claim_detection_job,
    claim_extraction_job,
    claim_hydrophone_detection_job,
    claim_processing_job,
    claim_search_job,
    claim_training_job,
    recover_stale_jobs,
)
from humpback.workers.retrain_worker import poll_retrain_workflows
from humpback.workers.search_worker import run_search_job

logger = logging.getLogger(__name__)


async def run_worker(settings: Settings | None = None) -> None:
    if settings is None:
        settings = Settings.from_repo_env()

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting worker...")

    engine = create_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = create_session_factory(engine)

    shutdown = asyncio.Event()

    def handle_signal():
        logger.info("Shutdown signal received")
        shutdown.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            pass  # Windows

    # Seed default model if registry is empty
    async with session_factory() as session:
        await seed_default_model(session)

    # Recover any jobs left in 'running' from a previous crash
    async with session_factory() as session:
        await recover_stale_jobs(session)

    # Periodic stale recovery interval (seconds)
    stale_recovery_interval = 60.0
    last_stale_check = time.monotonic()

    while not shutdown.is_set():
        # Periodically re-check for stale jobs (handles jobs that became stale
        # after startup, e.g., if worker crashed and restarted within the
        # stale timeout window)
        now = time.monotonic()
        if now - last_stale_check >= stale_recovery_interval:
            async with session_factory() as session:
                await recover_stale_jobs(session)
            last_stale_check = now

        claimed = False
        job = cjob = tjob = djob = None

        # Try search jobs first (sub-second interactive work)
        sjob = None
        async with session_factory() as session:
            sjob = await claim_search_job(session)
        if sjob:
            logger.info(f"Search job {sjob.id}")
            async with session_factory() as session:
                await run_search_job(session, sjob, settings)
            claimed = True

        if claimed:
            continue

        # Try processing jobs
        async with session_factory() as session:
            job = await claim_processing_job(session)
        if job:
            logger.info(f"Processing job {job.id} for audio {job.audio_file_id}")
            async with session_factory() as session:
                await run_processing_job(session, job, settings)
            claimed = True

        if claimed:
            continue

        # Then clustering jobs
        async with session_factory() as session:
            cjob = await claim_clustering_job(session)
        if cjob:
            logger.info(f"Clustering job {cjob.id}")
            async with session_factory() as session:
                await run_clustering_job(session, cjob, settings)
            claimed = True

        if claimed:
            continue

        # Then classifier training jobs
        async with session_factory() as session:
            tjob = await claim_training_job(session)
        if tjob:
            logger.info(f"Training job {tjob.id} ({tjob.name})")
            async with session_factory() as session:
                await run_training_job(session, tjob, settings)
            claimed = True

        if claimed:
            continue

        # Then detection jobs
        async with session_factory() as session:
            djob = await claim_detection_job(session)
        if djob:
            logger.info(f"Detection job {djob.id}")
            async with session_factory() as session:
                await run_detection_job(
                    session, djob, settings, session_factory=session_factory
                )
            claimed = True

        if claimed:
            continue

        # Then hydrophone detection jobs
        hjob = None
        async with session_factory() as session:
            hjob = await claim_hydrophone_detection_job(session)
        if hjob:
            logger.info(f"Hydrophone detection job {hjob.id}")
            async with session_factory() as session:
                await run_hydrophone_detection_job(
                    session, hjob, settings, session_factory=session_factory
                )
            claimed = True

        if claimed:
            continue

        # Then extraction jobs
        ejob = None
        async with session_factory() as session:
            ejob = await claim_extraction_job(session)
        if ejob:
            logger.info(f"Extraction job {ejob.id}")
            async with session_factory() as session:
                await run_extraction_job(session, ejob, settings)
            claimed = True

        if claimed:
            continue

        # Then detection embedding jobs
        dejob = None
        async with session_factory() as session:
            from humpback.workers.queue import claim_detection_embedding_job

            dejob = await claim_detection_embedding_job(session)
        if dejob:
            logger.info(f"Detection embedding job {dejob.id}")
            from humpback.workers.detection_embedding_worker import (
                run_detection_embedding_job,
            )

            async with session_factory() as session:
                await run_detection_embedding_job(session, dejob, settings)
            claimed = True

        if claimed:
            continue

        # Then label processing jobs
        lpjob = None
        async with session_factory() as session:
            from humpback.workers.queue import claim_label_processing_job

            lpjob = await claim_label_processing_job(session)
        if lpjob:
            logger.info(f"Label processing job {lpjob.id}")
            from humpback.workers.label_processing_worker import (
                run_label_processing_job,
            )

            async with session_factory() as session:
                await run_label_processing_job(session, lpjob, settings)
            claimed = True

        if claimed:
            continue

        # Then retrain workflows
        async with session_factory() as session:
            retrain_did_work = await poll_retrain_workflows(
                session, settings, session_factory
            )
        if retrain_did_work:
            claimed = True

        if claimed:
            continue

        # Then vocalization training jobs
        vtjob = None
        async with session_factory() as session:
            from humpback.workers.queue import claim_vocalization_training_job

            vtjob = await claim_vocalization_training_job(session)
        if vtjob:
            logger.info(f"Vocalization training job {vtjob.id}")
            from humpback.workers.vocalization_worker import (
                run_vocalization_training_job,
            )

            async with session_factory() as session:
                await run_vocalization_training_job(session, vtjob, settings)
            claimed = True

        if claimed:
            continue

        # Then vocalization inference jobs
        vijob = None
        async with session_factory() as session:
            from humpback.workers.queue import claim_vocalization_inference_job

            vijob = await claim_vocalization_inference_job(session)
        if vijob:
            logger.info(f"Vocalization inference job {vijob.id}")
            from humpback.workers.vocalization_worker import (
                run_vocalization_inference_job,
            )

            async with session_factory() as session:
                await run_vocalization_inference_job(session, vijob, settings)
            claimed = True

        if claimed:
            continue

        # No jobs found, wait before polling again
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=settings.worker_poll_interval
            )
        except asyncio.TimeoutError:
            pass

    await engine.dispose()
    logger.info("Worker stopped")


def main():
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
