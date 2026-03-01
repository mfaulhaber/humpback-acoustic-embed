"""Worker runner: polls for jobs and executes them."""

import asyncio
import logging
import signal
import sys

from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory
from humpback.services.model_registry_service import seed_default_model
from humpback.workers.clustering_worker import run_clustering_job
from humpback.workers.processing_worker import run_processing_job
from humpback.workers.queue import claim_clustering_job, claim_processing_job, recover_stale_jobs

logger = logging.getLogger(__name__)


async def run_worker(settings: Settings | None = None) -> None:
    if settings is None:
        settings = Settings()

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

    while not shutdown.is_set():
        async with session_factory() as session:
            # Try processing jobs first
            job = await claim_processing_job(session)
            if job:
                logger.info(f"Processing job {job.id} for audio {job.audio_file_id}")
                await run_processing_job(session, job, settings)
                continue

            # Then clustering jobs
            cjob = await claim_clustering_job(session)
            if cjob:
                logger.info(f"Clustering job {cjob.id}")
                await run_clustering_job(session, cjob, settings)
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
