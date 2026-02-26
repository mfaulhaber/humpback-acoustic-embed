from datetime import datetime, timedelta, timezone

from humpback.models.audio import AudioFile
from humpback.models.clustering import ClusteringJob
from humpback.models.processing import JobStatus, ProcessingJob
from humpback.workers.queue import (
    STALE_JOB_TIMEOUT,
    claim_processing_job,
    complete_processing_job,
    fail_processing_job,
    recover_stale_jobs,
)


async def test_claim_queued_job(session):
    af = AudioFile(filename="a.wav", checksum_sha256="q1")
    session.add(af)
    await session.flush()

    job = ProcessingJob(
        audio_file_id=af.id,
        encoding_signature="sig1",
        model_version="v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add(job)
    await session.commit()

    claimed = await claim_processing_job(session)
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status == JobStatus.running.value


async def test_no_jobs_returns_none(session):
    claimed = await claim_processing_job(session)
    assert claimed is None


async def test_complete_job(session):
    af = AudioFile(filename="a.wav", checksum_sha256="q2")
    session.add(af)
    await session.flush()

    job = ProcessingJob(
        audio_file_id=af.id,
        encoding_signature="sig2",
        model_version="v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        status=JobStatus.running.value,
    )
    session.add(job)
    await session.commit()

    await complete_processing_job(session, job.id)
    await session.refresh(job)
    assert job.status == JobStatus.complete.value


async def test_fail_job(session):
    af = AudioFile(filename="a.wav", checksum_sha256="q3")
    session.add(af)
    await session.flush()

    job = ProcessingJob(
        audio_file_id=af.id,
        encoding_signature="sig3",
        model_version="v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        status=JobStatus.running.value,
    )
    session.add(job)
    await session.commit()

    await fail_processing_job(session, job.id, "something broke")
    await session.refresh(job)
    assert job.status == JobStatus.failed.value
    assert job.error_message == "something broke"


async def test_skip_running_same_signature(session):
    af = AudioFile(filename="a.wav", checksum_sha256="q4")
    session.add(af)
    await session.flush()

    # Already running job with sig_a
    running = ProcessingJob(
        audio_file_id=af.id,
        encoding_signature="sig_a",
        model_version="v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        status=JobStatus.running.value,
    )
    # Queued job with same signature
    queued = ProcessingJob(
        audio_file_id=af.id,
        encoding_signature="sig_a",
        model_version="v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add_all([running, queued])
    await session.commit()

    # Should not claim the queued job because sig_a is already running
    claimed = await claim_processing_job(session)
    assert claimed is None


async def test_recover_stale_processing_job(session):
    af = AudioFile(filename="a.wav", checksum_sha256="q5")
    session.add(af)
    await session.flush()

    stale_time = datetime.now(timezone.utc) - STALE_JOB_TIMEOUT - timedelta(minutes=1)
    stale_job = ProcessingJob(
        audio_file_id=af.id,
        encoding_signature="sig_stale",
        model_version="v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        status=JobStatus.running.value,
        updated_at=stale_time,
    )
    session.add(stale_job)
    await session.commit()

    count = await recover_stale_jobs(session)
    assert count == 1

    await session.refresh(stale_job)
    assert stale_job.status == JobStatus.queued.value


async def test_recover_does_not_touch_recent_running_job(session):
    af = AudioFile(filename="a.wav", checksum_sha256="q6")
    session.add(af)
    await session.flush()

    recent_job = ProcessingJob(
        audio_file_id=af.id,
        encoding_signature="sig_recent",
        model_version="v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        status=JobStatus.running.value,
        updated_at=datetime.now(timezone.utc),
    )
    session.add(recent_job)
    await session.commit()

    count = await recover_stale_jobs(session)
    assert count == 0

    await session.refresh(recent_job)
    assert recent_job.status == JobStatus.running.value


async def test_recover_stale_clustering_job(session):
    stale_time = datetime.now(timezone.utc) - STALE_JOB_TIMEOUT - timedelta(minutes=1)
    cjob = ClusteringJob(
        status="running",
        embedding_set_ids="[]",
        updated_at=stale_time,
    )
    session.add(cjob)
    await session.commit()

    count = await recover_stale_jobs(session)
    assert count == 1

    await session.refresh(cjob)
    assert cjob.status == "queued"


async def test_recover_stale_unblocks_queue(session):
    """A stale running job should no longer block queued jobs with the same signature."""
    af = AudioFile(filename="a.wav", checksum_sha256="q7")
    session.add(af)
    await session.flush()

    stale_time = datetime.now(timezone.utc) - STALE_JOB_TIMEOUT - timedelta(minutes=1)
    stale_job = ProcessingJob(
        audio_file_id=af.id,
        encoding_signature="sig_block",
        model_version="v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        status=JobStatus.running.value,
        updated_at=stale_time,
    )
    queued_job = ProcessingJob(
        audio_file_id=af.id,
        encoding_signature="sig_block",
        model_version="v1",
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add_all([stale_job, queued_job])
    await session.commit()

    # Before recovery, the queued job is blocked
    claimed = await claim_processing_job(session)
    assert claimed is None

    await recover_stale_jobs(session)

    # After recovery, one of them can be claimed
    claimed = await claim_processing_job(session)
    assert claimed is not None
