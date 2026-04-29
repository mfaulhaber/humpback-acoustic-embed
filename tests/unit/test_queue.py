import asyncio
from datetime import datetime, timedelta, timezone

from humpback.database import create_session_factory
from humpback.models.clustering import ClusteringJob
from humpback.models.detection_embedding_job import DetectionEmbeddingJob
from humpback.models.hyperparameter import (
    HyperparameterManifest,
    HyperparameterSearchJob,
)
from humpback.models.sequence_models import ContinuousEmbeddingJob, HMMSequenceJob
from humpback.workers.queue import (
    STALE_JOB_TIMEOUT,
    claim_clustering_job,
    claim_detection_embedding_job,
    recover_stale_jobs,
)


async def test_claim_clustering_job(session):
    job = ClusteringJob(status="queued", detection_job_ids="[]")
    session.add(job)
    await session.commit()

    claimed = await claim_clustering_job(session)
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status == "running"


async def test_claim_detection_embedding_job(session):
    job = DetectionEmbeddingJob(
        status="queued",
        detection_job_id="det-1",
        model_version="tf2",
    )
    session.add(job)
    await session.commit()

    claimed = await claim_detection_embedding_job(session)
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status == "running"


async def test_recover_stale_jobs_requeues_retained_job_types(session):
    stale_time = datetime.now(timezone.utc) - STALE_JOB_TIMEOUT - timedelta(minutes=1)

    manifest = HyperparameterManifest(
        name="manifest",
        status="running",
        training_job_ids="[]",
        detection_job_ids="[]",
        embedding_model_version="tf2",
        split_ratio="[70, 15, 15]",
        seed=42,
        updated_at=stale_time,
    )
    search = HyperparameterSearchJob(
        name="search",
        status="running",
        manifest_id="manifest-1",
        search_space='{"classifier":["logreg"]}',
        n_trials=5,
        seed=42,
        updated_at=stale_time,
    )
    clustering = ClusteringJob(
        status="running",
        detection_job_ids="[]",
        updated_at=stale_time,
    )
    detection_embedding = DetectionEmbeddingJob(
        status="running",
        detection_job_id="det-1",
        model_version="tf2",
        updated_at=stale_time,
    )
    continuous = ContinuousEmbeddingJob(
        status="running",
        event_segmentation_job_id="seg-1",
        model_version="tf2",
        window_size_seconds=5.0,
        hop_seconds=1.0,
        pad_seconds=2.0,
        target_sample_rate=32000,
        encoding_signature="enc-1",
        updated_at=stale_time,
    )
    hmm = HMMSequenceJob(
        status="running",
        continuous_embedding_job_id="ce-1",
        n_states=4,
        pca_dims=8,
        updated_at=stale_time,
    )
    session.add_all(
        [manifest, search, clustering, detection_embedding, continuous, hmm]
    )
    await session.commit()

    count = await recover_stale_jobs(session)
    assert count == 6

    for job in (manifest, search, clustering, detection_embedding, continuous, hmm):
        await session.refresh(job)
        assert job.status == "queued"


async def test_recover_stale_jobs_ignores_recent_retained_jobs(session):
    recent_time = datetime.now(timezone.utc)
    clustering = ClusteringJob(
        status="running",
        detection_job_ids="[]",
        updated_at=recent_time,
    )
    detection_embedding = DetectionEmbeddingJob(
        status="running",
        detection_job_id="det-1",
        model_version="tf2",
        updated_at=recent_time,
    )
    session.add_all([clustering, detection_embedding])
    await session.commit()

    count = await recover_stale_jobs(session)
    assert count == 0

    await session.refresh(clustering)
    await session.refresh(detection_embedding)
    assert clustering.status == "running"
    assert detection_embedding.status == "running"


async def test_claim_clustering_job_is_atomic_across_sessions(session):
    job = ClusteringJob(status="queued", detection_job_ids="[]")
    session.add(job)
    await session.commit()

    factory = create_session_factory(session.bind)

    async def _claim_once() -> str | None:
        async with factory() as claim_session:
            claimed = await claim_clustering_job(claim_session)
            return claimed.id if claimed is not None else None

    c1, c2 = await asyncio.gather(_claim_once(), _claim_once())
    claimed_ids = [cid for cid in (c1, c2) if cid is not None]

    assert len(claimed_ids) == 1
    assert claimed_ids[0] == job.id
