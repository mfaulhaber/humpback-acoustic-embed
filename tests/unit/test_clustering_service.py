"""Tests for clustering service validation logic."""

import json

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from humpback.database import Base
from humpback.models.classifier import DetectionJob
from humpback.models.clustering import ClusteringJob
from humpback.models.detection_embedding_job import DetectionEmbeddingJob
from humpback.models.vocalization import (
    VocalizationClassifierModel,
    VocalizationInferenceJob,
)
from humpback.services.clustering_service import (
    create_clustering_job,
    create_vocalization_clustering_job,
    list_clustering_eligible_detection_jobs,
    list_vocalization_clustering_jobs,
)


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def test_clustering_rejects_legacy_embedding_set_jobs(session: AsyncSession):
    with pytest.raises(
        ValueError, match="Legacy embedding-set clustering has been removed"
    ):
        await create_clustering_job(session, ["es-1", "es-2"])


# ---- Vocalization Clustering ----


async def _setup_detection_with_inference(
    session: AsyncSession,
    *,
    inference_status: str = "complete",
    model_active: bool = True,
    embedding_status: str = "complete",
) -> tuple[DetectionJob, VocalizationClassifierModel]:
    """Helper to create a detection job with linked inference and embedding jobs."""
    model = VocalizationClassifierModel(
        name="test-voc-model",
        model_family="sklearn_perch_embedding",
        model_dir_path="/tmp/model",
        vocabulary_snapshot=json.dumps(["song", "call"]),
        per_class_thresholds=json.dumps({"song": 0.5, "call": 0.5}),
        is_active=model_active,
    )
    session.add(model)
    await session.flush()

    dj = DetectionJob(
        classifier_model_id="fake-classifier",
        hydrophone_name="Test Hydrophone",
        start_timestamp=1700000000.0,
        end_timestamp=1700003600.0,
        result_summary=json.dumps({"total_detections": 42}),
        status="complete",
    )
    session.add(dj)
    await session.flush()

    inf = VocalizationInferenceJob(
        vocalization_model_id=model.id,
        source_type="detection_job",
        source_id=dj.id,
        status=inference_status,
    )
    session.add(inf)

    emb = DetectionEmbeddingJob(
        detection_job_id=dj.id,
        model_version="perch_v2",
        status=embedding_status,
    )
    session.add(emb)
    await session.flush()

    return dj, model


async def test_vocalization_clustering_happy_path(session: AsyncSession):
    dj, _model = await _setup_detection_with_inference(session)
    job = await create_vocalization_clustering_job(session, [dj.id])
    assert job.id is not None
    assert job.detection_job_ids is not None
    assert json.loads(job.detection_job_ids) == [dj.id]


async def test_vocalization_clustering_rejects_missing_inference(
    session: AsyncSession,
):
    dj, _model = await _setup_detection_with_inference(
        session, inference_status="failed"
    )
    with pytest.raises(ValueError, match="no completed inference"):
        await create_vocalization_clustering_job(session, [dj.id])


async def test_vocalization_clustering_rejects_missing_embedding(
    session: AsyncSession,
):
    dj, _model = await _setup_detection_with_inference(
        session, embedding_status="failed"
    )
    with pytest.raises(ValueError, match="no completed embedding job"):
        await create_vocalization_clustering_job(session, [dj.id])


async def test_list_clustering_eligible_detection_jobs(session: AsyncSession):
    dj, _model = await _setup_detection_with_inference(session)
    eligible = await list_clustering_eligible_detection_jobs(session)
    assert len(eligible) == 1
    assert eligible[0].id == dj.id
    assert eligible[0].detection_count == 42


async def test_list_vocalization_clustering_jobs_excludes_standard(
    session: AsyncSession,
):
    dj, _model = await _setup_detection_with_inference(session)
    await create_vocalization_clustering_job(session, [dj.id])

    session.add(
        ClusteringJob(
            detection_job_ids=None,
        )
    )
    await session.commit()

    voc_jobs = await list_vocalization_clustering_jobs(session)
    assert len(voc_jobs) == 1
    assert voc_jobs[0].detection_job_ids is not None
    assert json.loads(voc_jobs[0].detection_job_ids) == [dj.id]
