"""Focused tests for retained vocalization clustering worker behavior."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from humpback.config import Settings
from humpback.database import Base
from humpback.models.clustering import ClusteringJob
from humpback.models.detection_embedding_job import DetectionEmbeddingJob
from humpback.storage import cluster_dir, detection_embeddings_path
from humpback.workers.clustering_worker import run_clustering_job


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        storage_root=tmp_path / "storage",
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
    )


def _write_detection_embeddings(settings: Settings, detection_job_id: str) -> None:
    path = detection_embeddings_path(settings.storage_root, detection_job_id, "tf2")
    path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.array(
        [
            [4.8, 4.9, 5.1, 5.0],
            [5.2, 5.1, 4.9, 5.0],
            [4.9, 5.0, 5.2, 4.8],
            [-4.9, -5.0, -5.1, -4.8],
            [-5.2, -4.9, -5.0, -5.1],
            [-5.0, -5.1, -4.8, -5.2],
        ],
        dtype=np.float32,
    )
    table = pa.table(
        {
            "row_id": [f"row-{idx}" for idx in range(len(embeddings))],
            "embedding": [row.tolist() for row in embeddings],
        }
    )
    pq.write_table(table, str(path))


async def test_worker_writes_detection_job_artifact_columns(session, settings):
    detection_job_id = "dj-cluster"
    _write_detection_embeddings(settings, detection_job_id)

    session.add(
        DetectionEmbeddingJob(
            detection_job_id=detection_job_id,
            model_version="tf2",
            status="complete",
        )
    )
    job = ClusteringJob(
        detection_job_ids=json.dumps([detection_job_id]),
        parameters=json.dumps(
            {
                "reduction_method": "pca",
                "clustering_algorithm": "kmeans",
                "n_clusters": 2,
            }
        ),
    )
    session.add(job)
    await session.commit()

    await run_clustering_job(session, job, settings)
    await session.refresh(job)

    assert job.status == "complete"

    output_dir = cluster_dir(settings.storage_root, job.id)
    umap_table = pq.read_table(str(output_dir / "umap_coords.parquet"))
    assignments_table = pq.read_table(str(output_dir / "assignments.parquet"))

    assert "detection_job_id" in umap_table.column_names
    assert "embedding_set_id" not in umap_table.column_names
    assert umap_table.column("detection_job_id").to_pylist() == [detection_job_id] * 6

    assert "detection_job_id" in assignments_table.column_names
    assert "embedding_set_id" not in assignments_table.column_names


async def test_worker_fails_legacy_embedding_set_jobs_clearly(session, settings):
    job = ClusteringJob(detection_job_ids=None)
    session.add(job)
    await session.commit()

    await run_clustering_job(session, job, settings)
    await session.refresh(job)

    assert job.status == "failed"
    assert job.error_message is not None
    assert "Legacy embedding-set clustering jobs are no longer supported" in (
        job.error_message
    )
