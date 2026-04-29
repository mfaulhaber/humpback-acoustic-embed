"""Unit tests for vocalization training worker with detection-job datasets."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import select

from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.labeling import VocalizationLabel
from humpback.models.training_dataset import TrainingDataset, TrainingDatasetLabel
from humpback.models.vocalization import (
    VocalizationClassifierModel,
    VocalizationTrainingJob,
)
from humpback.workers.vocalization_worker import run_vocalization_training_job


DIM = 16


@pytest.fixture
async def session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


def _make_settings(tmp_path: Path) -> Settings:
    storage = tmp_path / "storage"
    storage.mkdir(exist_ok=True)
    return Settings(
        storage_root=storage, database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db"
    )


def _make_separable_embeddings(
    n: int, center: np.ndarray, seed: int
) -> list[list[float]]:
    rng = np.random.RandomState(seed)
    return [
        (center + rng.randn(DIM) * 0.3).astype(np.float32).tolist() for _ in range(n)
    ]


def _make_detection_embeddings(
    storage_root: Path,
    det_job_id: str,
    row_ids: list[str],
    embeddings: list[list[float]],
    model_version: str = "v1",
) -> None:
    emb_dir = storage_root / "detections" / det_job_id / "embeddings" / model_version
    emb_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.string()),
            "embedding": pa.array(embeddings, type=pa.list_(pa.float32())),
            "confidence": pa.array([0.9] * len(row_ids), type=pa.float32()),
        }
    )
    pq.write_table(table, str(emb_dir / "detection_embeddings.parquet"))


async def _seed_detection_job(
    session,
    settings: Settings,
    det_job_id: str,
    label: str,
    embeddings: list[list[float]],
) -> list[str]:
    row_ids = [f"{det_job_id}-row-{i}" for i in range(len(embeddings))]
    _make_detection_embeddings(settings.storage_root, det_job_id, row_ids, embeddings)
    session.add(
        ClassifierModel(
            id=f"cm-{det_job_id}",
            name=f"cm-{det_job_id}",
            model_path="/tmp/model",
            model_version="v1",
            vector_dim=DIM,
            window_size_seconds=5.0,
            target_sample_rate=32000,
        )
    )
    session.add(
        DetectionJob(
            id=det_job_id,
            status="complete",
            classifier_model_id=f"cm-{det_job_id}",
        )
    )
    for row_id in row_ids:
        session.add(
            VocalizationLabel(
                detection_job_id=det_job_id,
                row_id=row_id,
                label=label,
            )
        )
    return row_ids


@pytest.mark.asyncio
async def test_training_with_source_config_creates_dataset(session_factory, tmp_path):
    settings = _make_settings(tmp_path)
    rng = np.random.RandomState(42)
    center_a = rng.randn(DIM).astype(np.float32) * 2
    center_b = rng.randn(DIM).astype(np.float32) * 2 + 5.0

    async with session_factory() as session:
        await _seed_detection_job(
            session,
            settings,
            "dj1",
            "Whup",
            _make_separable_embeddings(15, center_a, 1),
        )
        await _seed_detection_job(
            session,
            settings,
            "dj2",
            "Moan",
            _make_separable_embeddings(15, center_b, 2),
        )
        await session.flush()

        job = VocalizationTrainingJob(
            source_config=json.dumps({"detection_job_ids": ["dj1", "dj2"]}),
        )
        session.add(job)
        await session.flush()

        await run_vocalization_training_job(session, job, settings)

        updated_job = (
            await session.execute(
                select(VocalizationTrainingJob).where(
                    VocalizationTrainingJob.id == job.id
                )
            )
        ).scalar_one()
        assert updated_job.status == "complete"
        assert updated_job.training_dataset_id is not None

        model = (
            await session.execute(
                select(VocalizationClassifierModel).where(
                    VocalizationClassifierModel.id == updated_job.vocalization_model_id
                )
            )
        ).scalar_one()
        vocab = json.loads(model.vocabulary_snapshot)
        assert "Whup" in vocab
        assert "Moan" in vocab

        dataset = (
            await session.execute(
                select(TrainingDataset).where(
                    TrainingDataset.id == updated_job.training_dataset_id
                )
            )
        ).scalar_one()
        assert dataset.total_rows == 30


@pytest.mark.asyncio
async def test_training_with_dataset_id_reuses_dataset(session_factory, tmp_path):
    settings = _make_settings(tmp_path)
    rng = np.random.RandomState(42)
    center_a = rng.randn(DIM).astype(np.float32) * 2
    center_b = rng.randn(DIM).astype(np.float32) * 2 + 5.0

    async with session_factory() as session:
        await _seed_detection_job(
            session,
            settings,
            "dj1",
            "Whup",
            _make_separable_embeddings(15, center_a, 1),
        )
        await _seed_detection_job(
            session,
            settings,
            "dj2",
            "Moan",
            _make_separable_embeddings(15, center_b, 2),
        )
        await session.flush()

        job1 = VocalizationTrainingJob(
            source_config=json.dumps({"detection_job_ids": ["dj1", "dj2"]}),
        )
        session.add(job1)
        await session.flush()
        await run_vocalization_training_job(session, job1, settings)

        dataset_id = (
            await session.execute(
                select(VocalizationTrainingJob.training_dataset_id).where(
                    VocalizationTrainingJob.id == job1.id
                )
            )
        ).scalar_one()

        job2 = VocalizationTrainingJob(
            source_config=json.dumps({}),
            training_dataset_id=dataset_id,
        )
        session.add(job2)
        await session.flush()
        await run_vocalization_training_job(session, job2, settings)

        updated_job = (
            await session.execute(
                select(VocalizationTrainingJob).where(
                    VocalizationTrainingJob.id == job2.id
                )
            )
        ).scalar_one()
        assert updated_job.status == "complete"
        assert updated_job.training_dataset_id == dataset_id


@pytest.mark.asyncio
async def test_label_edits_reflected_in_retrain(session_factory, tmp_path):
    settings = _make_settings(tmp_path)
    rng = np.random.RandomState(42)
    center_a = rng.randn(DIM).astype(np.float32) * 2
    center_b = rng.randn(DIM).astype(np.float32) * 2 + 5.0

    async with session_factory() as session:
        await _seed_detection_job(
            session,
            settings,
            "dj1",
            "Whup",
            _make_separable_embeddings(15, center_a, 1),
        )
        await _seed_detection_job(
            session,
            settings,
            "dj2",
            "Moan",
            _make_separable_embeddings(15, center_b, 2),
        )
        await session.flush()

        job1 = VocalizationTrainingJob(
            source_config=json.dumps({"detection_job_ids": ["dj1", "dj2"]}),
        )
        session.add(job1)
        await session.flush()
        await run_vocalization_training_job(session, job1, settings)

        dataset_id = (
            await session.execute(
                select(VocalizationTrainingJob.training_dataset_id).where(
                    VocalizationTrainingJob.id == job1.id
                )
            )
        ).scalar_one()

        for i in range(5):
            session.add(
                TrainingDatasetLabel(
                    training_dataset_id=dataset_id,
                    row_index=i,
                    label="Shriek",
                    source="manual",
                )
            )
        await session.flush()

        job2 = VocalizationTrainingJob(
            source_config=json.dumps({}),
            training_dataset_id=dataset_id,
        )
        session.add(job2)
        await session.flush()
        await run_vocalization_training_job(session, job2, settings)

        updated_job = (
            await session.execute(
                select(VocalizationTrainingJob).where(
                    VocalizationTrainingJob.id == job2.id
                )
            )
        ).scalar_one()
        assert updated_job.status == "complete"

        model = (
            await session.execute(
                select(VocalizationClassifierModel).where(
                    VocalizationClassifierModel.id == updated_job.vocalization_model_id
                )
            )
        ).scalar_one()
        vocab = json.loads(model.vocabulary_snapshot)
        assert "Whup" in vocab
        assert "Moan" in vocab
        assert "Shriek" in vocab


@pytest.mark.asyncio
async def test_retired_embedding_set_source_config_fails_fast(
    session_factory, tmp_path
):
    settings = _make_settings(tmp_path)
    async with session_factory() as session:
        job = VocalizationTrainingJob(
            source_config=json.dumps({"embedding_set_ids": ["es1"]}),
        )
        session.add(job)
        await session.flush()
        await run_vocalization_training_job(session, job, settings)

        updated_job = (
            await session.execute(
                select(VocalizationTrainingJob).where(
                    VocalizationTrainingJob.id == job.id
                )
            )
        ).scalar_one()
        assert updated_job.status == "failed"
        assert "retired" in (updated_job.error_message or "")


@pytest.mark.asyncio
async def test_empty_dataset_fails_gracefully(session_factory, tmp_path):
    settings = _make_settings(tmp_path)
    async with session_factory() as session:
        job = VocalizationTrainingJob(
            source_config=json.dumps({"detection_job_ids": []}),
        )
        session.add(job)
        await session.flush()
        await run_vocalization_training_job(session, job, settings)

        updated_job = (
            await session.execute(
                select(VocalizationTrainingJob).where(
                    VocalizationTrainingJob.id == job.id
                )
            )
        ).scalar_one()
        assert updated_job.status == "failed"
        assert updated_job.error_message is not None
