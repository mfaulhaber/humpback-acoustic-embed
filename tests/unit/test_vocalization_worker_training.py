"""Unit tests for vocalization training worker with training datasets."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import select

from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.labeling import VocalizationLabel
from humpback.models.processing import EmbeddingSet
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
    """Generate n embeddings clustered around center."""
    rng = np.random.RandomState(seed)
    return [
        (center + rng.randn(DIM) * 0.3).astype(np.float32).tolist() for _ in range(n)
    ]


def _make_embedding_set_parquet(path: Path, embeddings: list[list[float]]) -> None:
    table = pa.table(
        {
            "row_index": pa.array(list(range(len(embeddings))), type=pa.int32()),
            "embedding": pa.array(embeddings, type=pa.list_(pa.float32())),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _make_detection_embeddings(
    storage_root: Path,
    det_job_id: str,
    row_ids: list[str],
    embeddings: list[list[float]],
) -> None:
    det_dir = storage_root / "detections" / det_job_id
    det_dir.mkdir(parents=True, exist_ok=True)
    n = len(row_ids)
    table = pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.string()),
            "embedding": pa.array(embeddings, type=pa.list_(pa.float32())),
            "confidence": pa.array([0.9] * n, type=pa.float32()),
        }
    )
    pq.write_table(table, str(det_dir / "detection_embeddings.parquet"))


@pytest.mark.asyncio
async def test_training_with_source_config_creates_dataset(session_factory, tmp_path):
    """Mode A: source_config creates a training dataset, then trains."""
    settings = _make_settings(tmp_path)
    rng = np.random.RandomState(42)
    center_a = rng.randn(DIM).astype(np.float32) * 2
    center_b = rng.randn(DIM).astype(np.float32) * 2 + 5.0

    # Create embedding set for type "Whup"
    es_path = settings.storage_root / "embeddings" / "v1" / "af1" / "sig.parquet"
    _make_embedding_set_parquet(es_path, _make_separable_embeddings(15, center_a, 1))

    # Create detection job with labeled "Moan" windows
    n_det = 15
    det_row_ids = [f"dj1-row-{i}" for i in range(n_det)]
    _make_detection_embeddings(
        settings.storage_root,
        "dj1",
        det_row_ids,
        _make_separable_embeddings(n_det, center_b, 2),
    )

    async with session_factory() as session:
        af = AudioFile(
            id="af1",
            filename="a.wav",
            folder_path="/data/Whup",
            checksum_sha256="a1",
        )
        session.add(af)
        es = EmbeddingSet(
            id="es1",
            audio_file_id="af1",
            encoding_signature="sig",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=DIM,
            parquet_path=str(es_path),
        )
        session.add(es)

        # Add vocalization labels for detection job
        for i in range(n_det):
            session.add(
                VocalizationLabel(
                    detection_job_id="dj1",
                    row_id=det_row_ids[i],
                    label="Moan",
                )
            )
        await session.flush()

        # Create training job with source_config
        job = VocalizationTrainingJob(
            source_config=json.dumps(
                {
                    "embedding_set_ids": ["es1"],
                    "detection_job_ids": ["dj1"],
                }
            ),
        )
        session.add(job)
        await session.flush()

        await run_vocalization_training_job(session, job, settings)

        # Verify job completed
        job_result = await session.execute(
            select(VocalizationTrainingJob).where(VocalizationTrainingJob.id == job.id)
        )
        updated_job = job_result.scalar_one()
        assert updated_job.status == "complete"
        assert updated_job.training_dataset_id is not None

        # Verify model has training_dataset_id
        model_result = await session.execute(
            select(VocalizationClassifierModel).where(
                VocalizationClassifierModel.id == updated_job.vocalization_model_id
            )
        )
        model = model_result.scalar_one()
        assert model.training_dataset_id == updated_job.training_dataset_id
        vocab = json.loads(model.vocabulary_snapshot)
        assert "Whup" in vocab
        assert "Moan" in vocab

        # Verify training dataset was created
        ds_result = await session.execute(
            select(TrainingDataset).where(
                TrainingDataset.id == updated_job.training_dataset_id
            )
        )
        dataset = ds_result.scalar_one()
        assert dataset.total_rows == 30  # 15 from es + 15 from detection


@pytest.mark.asyncio
async def test_training_with_dataset_id_reuses_dataset(session_factory, tmp_path):
    """Mode B: training_dataset_id reuses existing dataset."""
    settings = _make_settings(tmp_path)
    rng = np.random.RandomState(42)
    center_a = rng.randn(DIM).astype(np.float32) * 2
    center_b = rng.randn(DIM).astype(np.float32) * 2 + 5.0

    es_path_a = settings.storage_root / "embeddings" / "v1" / "af1" / "sig.parquet"
    _make_embedding_set_parquet(es_path_a, _make_separable_embeddings(15, center_a, 1))
    es_path_b = settings.storage_root / "embeddings" / "v1" / "af2" / "sig.parquet"
    _make_embedding_set_parquet(es_path_b, _make_separable_embeddings(15, center_b, 2))

    async with session_factory() as session:
        af1 = AudioFile(
            id="af1",
            filename="a.wav",
            folder_path="/data/Whup",
            checksum_sha256="a1",
        )
        af2 = AudioFile(
            id="af2",
            filename="b.wav",
            folder_path="/data/Moan",
            checksum_sha256="a2",
        )
        session.add_all([af1, af2])
        es1 = EmbeddingSet(
            id="es1",
            audio_file_id="af1",
            encoding_signature="sig",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=DIM,
            parquet_path=str(es_path_a),
        )
        es2 = EmbeddingSet(
            id="es2",
            audio_file_id="af2",
            encoding_signature="sig",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=DIM,
            parquet_path=str(es_path_b),
        )
        session.add_all([es1, es2])
        await session.flush()

        # First training: creates dataset
        job1 = VocalizationTrainingJob(
            source_config=json.dumps(
                {"embedding_set_ids": ["es1", "es2"], "detection_job_ids": []}
            ),
        )
        session.add(job1)
        await session.flush()
        await run_vocalization_training_job(session, job1, settings)

        job1_result = await session.execute(
            select(VocalizationTrainingJob).where(VocalizationTrainingJob.id == job1.id)
        )
        job1_updated = job1_result.scalar_one()
        dataset_id = job1_updated.training_dataset_id

        # Second training: reuses dataset
        job2 = VocalizationTrainingJob(
            source_config=json.dumps({}),
            training_dataset_id=dataset_id,
        )
        session.add(job2)
        await session.flush()
        await run_vocalization_training_job(session, job2, settings)

        job2_result = await session.execute(
            select(VocalizationTrainingJob).where(VocalizationTrainingJob.id == job2.id)
        )
        job2_updated = job2_result.scalar_one()
        assert job2_updated.status == "complete"
        assert job2_updated.training_dataset_id == dataset_id

        # Both models point to same dataset
        model_result = await session.execute(
            select(VocalizationClassifierModel).where(
                VocalizationClassifierModel.id == job2_updated.vocalization_model_id
            )
        )
        model2 = model_result.scalar_one()
        assert model2.training_dataset_id == dataset_id


@pytest.mark.asyncio
async def test_label_edits_reflected_in_retrain(session_factory, tmp_path):
    """Editing training_dataset_labels between trains changes the outcome."""
    settings = _make_settings(tmp_path)
    rng = np.random.RandomState(42)
    center_a = rng.randn(DIM).astype(np.float32) * 2
    center_b = rng.randn(DIM).astype(np.float32) * 2 + 5.0

    es_path_a = settings.storage_root / "embeddings" / "v1" / "af1" / "sig.parquet"
    _make_embedding_set_parquet(es_path_a, _make_separable_embeddings(15, center_a, 1))
    es_path_b = settings.storage_root / "embeddings" / "v1" / "af2" / "sig.parquet"
    _make_embedding_set_parquet(es_path_b, _make_separable_embeddings(15, center_b, 2))

    async with session_factory() as session:
        af1 = AudioFile(
            id="af1",
            filename="a.wav",
            folder_path="/data/Whup",
            checksum_sha256="a1",
        )
        af2 = AudioFile(
            id="af2",
            filename="b.wav",
            folder_path="/data/Moan",
            checksum_sha256="a2",
        )
        session.add_all([af1, af2])
        es1 = EmbeddingSet(
            id="es1",
            audio_file_id="af1",
            encoding_signature="sig",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=DIM,
            parquet_path=str(es_path_a),
        )
        es2 = EmbeddingSet(
            id="es2",
            audio_file_id="af2",
            encoding_signature="sig",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=DIM,
            parquet_path=str(es_path_b),
        )
        session.add_all([es1, es2])
        await session.flush()

        # First train — Whup + Moan
        job1 = VocalizationTrainingJob(
            source_config=json.dumps(
                {"embedding_set_ids": ["es1", "es2"], "detection_job_ids": []}
            ),
        )
        session.add(job1)
        await session.flush()
        await run_vocalization_training_job(session, job1, settings)

        job1_result = await session.execute(
            select(VocalizationTrainingJob).where(VocalizationTrainingJob.id == job1.id)
        )
        job1_updated = job1_result.scalar_one()
        assert job1_updated.status == "complete"
        dataset_id = job1_updated.training_dataset_id

        # Add a new label type "Shriek" to some Whup rows (multi-label)
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

        # Retrain with edited labels
        job2 = VocalizationTrainingJob(
            source_config=json.dumps({}),
            training_dataset_id=dataset_id,
        )
        session.add(job2)
        await session.flush()
        await run_vocalization_training_job(session, job2, settings)

        job2_result = await session.execute(
            select(VocalizationTrainingJob).where(VocalizationTrainingJob.id == job2.id)
        )
        job2_updated = job2_result.scalar_one()
        assert job2_updated.status == "complete"

        # New model should have Whup, Moan, and Shriek in vocabulary
        model_result = await session.execute(
            select(VocalizationClassifierModel).where(
                VocalizationClassifierModel.id == job2_updated.vocalization_model_id
            )
        )
        model = model_result.scalar_one()
        vocab = json.loads(model.vocabulary_snapshot)
        assert "Whup" in vocab
        assert "Moan" in vocab
        assert "Shriek" in vocab


@pytest.mark.asyncio
async def test_empty_dataset_fails_gracefully(session_factory, tmp_path):
    """Training with an empty dataset sets status to failed."""
    settings = _make_settings(tmp_path)
    async with session_factory() as session:
        job = VocalizationTrainingJob(
            source_config=json.dumps(
                {"embedding_set_ids": [], "detection_job_ids": []}
            ),
        )
        session.add(job)
        await session.flush()
        await run_vocalization_training_job(session, job, settings)

        job_result = await session.execute(
            select(VocalizationTrainingJob).where(VocalizationTrainingJob.id == job.id)
        )
        updated_job = job_result.scalar_one()
        assert updated_job.status == "failed"
        assert "No embeddings" in (updated_job.error_message or "")


# ---- pytorch_event_cnn rejection test ------------------------------------


@pytest.mark.asyncio
async def test_pytorch_event_cnn_rejected_by_vocalization_worker(
    session_factory, tmp_path
):
    """model_family='pytorch_event_cnn' is no longer handled by vocalization worker."""
    settings = _make_settings(tmp_path)

    async with session_factory() as session:
        job = VocalizationTrainingJob(
            source_config="{}",
            parameters=json.dumps({"samples": []}),
            model_family="pytorch_event_cnn",
            input_mode="segmented_event",
        )
        session.add(job)
        await session.flush()

        await run_vocalization_training_job(session, job, settings)

        job_result = await session.execute(
            select(VocalizationTrainingJob).where(VocalizationTrainingJob.id == job.id)
        )
        updated_job = job_result.scalar_one()
        assert updated_job.status == "failed"
        assert "Unsupported model_family" in (updated_job.error_message or "")
