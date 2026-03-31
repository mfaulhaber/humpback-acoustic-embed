"""Unit tests for training dataset snapshot and extend service."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import select

from humpback.database import Base, create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.labeling import VocalizationLabel
from humpback.models.processing import EmbeddingSet
from humpback.models.training_dataset import TrainingDatasetLabel
from humpback.services.training_dataset import (
    create_training_dataset_snapshot,
    extend_training_dataset,
)


@pytest.fixture
async def session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


def _make_embedding_set_parquet(path: Path, n_rows: int, dim: int = 8) -> None:
    """Write a minimal embedding set parquet (row_index, embedding)."""
    table = pa.table(
        {
            "row_index": pa.array(list(range(n_rows)), type=pa.int32()),
            "embedding": pa.array(
                [
                    np.random.default_rng(i).random(dim).astype(np.float32).tolist()
                    for i in range(n_rows)
                ],
                type=pa.list_(pa.float32()),
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _make_detection_embeddings_parquet(
    storage_root: Path,
    det_job_id: str,
    filenames: list[str],
    start_secs: list[float],
    end_secs: list[float],
    dim: int = 8,
) -> None:
    """Write a detection_embeddings.parquet for a detection job."""
    n = len(filenames)
    det_dir = storage_root / "detections" / det_job_id
    det_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "filename": pa.array(filenames, type=pa.string()),
            "start_sec": pa.array(start_secs, type=pa.float32()),
            "end_sec": pa.array(end_secs, type=pa.float32()),
            "embedding": pa.array(
                [
                    np.random.default_rng(i).random(dim).astype(np.float32).tolist()
                    for i in range(n)
                ],
                type=pa.list_(pa.float32()),
            ),
            "confidence": pa.array(
                [0.9 - i * 0.1 for i in range(n)], type=pa.float32()
            ),
        }
    )
    pq.write_table(table, str(det_dir / "detection_embeddings.parquet"))


# ---- Snapshot tests ----


@pytest.mark.asyncio
async def test_snapshot_from_embedding_set(session_factory, tmp_path):
    """Embedding set source produces correct rows with folder-inferred type."""
    storage_root = tmp_path / "storage"
    es_parquet = storage_root / "embeddings" / "v1" / "af1" / "sig.parquet"
    _make_embedding_set_parquet(es_parquet, n_rows=3, dim=8)

    async with session_factory() as session:
        af = AudioFile(
            id="af1",
            filename="recording_20250101T120000Z.wav",
            folder_path="/data/Whup",
            checksum_sha256="abc123",
        )
        session.add(af)
        es = EmbeddingSet(
            id="es1",
            audio_file_id="af1",
            encoding_signature="sig",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=8,
            parquet_path=str(es_parquet),
        )
        session.add(es)
        await session.flush()

        dataset = await create_training_dataset_snapshot(
            session,
            {"embedding_set_ids": ["es1"], "detection_job_ids": []},
            storage_root,
        )

        assert dataset.total_rows == 3
        assert "Whup" in json.loads(dataset.vocabulary)

        # Check parquet
        table = pq.read_table(dataset.parquet_path)
        assert table.num_rows == 3
        assert table.column("source_type").to_pylist() == ["embedding_set"] * 3
        assert table.column("source_id").to_pylist() == ["es1"] * 3
        # Window positions: 0-5, 5-10, 10-15
        start_secs = table.column("start_sec").to_pylist()
        assert [float(s) for s in start_secs] == [0.0, 5.0, 10.0]

        # Check labels
        labels_result = await session.execute(
            select(TrainingDatasetLabel).where(
                TrainingDatasetLabel.training_dataset_id == dataset.id
            )
        )
        labels = labels_result.scalars().all()
        assert len(labels) == 3
        assert all(lbl.label == "Whup" for lbl in labels)
        assert all(lbl.source == "snapshot" for lbl in labels)


@pytest.mark.asyncio
async def test_snapshot_from_detection_job(session_factory, tmp_path):
    """Detection job source matches labels by UTC key and excludes unlabeled rows."""
    storage_root = tmp_path / "storage"
    # Filename with parseable timestamp: 2025-01-01 12:00:00 UTC = 1735732800.0
    fname = "recording_20250101T120000Z.wav"
    _make_detection_embeddings_parquet(
        storage_root,
        "dj1",
        filenames=[fname, fname, fname],
        start_secs=[0.0, 5.0, 10.0],
        end_secs=[5.0, 10.0, 15.0],
    )

    async with session_factory() as session:
        # Label only the first two rows; third is unlabeled
        base_epoch = 1735732800.0
        session.add(
            VocalizationLabel(
                detection_job_id="dj1",
                start_utc=base_epoch + 0.0,
                end_utc=base_epoch + 5.0,
                label="Moan",
            )
        )
        session.add(
            VocalizationLabel(
                detection_job_id="dj1",
                start_utc=base_epoch + 5.0,
                end_utc=base_epoch + 10.0,
                label="(Negative)",
            )
        )
        await session.flush()

        dataset = await create_training_dataset_snapshot(
            session,
            {"embedding_set_ids": [], "detection_job_ids": ["dj1"]},
            storage_root,
        )

        # Only 2 labeled rows included (third is unlabeled)
        assert dataset.total_rows == 2

        # Check labels
        labels_result = await session.execute(
            select(TrainingDatasetLabel)
            .where(TrainingDatasetLabel.training_dataset_id == dataset.id)
            .order_by(TrainingDatasetLabel.row_index)
        )
        labels = labels_result.scalars().all()
        assert len(labels) == 2
        assert labels[0].label == "Moan"
        assert labels[0].row_index == 0
        assert labels[1].label == "(Negative)"
        assert labels[1].row_index == 1

        # Vocabulary should not include "(Negative)"
        vocab = json.loads(dataset.vocabulary)
        assert "(Negative)" not in vocab
        assert "Moan" in vocab


@pytest.mark.asyncio
async def test_snapshot_deduplication(session_factory, tmp_path):
    """Duplicate embedding set rows are not included twice."""
    storage_root = tmp_path / "storage"
    es_parquet = storage_root / "embeddings" / "v1" / "af1" / "sig.parquet"
    _make_embedding_set_parquet(es_parquet, n_rows=2, dim=8)

    async with session_factory() as session:
        af = AudioFile(
            id="af1",
            filename="rec.wav",
            folder_path="/data/Whup",
            checksum_sha256="abc123",
        )
        session.add(af)
        es = EmbeddingSet(
            id="es1",
            audio_file_id="af1",
            encoding_signature="sig",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=8,
            parquet_path=str(es_parquet),
        )
        session.add(es)
        await session.flush()

        # Pass the same embedding set ID twice
        dataset = await create_training_dataset_snapshot(
            session,
            {"embedding_set_ids": ["es1", "es1"], "detection_job_ids": []},
            storage_root,
        )
        assert dataset.total_rows == 2  # not 4


@pytest.mark.asyncio
async def test_snapshot_empty_sources_raises(session_factory, tmp_path):
    """Empty sources should raise ValueError."""
    storage_root = tmp_path / "storage"
    async with session_factory() as session:
        with pytest.raises(ValueError, match="No embeddings"):
            await create_training_dataset_snapshot(
                session,
                {"embedding_set_ids": [], "detection_job_ids": []},
                storage_root,
            )


# ---- Extend tests ----


@pytest.mark.asyncio
async def test_extend_appends_rows(session_factory, tmp_path):
    """Extend adds new rows with correct row_index continuation."""
    storage_root = tmp_path / "storage"
    es_parquet1 = storage_root / "embeddings" / "v1" / "af1" / "sig.parquet"
    _make_embedding_set_parquet(es_parquet1, n_rows=2, dim=8)
    es_parquet2 = storage_root / "embeddings" / "v1" / "af2" / "sig.parquet"
    _make_embedding_set_parquet(es_parquet2, n_rows=3, dim=8)

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
            vector_dim=8,
            parquet_path=str(es_parquet1),
        )
        session.add(es1)
        await session.flush()

        # Create initial dataset with es1
        dataset = await create_training_dataset_snapshot(
            session,
            {"embedding_set_ids": ["es1"], "detection_job_ids": []},
            storage_root,
        )
        assert dataset.total_rows == 2

        # Add es2 for extend
        es2 = EmbeddingSet(
            id="es2",
            audio_file_id="af2",
            encoding_signature="sig",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=8,
            parquet_path=str(es_parquet2),
        )
        session.add(es2)
        await session.flush()

        # Extend with es2
        dataset = await extend_training_dataset(
            session,
            dataset,
            {"embedding_set_ids": ["es2"], "detection_job_ids": []},
            storage_root,
        )
        assert dataset.total_rows == 5

        # Check parquet has combined rows
        table = pq.read_table(dataset.parquet_path)
        assert table.num_rows == 5
        row_indices = table.column("row_index").to_pylist()
        assert row_indices == [0, 1, 2, 3, 4]

        # Vocabulary should include both types
        vocab = json.loads(dataset.vocabulary)
        assert "Whup" in vocab
        assert "Moan" in vocab

        # Source config should be merged
        config = json.loads(dataset.source_config)
        assert set(config["embedding_set_ids"]) == {"es1", "es2"}

        # Labels: 2 for es1 (Whup) + 3 for es2 (Moan)
        labels_result = await session.execute(
            select(TrainingDatasetLabel)
            .where(TrainingDatasetLabel.training_dataset_id == dataset.id)
            .order_by(TrainingDatasetLabel.row_index)
        )
        labels = labels_result.scalars().all()
        assert len(labels) == 5
        assert labels[0].label == "Whup"
        assert labels[2].label == "Moan"
        assert labels[2].row_index == 2
