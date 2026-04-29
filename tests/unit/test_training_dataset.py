"""Unit tests for detection-job-based training dataset snapshots."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import select

from humpback.database import Base, create_engine, create_session_factory
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.labeling import VocalizationLabel
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


def _make_detection_embeddings_parquet(
    storage_root,
    det_job_id: str,
    row_ids: list[str],
    dim: int = 8,
    model_version: str = "v1",
) -> None:
    emb_dir = storage_root / "detections" / det_job_id / "embeddings" / model_version
    emb_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.string()),
            "embedding": pa.array(
                [
                    np.random.default_rng(i).random(dim).astype(np.float32).tolist()
                    for i in range(len(row_ids))
                ],
                type=pa.list_(pa.float32()),
            ),
            "confidence": pa.array(
                [0.9 - i * 0.1 for i in range(len(row_ids))], type=pa.float32()
            ),
        }
    )
    pq.write_table(table, str(emb_dir / "detection_embeddings.parquet"))


async def _seed_detection_job(
    session,
    storage_root,
    det_job_id: str,
    labels_by_row_id: dict[str, str],
    model_version: str = "v1",
) -> None:
    row_ids = list(labels_by_row_id.keys())
    _make_detection_embeddings_parquet(
        storage_root, det_job_id, row_ids, model_version=model_version
    )

    session.add(
        ClassifierModel(
            id=f"cm-{det_job_id}",
            name=f"cm-{det_job_id}",
            model_path="/tmp/model",
            model_version=model_version,
            vector_dim=8,
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
    for row_id, label in labels_by_row_id.items():
        session.add(
            VocalizationLabel(
                detection_job_id=det_job_id,
                row_id=row_id,
                label=label,
            )
        )


@pytest.mark.asyncio
async def test_snapshot_from_detection_jobs(session_factory, tmp_path):
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        await _seed_detection_job(
            session,
            storage_root,
            "dj1",
            {
                "dj1-row-1": "Moan",
                "dj1-row-2": "(Negative)",
                "dj1-row-3": "Whup",
            },
        )
        await session.flush()

        dataset = await create_training_dataset_snapshot(
            session,
            {"detection_job_ids": ["dj1"]},
            storage_root,
        )

        assert dataset.total_rows == 3
        assert set(json.loads(dataset.vocabulary)) == {"Moan", "Whup"}

        table = pq.read_table(dataset.parquet_path)
        assert table.column("source_type").to_pylist() == ["detection_job"] * 3
        assert table.column("source_id").to_pylist() == ["dj1"] * 3

        labels_result = await session.execute(
            select(TrainingDatasetLabel)
            .where(TrainingDatasetLabel.training_dataset_id == dataset.id)
            .order_by(TrainingDatasetLabel.row_index)
        )
        labels = labels_result.scalars().all()
        assert [lbl.label for lbl in labels] == ["Moan", "(Negative)", "Whup"]


@pytest.mark.asyncio
async def test_snapshot_deduplicates_detection_jobs(session_factory, tmp_path):
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        await _seed_detection_job(
            session,
            storage_root,
            "dj1",
            {
                "dj1-row-1": "Moan",
                "dj1-row-2": "Whup",
            },
        )
        await session.flush()

        dataset = await create_training_dataset_snapshot(
            session,
            {"detection_job_ids": ["dj1", "dj1"]},
            storage_root,
        )
        assert dataset.total_rows == 2


@pytest.mark.asyncio
async def test_snapshot_rejects_retired_embedding_set_sources(
    session_factory, tmp_path
):
    storage_root = tmp_path / "storage"
    async with session_factory() as session:
        with pytest.raises(
            ValueError, match="Embedding-set vocalization training sources are retired"
        ):
            await create_training_dataset_snapshot(
                session,
                {"embedding_set_ids": ["es1"], "detection_job_ids": ["dj1"]},
                storage_root,
            )


@pytest.mark.asyncio
async def test_snapshot_empty_sources_raises(session_factory, tmp_path):
    storage_root = tmp_path / "storage"
    async with session_factory() as session:
        with pytest.raises(ValueError, match="No embeddings"):
            await create_training_dataset_snapshot(
                session,
                {"detection_job_ids": []},
                storage_root,
            )


@pytest.mark.asyncio
async def test_extend_appends_detection_rows(session_factory, tmp_path):
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        await _seed_detection_job(
            session,
            storage_root,
            "dj1",
            {
                "dj1-row-1": "Whup",
                "dj1-row-2": "Whup",
            },
        )
        await _seed_detection_job(
            session,
            storage_root,
            "dj2",
            {
                "dj2-row-1": "Moan",
                "dj2-row-2": "Moan",
                "dj2-row-3": "(Negative)",
            },
        )
        await session.flush()

        dataset = await create_training_dataset_snapshot(
            session,
            {"detection_job_ids": ["dj1"]},
            storage_root,
        )
        assert dataset.total_rows == 2

        dataset = await extend_training_dataset(
            session,
            dataset,
            {"detection_job_ids": ["dj2"]},
            storage_root,
        )
        assert dataset.total_rows == 5

        table = pq.read_table(dataset.parquet_path)
        assert table.num_rows == 5
        assert table.column("row_index").to_pylist() == [0, 1, 2, 3, 4]

        config = json.loads(dataset.source_config)
        assert set(config["detection_job_ids"]) == {"dj1", "dj2"}
