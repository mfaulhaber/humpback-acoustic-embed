"""Integration tests for the Pass 2 segmentation training worker.

Exercises the happy path (claim -> train -> checkpoint + model row),
the failure path (exception mid-training -> status failed, no partial
artifacts), the missing-dataset guard, and the stale-job recovery
sweep.  Audio is synthesized procedurally per-test to keep the suite
self-contained.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import SegmentationModel
from humpback.models.segmentation_training import (
    SegmentationTrainingDataset,
    SegmentationTrainingJob,
    SegmentationTrainingSample,
)
from humpback.schemas.call_parsing import SegmentationTrainingConfig
from humpback.workers.queue import recover_stale_jobs
from humpback.workers.segmentation_training_worker import run_one_iteration


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
    storage.mkdir(parents=True, exist_ok=True)
    return Settings(
        storage_root=storage,
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
    )


def _write_tone_wav(
    path: Path,
    duration_sec: float,
    freq: float,
    sample_rate: int = 16000,
) -> None:
    n = int(sample_rate * duration_sec)
    samples = [
        int(32767 * 0.7 * math.sin(2 * math.pi * freq * i / sample_rate))
        for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))


def _write_silence_wav(
    path: Path,
    duration_sec: float,
    sample_rate: int = 16000,
) -> None:
    n = int(sample_rate * duration_sec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))


async def _seed_training_job(
    session_factory,
    tmp_path: Path,
    *,
    n_sources: int = 4,
    duration_sec: float = 1.0,
) -> tuple[str, str]:
    """Seed an audio-file dataset with ``n_sources`` positive+negative pairs.

    Returns ``(dataset_id, job_id)``.
    """
    audio_dir = tmp_path / "audio_src"
    async with session_factory() as session:
        dataset = SegmentationTrainingDataset(
            name="test-dataset",
            description="synthetic tone + silence",
        )
        session.add(dataset)
        await session.flush()
        dataset_id = dataset.id

        for i in range(n_sources):
            pos_path = audio_dir / f"pos_{i}.wav"
            neg_path = audio_dir / f"neg_{i}.wav"
            _write_tone_wav(pos_path, duration_sec=duration_sec, freq=440.0)
            _write_silence_wav(neg_path, duration_sec=duration_sec)

            pos_af = AudioFile(
                filename=pos_path.name,
                folder_path="",
                source_folder=str(audio_dir),
                checksum_sha256=f"sum-pos-{i}-{tmp_path.name}",
                duration_seconds=duration_sec,
            )
            neg_af = AudioFile(
                filename=neg_path.name,
                folder_path="",
                source_folder=str(audio_dir),
                checksum_sha256=f"sum-neg-{i}-{tmp_path.name}",
                duration_seconds=duration_sec,
            )
            session.add_all([pos_af, neg_af])
            await session.flush()

            session.add(
                SegmentationTrainingSample(
                    training_dataset_id=dataset_id,
                    audio_file_id=pos_af.id,
                    crop_start_sec=0.0,
                    crop_end_sec=duration_sec,
                    events_json=json.dumps(
                        [{"start_sec": 0.0, "end_sec": duration_sec}]
                    ),
                    source="test",
                )
            )
            session.add(
                SegmentationTrainingSample(
                    training_dataset_id=dataset_id,
                    audio_file_id=neg_af.id,
                    crop_start_sec=0.0,
                    crop_end_sec=duration_sec,
                    events_json="[]",
                    source="test",
                )
            )

        config = SegmentationTrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-2,
            weight_decay=0.0,
            early_stopping_patience=100,
            grad_clip=1.0,
            seed=0,
            val_fraction=0.25,
            conv_channels=[8],
            gru_hidden=8,
            gru_layers=1,
        )
        job = SegmentationTrainingJob(
            training_dataset_id=dataset_id,
            config_json=config.model_dump_json(),
            status="queued",
        )
        session.add(job)
        await session.commit()
        return dataset_id, job.id


async def test_worker_happy_path_persists_model_and_checkpoint(
    session_factory, tmp_path
):
    _, job_id = await _seed_training_job(session_factory, tmp_path)
    settings = _make_settings(tmp_path)

    async with session_factory() as session:
        claimed = await run_one_iteration(session, settings)
    assert claimed is not None
    assert claimed.id == job_id

    async with session_factory() as session:
        refreshed = await session.get(SegmentationTrainingJob, job_id)
        assert refreshed is not None
        assert refreshed.status == "complete"
        assert refreshed.error_message is None
        assert refreshed.segmentation_model_id is not None
        assert refreshed.completed_at is not None
        assert refreshed.result_summary is not None

        summary = json.loads(refreshed.result_summary)
        assert "train_losses" in summary
        assert "framewise" in summary
        assert "event" in summary
        assert summary["event"]["iou_threshold"] == 0.3
        assert summary["n_train_samples"] > 0

        model = await session.get(SegmentationModel, refreshed.segmentation_model_id)
        assert model is not None
        assert model.model_family == "pytorch_crnn"
        assert model.training_job_id == job_id
        assert model.config_json is not None

        model_config = json.loads(model.config_json)
        assert model_config["model_type"] == "SegmentationCRNN"
        assert "framewise_f1" in model_config["metrics"]
        assert "event_f1_iou_0_3" in model_config["metrics"]

    checkpoint_path = Path(model.model_path)
    assert checkpoint_path.exists()
    assert checkpoint_path.name == "checkpoint.pt"
    assert (checkpoint_path.parent / "config.json").exists()


async def test_worker_failure_path_cleans_up_and_marks_failed(
    session_factory, tmp_path, monkeypatch
):
    _, job_id = await _seed_training_job(session_factory, tmp_path)
    settings = _make_settings(tmp_path)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("trainer exploded")

    monkeypatch.setattr(
        "humpback.workers.segmentation_training_worker.train_model",
        _boom,
    )

    async with session_factory() as session:
        claimed = await run_one_iteration(session, settings)
    assert claimed is not None

    async with session_factory() as session:
        refreshed = await session.get(SegmentationTrainingJob, job_id)
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert refreshed.error_message is not None
        assert "trainer exploded" in refreshed.error_message
        assert refreshed.segmentation_model_id is None

        from sqlalchemy import select

        models = await session.execute(select(SegmentationModel))
        assert models.scalar_one_or_none() is None

    seg_root = settings.storage_root / "segmentation_models"
    if seg_root.exists():
        # If the directory was ever created, it must be empty after cleanup.
        assert list(seg_root.iterdir()) == []


async def test_worker_fails_when_training_dataset_missing(session_factory, tmp_path):
    settings = _make_settings(tmp_path)
    async with session_factory() as session:
        config = SegmentationTrainingConfig(
            epochs=1,
            conv_channels=[8],
            gru_hidden=8,
            gru_layers=1,
        )
        job = SegmentationTrainingJob(
            training_dataset_id="does-not-exist",
            config_json=config.model_dump_json(),
            status="queued",
        )
        session.add(job)
        await session.commit()
        job_id = job.id

    async with session_factory() as session:
        claimed = await run_one_iteration(session, settings)
    assert claimed is not None

    async with session_factory() as session:
        refreshed = await session.get(SegmentationTrainingJob, job_id)
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert refreshed.error_message is not None
        assert "does-not-exist" in refreshed.error_message


async def test_stale_segmentation_training_job_is_recovered(session_factory):
    stale_ts = datetime.now(timezone.utc) - timedelta(minutes=20)
    async with session_factory() as session:
        job = SegmentationTrainingJob(
            training_dataset_id="anything",
            config_json="{}",
            status="running",
            updated_at=stale_ts,
        )
        session.add(job)
        await session.commit()
        job_id = job.id

    async with session_factory() as session:
        recovered = await recover_stale_jobs(session)
    assert recovered >= 1

    async with session_factory() as session:
        refreshed = await session.get(SegmentationTrainingJob, job_id)
        assert refreshed is not None
        assert refreshed.status == "queued"
