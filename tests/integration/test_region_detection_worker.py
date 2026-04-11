"""Integration tests for the Pass 1 region detection worker.

End-to-end coverage with a short fixture audio file, a synthetic real
sklearn classifier, and the deterministic ``FakeTFLiteModel`` embedding
stub. Exercises the happy path, the failure path, and the queue-level
stale recovery sweep for ``RegionDetectionJob`` rows.

Hydrophone-source worker integration is deferred — see
``docs/plans/backlog.md``.
"""

from __future__ import annotations

import math
import struct
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from humpback.call_parsing.storage import region_job_dir
from humpback.classifier.trainer import train_binary_classifier
from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.classifier import ClassifierModel
from humpback.processing.inference import FakeTFLiteModel
from humpback.schemas.call_parsing import RegionDetectionConfig
from humpback.workers.queue import recover_stale_jobs
from humpback.workers.region_detection_worker import run_one_iteration


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


def _write_sine_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    n = int(sample_rate * duration_sec)
    samples = [
        int(32767 * 0.7 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))


def _synthetic_classifier() -> Pipeline:
    """Real sklearn Pipeline trained to fire on sin-shaped embeddings.

    Matches the fixture in ``tests/unit/test_detector_refactor.py`` so the
    Pass 1 worker produces at least one above-threshold event on the sine
    audio the integration fixture writes.
    """
    rng = np.random.RandomState(42)
    seed_embedding = np.sin(np.arange(64) * (1 + 1) / 64).astype(np.float32)
    pos = np.tile(seed_embedding, (20, 1)) + rng.randn(20, 64) * 0.01
    neg = rng.randn(20, 64) * 0.5 - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)
    return pipeline


async def _seed_fixture(session_factory, tmp_path: Path) -> tuple[str, str, str]:
    audio_dir = tmp_path / "audio_src"
    audio_path = audio_dir / "sample_20260411T000000Z.wav"
    _write_sine_wav(audio_path, duration_sec=12.0)

    async with session_factory() as session:
        audio_file = AudioFile(
            filename="sample_20260411T000000Z.wav",
            folder_path="",
            source_folder=str(audio_dir),
            checksum_sha256=f"sum-{tmp_path.name}",
        )
        session.add(audio_file)
        await session.flush()

        cm = ClassifierModel(
            name="fake-cm",
            model_path="/tmp/not-a-real-file.joblib",
            model_version="perch_v1",
            vector_dim=64,
            window_size_seconds=5.0,
            target_sample_rate=16000,
        )
        session.add(cm)
        await session.flush()

        config = RegionDetectionConfig(
            window_size_seconds=5.0,
            hop_seconds=1.0,
            high_threshold=0.70,
            low_threshold=0.45,
            padding_sec=1.0,
            min_region_duration_sec=0.0,
        )
        job = RegionDetectionJob(
            audio_file_id=audio_file.id,
            classifier_model_id=cm.id,
            model_config_id="perch_v1_mc",
            config_json=config.model_dump_json(),
            status="queued",
        )
        session.add(job)
        await session.commit()
        return audio_file.id, cm.id, job.id


@pytest.fixture
def patched_models(monkeypatch):
    pipeline = _synthetic_classifier()
    model = FakeTFLiteModel(vector_dim=64)

    def _fake_joblib_load(_path):
        return pipeline

    async def _fake_get_model_by_version(_session, _model_version, _settings):
        return model, "spectrogram"

    monkeypatch.setattr(
        "humpback.workers.region_detection_worker.joblib.load",
        _fake_joblib_load,
    )
    monkeypatch.setattr(
        "humpback.workers.region_detection_worker.get_model_by_version",
        _fake_get_model_by_version,
    )
    return pipeline, model


async def test_worker_happy_path_writes_trace_and_regions(
    session_factory, tmp_path, patched_models
):
    _, _, job_id = await _seed_fixture(session_factory, tmp_path)
    settings = _make_settings(tmp_path)

    async with session_factory() as session:
        claimed = await run_one_iteration(session, settings)
    assert claimed is not None
    assert claimed.id == job_id

    async with session_factory() as session:
        refreshed = await session.get(RegionDetectionJob, job_id)
        assert refreshed is not None
        assert refreshed.status == "complete"
        assert refreshed.error_message is None
        assert refreshed.trace_row_count is not None
        assert refreshed.trace_row_count > 0
        assert refreshed.region_count is not None
        assert refreshed.region_count >= 1
        assert refreshed.completed_at is not None

    job_dir = region_job_dir(settings.storage_root, job_id)
    trace_path = job_dir / "trace.parquet"
    regions_path = job_dir / "regions.parquet"
    assert trace_path.exists()
    assert regions_path.exists()

    from humpback.call_parsing.storage import read_regions, read_trace

    trace_rows = read_trace(trace_path)
    region_rows = read_regions(regions_path)
    assert len(trace_rows) == refreshed.trace_row_count
    assert len(region_rows) == refreshed.region_count
    assert len(region_rows) >= 1
    for region in region_rows:
        assert 0.0 <= region.padded_start_sec <= region.padded_end_sec <= 12.0
        assert region.start_sec <= region.end_sec
        assert region.n_windows >= 1


async def test_worker_failure_path_cleans_up_partial_artifacts(
    session_factory, tmp_path, monkeypatch
):
    _, _, job_id = await _seed_fixture(session_factory, tmp_path)
    settings = _make_settings(tmp_path)

    class _BrokenPipeline:
        def predict_proba(self, _X):
            raise RuntimeError("classifier exploded")

    def _fake_joblib_load(_path):
        return _BrokenPipeline()

    async def _fake_get_model_by_version(_session, _model_version, _settings):
        return FakeTFLiteModel(vector_dim=64), "spectrogram"

    monkeypatch.setattr(
        "humpback.workers.region_detection_worker.joblib.load",
        _fake_joblib_load,
    )
    monkeypatch.setattr(
        "humpback.workers.region_detection_worker.get_model_by_version",
        _fake_get_model_by_version,
    )

    async with session_factory() as session:
        claimed = await run_one_iteration(session, settings)
    assert claimed is not None

    async with session_factory() as session:
        refreshed = await session.get(RegionDetectionJob, job_id)
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert refreshed.error_message is not None
        assert "classifier exploded" in refreshed.error_message
        assert refreshed.trace_row_count is None
        assert refreshed.region_count is None

    job_dir = region_job_dir(settings.storage_root, job_id)
    assert not (job_dir / "trace.parquet").exists()
    assert not (job_dir / "regions.parquet").exists()
    assert list(job_dir.glob("*.tmp")) == []


async def test_stale_region_detection_job_is_recovered(session_factory):
    stale_ts = datetime.now(timezone.utc) - timedelta(minutes=20)
    async with session_factory() as session:
        job = RegionDetectionJob(
            audio_file_id="audio-x",
            status="running",
            updated_at=stale_ts,
        )
        session.add(job)
        await session.commit()
        job_id = job.id

    async with session_factory() as session:
        recovered_count = await recover_stale_jobs(session)
    assert recovered_count >= 1

    async with session_factory() as session:
        refreshed = await session.get(RegionDetectionJob, job_id)
        assert refreshed is not None
        assert refreshed.status == "queued"
