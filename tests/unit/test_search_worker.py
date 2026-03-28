"""Unit tests for the search worker."""

import json
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy import select

from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.search import SearchJob
from humpback.workers.search_worker import _resolve_audio, run_search_job


@pytest.fixture
async def db_session(tmp_path):
    """Create an in-memory database with tables, return session factory."""
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = create_session_factory(engine)
    yield sf
    await engine.dispose()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        storage_root=tmp_path / "storage",
        use_real_model=False,
    )


@pytest.fixture
async def seed_data(db_session, tmp_path):
    """Create a detection job + classifier model for search tests."""
    # Create a small WAV file
    import struct
    import wave

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    wav_path = audio_dir / "20240615T080000Z.wav"
    sr = 16000
    duration = 6.0
    n = int(sr * duration)
    samples = [int(32767 * np.sin(2 * np.pi * 440 * i / sr)) for i in range(n)]
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *samples))

    async with db_session() as session:
        # Register model in model_configs so model_cache resolves vector_dim
        from humpback.models.model_registry import ModelConfig

        mc = ModelConfig(
            name="search_test_v1",
            display_name="Perch v1 (test)",
            path="/fake/model.tflite",
            vector_dim=4,
            is_default=True,
        )
        session.add(mc)

        cm = ClassifierModel(
            name="test_classifier",
            model_path="/fake/model.tflite",
            model_version="search_test_v1",
            vector_dim=4,
            window_size_seconds=5.0,
            target_sample_rate=32000,
        )
        session.add(cm)
        await session.flush()

        dj = DetectionJob(
            classifier_model_id=cm.id,
            audio_folder=str(audio_dir),
            status="complete",
        )
        session.add(dj)
        await session.flush()

        ids = {"cm_id": cm.id, "dj_id": dj.id}
        await session.commit()

    return ids


async def test_run_search_job_success(db_session, settings, seed_data):
    """Search worker successfully encodes audio and completes the job."""
    ids = seed_data

    async with db_session() as session:
        sj = SearchJob(
            detection_job_id=ids["dj_id"],
            start_utc=1718438400.0,
            end_utc=1718438405.0,
        )
        session.add(sj)
        await session.commit()
        sj_id = sj.id

    async with db_session() as session:
        await run_search_job(session, sj, settings)

    async with db_session() as session:
        result = await session.execute(select(SearchJob).where(SearchJob.id == sj_id))
        job = result.scalar_one_or_none()
        assert job is not None
        assert job.status == "complete"
        assert job.model_version == "search_test_v1"
        assert job.embedding_vector is not None
        vector = json.loads(job.embedding_vector)
        assert isinstance(vector, list)
        assert len(vector) == 4  # FakeTFLiteModel with vector_dim=4


async def test_run_search_job_missing_detection_job(db_session, settings):
    """Search worker fails if detection job doesn't exist."""
    async with db_session() as session:
        sj = SearchJob(
            detection_job_id="nonexistent-id",
            start_utc=1718438400.0,
            end_utc=1718438405.0,
        )
        session.add(sj)
        await session.commit()
        sj_id = sj.id

    async with db_session() as session:
        await run_search_job(session, sj, settings)

    async with db_session() as session:
        result = await session.execute(select(SearchJob).where(SearchJob.id == sj_id))
        job = result.scalar_one_or_none()
        assert job is not None
        assert job.status == "failed"
        assert "not found" in (job.error_message or "")


async def test_run_search_job_missing_classifier_model(db_session, settings):
    """Search worker fails if classifier model doesn't exist."""
    async with db_session() as session:
        dj = DetectionJob(
            classifier_model_id="nonexistent-model",
            audio_folder="/fake",
            status="complete",
        )
        session.add(dj)
        await session.flush()

        sj = SearchJob(
            detection_job_id=dj.id,
            start_utc=1718438400.0,
            end_utc=1718438405.0,
        )
        session.add(sj)
        await session.commit()
        sj_id = sj.id

    async with db_session() as session:
        await run_search_job(session, sj, settings)

    async with db_session() as session:
        result = await session.execute(select(SearchJob).where(SearchJob.id == sj_id))
        job = result.scalar_one_or_none()
        assert job is not None
        assert job.status == "failed"
        assert "not found" in (job.error_message or "")


async def test_resolve_audio_hydrophone_passes_utc_directly(settings, monkeypatch):
    """Hydrophone search audio passes start_sec directly as absolute UTC timestamp."""
    captured: dict[str, float] = {}
    expected_audio = np.ones(160000, dtype=np.float32)

    def fake_build_archive_playback_provider(*args, **kwargs):
        return object()

    def fake_resolve_audio_slice(
        provider,
        stream_start_ts,
        stream_end_ts,
        start_utc,
        duration_sec,
        target_sr=32000,
        timeline=None,
    ):
        captured["start_utc"] = start_utc
        captured["duration_sec"] = duration_sec
        captured["stream_start_ts"] = stream_start_ts
        captured["stream_end_ts"] = stream_end_ts
        return expected_audio

    monkeypatch.setattr(
        "humpback.classifier.providers.build_archive_playback_provider",
        fake_build_archive_playback_provider,
    )
    monkeypatch.setattr(
        "humpback.classifier.s3_stream.resolve_audio_slice",
        fake_resolve_audio_slice,
    )

    det_job = DetectionJob(
        classifier_model_id="model-1",
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=1751439600.0,
        end_timestamp=1751443200.0,
        status="complete",
    )

    # Absolute UTC timestamp for the chunk start
    chunk_start_utc = (
        datetime.strptime("20250702T080118", "%Y%m%dT%H%M%S")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    audio, sr = await _resolve_audio(
        det_job,
        settings,
        chunk_start_utc,
        chunk_start_utc + 5.0,
    )

    assert sr == 32000
    assert np.array_equal(audio, expected_audio)
    assert captured["start_utc"] == pytest.approx(chunk_start_utc)
    assert captured["duration_sec"] == pytest.approx(5.0)
    assert captured["stream_start_ts"] == pytest.approx(det_job.start_timestamp)
    assert captured["stream_end_ts"] == pytest.approx(det_job.end_timestamp)
