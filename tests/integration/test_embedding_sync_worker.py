"""Integration tests for sync mode of detection_embedding_worker."""

import json
import math
import struct
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from humpback.classifier.detection_rows import (
    ROW_STORE_FIELDNAMES,
    write_detection_row_store,
)
from humpback.classifier.detector import (
    diff_row_store_vs_embeddings,
    write_detection_embeddings,
)
from humpback.database import Base, create_engine, create_session_factory
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.detection_embedding_job import DetectionEmbeddingJob
from humpback.storage import (
    detection_dir,
    detection_embeddings_path,
    detection_row_store_path,
    ensure_dir,
)
from humpback.workers.detection_embedding_worker import run_detection_embedding_job

# 2021-11-01T08:50:00Z
_BASE_EPOCH = 1635756600.0
_SAMPLE_RATE = 32000


def _write_wav(path: Path, duration: float = 10.0, sample_rate: int = _SAMPLE_RATE):
    n_samples = int(sample_rate * duration)
    samples = [
        int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(n_samples)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


def _make_row(start_utc: float, end_utc: float, label: str = "") -> dict[str, str]:
    row = {f: "" for f in ROW_STORE_FIELDNAMES}
    row["start_utc"] = str(start_utc)
    row["end_utc"] = str(end_utc)
    if label:
        row[label] = "1"
    return row


def _make_emb(filename: str, start_sec: float, end_sec: float, dim: int = 8) -> dict:
    return {
        "filename": filename,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "embedding": np.random.randn(dim).astype(np.float32).tolist(),
        "confidence": 0.9,
    }


async def _init_db(engine):
    """Create all tables in the test database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _setup_detection_job(sf, settings, audio_folder: Path):
    """Create a detection job with a fake classifier model."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Create a minimal fake pipeline
    model_path = settings.storage_root / "test_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    # Fit on dummy data so predict_proba works
    X = np.random.randn(10, 8).astype(np.float32)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    pipe.fit(X, y)
    joblib.dump(pipe, model_path)

    async with sf() as session:
        cm = ClassifierModel(
            name="test-model",
            model_path=str(model_path),
            model_version="fake_tflite",
            vector_dim=8,
            window_size_seconds=5.0,
            target_sample_rate=_SAMPLE_RATE,
        )
        session.add(cm)
        await session.flush()

        dj = DetectionJob(
            classifier_model_id=cm.id,
            audio_folder=str(audio_folder),
            confidence_threshold=0.5,
            hop_seconds=1.0,
            high_threshold=0.7,
            low_threshold=0.45,
            detection_mode="windowed",
            status="complete",
        )
        session.add(dj)
        await session.commit()
        return dj.id, cm.id


@pytest.mark.asyncio
async def test_sync_adds_missing_and_removes_orphans(app_settings):
    """Sync generates embeddings for added rows and removes orphaned ones."""
    engine = create_engine(app_settings.database_url)
    await _init_db(engine)
    sf = create_session_factory(engine)

    # Create audio folder with a 10-second file
    audio_folder = app_settings.storage_root / "test-audio"
    audio_folder.mkdir(parents=True, exist_ok=True)
    _write_wav(audio_folder / "20211101T085000Z.wav", duration=10.0)

    job_id, cm_id = await _setup_detection_job(sf, app_settings, audio_folder)

    # Set up initial state: row store and embeddings with drift
    ensure_dir(detection_dir(app_settings.storage_root, job_id))
    rs_path = detection_row_store_path(app_settings.storage_root, job_id)
    emb_path = detection_embeddings_path(app_settings.storage_root, job_id)

    fname = "20211101T085000Z.wav"
    # Row store: row at 0-5 (matched) + row at 5-10 (missing — manually added)
    rows = [
        _make_row(_BASE_EPOCH + 0, _BASE_EPOCH + 5, "humpback"),
        _make_row(_BASE_EPOCH + 5, _BASE_EPOCH + 10, "humpback"),
    ]
    write_detection_row_store(rs_path, rows)

    # Embeddings: emb at 0-5 (matched) + emb at 100-105 (orphaned — row deleted)
    embs = [
        _make_emb(fname, 0.0, 5.0, dim=8),
        _make_emb(fname, 100.0, 105.0, dim=8),
    ]
    write_detection_embeddings(embs, emb_path)

    # Create and run sync job
    async with sf() as session:
        sync_job = DetectionEmbeddingJob(
            detection_job_id=job_id,
            mode="sync",
        )
        session.add(sync_job)
        await session.commit()
        sync_job_id = sync_job.id

    async with sf() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == sync_job_id)
        )
        sync_job = result.scalar_one()
        await run_detection_embedding_job(session, sync_job, app_settings)

    # Verify result
    async with sf() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == sync_job_id)
        )
        completed_job = result.scalar_one()
        assert completed_job.status == "complete"
        assert completed_job.result_summary is not None

        summary = json.loads(completed_job.result_summary)
        assert summary["added"] == 1
        assert summary["removed"] == 1
        assert summary["unchanged"] == 1
        assert summary["skipped"] == 0

    # Verify embeddings parquet
    table = pq.read_table(str(emb_path))
    assert table.num_rows == 2  # 1 kept + 1 added

    # The orphaned row (100-105) should be gone
    start_secs = table.column("start_sec").to_pylist()
    assert 100.0 not in [float(s) for s in start_secs]


@pytest.mark.asyncio
async def test_sync_already_in_sync(app_settings):
    """Sync completes immediately when row store and embeddings match."""
    engine = create_engine(app_settings.database_url)
    await _init_db(engine)
    sf = create_session_factory(engine)

    audio_folder = app_settings.storage_root / "test-audio"
    audio_folder.mkdir(parents=True, exist_ok=True)
    _write_wav(audio_folder / "20211101T085000Z.wav", duration=10.0)

    job_id, cm_id = await _setup_detection_job(sf, app_settings, audio_folder)

    ensure_dir(detection_dir(app_settings.storage_root, job_id))
    rs_path = detection_row_store_path(app_settings.storage_root, job_id)
    emb_path = detection_embeddings_path(app_settings.storage_root, job_id)

    fname = "20211101T085000Z.wav"
    rows = [_make_row(_BASE_EPOCH + 0, _BASE_EPOCH + 5)]
    write_detection_row_store(rs_path, rows)
    write_detection_embeddings([_make_emb(fname, 0.0, 5.0, dim=8)], emb_path)

    async with sf() as session:
        sync_job = DetectionEmbeddingJob(detection_job_id=job_id, mode="sync")
        session.add(sync_job)
        await session.commit()
        sync_job_id = sync_job.id

    async with sf() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == sync_job_id)
        )
        sync_job = result.scalar_one()
        await run_detection_embedding_job(session, sync_job, app_settings)

    async with sf() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == sync_job_id)
        )
        completed = result.scalar_one()
        assert completed.status == "complete"
        assert completed.result_summary is not None
        summary = json.loads(completed.result_summary)
        assert summary["added"] == 0
        assert summary["removed"] == 0
        assert summary["unchanged"] == 1


@pytest.mark.asyncio
async def test_sync_skips_when_audio_unavailable(app_settings):
    """Sync records skipped rows when audio can't be resolved."""
    engine = create_engine(app_settings.database_url)
    await _init_db(engine)
    sf = create_session_factory(engine)

    audio_folder = app_settings.storage_root / "test-audio"
    audio_folder.mkdir(parents=True, exist_ok=True)
    # Only 10s of audio
    _write_wav(audio_folder / "20211101T085000Z.wav", duration=10.0)

    job_id, _ = await _setup_detection_job(sf, app_settings, audio_folder)

    ensure_dir(detection_dir(app_settings.storage_root, job_id))
    rs_path = detection_row_store_path(app_settings.storage_root, job_id)
    emb_path = detection_embeddings_path(app_settings.storage_root, job_id)

    fname = "20211101T085000Z.wav"
    # Row at an impossible time (1 hour later — no audio file covers it)
    rows = [
        _make_row(_BASE_EPOCH + 0, _BASE_EPOCH + 5),
        _make_row(_BASE_EPOCH + 3600, _BASE_EPOCH + 3605),  # no audio for this
    ]
    write_detection_row_store(rs_path, rows)
    write_detection_embeddings([_make_emb(fname, 0.0, 5.0, dim=8)], emb_path)

    async with sf() as session:
        sync_job = DetectionEmbeddingJob(detection_job_id=job_id, mode="sync")
        session.add(sync_job)
        await session.commit()
        sync_job_id = sync_job.id

    async with sf() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == sync_job_id)
        )
        sync_job = result.scalar_one()
        await run_detection_embedding_job(session, sync_job, app_settings)

    async with sf() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == sync_job_id)
        )
        completed = result.scalar_one()
        assert completed.status == "complete"
        assert completed.result_summary is not None
        summary = json.loads(completed.result_summary)
        assert summary["added"] == 0
        assert summary["skipped"] == 1
        assert len(summary["skipped_reasons"]) == 1
        assert summary["unchanged"] == 1


@pytest.mark.asyncio
async def test_sync_fractional_second_timestamps(app_settings):
    """Sync correctly handles row-store rows with .5 fractional-second timestamps.

    Regression test: rows whose start_utc has sub-second precision caused an
    infinite sync loop because the sync worker truncated the synthetic filename
    to integer seconds and set start_sec=0.0, producing a 0.5s delta that failed
    the strict < tolerance check.
    """
    engine = create_engine(app_settings.database_url)
    await _init_db(engine)
    sf = create_session_factory(engine)

    audio_folder = app_settings.storage_root / "test-audio"
    audio_folder.mkdir(parents=True, exist_ok=True)
    _write_wav(audio_folder / "20211101T085000Z.wav", duration=15.0)

    job_id, _ = await _setup_detection_job(sf, app_settings, audio_folder)

    ensure_dir(detection_dir(app_settings.storage_root, job_id))
    rs_path = detection_row_store_path(app_settings.storage_root, job_id)
    emb_path = detection_embeddings_path(app_settings.storage_root, job_id)

    fname = "20211101T085000Z.wav"
    # Row store: one integer-second row (matched) + one .5-second row (missing)
    rows = [
        _make_row(_BASE_EPOCH + 0, _BASE_EPOCH + 5),
        _make_row(_BASE_EPOCH + 5.5, _BASE_EPOCH + 10.5),
    ]
    write_detection_row_store(rs_path, rows)

    # Embeddings: only the integer-second row exists
    embs = [_make_emb(fname, 0.0, 5.0, dim=8)]
    write_detection_embeddings(embs, emb_path)

    # --- First sync: should add the .5-second row ---
    async with sf() as session:
        sync_job = DetectionEmbeddingJob(detection_job_id=job_id, mode="sync")
        session.add(sync_job)
        await session.commit()
        sync_job_id = sync_job.id

    async with sf() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == sync_job_id)
        )
        sync_job = result.scalar_one()
        await run_detection_embedding_job(session, sync_job, app_settings)

    async with sf() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == sync_job_id)
        )
        completed = result.scalar_one()
        assert completed.status == "complete"
        assert completed.result_summary is not None
        summary = json.loads(completed.result_summary)
        assert summary["added"] == 1
        assert summary["removed"] == 0
        assert summary["unchanged"] == 1

    # --- Verify the diff now shows in-sync ---
    diff = diff_row_store_vs_embeddings(rs_path, emb_path)
    assert diff.missing == [], (
        f"Expected no missing rows after sync, got {len(diff.missing)}"
    )
    assert diff.orphaned_indices == [], (
        f"Expected no orphans after sync, got {len(diff.orphaned_indices)}"
    )

    # --- Second sync: should be a no-op ---
    async with sf() as session:
        sync_job2 = DetectionEmbeddingJob(detection_job_id=job_id, mode="sync")
        session.add(sync_job2)
        await session.commit()
        sync_job2_id = sync_job2.id

    async with sf() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(DetectionEmbeddingJob).where(
                DetectionEmbeddingJob.id == sync_job2_id
            )
        )
        sync_job2 = result.scalar_one()
        await run_detection_embedding_job(session, sync_job2, app_settings)

    async with sf() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(DetectionEmbeddingJob).where(
                DetectionEmbeddingJob.id == sync_job2_id
            )
        )
        completed2 = result.scalar_one()
        assert completed2.status == "complete"
        assert completed2.result_summary is not None
        summary2 = json.loads(completed2.result_summary)
        assert summary2["added"] == 0
        assert summary2["removed"] == 0
        assert summary2["unchanged"] == 2
