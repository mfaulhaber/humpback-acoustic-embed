"""End-to-end tests for the detection re-embedding worker (full mode).

Uses the fake embedding model (``use_real_model=False``) and a fresh row store
to verify that the worker:
  - writes a parquet keyed by ``row_id`` to the model-versioned path,
  - sets ``rows_total`` then updates ``rows_processed``,
  - is idempotent for completed jobs,
  - persists ``error_message`` on failure and clears it on retry.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from sqlalchemy import select

from humpback.classifier.detection_rows import (
    ROW_STORE_FIELDNAMES,
    write_detection_row_store,
)
from humpback.database import Base, create_engine, create_session_factory
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.detection_embedding_job import DetectionEmbeddingJob
from humpback.services.detection_embedding_service import create_reembedding_job
from humpback.storage import (
    detection_dir,
    detection_embeddings_path,
    detection_row_store_path,
    ensure_dir,
)
from humpback.workers.detection_embedding_worker import run_detection_embedding_job

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


_ROW_COUNTER = 0


def _make_row(start_utc: float, end_utc: float, row_id: str) -> dict[str, str]:
    global _ROW_COUNTER  # noqa: PLW0603
    _ROW_COUNTER += 1
    row = {f: "" for f in ROW_STORE_FIELDNAMES}
    row["row_id"] = row_id
    row["start_utc"] = str(start_utc)
    row["end_utc"] = str(end_utc)
    row["humpback"] = "1"
    return row


async def _init_db(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _setup_detection_job(sf, settings, audio_folder: Path) -> str:
    """Create a detection job with a fake classifier and row store."""
    async with sf() as session:
        cm = ClassifierModel(
            name="test-model",
            model_path=str(settings.storage_root / "classifier.joblib"),
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
        return dj.id


@pytest.mark.asyncio
async def test_full_mode_writes_model_versioned_parquet(app_settings):
    engine = create_engine(app_settings.database_url)
    await _init_db(engine)
    sf = create_session_factory(engine)

    audio_folder = app_settings.storage_root / "audio"
    audio_folder.mkdir(parents=True, exist_ok=True)
    _write_wav(audio_folder / "20211101T085000Z.wav", duration=10.0)

    det_job_id = await _setup_detection_job(sf, app_settings, audio_folder)

    ensure_dir(detection_dir(app_settings.storage_root, det_job_id))
    rs_path = detection_row_store_path(app_settings.storage_root, det_job_id)
    write_detection_row_store(
        rs_path,
        [
            _make_row(_BASE_EPOCH + 0, _BASE_EPOCH + 5, "r-1"),
            _make_row(_BASE_EPOCH + 5, _BASE_EPOCH + 10, "r-2"),
        ],
    )

    async with sf() as session:
        job = await create_reembedding_job(session, det_job_id, "perch_v2")
        job_id = job.id

    async with sf() as session:
        job = (
            await session.execute(
                select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == job_id)
            )
        ).scalar_one()
        await run_detection_embedding_job(session, job, app_settings)

    async with sf() as session:
        finished = (
            await session.execute(
                select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == job_id)
            )
        ).scalar_one()
        assert finished.status == "complete"
        assert finished.rows_total == 2
        assert finished.rows_processed == 2
        assert finished.error_message is None
        summary = json.loads(finished.result_summary or "{}")
        assert summary["total"] == 2

    emb_path = detection_embeddings_path(
        app_settings.storage_root, det_job_id, "perch_v2"
    )
    assert emb_path.exists(), "parquet should be under embeddings/<model_version>/"
    table = pq.read_table(str(emb_path))
    assert set(table.column_names) == {"row_id", "embedding", "confidence"}
    assert set(table.column("row_id").to_pylist()) == {"r-1", "r-2"}


@pytest.mark.asyncio
async def test_full_mode_idempotent_when_complete(app_settings):
    engine = create_engine(app_settings.database_url)
    await _init_db(engine)
    sf = create_session_factory(engine)

    audio_folder = app_settings.storage_root / "audio"
    audio_folder.mkdir(parents=True, exist_ok=True)
    _write_wav(audio_folder / "20211101T085000Z.wav", duration=10.0)

    det_job_id = await _setup_detection_job(sf, app_settings, audio_folder)

    ensure_dir(detection_dir(app_settings.storage_root, det_job_id))
    rs_path = detection_row_store_path(app_settings.storage_root, det_job_id)
    write_detection_row_store(
        rs_path,
        [_make_row(_BASE_EPOCH + 0, _BASE_EPOCH + 5, "r-a")],
    )

    async with sf() as session:
        job = await create_reembedding_job(session, det_job_id, "perch_v2")
        job_id = job.id

    async with sf() as session:
        job = (
            await session.execute(
                select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == job_id)
            )
        ).scalar_one()
        await run_detection_embedding_job(session, job, app_settings)

    # Re-enqueue — should be a no-op (returns the completed row).
    async with sf() as session:
        retry = await create_reembedding_job(session, det_job_id, "perch_v2")
        assert retry.id == job_id
        assert retry.status == "complete"


@pytest.mark.asyncio
async def test_full_mode_persists_error_and_retries_clear_it(app_settings):
    engine = create_engine(app_settings.database_url)
    await _init_db(engine)
    sf = create_session_factory(engine)

    det_job_id = await _setup_detection_job(
        sf, app_settings, app_settings.storage_root / "audio"
    )
    # No row store → worker will raise, landing the job in failed state.

    async with sf() as session:
        job = await create_reembedding_job(session, det_job_id, "perch_v2")
        job_id = job.id

    async with sf() as session:
        job = (
            await session.execute(
                select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == job_id)
            )
        ).scalar_one()
        await run_detection_embedding_job(session, job, app_settings)

    async with sf() as session:
        failed = (
            await session.execute(
                select(DetectionEmbeddingJob).where(DetectionEmbeddingJob.id == job_id)
            )
        ).scalar_one()
        assert failed.status == "failed"
        assert failed.error_message is not None
        assert "Row store not found" in failed.error_message

    # Retry via the service — should reset status and clear error_message.
    async with sf() as session:
        retried = await create_reembedding_job(session, det_job_id, "perch_v2")
        assert retried.id == job_id
        assert retried.status == "queued"
        assert retried.error_message is None
