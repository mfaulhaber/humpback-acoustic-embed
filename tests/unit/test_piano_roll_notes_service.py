"""Tests for the Piano Roll Notes service and Phase A worker scaffold."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from humpback.models.piano_roll_notes import (
    DEFAULT_EXTRACTOR_VERSION,
    PianoRollNotesJob,
)
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import EventEncoderJob
from humpback.schemas.piano_roll_notes import (
    PianoRollNotesJobRead,
    PianoRollNotesStatusAbsent,
)
from humpback.services.piano_roll_notes_service import (
    PianoRollNotesJobConflict,
    auto_enqueue_after_encoder_complete,
    enqueue_piano_roll_notes_job,
    latest_for_encoder_job,
)
from humpback.workers.piano_roll_notes_worker import (
    NOTES_SCHEMA,
    run_piano_roll_notes_job,
)


async def _make_encoder_job(session, *, status: str = JobStatus.complete.value) -> str:
    job = EventEncoderJob(
        status=status,
        event_segmentation_job_id="seg-1",
        event_source_mode="raw",
        continuous_embedding_job_id="cej-1",
        continuous_embedding_signature="cej-sig",
        tokenizer_version="crnn-event-encoder-v2",
        pooling_config_json="{}",
        descriptor_config_json="{}",
        preprocessing_config_json="{}",
        k_values_json="[50]",
        random_seed=0,
        tokenization_signature=f"tok-sig-{datetime.now(timezone.utc).timestamp()}",
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job.id


# ---------- Service tests ----------


@pytest.mark.asyncio
async def test_enqueue_creates_new_row(session) -> None:
    encoder_id = await _make_encoder_job(session)
    job, created = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder_id
    )

    assert created is True
    assert job.event_encoder_job_id == encoder_id
    assert job.extractor_version == DEFAULT_EXTRACTOR_VERSION
    assert job.status == JobStatus.queued.value
    assert job.params_json == "{}"


@pytest.mark.asyncio
async def test_enqueue_complete_returns_existing(session) -> None:
    encoder_id = await _make_encoder_job(session)
    first, created = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder_id
    )
    assert created is True

    first.status = JobStatus.complete.value
    await session.commit()

    second, created_again = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder_id
    )
    assert created_again is False
    assert second.id == first.id
    assert second.status == JobStatus.complete.value


@pytest.mark.asyncio
async def test_enqueue_running_raises_conflict(session) -> None:
    encoder_id = await _make_encoder_job(session)
    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder_id
    )
    job.status = JobStatus.running.value
    await session.commit()

    with pytest.raises(PianoRollNotesJobConflict):
        await enqueue_piano_roll_notes_job(session, event_encoder_job_id=encoder_id)


@pytest.mark.asyncio
async def test_enqueue_failed_resets_row(session) -> None:
    encoder_id = await _make_encoder_job(session)
    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder_id
    )
    job.status = JobStatus.failed.value
    job.error_message = "boom"
    job.finished_at = datetime.now(timezone.utc)
    job.notes_path = "/tmp/old.parquet"
    job.n_events = 99
    job.n_notes = 1234
    job.compute_seconds = 12.5
    await session.commit()

    reset, created = await enqueue_piano_roll_notes_job(
        session,
        event_encoder_job_id=encoder_id,
        params={"k_noise": 4.0},
    )
    assert created is False
    assert reset.id == job.id
    assert reset.status == JobStatus.queued.value
    assert reset.error_message is None
    assert reset.notes_path is None
    assert reset.n_events is None
    assert reset.n_notes is None
    assert reset.compute_seconds is None
    assert json.loads(reset.params_json) == {"k_noise": 4.0}


@pytest.mark.asyncio
async def test_enqueue_unknown_encoder_raises(session) -> None:
    with pytest.raises(ValueError):
        await enqueue_piano_roll_notes_job(
            session, event_encoder_job_id="does-not-exist"
        )


@pytest.mark.asyncio
async def test_latest_for_encoder_job_prefers_complete(session) -> None:
    encoder_id = await _make_encoder_job(session)
    older, _ = await enqueue_piano_roll_notes_job(
        session,
        event_encoder_job_id=encoder_id,
        extractor_version="v1",
    )
    older.status = JobStatus.complete.value
    older.finished_at = datetime(2026, 5, 1, tzinfo=timezone.utc)
    await session.commit()

    newer, _ = await enqueue_piano_roll_notes_job(
        session,
        event_encoder_job_id=encoder_id,
        extractor_version="v2-experimental",
    )
    # newer is still queued
    await session.commit()

    latest = await latest_for_encoder_job(session, event_encoder_job_id=encoder_id)
    assert latest is not None
    assert latest.id == older.id  # complete row wins over a newer queued one


@pytest.mark.asyncio
async def test_latest_for_encoder_job_falls_back_to_any(session) -> None:
    encoder_id = await _make_encoder_job(session)
    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder_id
    )
    latest = await latest_for_encoder_job(session, event_encoder_job_id=encoder_id)
    assert latest is not None
    assert latest.id == job.id


@pytest.mark.asyncio
async def test_latest_for_encoder_job_returns_none_when_absent(session) -> None:
    encoder_id = await _make_encoder_job(session)
    latest = await latest_for_encoder_job(session, event_encoder_job_id=encoder_id)
    assert latest is None


@pytest.mark.asyncio
async def test_auto_enqueue_after_encoder_complete_creates_row(session) -> None:
    encoder_id = await _make_encoder_job(session)
    job = await auto_enqueue_after_encoder_complete(
        session, event_encoder_job_id=encoder_id
    )
    assert job is not None
    assert job.event_encoder_job_id == encoder_id
    assert job.status == JobStatus.queued.value


@pytest.mark.asyncio
async def test_auto_enqueue_swallows_conflict(session) -> None:
    encoder_id = await _make_encoder_job(session)
    initial, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder_id
    )
    initial.status = JobStatus.running.value
    await session.commit()

    # Auto-enqueue should not raise even though enqueue would conflict.
    result = await auto_enqueue_after_encoder_complete(
        session, event_encoder_job_id=encoder_id
    )
    assert result is None


# ---------- Worker stub tests ----------


@pytest.mark.asyncio
async def test_worker_completes_and_writes_empty_parquet(
    session, settings, tmp_path
) -> None:
    encoder_id = await _make_encoder_job(session)
    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder_id
    )
    job.status = JobStatus.running.value
    await session.commit()

    await run_piano_roll_notes_job(session, job, settings)

    refreshed = await session.get(PianoRollNotesJob, job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.complete.value
    assert refreshed.error_message is None
    assert refreshed.notes_path is not None
    assert refreshed.n_events == 0
    assert refreshed.n_notes == 0
    assert refreshed.compute_seconds is not None

    parquet_path = Path(refreshed.notes_path)
    assert parquet_path.exists()
    table = pq.read_table(parquet_path)
    assert table.num_rows == 0
    assert table.schema.equals(NOTES_SCHEMA)


@pytest.mark.asyncio
async def test_worker_fails_when_encoder_not_complete(session, settings) -> None:
    encoder_id = await _make_encoder_job(session, status=JobStatus.queued.value)
    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder_id
    )
    job.status = JobStatus.running.value
    await session.commit()

    await run_piano_roll_notes_job(session, job, settings)

    refreshed = await session.get(PianoRollNotesJob, job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.failed.value
    assert refreshed.error_message is not None
    assert "requires a completed event_encoder_job" in refreshed.error_message


# ---------- Schema round-trip ----------


def test_schema_round_trip_from_orm() -> None:
    row = PianoRollNotesJob(
        id="abc",
        event_encoder_job_id="eej-1",
        extractor_version="v1",
        status=JobStatus.complete.value,
        started_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 5, 20, 12, 1, tzinfo=timezone.utc),
        error_message=None,
        notes_path="/storage/event_encoders/eej-1/event_notes_v1.parquet",
        n_events=42,
        n_notes=137,
        compute_seconds=1.23,
        params_json='{"k_noise": 3.0}',
        created_at=datetime(2026, 5, 20, 11, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 20, 12, 1, tzinfo=timezone.utc),
    )
    schema = PianoRollNotesJobRead.model_validate(row)
    payload = schema.model_dump_json()
    parsed = json.loads(payload)
    assert parsed["status"] == "complete"
    assert parsed["n_events"] == 42
    assert parsed["n_notes"] == 137
    assert parsed["compute_seconds"] == 1.23


def test_absent_status_schema_default() -> None:
    payload = PianoRollNotesStatusAbsent().model_dump()
    assert payload == {"status": "absent"}
