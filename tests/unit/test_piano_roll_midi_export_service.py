"""Tests for the Piano Roll MIDI export service."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from humpback.models.piano_roll_notes import (
    DEFAULT_EXTRACTOR_VERSION,
    PianoRollNotesJob,
)
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import EventEncoderJob
from humpback.services.piano_roll_midi_export_service import (
    PianoRollMidiExportConflict,
    complete_for_encoder_job_version,
    enqueue_piano_roll_midi_export,
    latest_for_encoder_job,
)


_WIN_START: float = 1_000.0
_WIN_END: float = 1_060.0


async def _make_encoder_job(session) -> str:
    job = EventEncoderJob(
        status=JobStatus.complete.value,
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


async def _make_complete_notes_job(
    session,
    *,
    event_encoder_job_id: str,
    extractor_version: str = DEFAULT_EXTRACTOR_VERSION,
    finished_at: datetime | None = None,
) -> PianoRollNotesJob:
    notes = PianoRollNotesJob(
        event_encoder_job_id=event_encoder_job_id,
        extractor_version=extractor_version,
        status=JobStatus.complete.value,
        finished_at=finished_at or datetime.now(timezone.utc),
        notes_path=f"event_encoders/{event_encoder_job_id}/event_notes_{extractor_version}.parquet",
        n_notes=10,
    )
    session.add(notes)
    await session.commit()
    await session.refresh(notes)
    return notes


# ---------- enqueue behavior matrix ----------


@pytest.mark.asyncio
async def test_enqueue_creates_new_row(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(session, event_encoder_job_id=encoder_id)

    row, created = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )

    assert created is True
    assert row.event_encoder_job_id == encoder_id
    assert row.extractor_version == DEFAULT_EXTRACTOR_VERSION
    assert row.status == JobStatus.queued.value
    assert row.params_json == "{}"
    assert row.window_start_utc == _WIN_START
    assert row.window_end_utc == _WIN_END


@pytest.mark.asyncio
async def test_enqueue_complete_with_matching_window_returns_existing(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(session, event_encoder_job_id=encoder_id)
    first, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    first.status = JobStatus.complete.value
    first.midi_path = "exports/event_encoders/x/notes_v2.mid"
    first.audio_path = "exports/event_encoders/x/audio_v2.flac"
    first.audio_size_bytes = 4096
    first.audio_sample_rate = 32_000
    first.audio_duration_s = 60.0
    first.finished_at = datetime.now(timezone.utc)
    await session.commit()

    second, created = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    assert created is False
    assert second.id == first.id
    assert second.status == JobStatus.complete.value
    assert second.midi_path == "exports/event_encoders/x/notes_v2.mid"
    assert second.audio_path == "exports/event_encoders/x/audio_v2.flac"


@pytest.mark.asyncio
async def test_enqueue_complete_with_different_window_resets(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(session, event_encoder_job_id=encoder_id)
    first, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    first.status = JobStatus.complete.value
    first.midi_path = "exports/event_encoders/x/notes_v2.mid"
    first.audio_path = "exports/event_encoders/x/audio_v2.flac"
    first.audio_size_bytes = 4096
    first.audio_sample_rate = 32_000
    first.audio_duration_s = 60.0
    first.finished_at = datetime.now(timezone.utc)
    first.n_notes = 5
    first.n_bytes = 2048
    await session.commit()

    reset, created = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=2_000.0,
        window_end_utc=2_120.0,
    )
    assert created is False
    assert reset.id == first.id
    assert reset.status == JobStatus.queued.value
    assert reset.midi_path is None
    assert reset.audio_path == ""
    assert reset.audio_size_bytes == 0
    assert reset.audio_sample_rate == 0
    assert reset.audio_duration_s == 0.0
    assert reset.window_start_utc == 2_000.0
    assert reset.window_end_utc == 2_120.0
    assert reset.n_notes is None
    assert reset.n_bytes is None


@pytest.mark.asyncio
async def test_enqueue_complete_with_force_resets(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(session, event_encoder_job_id=encoder_id)
    row, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    row.status = JobStatus.complete.value
    row.midi_path = "exports/event_encoders/x/notes_v2.mid"
    row.finished_at = datetime.now(timezone.utc)
    row.n_notes = 42
    row.n_bytes = 1024
    row.compute_seconds = 1.5
    row.audio_path = "exports/event_encoders/x/audio_v2.flac"
    row.audio_size_bytes = 9999
    row.audio_sample_rate = 32_000
    row.audio_duration_s = 60.0
    await session.commit()

    reset, created = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        force=True,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    assert created is False
    assert reset.id == row.id
    assert reset.status == JobStatus.queued.value
    assert reset.midi_path is None
    assert reset.finished_at is None
    assert reset.n_notes is None
    assert reset.n_bytes is None
    assert reset.compute_seconds is None
    assert reset.audio_path == ""
    assert reset.audio_size_bytes == 0
    assert reset.audio_sample_rate == 0
    assert reset.audio_duration_s == 0.0


@pytest.mark.asyncio
async def test_enqueue_running_raises_conflict(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(session, event_encoder_job_id=encoder_id)
    row, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    row.status = JobStatus.running.value
    await session.commit()

    with pytest.raises(PianoRollMidiExportConflict):
        await enqueue_piano_roll_midi_export(
            session,
            event_encoder_job_id=encoder_id,
            window_start_utc=_WIN_START,
            window_end_utc=_WIN_END,
        )


@pytest.mark.asyncio
async def test_enqueue_queued_raises_conflict(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(session, event_encoder_job_id=encoder_id)
    await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )

    with pytest.raises(PianoRollMidiExportConflict):
        await enqueue_piano_roll_midi_export(
            session,
            event_encoder_job_id=encoder_id,
            window_start_utc=_WIN_START,
            window_end_utc=_WIN_END,
        )


@pytest.mark.asyncio
async def test_enqueue_failed_resets_row(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(session, event_encoder_job_id=encoder_id)
    row, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    row.status = JobStatus.failed.value
    row.error_message = "boom"
    row.finished_at = datetime.now(timezone.utc)
    row.midi_path = "stale.mid"
    row.n_notes = 1
    row.n_bytes = 2
    row.compute_seconds = 3.0
    await session.commit()

    reset, created = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        params={"foo": 1},
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    assert created is False
    assert reset.id == row.id
    assert reset.status == JobStatus.queued.value
    assert reset.error_message is None
    assert reset.midi_path is None
    assert reset.n_notes is None
    assert reset.n_bytes is None
    assert reset.compute_seconds is None
    assert json.loads(reset.params_json) == {"foo": 1}


@pytest.mark.asyncio
async def test_enqueue_canceled_resets_row(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(session, event_encoder_job_id=encoder_id)
    row, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    row.status = JobStatus.canceled.value
    await session.commit()

    reset, created = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    assert created is False
    assert reset.id == row.id
    assert reset.status == JobStatus.queued.value


@pytest.mark.asyncio
async def test_enqueue_rejects_non_positive_window(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(session, event_encoder_job_id=encoder_id)
    with pytest.raises(ValueError):
        await enqueue_piano_roll_midi_export(
            session,
            event_encoder_job_id=encoder_id,
            window_start_utc=2_000.0,
            window_end_utc=2_000.0,
        )


# ---------- version resolution ----------


@pytest.mark.asyncio
async def test_enqueue_resolves_version_when_none(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(
        session,
        event_encoder_job_id=encoder_id,
        extractor_version="v2-experimental",
        finished_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )
    await _make_complete_notes_job(
        session,
        event_encoder_job_id=encoder_id,
        extractor_version="v2",
        finished_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    row, created = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    assert created is True
    assert row.extractor_version == "v2-experimental"


@pytest.mark.asyncio
async def test_enqueue_no_complete_notes_raises(session) -> None:
    encoder_id = await _make_encoder_job(session)
    notes = PianoRollNotesJob(
        event_encoder_job_id=encoder_id,
        status=JobStatus.queued.value,
    )
    session.add(notes)
    await session.commit()

    with pytest.raises(ValueError):
        await enqueue_piano_roll_midi_export(
            session,
            event_encoder_job_id=encoder_id,
            window_start_utc=_WIN_START,
            window_end_utc=_WIN_END,
        )


@pytest.mark.asyncio
async def test_enqueue_version_not_complete_raises(session) -> None:
    encoder_id = await _make_encoder_job(session)
    notes = PianoRollNotesJob(
        event_encoder_job_id=encoder_id,
        extractor_version="v2",
        status=JobStatus.queued.value,
    )
    session.add(notes)
    await session.commit()

    with pytest.raises(ValueError):
        await enqueue_piano_roll_midi_export(
            session,
            event_encoder_job_id=encoder_id,
            extractor_version="v2",
            window_start_utc=_WIN_START,
            window_end_utc=_WIN_END,
        )


@pytest.mark.asyncio
async def test_enqueue_unknown_encoder_raises(session) -> None:
    with pytest.raises(ValueError):
        await enqueue_piano_roll_midi_export(
            session,
            event_encoder_job_id="does-not-exist",
            window_start_utc=_WIN_START,
            window_end_utc=_WIN_END,
        )


# ---------- query helpers ----------


@pytest.mark.asyncio
async def test_latest_for_encoder_job_prefers_complete(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(
        session, event_encoder_job_id=encoder_id, extractor_version="v1"
    )
    await _make_complete_notes_job(
        session, event_encoder_job_id=encoder_id, extractor_version="v2"
    )
    older, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        extractor_version="v1",
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    older.status = JobStatus.complete.value
    older.finished_at = datetime(2026, 5, 1, tzinfo=timezone.utc)
    newer, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        extractor_version="v2",
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    await session.commit()

    latest = await latest_for_encoder_job(session, event_encoder_job_id=encoder_id)
    assert latest is not None
    assert latest.id == older.id


@pytest.mark.asyncio
async def test_latest_for_encoder_job_falls_back_to_any(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(session, event_encoder_job_id=encoder_id)
    row, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    latest = await latest_for_encoder_job(session, event_encoder_job_id=encoder_id)
    assert latest is not None
    assert latest.id == row.id


@pytest.mark.asyncio
async def test_latest_for_encoder_job_returns_none_when_absent(session) -> None:
    encoder_id = await _make_encoder_job(session)
    latest = await latest_for_encoder_job(session, event_encoder_job_id=encoder_id)
    assert latest is None


@pytest.mark.asyncio
async def test_complete_for_encoder_job_version(session) -> None:
    encoder_id = await _make_encoder_job(session)
    await _make_complete_notes_job(
        session, event_encoder_job_id=encoder_id, extractor_version="v2"
    )
    row, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        extractor_version="v2",
        window_start_utc=_WIN_START,
        window_end_utc=_WIN_END,
    )
    row.status = JobStatus.complete.value
    row.finished_at = datetime.now(timezone.utc)
    await session.commit()

    pinned = await complete_for_encoder_job_version(
        session, event_encoder_job_id=encoder_id, extractor_version="v2"
    )
    assert pinned is not None
    assert pinned.id == row.id

    missing = await complete_for_encoder_job_version(
        session, event_encoder_job_id=encoder_id, extractor_version="v99"
    )
    assert missing is None
