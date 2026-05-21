"""Tests for the Piano Roll MIDI export worker."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import mido
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from humpback.models.piano_roll_notes import (
    DEFAULT_EXTRACTOR_VERSION,
    PianoRollNotesJob,
)
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import EventEncoderJob
from humpback.services.piano_roll_midi_export_service import (
    enqueue_piano_roll_midi_export,
)
from humpback.storage import (
    event_encoder_midi_export_path,
    event_encoder_notes_path,
)
from humpback.workers import piano_roll_midi_export_worker
from humpback.workers.piano_roll_midi_export_worker import (
    run_piano_roll_midi_export,
)
from humpback.workers.piano_roll_notes_worker import NOTES_SCHEMA


async def _make_encoder_and_notes(
    session,
    *,
    extractor_version: str = DEFAULT_EXTRACTOR_VERSION,
) -> str:
    encoder = EventEncoderJob(
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
    session.add(encoder)
    await session.commit()
    await session.refresh(encoder)

    notes = PianoRollNotesJob(
        event_encoder_job_id=encoder.id,
        extractor_version=extractor_version,
        status=JobStatus.complete.value,
        finished_at=datetime.now(timezone.utc),
        n_notes=2,
    )
    session.add(notes)
    await session.commit()
    return encoder.id


def _write_notes_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = {
        "event_token": 0,
        "start_offset_s": 0.0,
        "peak_magnitude": 0.0,
        "track_id": 0,
    }
    table = pa.Table.from_pylist(
        [
            {
                field.name: row.get(field.name, defaults.get(field.name))
                for field in NOTES_SCHEMA
            }
            for row in rows
        ],
        schema=NOTES_SCHEMA,
    )
    pq.write_table(table, path)


@pytest.mark.asyncio
async def test_worker_writes_midi_and_marks_complete(session, settings) -> None:
    encoder_id = await _make_encoder_and_notes(session)
    notes_path = event_encoder_notes_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    _write_notes_parquet(
        notes_path,
        [
            {
                "event_id": "e0",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": 0.0,
                "duration_s": 0.5,
                "velocity": 80,
            },
            {
                "event_id": "e0",
                "partial_index": 1,
                "midi_pitch": 72,
                "start_utc": 0.0,
                "duration_s": 0.5,
                "velocity": 40,
            },
        ],
    )

    job, _ = await enqueue_piano_roll_midi_export(
        session, event_encoder_job_id=encoder_id
    )

    await run_piano_roll_midi_export(session, job, settings)

    await session.refresh(job)
    assert job.status == JobStatus.complete.value
    assert job.error_message is None
    assert job.midi_path is not None
    assert job.n_notes == 2
    assert job.n_bytes is not None and job.n_bytes > 0
    assert job.compute_seconds is not None
    assert job.finished_at is not None

    midi_path = event_encoder_midi_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    assert midi_path.exists()
    assert midi_path.stat().st_size == job.n_bytes

    # File parses as a valid MIDI file with the expected notes spread
    # across the per-channel tracks (F0 + 2nd harmonic in this fixture).
    parsed = mido.MidiFile(file=io.BytesIO(midi_path.read_bytes()))
    assert parsed.type == 1
    note_ons = [
        m
        for track in parsed.tracks[1:]
        for m in track
        if m.type == "note_on" and m.velocity > 0
    ]
    assert {m.note for m in note_ons} == {60, 72}


@pytest.mark.asyncio
async def test_worker_fails_when_parquet_missing(session, settings) -> None:
    encoder_id = await _make_encoder_and_notes(session)
    job, _ = await enqueue_piano_roll_midi_export(
        session, event_encoder_job_id=encoder_id
    )

    await run_piano_roll_midi_export(session, job, settings)

    await session.refresh(job)
    assert job.status == JobStatus.failed.value
    assert job.error_message is not None
    assert "notes parquet not found" in job.error_message
    assert job.midi_path is None

    midi_path = event_encoder_midi_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    assert not midi_path.exists()


@pytest.mark.asyncio
async def test_worker_cleans_up_partial_on_synthesis_exception(
    session, settings, monkeypatch
) -> None:
    encoder_id = await _make_encoder_and_notes(session)
    notes_path = event_encoder_notes_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    _write_notes_parquet(
        notes_path,
        [
            {
                "event_id": "e0",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": 0.0,
                "duration_s": 0.5,
                "velocity": 80,
            }
        ],
    )

    def boom(_table):
        raise RuntimeError("synth blew up")

    monkeypatch.setattr(
        piano_roll_midi_export_worker, "notes_table_to_midi_bytes", boom
    )

    job, _ = await enqueue_piano_roll_midi_export(
        session, event_encoder_job_id=encoder_id
    )

    await run_piano_roll_midi_export(session, job, settings)

    await session.refresh(job)
    assert job.status == JobStatus.failed.value
    assert "synth blew up" in (job.error_message or "")

    midi_path = event_encoder_midi_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    assert not midi_path.exists()
    assert not midi_path.with_suffix(midi_path.suffix + ".tmp").exists()


@pytest.mark.asyncio
async def test_queue_claim_is_atomic(session, settings) -> None:
    from humpback.workers.queue import claim_piano_roll_midi_export

    encoder_id = await _make_encoder_and_notes(session)
    await enqueue_piano_roll_midi_export(session, event_encoder_job_id=encoder_id)

    first = await claim_piano_roll_midi_export(session)
    assert first is not None
    assert first.status == JobStatus.running.value

    second = await claim_piano_roll_midi_export(session)
    assert second is None


@pytest.mark.asyncio
async def test_stale_running_recovered(session, settings) -> None:
    from humpback.workers.queue import recover_stale_jobs

    encoder_id = await _make_encoder_and_notes(session)
    job, _ = await enqueue_piano_roll_midi_export(
        session, event_encoder_job_id=encoder_id
    )
    job.status = JobStatus.running.value
    # Backdate updated_at past the 10-minute cutoff.
    job.updated_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    await session.commit()

    recovered = await recover_stale_jobs(session)
    assert recovered >= 1

    await session.refresh(job)
    assert job.status == JobStatus.queued.value
