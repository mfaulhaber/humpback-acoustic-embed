"""Tests for the Piano Roll MIDI export worker."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import mido
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
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
    event_encoder_audio_export_path,
    event_encoder_midi_export_path,
    event_encoder_notes_path,
)
from humpback.workers import piano_roll_midi_export_worker
from humpback.workers.piano_roll_midi_export_worker import (
    EXPORT_SAMPLE_RATE,
    run_piano_roll_midi_export,
)
from humpback.workers.piano_roll_notes_worker import NOTES_SCHEMA


_JOB_START = 1_750_000_000.0
_JOB_END = _JOB_START + 600.0
_WINDOW_START = _JOB_START + 100.0
_WINDOW_END = _WINDOW_START + 60.0


def _stub_resolve_window_audio(
    *, duration_sec: float | None = None, value: float = 0.5
):
    """Return a monkeypatch target that produces a constant-amplitude sine clip."""

    def _stub(
        *,
        region_job,
        settings,
        window_start_utc,
        window_end_utc,
    ):
        actual_duration = (
            duration_sec
            if duration_sec is not None
            else (window_end_utc - window_start_utc)
        )
        n_samples = int(round(actual_duration * EXPORT_SAMPLE_RATE))
        return np.full(n_samples, value, dtype=np.float32)

    return _stub


async def _make_encoder_and_notes(
    session,
    *,
    extractor_version: str = DEFAULT_EXTRACTOR_VERSION,
    hydrophone_id: str = "test-hydrophone",
    job_start_timestamp: float = _JOB_START,
    job_end_timestamp: float = _JOB_END,
) -> str:
    region = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id=hydrophone_id,
        start_timestamp=job_start_timestamp,
        end_timestamp=job_end_timestamp,
    )
    session.add(region)
    await session.commit()
    await session.refresh(region)

    seg = EventSegmentationJob(
        status=JobStatus.complete.value,
        region_detection_job_id=region.id,
    )
    session.add(seg)
    await session.commit()
    await session.refresh(seg)

    encoder = EventEncoderJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id=seg.id,
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
async def test_worker_writes_midi_and_flac_and_marks_complete(
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
                "start_utc": _WINDOW_START + 1.0,
                "duration_s": 0.5,
                "velocity": 80,
            },
            {
                "event_id": "e0",
                "partial_index": 1,
                "midi_pitch": 72,
                "start_utc": _WINDOW_START + 1.0,
                "duration_s": 0.5,
                "velocity": 40,
            },
        ],
    )
    monkeypatch.setattr(
        piano_roll_midi_export_worker,
        "_resolve_window_audio",
        _stub_resolve_window_audio(value=0.25),
    )

    job, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
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
    assert job.audio_sample_rate == EXPORT_SAMPLE_RATE
    expected_duration = _WINDOW_END - _WINDOW_START
    assert abs(job.audio_duration_s - expected_duration) < 1e-3

    midi_path = event_encoder_midi_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    audio_path = event_encoder_audio_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    assert midi_path.exists()
    assert audio_path.exists()
    assert midi_path.stat().st_size == job.n_bytes
    assert audio_path.stat().st_size == job.audio_size_bytes

    parsed = mido.MidiFile(file=io.BytesIO(midi_path.read_bytes()))
    assert parsed.type == 1
    note_ons = [
        m
        for track in parsed.tracks[1:]
        for m in track
        if m.type == "note_on" and m.velocity > 0
    ]
    assert {m.note for m in note_ons} == {60, 72}

    samples, sr = sf.read(str(audio_path), dtype="float32")
    assert sr == EXPORT_SAMPLE_RATE
    assert samples.ndim == 1
    assert abs(len(samples) - int(round(expected_duration * EXPORT_SAMPLE_RATE))) <= 1
    # Unormalized: 16-bit PCM quantization of 0.25 round-trips within tolerance.
    assert np.max(np.abs(samples - 0.25)) < 1e-3


@pytest.mark.asyncio
async def test_worker_filters_and_clips_notes_to_window(
    session, settings, monkeypatch
) -> None:
    encoder_id = await _make_encoder_and_notes(session)
    notes_path = event_encoder_notes_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    _write_notes_parquet(
        notes_path,
        [
            # Inside the window, near the window start.
            {
                "event_id": "in",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": _WINDOW_START + 1.0,
                "duration_s": 0.5,
                "velocity": 80,
            },
            # Starts before the window; clipped at window start.
            {
                "event_id": "edge-left",
                "partial_index": 0,
                "midi_pitch": 62,
                "start_utc": _WINDOW_START - 0.4,
                "duration_s": 0.8,
                "velocity": 80,
            },
            # Ends after the window; clipped at window end.
            {
                "event_id": "edge-right",
                "partial_index": 0,
                "midi_pitch": 64,
                "start_utc": _WINDOW_END - 0.1,
                "duration_s": 1.0,
                "velocity": 80,
            },
            # Entirely outside the window — dropped.
            {
                "event_id": "outside",
                "partial_index": 0,
                "midi_pitch": 66,
                "start_utc": _WINDOW_END + 5.0,
                "duration_s": 1.0,
                "velocity": 80,
            },
        ],
    )
    monkeypatch.setattr(
        piano_roll_midi_export_worker,
        "_resolve_window_audio",
        _stub_resolve_window_audio(),
    )

    job, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
    )

    await run_piano_roll_midi_export(session, job, settings)

    await session.refresh(job)
    assert job.status == JobStatus.complete.value
    assert job.n_notes == 3

    midi_path = event_encoder_midi_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    parsed = mido.MidiFile(file=io.BytesIO(midi_path.read_bytes()))
    pitches = {
        m.note
        for track in parsed.tracks[1:]
        for m in track
        if m.type == "note_on" and m.velocity > 0
    }
    assert pitches == {60, 62, 64}


@pytest.mark.asyncio
async def test_worker_uses_window_start_as_midi_time_origin(
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
                "event_id": "anchor",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": _WINDOW_START,
                "duration_s": 0.5,
                "velocity": 80,
            },
            {
                "event_id": "one-second",
                "partial_index": 0,
                "midi_pitch": 62,
                "start_utc": _WINDOW_START + 1.0,
                "duration_s": 0.5,
                "velocity": 80,
            },
        ],
    )
    monkeypatch.setattr(
        piano_roll_midi_export_worker,
        "_resolve_window_audio",
        _stub_resolve_window_audio(),
    )

    job, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
    )
    await run_piano_roll_midi_export(session, job, settings)
    await session.refresh(job)
    assert job.status == JobStatus.complete.value

    midi_path = event_encoder_midi_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    parsed = mido.MidiFile(file=io.BytesIO(midi_path.read_bytes()))

    abs_ticks_by_pitch: dict[int, int] = {}
    for track in parsed.tracks[1:]:
        absolute = 0
        for msg in track:
            absolute += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                abs_ticks_by_pitch.setdefault(msg.note, absolute)

    assert abs_ticks_by_pitch[60] == 0
    # 1.0 s at 120 BPM = 2 quarter notes = 2 * 480 ticks.
    assert abs_ticks_by_pitch[62] == 960


@pytest.mark.asyncio
async def test_worker_succeeds_with_empty_window(
    session, settings, monkeypatch
) -> None:
    encoder_id = await _make_encoder_and_notes(session)
    notes_path = event_encoder_notes_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    _write_notes_parquet(
        notes_path,
        [
            # Far outside the window.
            {
                "event_id": "far",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": _WINDOW_END + 100.0,
                "duration_s": 1.0,
                "velocity": 80,
            },
        ],
    )
    monkeypatch.setattr(
        piano_roll_midi_export_worker,
        "_resolve_window_audio",
        _stub_resolve_window_audio(value=0.0),
    )

    job, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
    )
    await run_piano_roll_midi_export(session, job, settings)
    await session.refresh(job)
    assert job.status == JobStatus.complete.value
    assert job.n_notes == 0

    midi_path = event_encoder_midi_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    audio_path = event_encoder_audio_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    assert midi_path.exists()
    assert audio_path.exists()
    parsed = mido.MidiFile(file=io.BytesIO(midi_path.read_bytes()))
    note_ons = [
        m
        for track in parsed.tracks[1:]
        for m in track
        if m.type == "note_on" and m.velocity > 0
    ]
    assert note_ons == []


@pytest.mark.asyncio
async def test_worker_fails_when_parquet_missing(
    session, settings, monkeypatch
) -> None:
    encoder_id = await _make_encoder_and_notes(session)
    monkeypatch.setattr(
        piano_roll_midi_export_worker,
        "_resolve_window_audio",
        _stub_resolve_window_audio(),
    )
    job, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
    )

    await run_piano_roll_midi_export(session, job, settings)

    await session.refresh(job)
    assert job.status == JobStatus.failed.value
    assert job.error_message is not None
    assert "notes parquet not found" in job.error_message

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
                "start_utc": _WINDOW_START + 0.1,
                "duration_s": 0.5,
                "velocity": 80,
            }
        ],
    )

    def boom(_table, **_kwargs):
        raise RuntimeError("synth blew up")

    monkeypatch.setattr(
        piano_roll_midi_export_worker, "notes_table_to_midi_bytes", boom
    )
    monkeypatch.setattr(
        piano_roll_midi_export_worker,
        "_resolve_window_audio",
        _stub_resolve_window_audio(),
    )

    job, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
    )

    await run_piano_roll_midi_export(session, job, settings)

    await session.refresh(job)
    assert job.status == JobStatus.failed.value
    assert "synth blew up" in (job.error_message or "")

    midi_path = event_encoder_midi_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    audio_path = event_encoder_audio_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    assert not midi_path.exists()
    assert not midi_path.with_suffix(midi_path.suffix + ".tmp").exists()
    assert not audio_path.exists()


@pytest.mark.asyncio
async def test_prior_successful_export_survives_failed_re_export(
    session, settings, monkeypatch
) -> None:
    """A failed re-export must NOT delete the previous successful pair on disk.

    Regression guard: a buggy version of the worker renamed the MIDI to its
    final path BEFORE writing the FLAC, so a FLAC write failure left the
    prior successful MIDI permanently overwritten. With the staged
    "write both temps first, then rename both" sequence the previous
    artifacts must remain on disk and downloadable.
    """
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
                "start_utc": _WINDOW_START + 0.1,
                "duration_s": 0.5,
                "velocity": 80,
            }
        ],
    )
    monkeypatch.setattr(
        piano_roll_midi_export_worker,
        "_resolve_window_audio",
        _stub_resolve_window_audio(value=0.4),
    )

    # First export — runs to completion and produces both artifacts.
    job, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
    )
    await run_piano_roll_midi_export(session, job, settings)
    await session.refresh(job)
    assert job.status == JobStatus.complete.value

    midi_path = event_encoder_midi_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    audio_path = event_encoder_audio_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    prior_midi_bytes = midi_path.read_bytes()
    prior_audio_bytes = audio_path.read_bytes()
    assert len(prior_midi_bytes) > 0
    assert len(prior_audio_bytes) > 0

    # Second export — same row, force a re-run, but mock the FLAC writer
    # to blow up. The worker MUST leave the prior MIDI + FLAC on disk.
    def explode(*_args, **_kwargs):
        raise RuntimeError("flac write failed on re-export")

    monkeypatch.setattr(piano_roll_midi_export_worker, "write_flac_samples", explode)

    job2, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
        force=True,
    )
    await run_piano_roll_midi_export(session, job2, settings)
    await session.refresh(job2)

    assert job2.status == JobStatus.failed.value
    assert "flac write failed on re-export" in (job2.error_message or "")

    # The previous successful pair must still be readable byte-for-byte.
    assert midi_path.exists()
    assert audio_path.exists()
    assert midi_path.read_bytes() == prior_midi_bytes
    assert audio_path.read_bytes() == prior_audio_bytes
    # No leftover tmp files.
    assert not midi_path.with_suffix(midi_path.suffix + ".tmp").exists()
    assert not audio_path.with_suffix(audio_path.suffix + ".tmp").exists()


@pytest.mark.asyncio
async def test_worker_rolls_back_midi_when_flac_write_fails(
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
                "start_utc": _WINDOW_START + 0.1,
                "duration_s": 0.5,
                "velocity": 80,
            }
        ],
    )

    monkeypatch.setattr(
        piano_roll_midi_export_worker,
        "_resolve_window_audio",
        _stub_resolve_window_audio(),
    )

    def explode(*_args, **_kwargs):
        raise RuntimeError("flac write failed")

    monkeypatch.setattr(piano_roll_midi_export_worker, "write_flac_samples", explode)

    job, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
    )
    await run_piano_roll_midi_export(session, job, settings)
    await session.refresh(job)

    assert job.status == JobStatus.failed.value
    assert "flac write failed" in (job.error_message or "")

    midi_path = event_encoder_midi_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    audio_path = event_encoder_audio_export_path(
        settings.storage_root, encoder_id, DEFAULT_EXTRACTOR_VERSION
    )
    # No prior export → no final artifacts on disk, only temp leftovers cleaned.
    assert not midi_path.exists()
    assert not audio_path.exists()
    assert not midi_path.with_suffix(midi_path.suffix + ".tmp").exists()
    assert not audio_path.with_suffix(audio_path.suffix + ".tmp").exists()


@pytest.mark.asyncio
async def test_queue_claim_is_atomic(session, settings) -> None:
    from humpback.workers.queue import claim_piano_roll_midi_export

    encoder_id = await _make_encoder_and_notes(session)
    await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
    )

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
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
    )
    job.status = JobStatus.running.value
    job.updated_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    await session.commit()

    recovered = await recover_stale_jobs(session)
    assert recovered >= 1

    await session.refresh(job)
    assert job.status == JobStatus.queued.value


# --------------------------------------------------------------------------- #
# v3 MPE export branch
# --------------------------------------------------------------------------- #


from humpback.storage import event_encoder_note_contours_path  # noqa: E402
from humpback.workers.piano_roll_notes_worker import (  # noqa: E402
    NOTE_CONTOURS_V3_SCHEMA,
    NOTES_V3_SCHEMA,
)


def _write_v3_parquets(
    notes_path: Path, contours_path: Path, *, notes: list[dict], contours: list[dict]
) -> None:
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    contours_path.parent.mkdir(parents=True, exist_ok=True)
    note_defaults = {
        "event_token": 0,
        "start_offset_s": 0.0,
        "peak_magnitude": 0.0,
        "track_id": 0,
        "f0_track_id": 0,
        "contour_frame_count": 0,
    }
    notes_table = pa.Table.from_pylist(
        [
            {
                field.name: row.get(field.name, note_defaults.get(field.name))
                for field in NOTES_V3_SCHEMA
            }
            for row in notes
        ],
        schema=NOTES_V3_SCHEMA,
    )
    pq.write_table(notes_table, notes_path)

    contour_defaults = {
        "time_offset_s": 0.0,
        "harmonic_strength": 0.0,
        "subharmonic_octave": 0,
    }
    contour_table = pa.Table.from_pylist(
        [
            {
                field.name: row.get(field.name, contour_defaults.get(field.name))
                for field in NOTE_CONTOURS_V3_SCHEMA
            }
            for row in contours
        ],
        schema=NOTE_CONTOURS_V3_SCHEMA,
    )
    pq.write_table(contour_table, contours_path)


@pytest.mark.asyncio
async def test_v3_export_reads_contour_sidecar(session, settings, monkeypatch) -> None:
    """v3 export must consume the contour sidecar and emit an MPE SMF."""
    encoder_id = await _make_encoder_and_notes(session, extractor_version="v3")
    notes_path = event_encoder_notes_path(settings.storage_root, encoder_id, "v3")
    contours_path = event_encoder_note_contours_path(
        settings.storage_root, encoder_id, "v3"
    )
    _write_v3_parquets(
        notes_path,
        contours_path,
        notes=[
            {
                "event_id": "ev-1",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": _WINDOW_START + 1.0,
                "duration_s": 0.5,
                "velocity": 80,
                "note_uid": "uid-1",
                "track_id": 0,
                "f0_track_id": 0,
                "contour_frame_count": 2,
            },
        ],
        contours=[
            {"note_uid": "uid-1", "frame_index": 0, "cents_from_pitch": 0.0},
            {"note_uid": "uid-1", "frame_index": 1, "cents_from_pitch": 0.0},
        ],
    )
    monkeypatch.setattr(
        piano_roll_midi_export_worker,
        "_resolve_window_audio",
        _stub_resolve_window_audio(value=0.25),
    )

    job, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
        extractor_version="v3",
    )
    await run_piano_roll_midi_export(session, job, settings)

    await session.refresh(job)
    assert job.status == JobStatus.complete.value, job.error_message
    midi_path_abs = event_encoder_midi_export_path(
        settings.storage_root, encoder_id, "v3"
    )
    parsed = mido.MidiFile(file=io.BytesIO(midi_path_abs.read_bytes()))
    # MPE shape: tempo + master + 15 members.
    assert len(parsed.tracks) == 1 + 1 + 15


@pytest.mark.asyncio
async def test_v3_export_fails_without_contour_sidecar(
    session, settings, monkeypatch
) -> None:
    """v3 export must fail with a clear error when the sidecar is missing."""
    encoder_id = await _make_encoder_and_notes(session, extractor_version="v3")
    notes_path = event_encoder_notes_path(settings.storage_root, encoder_id, "v3")
    # Write notes only — no contour parquet.
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_table = pa.Table.from_pylist(
        [
            {field.name: 0 for field in NOTES_V3_SCHEMA}
            | {
                "event_id": "ev-1",
                "note_uid": "uid-1",
                "midi_pitch": 60,
                "start_utc": _WINDOW_START + 1.0,
                "duration_s": 0.5,
                "velocity": 80,
            }
        ],
        schema=NOTES_V3_SCHEMA,
    )
    pq.write_table(notes_table, notes_path)
    monkeypatch.setattr(
        piano_roll_midi_export_worker,
        "_resolve_window_audio",
        _stub_resolve_window_audio(),
    )

    job, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
        extractor_version="v3",
    )
    await run_piano_roll_midi_export(session, job, settings)

    await session.refresh(job)
    assert job.status == JobStatus.failed.value
    assert job.error_message is not None
    assert "contour" in job.error_message.lower()


@pytest.mark.asyncio
async def test_v3_export_clips_notes_to_window(session, settings, monkeypatch) -> None:
    """A v3 note whose end exceeds the window must be truncated."""
    encoder_id = await _make_encoder_and_notes(session, extractor_version="v3")
    notes_path = event_encoder_notes_path(settings.storage_root, encoder_id, "v3")
    contours_path = event_encoder_note_contours_path(
        settings.storage_root, encoder_id, "v3"
    )
    _write_v3_parquets(
        notes_path,
        contours_path,
        notes=[
            # Note runs past window_end_utc; should be clipped.
            {
                "event_id": "ev-1",
                "partial_index": 0,
                "midi_pitch": 60,
                "start_utc": _WINDOW_END - 0.2,
                "duration_s": 1.0,
                "velocity": 80,
                "note_uid": "uid-1",
                "track_id": 0,
                "f0_track_id": 0,
                "contour_frame_count": 3,
            },
            # Note fully outside; dropped.
            {
                "event_id": "ev-2",
                "partial_index": 0,
                "midi_pitch": 62,
                "start_utc": _WINDOW_END + 5.0,
                "duration_s": 1.0,
                "velocity": 80,
                "note_uid": "uid-2",
                "track_id": 1,
                "f0_track_id": 1,
                "contour_frame_count": 2,
            },
        ],
        contours=[
            {"note_uid": "uid-1", "frame_index": 0, "cents_from_pitch": 0.0},
            {"note_uid": "uid-1", "frame_index": 1, "cents_from_pitch": 0.0},
            {"note_uid": "uid-1", "frame_index": 2, "cents_from_pitch": 0.0},
            {"note_uid": "uid-2", "frame_index": 0, "cents_from_pitch": 0.0},
        ],
    )
    monkeypatch.setattr(
        piano_roll_midi_export_worker,
        "_resolve_window_audio",
        _stub_resolve_window_audio(),
    )

    job, _ = await enqueue_piano_roll_midi_export(
        session,
        event_encoder_job_id=encoder_id,
        window_start_utc=_WINDOW_START,
        window_end_utc=_WINDOW_END,
        extractor_version="v3",
    )
    await run_piano_roll_midi_export(session, job, settings)

    await session.refresh(job)
    assert job.status == JobStatus.complete.value, job.error_message
    assert job.n_notes == 1
