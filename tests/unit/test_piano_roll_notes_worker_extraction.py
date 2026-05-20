"""End-to-end test for the Piano Roll Notes worker extraction pipeline.

Builds a complete dependency graph (AudioFile → RegionDetectionJob →
EventSegmentationJob with events.parquet → EventEncoderJob with
event_tokens.parquet), monkey-patches the worker's audio provider to
return synthetic harmonic-stack audio, runs the worker, and asserts the
sidecar parquet contains the expected MIDI notes.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from humpback.call_parsing.storage import write_events
from humpback.call_parsing.types import Event
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.piano_roll_notes import PianoRollNotesJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import EventEncoderJob
from humpback.services.piano_roll_notes_service import (
    enqueue_piano_roll_notes_job,
)
from humpback.storage import event_encoder_dir, event_encoder_tokens_path
from humpback.workers import piano_roll_notes_worker
from humpback.workers.piano_roll_notes_worker import run_piano_roll_notes_job


SAMPLE_RATE = 22050


def _harmonic_stack(
    *,
    fundamental_hz: float,
    duration_s: float,
    n_harmonics: int = 3,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    audio = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        audio += (1.0 / k) * np.sin(2 * np.pi * fundamental_hz * k * t)
    return (0.5 * audio).astype(np.float32)


def _build_two_event_audio() -> tuple[np.ndarray, list[Event]]:
    """Return (full_buffer, [event1, event2]) at 22050 Hz.

    Event 1 (0.10–0.50 s): A3 (220 Hz) fundamental + 2 harmonics → expect
    MIDI 57 and 69 (and possibly 76 for the third partial).
    Event 2 (1.00–1.50 s): E4 (330 Hz) fundamental + 1 harmonic → expect
    MIDI 64 (and 76 for the second partial).
    """
    total_seconds = 2.0
    buffer = np.zeros(int(total_seconds * SAMPLE_RATE), dtype=np.float32)

    def _insert(audio: np.ndarray, start_s: float) -> None:
        start_idx = int(start_s * SAMPLE_RATE)
        end_idx = start_idx + audio.shape[0]
        buffer[start_idx:end_idx] += audio

    _insert(_harmonic_stack(fundamental_hz=220.0, duration_s=0.40), start_s=0.10)
    _insert(
        _harmonic_stack(fundamental_hz=330.0, duration_s=0.40, n_harmonics=2),
        start_s=1.00,
    )

    events = [
        Event(
            event_id="ev-1",
            region_id="rg-1",
            start_sec=0.10,
            end_sec=0.50,
            center_sec=0.30,
            segmentation_confidence=0.9,
        ),
        Event(
            event_id="ev-2",
            region_id="rg-1",
            start_sec=1.00,
            end_sec=1.50,
            center_sec=1.25,
            segmentation_confidence=0.9,
        ),
    ]
    return buffer, events


def _write_event_tokens_parquet(
    path: Path, *, event_ids: Iterable[str], token_id: int, k: int
) -> None:
    schema = pa.schema(
        [
            pa.field("k", pa.int32(), nullable=False),
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("token_id", pa.int32(), nullable=False),
        ]
    )
    rows = [{"k": k, "event_id": eid, "token_id": token_id} for eid in event_ids]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


@pytest.mark.asyncio
async def test_worker_extracts_expected_notes(
    session, settings, tmp_path, monkeypatch
) -> None:
    # ---------- Build the SQL dependency graph ----------
    audio = AudioFile(
        filename="synthetic.wav",
        folder_path=str(tmp_path),
        checksum_sha256="deadbeef",
        duration_seconds=2.0,
        sample_rate_original=SAMPLE_RATE,
    )
    session.add(audio)
    await session.commit()
    await session.refresh(audio)

    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        audio_file_id=audio.id,
        start_timestamp=1000.0,
        end_timestamp=1002.0,
    )
    session.add(region_job)
    await session.commit()
    await session.refresh(region_job)

    seg_job = EventSegmentationJob(
        status=JobStatus.complete.value,
        region_detection_job_id=region_job.id,
    )
    session.add(seg_job)
    await session.commit()
    await session.refresh(seg_job)

    encoder = EventEncoderJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id=seg_job.id,
        event_source_mode="raw",
        continuous_embedding_job_id="cej-stub",
        continuous_embedding_signature="cej-sig",
        tokenizer_version="crnn-event-encoder-v2",
        pooling_config_json="{}",
        descriptor_config_json="{}",
        preprocessing_config_json="{}",
        k_values_json="[50]",
        random_seed=0,
        tokenization_signature="tok-sig-worker-extraction-test",
    )
    session.add(encoder)
    await session.commit()
    await session.refresh(encoder)

    # ---------- Write events.parquet + event_tokens.parquet ----------
    buffer, events = _build_two_event_audio()
    from humpback.call_parsing.storage import segmentation_job_dir as seg_dir

    seg_path = seg_dir(settings.storage_root, seg_job.id) / "events.parquet"
    write_events(seg_path, events)

    encoder_dir = event_encoder_dir(settings.storage_root, encoder.id)
    encoder_dir.mkdir(parents=True, exist_ok=True)
    _write_event_tokens_parquet(
        event_encoder_tokens_path(settings.storage_root, encoder.id),
        event_ids=[e.event_id for e in events],
        token_id=7,
        k=50,
    )

    # ---------- Monkeypatch the audio provider ----------
    async def _fake_build_audio_provider(*_args, **_kwargs):
        def _provider(event):
            return buffer, 0.0  # buffer_start_relative = 0

        return _provider

    monkeypatch.setattr(
        piano_roll_notes_worker, "_build_audio_provider", _fake_build_audio_provider
    )

    # ---------- Enqueue + run the worker ----------
    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id
    )
    job.status = JobStatus.running.value
    await session.commit()

    await run_piano_roll_notes_job(session, job, settings)

    refreshed = await session.get(PianoRollNotesJob, job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.complete.value, (
        f"expected complete, error: {refreshed.error_message}"
    )
    assert refreshed.notes_path is not None
    assert refreshed.n_events == 2
    assert refreshed.n_notes is not None and refreshed.n_notes >= 3

    notes = pq.read_table(refreshed.notes_path).to_pylist()
    assert notes, "expected at least one note row"

    by_event: dict[str, list[dict]] = {}
    for row in notes:
        by_event.setdefault(row["event_id"], []).append(row)

    # Event 1 should produce F0 = MIDI 57 (A3) and 2x harmonic = MIDI 69 (A4)
    ev1_pitches = sorted({row["midi_pitch"] for row in by_event.get("ev-1", [])})
    assert 57 in ev1_pitches, ev1_pitches
    assert 69 in ev1_pitches, ev1_pitches

    # Event 2 should produce F0 = MIDI 64 (E4) at minimum.
    ev2_pitches = sorted({row["midi_pitch"] for row in by_event.get("ev-2", [])})
    assert 64 in ev2_pitches, ev2_pitches

    # All notes share the token_id we wrote into event_tokens.parquet.
    assert {row["event_token"] for row in notes} == {7}

    # Velocities are within MIDI range.
    velocities = [row["velocity"] for row in notes]
    assert all(1 <= v <= 127 for v in velocities)

    # Notes are sorted by (start_utc, midi_pitch).
    keys = [(row["start_utc"], row["midi_pitch"]) for row in notes]
    assert keys == sorted(keys)

    # First event's notes start near region_offset + event.start_sec (1000.10),
    # second event's near 1001.00.
    starts_ev1 = [row["start_utc"] for row in by_event["ev-1"]]
    starts_ev2 = [row["start_utc"] for row in by_event["ev-2"]]
    assert all(999.9 < s < 1000.6 for s in starts_ev1)
    assert all(1000.9 < s < 1001.6 for s in starts_ev2)


@pytest.mark.asyncio
async def test_worker_partial_failure_still_completes(
    session, settings, tmp_path, monkeypatch
) -> None:
    """One event raises during extraction; the other still produces notes."""
    audio = AudioFile(
        filename="synthetic.wav",
        folder_path=str(tmp_path),
        checksum_sha256="deadbeef-partial",
        duration_seconds=2.0,
        sample_rate_original=SAMPLE_RATE,
    )
    session.add(audio)
    await session.commit()
    await session.refresh(audio)

    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        audio_file_id=audio.id,
        start_timestamp=2000.0,
        end_timestamp=2002.0,
    )
    session.add(region_job)
    await session.commit()
    await session.refresh(region_job)
    seg_job = EventSegmentationJob(
        status=JobStatus.complete.value,
        region_detection_job_id=region_job.id,
    )
    session.add(seg_job)
    await session.commit()
    await session.refresh(seg_job)

    encoder = EventEncoderJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id=seg_job.id,
        event_source_mode="raw",
        continuous_embedding_job_id="cej-stub",
        continuous_embedding_signature="cej-sig",
        tokenizer_version="crnn-event-encoder-v2",
        pooling_config_json="{}",
        descriptor_config_json="{}",
        preprocessing_config_json="{}",
        k_values_json="[50]",
        random_seed=0,
        tokenization_signature="tok-sig-partial-failure",
    )
    session.add(encoder)
    await session.commit()
    await session.refresh(encoder)

    buffer, events = _build_two_event_audio()
    from humpback.call_parsing.storage import segmentation_job_dir as seg_dir

    write_events(seg_dir(settings.storage_root, seg_job.id) / "events.parquet", events)

    async def _fake_build_audio_provider(*_args, **_kwargs):
        def _provider(event):
            if event.event_id == "ev-2":
                raise RuntimeError("simulated audio load failure")
            return buffer, 0.0

        return _provider

    monkeypatch.setattr(
        piano_roll_notes_worker, "_build_audio_provider", _fake_build_audio_provider
    )

    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id
    )
    job.status = JobStatus.running.value
    await session.commit()

    await run_piano_roll_notes_job(session, job, settings)

    refreshed = await session.get(PianoRollNotesJob, job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.complete.value
    assert refreshed.error_message is not None
    assert "ev-2" in refreshed.error_message
    notes = pq.read_table(refreshed.notes_path).to_pylist()
    event_ids_in_notes = {row["event_id"] for row in notes}
    assert event_ids_in_notes == {"ev-1"}


@pytest.mark.asyncio
async def test_worker_deterministic_output(
    session, settings, tmp_path, monkeypatch
) -> None:
    """Two runs on the same encoder job produce byte-equal parquet."""
    audio = AudioFile(
        filename="synthetic.wav",
        folder_path=str(tmp_path),
        checksum_sha256="deadbeef-deterministic",
        duration_seconds=2.0,
        sample_rate_original=SAMPLE_RATE,
    )
    session.add(audio)
    await session.commit()
    await session.refresh(audio)

    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        audio_file_id=audio.id,
        start_timestamp=3000.0,
        end_timestamp=3002.0,
    )
    session.add(region_job)
    await session.commit()
    await session.refresh(region_job)
    seg_job = EventSegmentationJob(
        status=JobStatus.complete.value,
        region_detection_job_id=region_job.id,
    )
    session.add(seg_job)
    await session.commit()
    await session.refresh(seg_job)

    encoder = EventEncoderJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id=seg_job.id,
        event_source_mode="raw",
        continuous_embedding_job_id="cej-stub",
        continuous_embedding_signature="cej-sig",
        tokenizer_version="crnn-event-encoder-v2",
        pooling_config_json="{}",
        descriptor_config_json="{}",
        preprocessing_config_json="{}",
        k_values_json="[50]",
        random_seed=0,
        tokenization_signature="tok-sig-deterministic",
    )
    session.add(encoder)
    await session.commit()
    await session.refresh(encoder)

    buffer, events = _build_two_event_audio()
    from humpback.call_parsing.storage import segmentation_job_dir as seg_dir

    write_events(seg_dir(settings.storage_root, seg_job.id) / "events.parquet", events)

    async def _fake_build_audio_provider(*_args, **_kwargs):
        def _provider(event):
            return buffer, 0.0

        return _provider

    monkeypatch.setattr(
        piano_roll_notes_worker, "_build_audio_provider", _fake_build_audio_provider
    )

    job1, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id
    )
    job1.status = JobStatus.running.value
    await session.commit()
    await run_piano_roll_notes_job(session, job1, settings)
    refreshed1 = await session.get(PianoRollNotesJob, job1.id)
    first_bytes = Path(refreshed1.notes_path).read_bytes()

    # Reset and re-run.
    job1_again = await session.get(PianoRollNotesJob, job1.id)
    job1_again.status = JobStatus.failed.value
    await session.commit()
    job_reset, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id
    )
    assert job_reset.id == job1.id
    job_reset.status = JobStatus.running.value
    await session.commit()
    await run_piano_roll_notes_job(session, job_reset, settings)
    refreshed2 = await session.get(PianoRollNotesJob, job_reset.id)
    second_bytes = Path(refreshed2.notes_path).read_bytes()

    assert first_bytes == second_bytes


# ---------- Fixture-backed end-to-end coverage ----------

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "piano_roll"


def _load_fixture_audio() -> tuple[np.ndarray, dict]:
    wav_path = _FIXTURE_DIR / "synthetic_three_events.wav"
    meta_path = _FIXTURE_DIR / "synthetic_three_events.json"
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        frames = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16)
        audio = (pcm.astype(np.float32) / 32767.0).astype(np.float32)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return audio, metadata


@pytest.mark.asyncio
async def test_worker_recovers_fixture_three_events(
    session, settings, tmp_path, monkeypatch
) -> None:
    """Backend integration check against the committed three-event fixture."""
    buffer, metadata = _load_fixture_audio()
    events = [
        Event(
            event_id=ev["event_id"],
            region_id="rg-fixture",
            start_sec=float(ev["start_s"]),
            end_sec=float(ev["end_s"]),
            center_sec=(float(ev["start_s"]) + float(ev["end_s"])) / 2,
            segmentation_confidence=0.9,
        )
        for ev in metadata["events"]
    ]

    audio = AudioFile(
        filename="synthetic_three_events.wav",
        folder_path=str(tmp_path),
        checksum_sha256="fixture-three-events",
        duration_seconds=float(metadata["duration_s"]),
        sample_rate_original=int(metadata["sample_rate"]),
    )
    session.add(audio)
    await session.commit()
    await session.refresh(audio)

    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        audio_file_id=audio.id,
        start_timestamp=5000.0,
        end_timestamp=5000.0 + float(metadata["duration_s"]),
    )
    session.add(region_job)
    await session.commit()
    await session.refresh(region_job)

    seg_job = EventSegmentationJob(
        status=JobStatus.complete.value,
        region_detection_job_id=region_job.id,
    )
    session.add(seg_job)
    await session.commit()
    await session.refresh(seg_job)

    encoder = EventEncoderJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id=seg_job.id,
        event_source_mode="raw",
        continuous_embedding_job_id="cej-stub",
        continuous_embedding_signature="cej-sig",
        tokenizer_version="crnn-event-encoder-v3",
        pooling_config_json="{}",
        descriptor_config_json="{}",
        preprocessing_config_json="{}",
        k_values_json="[50]",
        random_seed=0,
        tokenization_signature="tok-sig-fixture",
    )
    session.add(encoder)
    await session.commit()
    await session.refresh(encoder)

    from humpback.call_parsing.storage import segmentation_job_dir as seg_dir

    write_events(seg_dir(settings.storage_root, seg_job.id) / "events.parquet", events)
    _write_event_tokens_parquet(
        event_encoder_tokens_path(settings.storage_root, encoder.id),
        event_ids=[e.event_id for e in events],
        token_id=11,
        k=50,
    )

    async def _fake_build_audio_provider(*_args, **_kwargs):
        def _provider(_event):
            return buffer, 0.0

        return _provider

    monkeypatch.setattr(
        piano_roll_notes_worker, "_build_audio_provider", _fake_build_audio_provider
    )

    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id
    )
    job.status = JobStatus.running.value
    await session.commit()
    await run_piano_roll_notes_job(session, job, settings)

    refreshed = await session.get(PianoRollNotesJob, job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.complete.value, refreshed.error_message
    notes = pq.read_table(refreshed.notes_path).to_pylist()
    by_event: dict[str, set[int]] = {}
    for row in notes:
        by_event.setdefault(row["event_id"], set()).add(int(row["midi_pitch"]))
    for expected in metadata["expected_notes"]:
        pitches = by_event.get(expected["event_id"], set())
        assert int(expected["midi_pitch"]) in pitches, (
            f"missing MIDI {expected['midi_pitch']} for {expected['event_id']}: "
            f"got {sorted(pitches)}"
        )
