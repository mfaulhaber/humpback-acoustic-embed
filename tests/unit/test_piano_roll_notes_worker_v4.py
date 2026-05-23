"""End-to-end tests for the Piano Roll Notes worker v4 branch (ADR-070).

Mirrors the v3 worker tests but exercises ``extractor_version='v4'``,
verifying HPS dispatch, sidecar paths, ``params_json`` round-trip, the
mixed-version coexistence guarantee on disk, and the worker's resilience
when the encoder ridge sidecar is missing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from humpback.call_parsing.storage import segmentation_job_dir, write_events
from humpback.call_parsing.types import Event
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.piano_roll_notes import PianoRollNotesJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import EventEncoderJob
from humpback.services.piano_roll_notes_service import enqueue_piano_roll_notes_job
from humpback.storage import (
    event_encoder_dir,
    event_encoder_note_contours_path,
    event_encoder_notes_path,
    event_encoder_tokens_path,
)
from humpback.workers import piano_roll_notes_worker
from humpback.workers.piano_roll_notes_worker import (
    _resolve_params,
    _V4_EXTRACTOR_VERSION,
    run_piano_roll_notes_job,
)


SAMPLE_RATE = 22050


def _harmonic_stack(
    *,
    fundamental_hz: float,
    duration_s: float,
    harmonic_amplitudes: list[float],
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    audio = np.zeros_like(t)
    for n, amp in enumerate(harmonic_amplitudes, start=1):
        audio += amp * np.sin(2 * np.pi * fundamental_hz * n * t)
    return audio.astype(np.float32)


def _build_two_event_audio() -> tuple[np.ndarray, list[Event]]:
    total_seconds = 2.0
    buffer = np.zeros(int(total_seconds * SAMPLE_RATE), dtype=np.float32)

    def _insert(audio: np.ndarray, start_s: float) -> None:
        start_idx = int(start_s * SAMPLE_RATE)
        end_idx = start_idx + audio.shape[0]
        buffer[start_idx:end_idx] += audio

    # Event 1: F0 at 200 Hz with strong H2 — the v3 failure mode case.
    # HPS should pick d=2 and emit F0 ≈ MIDI 55.
    _insert(
        _harmonic_stack(
            fundamental_hz=200.0,
            duration_s=0.40,
            harmonic_amplitudes=[0.10, 0.40],
        ),
        start_s=0.10,
    )
    # Event 2: F0 at 330 Hz with three harmonics — should land near MIDI 64.
    _insert(
        _harmonic_stack(
            fundamental_hz=330.0,
            duration_s=0.40,
            harmonic_amplitudes=[0.40, 0.20, 0.10],
        ),
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


async def _build_encoder_graph(
    session,
    settings,
    tmp_path: Path,
    *,
    suffix: str,
) -> tuple[EventEncoderJob, list[Event], np.ndarray]:
    audio = AudioFile(
        filename="synthetic.wav",
        folder_path=str(tmp_path),
        checksum_sha256=f"v4-{suffix}",
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
        tokenizer_version="crnn-event-encoder-v3",
        pooling_config_json="{}",
        descriptor_config_json="{}",
        preprocessing_config_json="{}",
        k_values_json="[50]",
        random_seed=0,
        tokenization_signature=f"tok-sig-v4-{suffix}",
    )
    session.add(encoder)
    await session.commit()
    await session.refresh(encoder)

    buffer, events = _build_two_event_audio()
    seg_path = (
        segmentation_job_dir(settings.storage_root, seg_job.id) / "events.parquet"
    )
    write_events(seg_path, events)

    encoder_dir = event_encoder_dir(settings.storage_root, encoder.id)
    encoder_dir.mkdir(parents=True, exist_ok=True)
    _write_event_tokens_parquet(
        event_encoder_tokens_path(settings.storage_root, encoder.id),
        event_ids=[e.event_id for e in events],
        token_id=11,
        k=50,
    )
    return encoder, events, buffer


def _patch_audio_provider(monkeypatch, buffer: np.ndarray) -> None:
    async def _fake_build_audio_provider(*_args, **_kwargs):
        def _provider(_event):
            return buffer, 0.0

        return _provider

    monkeypatch.setattr(
        piano_roll_notes_worker,
        "_build_audio_provider",
        _fake_build_audio_provider,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v4_worker_writes_v4_sidecars_with_hps_params(
    session, settings, tmp_path, monkeypatch
) -> None:
    encoder, _events, buffer = await _build_encoder_graph(
        session, settings, tmp_path, suffix="happy"
    )
    _patch_audio_provider(monkeypatch, buffer)

    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id, extractor_version="v4"
    )
    job.status = JobStatus.running.value
    await session.commit()

    await run_piano_roll_notes_job(session, job, settings)

    refreshed = await session.get(PianoRollNotesJob, job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.complete.value, refreshed.error_message
    assert refreshed.notes_path is not None
    notes_path_obj = Path(refreshed.notes_path)
    # Path templates on extractor_version, so the v4 file lives at the
    # v4-suffixed name and not the v3 path.
    assert notes_path_obj.name == "event_notes_v4.parquet"
    assert notes_path_obj.exists()

    contours_path = event_encoder_note_contours_path(
        settings.storage_root, encoder.id, "v4"
    )
    assert contours_path.exists()

    notes = pq.read_table(refreshed.notes_path)
    contours = pq.read_table(contours_path)

    # Same schema columns as v3.
    note_columns = set(notes.schema.names)
    assert {"note_uid", "f0_track_id", "contour_frame_count"} <= note_columns
    contour_columns = set(contours.schema.names)
    assert {
        "note_uid",
        "frame_index",
        "time_offset_s",
        "cents_from_pitch",
        "harmonic_strength",
        "subharmonic_octave",
    } <= contour_columns

    # At least one F0 note recovered from the H2-dominated event 1 — and
    # for that event HPS should have shifted the ridge down (recorded as
    # subharmonic_octave > 0 on a majority of its F0 contour frames).
    notes_pylist = notes.to_pylist()
    contour_pylist = contours.to_pylist()
    ev1_f0_notes = [
        row
        for row in notes_pylist
        if row["event_id"] == "ev-1" and row["partial_index"] == 0
    ]
    assert ev1_f0_notes, "no F0 note emitted for ev-1"
    ev1_f0_uid = ev1_f0_notes[0]["note_uid"]
    ev1_f0_contours = [r for r in contour_pylist if r["note_uid"] == ev1_f0_uid]
    assert ev1_f0_contours
    shifted = sum(1 for r in ev1_f0_contours if r["subharmonic_octave"] >= 1)
    assert shifted >= len(ev1_f0_contours) // 2, (
        "expected HPS to mark a majority of ev-1 F0 frames as ridge-shifted "
        f"(divisor > 1), saw {shifted}/{len(ev1_f0_contours)}"
    )

    # The F0 should land near 200 Hz (MIDI 55), not 400 Hz (MIDI 67).
    assert ev1_f0_notes[0]["midi_pitch"] <= 60

    # No `partial_index = -1` regression carried over from v1/v2 era.
    assert all(row["partial_index"] >= 0 for row in notes_pylist)

    # params_json round-trips with the hps block.
    payload = json.loads(refreshed.params_json)
    assert "hps" in payload
    assert payload["hps"]["candidate_divisors"] == [1, 2, 3, 4, 5, 6]
    assert payload["hps"]["max_harmonic_dynamic_range_log"] == 3.0
    assert payload["stft"]["min_frequency_hz"] == 30.0  # v4 default
    # contours sidecar metadata also present.
    assert payload["contours_path"].endswith("event_note_contours_v4.parquet")
    assert payload["ridges_path"] == "absent"
    assert payload["n_contour_frames"] == len(contour_pylist)


@pytest.mark.asyncio
async def test_v4_and_v3_sidecars_coexist_on_disk(
    session, settings, tmp_path, monkeypatch
) -> None:
    encoder, _events, buffer = await _build_encoder_graph(
        session, settings, tmp_path, suffix="coexist"
    )
    _patch_audio_provider(monkeypatch, buffer)

    # Run v3 first.
    job_v3, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id, extractor_version="v3"
    )
    job_v3.status = JobStatus.running.value
    await session.commit()
    await run_piano_roll_notes_job(session, job_v3, settings)

    # Then v4 in the same encoder dir.
    job_v4, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id, extractor_version="v4"
    )
    job_v4.status = JobStatus.running.value
    await session.commit()
    await run_piano_roll_notes_job(session, job_v4, settings)

    v3_notes = event_encoder_notes_path(settings.storage_root, encoder.id, "v3")
    v4_notes = event_encoder_notes_path(settings.storage_root, encoder.id, "v4")
    v3_contours = event_encoder_note_contours_path(
        settings.storage_root, encoder.id, "v3"
    )
    v4_contours = event_encoder_note_contours_path(
        settings.storage_root, encoder.id, "v4"
    )

    # All four sidecars exist independently — v4 must not overwrite v3.
    assert v3_notes.exists()
    assert v4_notes.exists()
    assert v3_contours.exists()
    assert v4_contours.exists()

    # Final v4 job in the database is complete.
    refreshed = await session.get(PianoRollNotesJob, job_v4.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.complete.value


def test_resolve_params_round_trip_for_v4() -> None:
    """``_resolve_params`` reads its own JSON output back losslessly."""
    resolved = _resolve_params("", _V4_EXTRACTOR_VERSION)
    assert resolved.stft.min_frequency_hz == 30.0
    payload = resolved.to_json_dict()
    assert "hps" in payload
    # Re-resolve from the serialized payload.
    again = _resolve_params(json.dumps(payload), _V4_EXTRACTOR_VERSION)
    assert again.hps == resolved.hps
    assert again.stft == resolved.stft
    assert again.segmentation == resolved.segmentation
    assert again.harmonic_v3 == resolved.harmonic_v3
    assert again.midi_v3 == resolved.midi_v3


def test_resolve_params_legacy_min_freq_default_unchanged_for_v3() -> None:
    """Older v3 rows continue to default ``stft.min_frequency_hz`` to 100 Hz."""
    resolved = _resolve_params("", "v3")
    assert resolved.stft.min_frequency_hz == 100.0
    # Legacy rows with no version string still get the v3 (100 Hz)
    # default — only v4 jobs opt into the wider band.
    resolved_legacy = _resolve_params("")
    assert resolved_legacy.stft.min_frequency_hz == 100.0


def test_resolve_params_explicit_value_overrides_version_default() -> None:
    """An explicit ``stft.min_frequency_hz`` in params_json wins over the version default."""
    payload = json.dumps({"stft": {"min_frequency_hz": 75.0}})
    resolved = _resolve_params(payload, _V4_EXTRACTOR_VERSION)
    assert resolved.stft.min_frequency_hz == 75.0


def test_v4_harmonic_stack_pitch_check() -> None:
    """Smoke: a pure 200 Hz F0 + H2 audio buffer recovers the F0 in-process.

    Lighter weight than the worker fixture; confirms the HPS algorithm
    settles to MIDI 55 on a representative input without needing the
    full SQL graph.
    """
    duration_s = 0.40
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    audio = (
        0.10 * np.sin(2.0 * np.pi * 200.0 * t) + 0.40 * np.sin(2.0 * np.pi * 400.0 * t)
    ).astype(np.float32)
    from humpback.processing.note_extractor_v4 import (
        ExtractNotesV4Params,
        extract_notes_v4,
    )

    params = ExtractNotesV4Params(job_id="j", event_id="e", event_start_utc=0.0)
    result = extract_notes_v4(audio, SAMPLE_RATE, params=params)
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert f0_notes
    assert abs(f0_notes[0].midi_pitch - 55) <= 2
    # midi_pitch == 55 means HPS shifted ridge down by an octave.
    assert math.isclose(2 ** (math.log2(440.0) + (55 - 69) / 12.0), 196.0, rel_tol=0.02)
