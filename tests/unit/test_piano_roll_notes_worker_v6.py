"""End-to-end tests for the Piano Roll Notes worker v6 branch (ADR-072).

Mirrors the v4/v5 worker tests but exercises ``extractor_version='v6'``,
verifying de-spike dispatch, sidecar paths, ``params_json`` round-trip of
the ``despike`` block, the v6 version-conditional defaults, and the
mixed-version coexistence guarantee on disk (v6 must not overwrite v5).
"""

from __future__ import annotations

import json
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
    _V6_EXTRACTOR_VERSION,
    _resolve_params,
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

    _insert(
        _harmonic_stack(
            fundamental_hz=200.0,
            duration_s=0.40,
            harmonic_amplitudes=[0.40, 0.30, 0.20, 0.15],
        ),
        start_s=0.10,
    )
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
        checksum_sha256=f"v6-{suffix}",
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
        tokenization_signature=f"tok-sig-v6-{suffix}",
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
async def test_v6_worker_writes_v6_sidecars_with_despike_params(
    session, settings, tmp_path, monkeypatch
) -> None:
    encoder, _events, buffer = await _build_encoder_graph(
        session, settings, tmp_path, suffix="happy"
    )
    _patch_audio_provider(monkeypatch, buffer)

    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id, extractor_version="v6"
    )
    job.status = JobStatus.running.value
    await session.commit()

    await run_piano_roll_notes_job(session, job, settings)

    refreshed = await session.get(PianoRollNotesJob, job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.complete.value, refreshed.error_message
    assert refreshed.notes_path is not None
    notes_path_obj = Path(refreshed.notes_path)
    assert notes_path_obj.name == "event_notes_v6.parquet"
    assert notes_path_obj.exists()

    contours_path = event_encoder_note_contours_path(
        settings.storage_root, encoder.id, "v6"
    )
    assert contours_path.exists()

    notes = pq.read_table(refreshed.notes_path)
    contours = pq.read_table(contours_path)

    # Same schema columns as v3-v5.
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

    notes_pylist = notes.to_pylist()
    contour_pylist = contours.to_pylist()
    ev1_f0_notes = [
        row
        for row in notes_pylist
        if row["event_id"] == "ev-1" and row["partial_index"] == 0
    ]
    assert ev1_f0_notes, "no F0 note emitted for ev-1"
    # F0 near 200 Hz (MIDI 55).
    assert ev1_f0_notes[0]["midi_pitch"] <= 60
    # v6 reserves subharmonic_octave (always 0, like v5).
    assert all(r["subharmonic_octave"] == 0 for r in contour_pylist)
    assert all(row["partial_index"] >= 0 for row in notes_pylist)

    # params_json round-trips with the despike block and the v6 defaults.
    payload = json.loads(refreshed.params_json)
    assert payload["despike"]["enabled"] is True
    assert payload["despike"]["max_slope_oct_per_s"] == 6.0
    assert payload["despike"]["max_spike_frames"] == 12
    assert payload["stft"]["min_frequency_hz"] == 30.0  # v6 default
    assert payload["segmentation"]["min_break_frames"] == 6  # v6 default
    assert payload["audio"]["pad_seconds"] == 0.25  # v6 default
    assert payload["contours_path"].endswith("event_note_contours_v6.parquet")
    assert payload["n_contour_frames"] == len(contour_pylist)


@pytest.mark.asyncio
async def test_v6_and_v5_sidecars_coexist_on_disk(
    session, settings, tmp_path, monkeypatch
) -> None:
    encoder, _events, buffer = await _build_encoder_graph(
        session, settings, tmp_path, suffix="coexist"
    )
    _patch_audio_provider(monkeypatch, buffer)

    job_v5, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id, extractor_version="v5"
    )
    job_v5.status = JobStatus.running.value
    await session.commit()
    await run_piano_roll_notes_job(session, job_v5, settings)

    job_v6, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id, extractor_version="v6"
    )
    job_v6.status = JobStatus.running.value
    await session.commit()
    await run_piano_roll_notes_job(session, job_v6, settings)

    v5_notes = event_encoder_notes_path(settings.storage_root, encoder.id, "v5")
    v6_notes = event_encoder_notes_path(settings.storage_root, encoder.id, "v6")
    v5_contours = event_encoder_note_contours_path(
        settings.storage_root, encoder.id, "v5"
    )
    v6_contours = event_encoder_note_contours_path(
        settings.storage_root, encoder.id, "v6"
    )

    assert v5_notes.exists()
    assert v6_notes.exists()
    assert v5_contours.exists()
    assert v6_contours.exists()

    refreshed = await session.get(PianoRollNotesJob, job_v6.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.complete.value


def test_resolve_params_round_trip_for_v6() -> None:
    """``_resolve_params`` reads its own JSON output back losslessly for v6."""
    resolved = _resolve_params("", _V6_EXTRACTOR_VERSION)
    # v6 inherits the v5-like version defaults.
    assert resolved.stft.min_frequency_hz == 30.0
    assert resolved.segmentation.min_break_frames == 6
    assert resolved.audio.pad_seconds == 0.25
    # Default despike block.
    assert resolved.despike.enabled is True
    assert resolved.despike.max_slope_oct_per_s == 6.0
    assert resolved.despike.max_spike_frames == 12

    payload = resolved.to_json_dict()
    assert "despike" in payload
    again = _resolve_params(json.dumps(payload), _V6_EXTRACTOR_VERSION)
    assert again.despike == resolved.despike
    assert again.harmonic_viterbi == resolved.harmonic_viterbi
    assert again.segmentation == resolved.segmentation
    assert again.audio == resolved.audio


def test_resolve_params_despike_override_for_v6() -> None:
    """An explicit despike block in params_json overrides the defaults."""
    payload = json.dumps(
        {
            "despike": {
                "enabled": False,
                "max_slope_oct_per_s": 3.0,
                "max_spike_frames": 5,
            }
        }
    )
    resolved = _resolve_params(payload, _V6_EXTRACTOR_VERSION)
    assert resolved.despike.enabled is False
    assert resolved.despike.max_slope_oct_per_s == 3.0
    assert resolved.despike.max_spike_frames == 5
