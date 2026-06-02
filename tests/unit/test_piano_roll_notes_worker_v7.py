"""End-to-end tests for the Piano Roll Notes worker v7 branch."""

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
    event_encoder_ridges_path,
    event_encoder_tokens_path,
)
from humpback.workers import piano_roll_notes_worker
from humpback.workers.piano_roll_notes_worker import (
    _V6_EXTRACTOR_VERSION,
    _V7_EXTRACTOR_VERSION,
    _resolve_params,
    run_piano_roll_notes_job,
)

SAMPLE_RATE = 22050


def _harmonic_stack(
    *,
    fundamental_hz: float,
    duration_s: float,
    harmonic_amplitudes: list[float],
) -> np.ndarray:
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    audio = np.zeros_like(t)
    for n, amp in enumerate(harmonic_amplitudes, start=1):
        audio += amp * np.sin(2 * np.pi * fundamental_hz * n * t)
    return audio.astype(np.float32)


def _build_two_event_audio() -> tuple[np.ndarray, list[Event]]:
    buffer = np.zeros(int(2.0 * SAMPLE_RATE), dtype=np.float32)

    def _insert(audio: np.ndarray, start_s: float) -> None:
        start_idx = int(start_s * SAMPLE_RATE)
        buffer[start_idx : start_idx + audio.shape[0]] += audio

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

    return buffer, [
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


def _write_flat_ridges(path: Path, event_ids: Iterable[str]) -> None:
    rows = []
    for event_id in event_ids:
        for frame_index in range(12):
            rows.append(
                {
                    "event_id": event_id,
                    "frame_index": frame_index,
                    "frame_time_offset_s": frame_index * 0.02,
                    "log_frequency": math.log2(800.0),
                    "strength": 1.0,
                    "energy_ratio": 1.0,
                }
            )
    schema = pa.schema(
        [
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("frame_index", pa.int32(), nullable=False),
            pa.field("frame_time_offset_s", pa.float64(), nullable=False),
            pa.field("log_frequency", pa.float64(), nullable=False),
            pa.field("strength", pa.float64(), nullable=False),
            pa.field("energy_ratio", pa.float64(), nullable=False),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


async def _build_encoder_graph(
    session,
    settings,
    tmp_path: Path,
) -> tuple[EventEncoderJob, list[Event], np.ndarray]:
    audio = AudioFile(
        filename="synthetic.wav",
        folder_path=str(tmp_path),
        checksum_sha256="v7-worker",
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
        tokenization_signature="tok-sig-v7-worker",
    )
    session.add(encoder)
    await session.commit()
    await session.refresh(encoder)

    buffer, events = _build_two_event_audio()
    seg_path = (
        segmentation_job_dir(settings.storage_root, seg_job.id) / "events.parquet"
    )
    write_events(seg_path, events)

    event_encoder_dir(settings.storage_root, encoder.id).mkdir(
        parents=True, exist_ok=True
    )
    _write_event_tokens_parquet(
        event_encoder_tokens_path(settings.storage_root, encoder.id),
        event_ids=[e.event_id for e in events],
        token_id=11,
        k=50,
    )
    _write_flat_ridges(
        event_encoder_ridges_path(
            settings.storage_root, encoder.id, encoder.tokenizer_version
        ),
        event_ids=[e.event_id for e in events],
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


@pytest.mark.asyncio
async def test_v7_worker_writes_v7_sidecars_with_rescue_params(
    session, settings, tmp_path, monkeypatch
) -> None:
    encoder, _events, buffer = await _build_encoder_graph(session, settings, tmp_path)
    _patch_audio_provider(monkeypatch, buffer)

    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id, extractor_version="v7"
    )
    job.status = JobStatus.running.value
    await session.commit()

    await run_piano_roll_notes_job(session, job, settings)

    refreshed = await session.get(PianoRollNotesJob, job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.complete.value, refreshed.error_message
    assert refreshed.notes_path is not None
    assert Path(refreshed.notes_path).name == "event_notes_v7.parquet"

    contours_path = event_encoder_note_contours_path(
        settings.storage_root, encoder.id, "v7"
    )
    assert contours_path.exists()
    assert event_encoder_notes_path(settings.storage_root, encoder.id, "v7").exists()

    notes = pq.read_table(refreshed.notes_path)
    contours = pq.read_table(contours_path)
    assert len(notes) > 0
    assert len(contours) > 0
    assert {"note_uid", "f0_track_id", "contour_frame_count"} <= set(notes.schema.names)

    payload = json.loads(refreshed.params_json)
    assert payload["despike"]["enabled"] is True
    assert payload["discontinuity"]["enabled"] is True
    assert payload["discontinuity"]["max_continuous_slope_oct_per_s"] == 6.0
    assert payload["ridge_rescue"]["enabled"] is True
    assert payload["ridge_rescue"]["max_decoded_span_semitones"] == 2.0
    assert payload["ridge_rescue"]["min_ridge_span_semitones"] == 5.0
    assert payload["ridge_rescue"]["min_overlap_frames"] == 8
    assert payload["audio"]["pad_seconds"] == 0.25
    assert payload["segmentation"]["min_break_frames"] == 6
    assert payload["contours_path"].endswith("event_note_contours_v7.parquet")
    assert payload["ridges_path"].endswith("event_ridges_crnn-event-encoder-v3.parquet")


def test_resolve_params_round_trip_for_v7() -> None:
    resolved = _resolve_params("", _V7_EXTRACTOR_VERSION)
    assert resolved.stft.min_frequency_hz == 30.0
    assert resolved.segmentation.min_break_frames == 6
    assert resolved.audio.pad_seconds == 0.25
    assert resolved.discontinuity.enabled is True
    assert resolved.ridge_rescue.enabled is True

    payload = resolved.to_json_dict()
    again = _resolve_params(json.dumps(payload), _V7_EXTRACTOR_VERSION)
    assert again.despike == resolved.despike
    assert again.discontinuity == resolved.discontinuity
    assert again.ridge_rescue == resolved.ridge_rescue
    assert again.harmonic_viterbi == resolved.harmonic_viterbi


def test_resolve_params_v7_overrides_and_v6_defaults_remain_distinct() -> None:
    payload = json.dumps(
        {
            "discontinuity": {
                "enabled": False,
                "max_continuous_slope_oct_per_s": 4.5,
            },
            "ridge_rescue": {
                "enabled": False,
                "max_decoded_span_semitones": 1.0,
                "min_ridge_span_semitones": 7.0,
                "min_overlap_frames": 5,
                "max_ratio_mad_semitones": 0.75,
                "min_carrier_harmonic": 2,
                "max_carrier_harmonic": 12,
            },
        }
    )
    resolved = _resolve_params(payload, _V7_EXTRACTOR_VERSION)
    assert resolved.discontinuity.enabled is False
    assert resolved.discontinuity.max_continuous_slope_oct_per_s == 4.5
    assert resolved.ridge_rescue.enabled is False
    assert resolved.ridge_rescue.max_decoded_span_semitones == 1.0
    assert resolved.ridge_rescue.min_ridge_span_semitones == 7.0
    assert resolved.ridge_rescue.min_overlap_frames == 5
    assert resolved.ridge_rescue.max_ratio_mad_semitones == 0.75
    assert resolved.ridge_rescue.min_carrier_harmonic == 2
    assert resolved.ridge_rescue.max_carrier_harmonic == 12

    v6 = _resolve_params("", _V6_EXTRACTOR_VERSION)
    assert v6.segmentation.min_break_frames == 6
    assert v6.audio.pad_seconds == 0.25
