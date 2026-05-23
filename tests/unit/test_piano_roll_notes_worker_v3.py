"""End-to-end tests for the Piano Roll Notes worker v3 branch.

Builds the same SQL dependency graph as the v2 extraction test, but the
job is enqueued with ``extractor_version='v3'`` so the worker takes the
ridge-aware code path. Verifies that:

- The notes and contour sidecars are both written with the v3 schema.
- ``params_json`` records both sidecar paths and the contour-frame count.
- Encoder ridge sidecar consumption short-circuits in-process recompute.
- Missing ridge sidecar falls back to recompute and still succeeds.
- Partial-failure inside the contour writer cleans up both files.
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
    event_encoder_ridges_path,
    event_encoder_tokens_path,
)
from humpback.workers import piano_roll_notes_worker
from humpback.workers.piano_roll_notes_worker import run_piano_roll_notes_job


SAMPLE_RATE = 22050


def _harmonic_stack(
    *,
    fundamental_hz: float,
    duration_s: float,
    n_harmonics: int = 3,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.4,
) -> np.ndarray:
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    audio = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        audio += (amplitude / k) * np.sin(2 * np.pi * fundamental_hz * k * t)
    return audio.astype(np.float32)


def _build_two_event_audio() -> tuple[np.ndarray, list[Event]]:
    total_seconds = 2.0
    buffer = np.zeros(int(total_seconds * SAMPLE_RATE), dtype=np.float32)

    def _insert(audio: np.ndarray, start_s: float) -> None:
        start_idx = int(start_s * SAMPLE_RATE)
        end_idx = start_idx + audio.shape[0]
        buffer[start_idx:end_idx] += audio

    _insert(
        _harmonic_stack(fundamental_hz=220.0, duration_s=0.40, n_harmonics=3),
        start_s=0.10,
    )
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


async def _build_v3_graph(
    session,
    settings,
    tmp_path: Path,
    *,
    suffix: str,
    region_start_utc: float = 1000.0,
) -> tuple[EventEncoderJob, list[Event], np.ndarray]:
    audio = AudioFile(
        filename="synthetic.wav",
        folder_path=str(tmp_path),
        checksum_sha256=f"v3-{suffix}",
        duration_seconds=2.0,
        sample_rate_original=SAMPLE_RATE,
    )
    session.add(audio)
    await session.commit()
    await session.refresh(audio)

    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        audio_file_id=audio.id,
        start_timestamp=region_start_utc,
        end_timestamp=region_start_utc + 2.0,
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
        tokenization_signature=f"tok-sig-v3-{suffix}",
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
        token_id=7,
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


def _write_fake_ridge_sidecar(path: Path, events: list[Event]) -> None:
    """Manufacture a constant-frequency ridge sidecar at the F0 pitch.

    The encoder normally writes this; we synthesize it so the worker can
    take the cached-ridge path without re-encoding upstream.
    """
    rows: list[dict] = []
    for event in events:
        if event.event_id == "ev-1":
            log_freq = math.log2(220.0)
        else:
            log_freq = math.log2(330.0)
        for frame_index in range(20):
            rows.append(
                {
                    "event_id": event.event_id,
                    "frame_index": int(frame_index),
                    "frame_time_offset_s": float(frame_index * 512 / SAMPLE_RATE),
                    "log_frequency": float(log_freq),
                    "strength": 1.0,
                    "energy_ratio": 0.5,
                }
            )
    schema = pa.schema(
        [
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("frame_index", pa.uint32(), nullable=False),
            pa.field("frame_time_offset_s", pa.float32(), nullable=False),
            pa.field("log_frequency", pa.float32(), nullable=False),
            pa.field("strength", pa.float32(), nullable=False),
            pa.field("energy_ratio", pa.float32(), nullable=False),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v3_worker_writes_notes_and_contours(
    session, settings, tmp_path, monkeypatch
) -> None:
    encoder, _events, buffer = await _build_v3_graph(
        session, settings, tmp_path, suffix="happy"
    )
    _patch_audio_provider(monkeypatch, buffer)

    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id, extractor_version="v3"
    )
    job.status = JobStatus.running.value
    await session.commit()

    await run_piano_roll_notes_job(session, job, settings)

    refreshed = await session.get(PianoRollNotesJob, job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.complete.value, refreshed.error_message
    assert refreshed.notes_path is not None and Path(refreshed.notes_path).exists()
    assert refreshed.n_notes is not None and refreshed.n_notes >= 1

    contours_path = event_encoder_note_contours_path(
        settings.storage_root, encoder.id, "v3"
    )
    assert contours_path.exists()

    notes = pq.read_table(refreshed.notes_path)
    contours = pq.read_table(contours_path)

    # v3 schema columns present.
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

    # note_uids in the notes table align with contour rows.
    notes_pylist = notes.to_pylist()
    note_uids_in_notes = {row["note_uid"] for row in notes_pylist}
    note_uids_in_contours = {row["note_uid"] for row in contours.to_pylist()}
    assert note_uids_in_notes == note_uids_in_contours

    # contour_frame_count matches the actual count.
    counts_by_uid: dict[str, int] = {}
    for row in contours.to_pylist():
        counts_by_uid[row["note_uid"]] = counts_by_uid.get(row["note_uid"], 0) + 1
    for row in notes_pylist:
        assert counts_by_uid[row["note_uid"]] == row["contour_frame_count"]

    # No `partial_index = -1` regression.
    assert all(row["partial_index"] >= 0 for row in notes_pylist)

    # event_token propagates from the encoder's token parquet (k=50, token_id=7).
    assert {row["event_token"] for row in notes_pylist} == {7}

    # params_json carries sidecar paths and contour frame count.
    payload = json.loads(refreshed.params_json)
    assert "contours_path" in payload
    assert "ridges_path" in payload
    assert payload["ridges_path"] == "absent"
    assert payload["n_contour_frames"] == sum(counts_by_uid.values())


@pytest.mark.asyncio
async def test_v3_worker_uses_ridge_sidecar_when_present(
    session, settings, tmp_path, monkeypatch
) -> None:
    encoder, events, buffer = await _build_v3_graph(
        session, settings, tmp_path, suffix="sidecar"
    )
    _patch_audio_provider(monkeypatch, buffer)

    sidecar_path = event_encoder_ridges_path(
        settings.storage_root, encoder.id, encoder.tokenizer_version
    )
    _write_fake_ridge_sidecar(sidecar_path, events)

    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id, extractor_version="v3"
    )
    job.status = JobStatus.running.value
    await session.commit()

    await run_piano_roll_notes_job(session, job, settings)

    refreshed = await session.get(PianoRollNotesJob, job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.complete.value, refreshed.error_message

    payload = json.loads(refreshed.params_json)
    assert payload["ridges_path"] == str(sidecar_path)


@pytest.mark.asyncio
async def test_v3_worker_partial_failure_cleans_up_sidecars(
    session, settings, tmp_path, monkeypatch
) -> None:
    """Raising inside the contour writer should remove both v3 parquets."""
    encoder, _events, buffer = await _build_v3_graph(
        session, settings, tmp_path, suffix="partial"
    )
    _patch_audio_provider(monkeypatch, buffer)

    original = piano_roll_notes_worker._atomic_write_parquet
    call_count = {"n": 0}

    def _failing_write(table, dst):
        call_count["n"] += 1
        if call_count["n"] == 2:  # contour write
            raise RuntimeError("simulated contour-writer failure")
        return original(table, dst)

    monkeypatch.setattr(
        piano_roll_notes_worker, "_atomic_write_parquet", _failing_write
    )

    job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id, extractor_version="v3"
    )
    job.status = JobStatus.running.value
    await session.commit()

    await run_piano_roll_notes_job(session, job, settings)

    refreshed = await session.get(PianoRollNotesJob, job.id)
    assert refreshed is not None
    assert refreshed.status == JobStatus.failed.value
    assert refreshed.error_message is not None

    notes_path = piano_roll_notes_worker.event_encoder_notes_path(
        settings.storage_root, encoder.id, "v3"
    )
    contours_path = event_encoder_note_contours_path(
        settings.storage_root, encoder.id, "v3"
    )
    assert not notes_path.exists()
    assert not contours_path.exists()
    # No leftover .tmp artifacts.
    for path in (notes_path, contours_path):
        assert not path.with_suffix(path.suffix + ".tmp").exists()


@pytest.mark.asyncio
async def test_v2_and_v3_coexist_for_same_encoder_job(
    session, settings, tmp_path, monkeypatch
) -> None:
    """A v2 job submitted alongside v3 should still run the legacy path."""
    encoder, _events, buffer = await _build_v3_graph(
        session, settings, tmp_path, suffix="coexist"
    )
    _patch_audio_provider(monkeypatch, buffer)

    v2_job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id, extractor_version="v2"
    )
    v2_job.status = JobStatus.running.value
    await session.commit()
    await run_piano_roll_notes_job(session, v2_job, settings)
    v2_refreshed = await session.get(PianoRollNotesJob, v2_job.id)
    assert v2_refreshed.status == JobStatus.complete.value
    v2_path = Path(v2_refreshed.notes_path)
    assert "event_notes_v2.parquet" in str(v2_path)
    v2_table = pq.read_table(v2_path)
    assert "note_uid" not in set(v2_table.schema.names)

    v3_job, _ = await enqueue_piano_roll_notes_job(
        session, event_encoder_job_id=encoder.id, extractor_version="v3"
    )
    v3_job.status = JobStatus.running.value
    await session.commit()
    await run_piano_roll_notes_job(session, v3_job, settings)
    v3_refreshed = await session.get(PianoRollNotesJob, v3_job.id)
    assert v3_refreshed.status == JobStatus.complete.value
    v3_path = Path(v3_refreshed.notes_path)
    assert "event_notes_v3.parquet" in str(v3_path)
    v3_table = pq.read_table(v3_path)
    assert "note_uid" in set(v3_table.schema.names)
    # Both sidecars coexist on disk.
    assert v2_path.exists() and v3_path.exists()
