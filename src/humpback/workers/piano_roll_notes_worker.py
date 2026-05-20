"""Worker for Piano Roll Notes jobs.

Reads completed Event Encoder outputs and the source audio, then for each
event decomposes the time-frequency content into a set of MIDI notes via
CQT + per-frame peak picking + cross-frame tracking + harmonic-prior
labeling. Per-job notes are persisted to
``event_notes_{extractor_version}.parquet`` under the Event Encoder job
directory; one parquet row per note.

The worker never modifies Event Encoder outputs. Per-event audio failures
are aggregated into the job row's ``error_message`` but do not abort the
job — the job is only marked ``failed`` when no event yielded a note and
at least one event threw.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.audio_loader import build_event_audio_loader
from humpback.call_parsing.storage import read_events, segmentation_job_dir
from humpback.call_parsing.types import Event
from humpback.config import Settings
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.piano_roll_notes import PianoRollNotesJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import EventEncoderJob
from humpback.processing.piano_roll_cqt import (
    CQTParams,
    PeakParams,
    compute_event_cqt,
    pick_peaks_per_frame,
)
from humpback.processing.piano_roll_tracker import (
    HarmonicParams,
    MIDIQuantizeParams,
    TrackerParams,
    build_tracks,
    label_harmonics,
    quantize_to_midi,
)
from humpback.storage import (
    ensure_dir,
    event_encoder_dir,
    event_encoder_notes_path,
    event_encoder_tokens_path,
)

logger = logging.getLogger(__name__)


NOTES_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("event_token", pa.int32(), nullable=False),
        pa.field("partial_index", pa.int32(), nullable=False),
        pa.field("midi_pitch", pa.uint8(), nullable=False),
        pa.field("start_utc", pa.float64(), nullable=False),
        pa.field("start_offset_s", pa.float64(), nullable=False),
        pa.field("duration_s", pa.float64(), nullable=False),
        pa.field("velocity", pa.uint8(), nullable=False),
        pa.field("peak_magnitude", pa.float32(), nullable=False),
        pa.field("track_id", pa.uint32(), nullable=False),
    ]
)


_MAX_FAILURE_REPORT = 10
_UNKNOWN_TOKEN = -1


@dataclass(frozen=True, slots=True)
class _VelocityParams:
    floor_percentile: float = 5.0
    ceiling_percentile: float = 99.0
    floor: int = 1
    ceiling: int = 127


@dataclass(frozen=True, slots=True)
class _AudioParams:
    pad_seconds: float = 0.05
    min_event_duration_s: float = 0.03


@dataclass(frozen=True, slots=True)
class _ResolvedParams:
    cqt: CQTParams
    peak: PeakParams
    tracker: TrackerParams
    harmonic: HarmonicParams
    midi: MIDIQuantizeParams
    velocity: _VelocityParams
    audio: _AudioParams

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "cqt": {
                "target_sample_rate": self.cqt.target_sample_rate,
                "hop_length": self.cqt.hop_length,
                "fmin": self.cqt.fmin,
                "n_bins": self.cqt.n_bins,
                "bins_per_octave": self.cqt.bins_per_octave,
                "filter_scale": self.cqt.filter_scale,
            },
            "peak": {
                "k_noise": self.peak.k_noise,
                "top_k": self.peak.top_k,
            },
            "tracker": {
                "bin_tolerance": self.tracker.bin_tolerance,
                "miss_tolerance_frames": self.tracker.miss_tolerance_frames,
                "min_duration_s": self.tracker.min_duration_s,
                "amplitude_floor_percentile": self.tracker.amplitude_floor_percentile,
            },
            "harmonic": {
                "enabled": self.harmonic.enabled,
                "max_harmonic": self.harmonic.max_harmonic,
                "cents_tolerance": self.harmonic.cents_tolerance,
            },
            "midi": {
                "min_pitch": self.midi.min_pitch,
                "max_pitch": self.midi.max_pitch,
            },
            "velocity": {
                "floor_percentile": self.velocity.floor_percentile,
                "ceiling_percentile": self.velocity.ceiling_percentile,
                "floor": self.velocity.floor,
                "ceiling": self.velocity.ceiling,
            },
            "audio": {
                "pad_seconds": self.audio.pad_seconds,
                "min_event_duration_s": self.audio.min_event_duration_s,
            },
        }


def _atomic_write_parquet(table: pa.Table, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


async def run_piano_roll_notes_job(
    session: AsyncSession,
    job: PianoRollNotesJob,
    settings: Settings,
) -> None:
    """Execute one Piano Roll Notes job end-to-end."""
    job_id = job.id
    started_wall = time.monotonic()
    started_dt = datetime.now(timezone.utc)

    try:
        job = await session.merge(job)
        job.started_at = started_dt
        await session.commit()

        encoder = await session.get(EventEncoderJob, job.event_encoder_job_id)
        if encoder is None:
            raise ValueError(f"event_encoder_job not found: {job.event_encoder_job_id}")
        if encoder.status != JobStatus.complete.value:
            raise ValueError(
                "piano roll notes requires a completed event_encoder_job "
                f"(status={encoder.status!r})"
            )

        params = _resolve_params(job.params_json)
        notes_path = event_encoder_notes_path(
            settings.storage_root, encoder.id, job.extractor_version
        )
        ensure_dir(event_encoder_dir(settings.storage_root, encoder.id))

        rows, n_events, failures = await _extract_notes(
            session, encoder, params, settings
        )
        table = pa.Table.from_pylist(rows, schema=NOTES_SCHEMA)
        _atomic_write_parquet(table, notes_path)

        compute_seconds = time.monotonic() - started_wall
        refreshed = await session.get(PianoRollNotesJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = JobStatus.complete.value
        target.finished_at = datetime.now(timezone.utc)
        target.error_message = _summarize_failures(failures)
        target.notes_path = str(notes_path)
        target.n_events = n_events
        target.n_notes = len(rows)
        target.compute_seconds = compute_seconds
        target.params_json = json.dumps(
            params.to_json_dict(), sort_keys=True, separators=(",", ":")
        )
        await session.commit()

        logger.info(
            "piano_roll_notes | job=%s | complete | events=%d notes=%d "
            "failures=%d secs=%.2f",
            job_id,
            n_events,
            len(rows),
            len(failures),
            compute_seconds,
        )
    except Exception as exc:
        logger.exception("piano_roll_notes job %s failed", job_id)
        failed = await session.get(PianoRollNotesJob, job_id)
        target = failed if failed is not None else job
        target.status = JobStatus.failed.value
        target.finished_at = datetime.now(timezone.utc)
        target.error_message = _truncate(str(exc), limit=2048)
        target.notes_path = None
        target.compute_seconds = time.monotonic() - started_wall
        await session.commit()


def _resolve_params(params_json: str) -> _ResolvedParams:
    raw: dict[str, Any] = {}
    if params_json:
        try:
            loaded = json.loads(params_json)
            if isinstance(loaded, dict):
                raw = loaded
        except json.JSONDecodeError:
            raw = {}

    cqt_raw = _section(raw, "cqt")
    peak_raw = _section(raw, "peak")
    tracker_raw = _section(raw, "tracker")
    harmonic_raw = _section(raw, "harmonic")
    midi_raw = _section(raw, "midi")
    velocity_raw = _section(raw, "velocity")
    audio_raw = _section(raw, "audio")

    return _ResolvedParams(
        cqt=CQTParams(
            target_sample_rate=int(cqt_raw.get("target_sample_rate", 22050)),
            hop_length=int(cqt_raw.get("hop_length", 256)),
            fmin=float(cqt_raw.get("fmin", 27.5)),
            n_bins=int(cqt_raw.get("n_bins", 264)),
            bins_per_octave=int(cqt_raw.get("bins_per_octave", 36)),
            filter_scale=float(cqt_raw.get("filter_scale", 1.0)),
        ),
        peak=PeakParams(
            k_noise=float(peak_raw.get("k_noise", 3.0)),
            top_k=int(peak_raw.get("top_k", 8)),
        ),
        tracker=TrackerParams(
            bin_tolerance=int(tracker_raw.get("bin_tolerance", 3)),
            miss_tolerance_frames=int(tracker_raw.get("miss_tolerance_frames", 2)),
            min_duration_s=float(tracker_raw.get("min_duration_s", 0.05)),
            amplitude_floor_percentile=float(
                tracker_raw.get("amplitude_floor_percentile", 5.0)
            ),
        ),
        harmonic=HarmonicParams(
            enabled=bool(harmonic_raw.get("enabled", True)),
            max_harmonic=int(harmonic_raw.get("max_harmonic", 8)),
            cents_tolerance=float(harmonic_raw.get("cents_tolerance", 50.0)),
        ),
        midi=MIDIQuantizeParams(
            min_pitch=int(midi_raw.get("min_pitch", 21)),
            max_pitch=int(midi_raw.get("max_pitch", 108)),
        ),
        velocity=_VelocityParams(
            floor_percentile=float(velocity_raw.get("floor_percentile", 5.0)),
            ceiling_percentile=float(velocity_raw.get("ceiling_percentile", 99.0)),
            floor=int(velocity_raw.get("floor", 1)),
            ceiling=int(velocity_raw.get("ceiling", 127)),
        ),
        audio=_AudioParams(
            pad_seconds=float(audio_raw.get("pad_seconds", 0.05)),
            min_event_duration_s=float(audio_raw.get("min_event_duration_s", 0.03)),
        ),
    )


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


def _truncate(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _summarize_failures(failures: list[tuple[str, str]]) -> Optional[str]:
    if not failures:
        return None
    head = failures[:_MAX_FAILURE_REPORT]
    lines = [f"{event_id}: {message}" for event_id, message in head]
    summary = "; ".join(lines)
    if len(failures) > _MAX_FAILURE_REPORT:
        summary += f"; ... and {len(failures) - _MAX_FAILURE_REPORT} more"
    return _truncate(summary, limit=2048)


# ---------- Extraction ------------------------------------------------------


async def _extract_notes(
    session: AsyncSession,
    encoder: EventEncoderJob,
    params: _ResolvedParams,
    settings: Settings,
) -> tuple[list[dict[str, Any]], int, list[tuple[str, str]]]:
    """Read events, run extraction, return parquet-ready rows."""
    seg_job = await session.get(EventSegmentationJob, encoder.event_segmentation_job_id)
    if seg_job is None:
        raise ValueError(
            f"event_segmentation_job not found: {encoder.event_segmentation_job_id}"
        )

    region_job = await session.get(RegionDetectionJob, seg_job.region_detection_job_id)
    if region_job is None:
        raise ValueError(
            f"region_detection_job not found: {seg_job.region_detection_job_id}"
        )

    events_path = (
        segmentation_job_dir(settings.storage_root, encoder.event_segmentation_job_id)
        / "events.parquet"
    )
    if not events_path.exists():
        raise FileNotFoundError(
            f"events.parquet not found for segmentation job {seg_job.id}"
        )
    events = read_events(events_path)

    token_map = _load_event_token_map(settings, encoder.id)

    audio_provider = await _build_audio_provider(
        session, settings, region_job, events, params.cqt.target_sample_rate
    )

    region_offset = float(region_job.start_timestamp or 0.0)
    frame_seconds = params.cqt.hop_length / params.cqt.target_sample_rate

    failures: list[tuple[str, str]] = []
    per_frame_magnitudes: list[float] = []

    @dataclass
    class _PendingNote:
        row: dict[str, Any]
        raw_magnitude: float

    pending: list[_PendingNote] = []

    for event in events:
        if event.end_sec - event.start_sec < params.audio.min_event_duration_s:
            continue
        try:
            audio = _slice_event_audio(
                event,
                audio_provider,
                target_sr=params.cqt.target_sample_rate,
                pad_seconds=params.audio.pad_seconds,
            )
            if audio.size == 0:
                continue
            log_mag = compute_event_cqt(
                audio, params.cqt.target_sample_rate, params=params.cqt
            )
            if log_mag.size > 0:
                per_frame_magnitudes.extend(np.max(log_mag, axis=0).tolist())
            peaks = pick_peaks_per_frame(log_mag, params=params.peak)
            tracks = build_tracks(peaks, cqt_params=params.cqt, params=params.tracker)
            tracks = label_harmonics(
                tracks, cqt_params=params.cqt, params=params.harmonic
            )

            notes = []
            for track in tracks:
                note = quantize_to_midi(
                    track, cqt_params=params.cqt, midi_params=params.midi
                )
                if note is not None:
                    notes.append(note)

            if not notes:
                continue

            event_start_utc = region_offset + float(event.start_sec)
            pad = params.audio.pad_seconds
            token_id = int(token_map.get(event.event_id, _UNKNOWN_TOKEN))
            for note in notes:
                start_offset = note.start_offset_s - pad
                start_utc = event_start_utc + start_offset
                pending.append(
                    _PendingNote(
                        row={
                            "event_id": event.event_id,
                            "event_token": token_id,
                            "partial_index": int(note.partial_index),
                            "midi_pitch": int(note.midi_pitch),
                            "start_utc": float(start_utc),
                            "start_offset_s": float(start_offset),
                            "duration_s": float(note.duration_s),
                            "velocity": 0,  # filled below
                            "peak_magnitude": float(note.peak_magnitude),
                            "track_id": int(note.track_id),
                        },
                        raw_magnitude=float(note.peak_magnitude),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            failures.append((event.event_id, _truncate(str(exc), limit=200)))
            logger.warning(
                "piano_roll_notes | event=%s failed: %s", event.event_id, exc
            )

    del frame_seconds  # currently unused; reserved for future use

    if not pending and failures and not events:
        raise RuntimeError("piano roll notes had no events to process")
    if not pending and failures:
        raise RuntimeError(
            f"piano roll notes had no successful events ({len(failures)} failures)"
        )

    velocity_map = _velocity_mapper(per_frame_magnitudes, params.velocity)
    for entry in pending:
        entry.row["velocity"] = velocity_map(entry.raw_magnitude)

    rows = [entry.row for entry in pending]
    rows.sort(key=lambda r: (r["start_utc"], r["midi_pitch"]))
    return rows, len(events), failures


def _slice_event_audio(
    event: Event,
    audio_provider,
    *,
    target_sr: int,
    pad_seconds: float,
) -> np.ndarray:
    buffer, buffer_start = audio_provider(event)
    pad_samples = int(round(pad_seconds * target_sr))
    raw_start = int(round((float(event.start_sec) - buffer_start) * target_sr))
    raw_end = int(round((float(event.end_sec) - buffer_start) * target_sr))
    start_idx = max(0, raw_start - pad_samples)
    end_idx = min(buffer.shape[0], raw_end + pad_samples)
    if end_idx <= start_idx:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(buffer[start_idx:end_idx], dtype=np.float32)


def _velocity_mapper(per_frame_magnitudes: list[float], params: _VelocityParams):
    if not per_frame_magnitudes:
        # Without a reference distribution, every note gets the floor.
        def _fallback(_: float) -> int:
            return params.floor

        return _fallback

    arr = np.asarray(per_frame_magnitudes, dtype=np.float64)
    lo = float(np.percentile(arr, params.floor_percentile))
    hi = float(np.percentile(arr, params.ceiling_percentile))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:

        def _fallback(_: float) -> int:
            return params.floor

        return _fallback

    span = hi - lo
    floor = params.floor
    ceiling = params.ceiling

    def _map(value: float) -> int:
        normalized = (value - lo) / span
        normalized = max(0.0, min(1.0, normalized))
        scaled = floor + normalized * (ceiling - floor)
        return int(round(scaled))

    return _map


# ---------- Token map and audio provider ------------------------------------


def _load_event_token_map(settings: Settings, encoder_job_id: str) -> dict[str, int]:
    """Return ``{event_id: token_id}`` for the largest available k value.

    Missing parquet → empty map (notes will carry ``event_token = -1``).
    """
    tokens_path = event_encoder_tokens_path(settings.storage_root, encoder_job_id)
    if not tokens_path.exists():
        return {}
    table = pq.read_table(tokens_path, columns=["k", "event_id", "token_id"])
    if table.num_rows == 0:
        return {}
    ks = table.column("k").to_pylist()
    if not ks:
        return {}
    target_k = max(int(k) for k in ks)
    out: dict[str, int] = {}
    for k, event_id, token_id in zip(
        ks, table.column("event_id").to_pylist(), table.column("token_id").to_pylist()
    ):
        if int(k) != target_k:
            continue
        out[str(event_id)] = int(token_id)
    return out


async def _build_audio_provider(
    session: AsyncSession,
    settings: Settings,
    region_job: RegionDetectionJob,
    events: list[Event],
    target_sr: int,
):
    if region_job.audio_file_id:
        result = await session.execute(
            select(AudioFile).where(AudioFile.id == region_job.audio_file_id)
        )
        audio_file = result.scalar_one_or_none()
        if audio_file is None:
            raise ValueError(f"AudioFile {region_job.audio_file_id} not found")
        return build_event_audio_loader(
            target_sr=target_sr,
            settings=settings,
            audio_file=audio_file,
            storage_root=settings.storage_root,
        )
    if region_job.hydrophone_id:
        return build_event_audio_loader(
            target_sr=target_sr,
            settings=settings,
            hydrophone_id=region_job.hydrophone_id,
            job_start_ts=region_job.start_timestamp or 0.0,
            job_end_ts=region_job.end_timestamp or 0.0,
            preload_events=events,
        )
    raise ValueError(f"region_detection_job {region_job.id} has no audio source")


__all__ = ["run_piano_roll_notes_job", "NOTES_SCHEMA"]
