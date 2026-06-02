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
from humpback.processing.note_extractor_v3 import (
    ContourFrame,
    ExtractNotesV3Params,
    HarmonicSearchParams,
    MidiRangeParams,
    NoteV3,
    SegmentationParams,
    STFTParams,
    SubharmonicParams,
    extract_notes_v3,
)
from humpback.processing.note_extractor_v4 import (
    ExtractNotesV4Params,
    HPSParams,
    extract_notes_v4,
)
from humpback.processing.note_extractor_v5 import (
    ExtractNotesV5Params,
    HarmonicViterbiParams,
    extract_notes_v5,
)
from humpback.processing.note_extractor_v6 import (
    DespikeParams,
    ExtractNotesV6Params,
    extract_notes_v6,
)
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
    event_encoder_note_contours_path,
    event_encoder_notes_path,
    event_encoder_ridges_path,
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


NOTES_V3_SCHEMA = pa.schema(
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
        pa.field("note_uid", pa.string(), nullable=False),
        pa.field("f0_track_id", pa.uint32(), nullable=False),
        pa.field("contour_frame_count", pa.uint32(), nullable=False),
    ]
)


NOTE_CONTOURS_V3_SCHEMA = pa.schema(
    [
        pa.field("note_uid", pa.string(), nullable=False),
        pa.field("frame_index", pa.uint32(), nullable=False),
        pa.field("time_offset_s", pa.float32(), nullable=False),
        pa.field("cents_from_pitch", pa.float32(), nullable=False),
        pa.field("harmonic_strength", pa.float32(), nullable=False),
        pa.field("subharmonic_octave", pa.uint8(), nullable=False),
    ]
)


_MAX_FAILURE_REPORT = 10
_UNKNOWN_TOKEN = -1
_V3_EXTRACTOR_VERSION = "v3"
_V4_EXTRACTOR_VERSION = "v4"
_V5_EXTRACTOR_VERSION = "v5"
_V6_EXTRACTOR_VERSION = "v6"
# Versions that produce per-frame contour sidecars + MPE-compatible
# parquet rows (note_uid + f0_track_id + contour_frame_count). v3-v6
# share the same parquet schemas; only the underlying F0 algorithm
# differs.
_RIDGE_AWARE_VERSIONS = frozenset(
    {
        _V3_EXTRACTOR_VERSION,
        _V4_EXTRACTOR_VERSION,
        _V5_EXTRACTOR_VERSION,
        _V6_EXTRACTOR_VERSION,
    }
)


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
    stft: STFTParams
    subharmonic: SubharmonicParams
    segmentation: SegmentationParams
    harmonic_v3: HarmonicSearchParams
    midi_v3: MidiRangeParams
    hps: HPSParams
    harmonic_viterbi: HarmonicViterbiParams
    despike: DespikeParams

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
            "stft": {
                "n_fft": self.stft.n_fft,
                "hop_length": self.stft.hop_length,
                "min_frequency_hz": self.stft.min_frequency_hz,
                "max_frequency_hz": self.stft.max_frequency_hz,
                "candidate_count": self.stft.candidate_count,
                "smoothness_penalty": self.stft.smoothness_penalty,
                "peak_prominence_ratio": self.stft.peak_prominence_ratio,
            },
            "subharmonic": {
                "k_sub": self.subharmonic.k_sub,
                "max_halvings": self.subharmonic.max_halvings,
                "smoothing_frames": self.subharmonic.smoothing_frames,
                "min_relative_log_magnitude": (
                    self.subharmonic.min_relative_log_magnitude
                ),
            },
            "segmentation": {
                "amplitude_floor_percentile": (
                    self.segmentation.amplitude_floor_percentile
                ),
                "min_break_frames": self.segmentation.min_break_frames,
                "min_note_frames": self.segmentation.min_note_frames,
            },
            "harmonic_v3": {
                "min_harmonic": self.harmonic_v3.min_harmonic,
                "max_harmonic": self.harmonic_v3.max_harmonic,
                "cents_tolerance": self.harmonic_v3.cents_tolerance,
                "min_break_frames": self.harmonic_v3.min_break_frames,
                "min_note_frames": self.harmonic_v3.min_note_frames,
            },
            "midi_v3": {
                "min_pitch": self.midi_v3.min_pitch,
                "max_pitch": self.midi_v3.max_pitch,
                "cents_clip": self.midi_v3.cents_clip,
            },
            "hps": {
                "n_harmonics": self.hps.n_harmonics,
                "cents_tolerance": self.hps.cents_tolerance,
                "k_noise": self.hps.k_noise,
                "candidate_divisors": list(self.hps.candidate_divisors),
                "smoothing_frames": self.hps.smoothing_frames,
                "low_band_penalty": self.hps.low_band_penalty,
                "low_band_threshold_hz": self.hps.low_band_threshold_hz,
                "low_band_min_harmonics": self.hps.low_band_min_harmonics,
                "high_band_min_harmonics": self.hps.high_band_min_harmonics,
                "min_above_floor": self.hps.min_above_floor,
                "max_harmonic_dynamic_range_log": (
                    self.hps.max_harmonic_dynamic_range_log
                ),
            },
            "harmonic_viterbi": {
                "n_harmonics": self.harmonic_viterbi.n_harmonics,
                "harmonic_weight": self.harmonic_viterbi.harmonic_weight,
                "f0_min_hz": self.harmonic_viterbi.f0_min_hz,
                "f0_max_hz": self.harmonic_viterbi.f0_max_hz,
                "cents_tolerance": self.harmonic_viterbi.cents_tolerance,
                "k_noise": self.harmonic_viterbi.k_noise,
                "tau_voicing": self.harmonic_viterbi.tau_voicing,
                "transition_lambda": self.harmonic_viterbi.transition_lambda,
                "voicing_transition_cost": (
                    self.harmonic_viterbi.voicing_transition_cost
                ),
                "min_harmonics_present": self.harmonic_viterbi.min_harmonics_present,
                "max_h1_below_strongest": (
                    self.harmonic_viterbi.max_h1_below_strongest
                ),
                "background_source": self.harmonic_viterbi.background_source,
                "background_percentile": (self.harmonic_viterbi.background_percentile),
                "background_min_frames": (self.harmonic_viterbi.background_min_frames),
                "background_min_pad_frames": (
                    self.harmonic_viterbi.background_min_pad_frames
                ),
            },
            "despike": {
                "enabled": self.despike.enabled,
                "max_slope_oct_per_s": self.despike.max_slope_oct_per_s,
                "max_spike_frames": self.despike.max_spike_frames,
                "max_trailing_trim_frames": self.despike.max_trailing_trim_frames,
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


def _cleanup_partial_parquet(path: Path) -> None:
    """Remove the canonical parquet (and any leftover tmp) after a job failure."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    for candidate in (path, tmp):
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "piano_roll_notes | could not remove partial artifact %s",
                candidate,
                exc_info=True,
            )


async def run_piano_roll_notes_job(
    session: AsyncSession,
    job: PianoRollNotesJob,
    settings: Settings,
) -> None:
    """Execute one Piano Roll Notes job end-to-end."""
    job_id = job.id
    started_wall = time.monotonic()
    started_dt = datetime.now(timezone.utc)

    notes_path: Optional[Path] = None
    contours_path: Optional[Path] = None
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

        params = _resolve_params(job.params_json, job.extractor_version)
        notes_path = event_encoder_notes_path(
            settings.storage_root, encoder.id, job.extractor_version
        )
        ensure_dir(event_encoder_dir(settings.storage_root, encoder.id))

        is_ridge_aware = job.extractor_version in _RIDGE_AWARE_VERSIONS
        extra_meta: dict[str, Any] = {}
        if is_ridge_aware:
            contours_path = event_encoder_note_contours_path(
                settings.storage_root, encoder.id, job.extractor_version
            )
            extractor = _RIDGE_AWARE_EXTRACTORS[job.extractor_version]
            (
                rows,
                contour_rows,
                n_events,
                failures,
                ridges_status,
            ) = await extractor(session, encoder, job.id, params, settings)
            # v3, v4, and v5 share schema; only ``subharmonic_octave``
            # semantics differ (v3 = octave halvings per ADR-069,
            # v4 = divisor − 1 per ADR-070, v5 = always 0 per ADR-071).
            notes_table = pa.Table.from_pylist(rows, schema=NOTES_V3_SCHEMA)
            contours_table = pa.Table.from_pylist(
                contour_rows, schema=NOTE_CONTOURS_V3_SCHEMA
            )
            _atomic_write_parquet(notes_table, notes_path)
            try:
                _atomic_write_parquet(contours_table, contours_path)
            except BaseException:
                _cleanup_partial_parquet(notes_path)
                raise
            extra_meta = {
                "contours_path": str(contours_path),
                "ridges_path": ridges_status,
                "n_contour_frames": len(contour_rows),
            }
        else:
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
        params_dict = params.to_json_dict()
        params_dict.update(extra_meta)
        target.params_json = json.dumps(
            params_dict, sort_keys=True, separators=(",", ":")
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
        if notes_path is not None:
            _cleanup_partial_parquet(notes_path)
        if contours_path is not None:
            _cleanup_partial_parquet(contours_path)
        failed = await session.get(PianoRollNotesJob, job_id)
        target = failed if failed is not None else job
        target.status = JobStatus.failed.value
        target.finished_at = datetime.now(timezone.utc)
        target.error_message = _truncate(str(exc), limit=2048)
        target.notes_path = None
        target.compute_seconds = time.monotonic() - started_wall
        await session.commit()


def _resolve_params(params_json: str, extractor_version: str = "") -> _ResolvedParams:
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
    stft_raw = _section(raw, "stft")
    subharmonic_raw = _section(raw, "subharmonic")
    segmentation_raw = _section(raw, "segmentation")
    harmonic_v3_raw = _section(raw, "harmonic_v3")
    midi_v3_raw = _section(raw, "midi_v3")
    hps_raw = _section(raw, "hps")
    harmonic_viterbi_raw = _section(raw, "harmonic_viterbi")
    despike_raw = _section(raw, "despike")
    # v4-v6 lower the STFT ridge band floor to 30 Hz (ADR-070 §4.3,
    # ADR-071 §5.1). v3 keeps the historical 100 Hz floor so re-running
    # an old row reproduces v3's bytes.
    default_min_freq = (
        30.0
        if extractor_version
        in (_V4_EXTRACTOR_VERSION, _V5_EXTRACTOR_VERSION, _V6_EXTRACTOR_VERSION)
        else 100.0
    )
    # The harmonic-Viterbi voicing oracle (v5/v6) produces shorter, more
    # frequent unvoiced gaps than v3/v4's STFT-ridge prominence gate,
    # so the v3/v4 default fragments coherent contours. ADR-071 raises
    # the gap-bridging threshold; v6 inherits the v5 decode unchanged.
    _viterbi_versions = (_V5_EXTRACTOR_VERSION, _V6_EXTRACTOR_VERSION)
    default_min_break_frames = 6 if extractor_version in _viterbi_versions else 3
    # v5/v6 bump the worker audio pad to 0.25 s so the harmonic-Viterbi
    # background subtractor has enough pad frames to estimate the
    # per-bin chronic-noise baseline (ADR-071 §6).
    default_pad_seconds = 0.25 if extractor_version in _viterbi_versions else 0.05

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
            pad_seconds=float(audio_raw.get("pad_seconds", default_pad_seconds)),
            min_event_duration_s=float(audio_raw.get("min_event_duration_s", 0.03)),
        ),
        stft=STFTParams(
            n_fft=int(stft_raw.get("n_fft", 1024)),
            hop_length=int(stft_raw.get("hop_length", 512)),
            min_frequency_hz=float(stft_raw.get("min_frequency_hz", default_min_freq)),
            max_frequency_hz=float(stft_raw.get("max_frequency_hz", 6000.0)),
            candidate_count=int(stft_raw.get("candidate_count", 5)),
            smoothness_penalty=float(stft_raw.get("smoothness_penalty", 8.0)),
            peak_prominence_ratio=float(stft_raw.get("peak_prominence_ratio", 0.0)),
        ),
        subharmonic=SubharmonicParams(
            k_sub=float(subharmonic_raw.get("k_sub", 2.0)),
            max_halvings=int(subharmonic_raw.get("max_halvings", 3)),
            smoothing_frames=int(subharmonic_raw.get("smoothing_frames", 5)),
            min_relative_log_magnitude=float(
                subharmonic_raw.get("min_relative_log_magnitude", -2.5)
            ),
        ),
        segmentation=SegmentationParams(
            amplitude_floor_percentile=float(
                segmentation_raw.get("amplitude_floor_percentile", 5.0)
            ),
            min_break_frames=int(
                segmentation_raw.get("min_break_frames", default_min_break_frames)
            ),
            min_note_frames=int(segmentation_raw.get("min_note_frames", 3)),
        ),
        harmonic_v3=HarmonicSearchParams(
            min_harmonic=int(harmonic_v3_raw.get("min_harmonic", 2)),
            max_harmonic=int(harmonic_v3_raw.get("max_harmonic", 16)),
            cents_tolerance=float(harmonic_v3_raw.get("cents_tolerance", 75.0)),
            min_break_frames=int(harmonic_v3_raw.get("min_break_frames", 3)),
            min_note_frames=int(harmonic_v3_raw.get("min_note_frames", 3)),
        ),
        midi_v3=MidiRangeParams(
            min_pitch=int(midi_v3_raw.get("min_pitch", 12)),
            max_pitch=int(midi_v3_raw.get("max_pitch", 120)),
            cents_clip=float(midi_v3_raw.get("cents_clip", 9600.0)),
        ),
        hps=HPSParams(
            n_harmonics=int(hps_raw.get("n_harmonics", 8)),
            cents_tolerance=float(hps_raw.get("cents_tolerance", 50.0)),
            k_noise=float(hps_raw.get("k_noise", 2.0)),
            candidate_divisors=tuple(
                int(d) for d in hps_raw.get("candidate_divisors", (1, 2, 3, 4, 5, 6))
            ),
            smoothing_frames=int(hps_raw.get("smoothing_frames", 5)),
            low_band_penalty=float(hps_raw.get("low_band_penalty", 0.5)),
            low_band_threshold_hz=float(hps_raw.get("low_band_threshold_hz", 100.0)),
            low_band_min_harmonics=int(hps_raw.get("low_band_min_harmonics", 3)),
            high_band_min_harmonics=int(hps_raw.get("high_band_min_harmonics", 2)),
            min_above_floor=float(hps_raw.get("min_above_floor", 1.0)),
            max_harmonic_dynamic_range_log=float(
                hps_raw.get("max_harmonic_dynamic_range_log", 3.0)
            ),
        ),
        harmonic_viterbi=HarmonicViterbiParams(
            n_harmonics=int(harmonic_viterbi_raw.get("n_harmonics", 4)),
            harmonic_weight=str(
                harmonic_viterbi_raw.get("harmonic_weight", "inv_sqrt_k")
            ),  # type: ignore[arg-type]
            f0_min_hz=float(harmonic_viterbi_raw.get("f0_min_hz", 30.0)),
            f0_max_hz=float(harmonic_viterbi_raw.get("f0_max_hz", 600.0)),
            cents_tolerance=float(harmonic_viterbi_raw.get("cents_tolerance", 50.0)),
            k_noise=float(harmonic_viterbi_raw.get("k_noise", 2.0)),
            tau_voicing=float(harmonic_viterbi_raw.get("tau_voicing", 3.0)),
            transition_lambda=float(harmonic_viterbi_raw.get("transition_lambda", 6.0)),
            voicing_transition_cost=float(
                harmonic_viterbi_raw.get("voicing_transition_cost", 1.0)
            ),
            min_harmonics_present=int(
                harmonic_viterbi_raw.get("min_harmonics_present", 2)
            ),
            max_h1_below_strongest=float(
                harmonic_viterbi_raw.get("max_h1_below_strongest", 2.5)
            ),
            background_source=str(harmonic_viterbi_raw.get("background_source", "pad")),  # type: ignore[arg-type]
            background_percentile=float(
                harmonic_viterbi_raw.get("background_percentile", 25.0)
            ),
            background_min_frames=int(
                harmonic_viterbi_raw.get("background_min_frames", 20)
            ),
            background_min_pad_frames=int(
                harmonic_viterbi_raw.get("background_min_pad_frames", 8)
            ),
        ),
        despike=DespikeParams(
            enabled=bool(despike_raw.get("enabled", True)),
            max_slope_oct_per_s=float(despike_raw.get("max_slope_oct_per_s", 6.0)),
            max_spike_frames=int(despike_raw.get("max_spike_frames", 12)),
            max_trailing_trim_frames=int(
                despike_raw.get("max_trailing_trim_frames", 4)
            ),
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
                "piano_roll_notes | event=%s failed",
                event.event_id,
                exc_info=True,
            )

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


async def _extract_notes_v3(
    session: AsyncSession,
    encoder: EventEncoderJob,
    job_id: str,
    params: _ResolvedParams,
    settings: Settings,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    list[tuple[str, str]],
    str,
]:
    """Run the v3 ridge-aware extractor across an encoder job's events.

    Returns ``(note_rows, contour_rows, n_events, failures, ridges_status)``
    where ``ridges_status`` is the absolute path of the consumed ridge
    sidecar or the literal string ``"absent"`` when the worker fell back
    to in-process recomputation (recorded in ``params_json`` per spec
    §6.4).
    """
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

    ridges_path = event_encoder_ridges_path(
        settings.storage_root, encoder.id, encoder.tokenizer_version
    )
    sidecar_by_event = _load_ridge_sidecar(ridges_path)
    ridges_status = str(ridges_path) if sidecar_by_event is not None else "absent"

    @dataclass
    class _PendingNote:
        row: dict[str, Any]
        raw_magnitude: float

    pending: list[_PendingNote] = []
    contour_rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

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
            extract_params = ExtractNotesV3Params(
                job_id=job_id,
                event_id=event.event_id,
                event_start_utc=region_offset + float(event.start_sec),
                pad_seconds=params.audio.pad_seconds,
                cqt=params.cqt,
                stft=params.stft,
                subharmonic=params.subharmonic,
                segmentation=params.segmentation,
                harmonic=params.harmonic_v3,
                midi=params.midi_v3,
            )
            sidecar = (
                sidecar_by_event.get(event.event_id)
                if sidecar_by_event is not None
                else None
            )
            result = extract_notes_v3(
                audio,
                params.cqt.target_sample_rate,
                params=extract_params,
                ridge_sidecar_rows=sidecar,
            )
            if not result.notes:
                continue

            token_id = int(token_map.get(event.event_id, _UNKNOWN_TOKEN))
            for note in result.notes:
                pending.append(
                    _PendingNote(
                        row=_note_v3_row(note, event.event_id, token_id),
                        raw_magnitude=float(note.peak_magnitude),
                    )
                )
            contour_rows.extend(_contour_v3_row(c) for c in result.contours)
        except Exception as exc:  # noqa: BLE001
            failures.append((event.event_id, _truncate(str(exc), limit=200)))
            logger.warning(
                "piano_roll_notes_v3 | event=%s failed",
                event.event_id,
                exc_info=True,
            )

    if not pending and failures:
        raise RuntimeError(
            f"piano roll notes v3 had no successful events ({len(failures)} failures)"
        )

    velocity_map = _velocity_mapper(
        [entry.raw_magnitude for entry in pending], params.velocity
    )
    for entry in pending:
        entry.row["velocity"] = velocity_map(entry.raw_magnitude)

    rows = [entry.row for entry in pending]
    rows.sort(key=lambda r: (r["start_utc"], r["midi_pitch"]))
    contour_rows.sort(key=lambda r: (r["note_uid"], r["frame_index"]))
    return rows, contour_rows, len(events), failures, ridges_status


async def _extract_notes_v4(
    session: AsyncSession,
    encoder: EventEncoderJob,
    job_id: str,
    params: _ResolvedParams,
    settings: Settings,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    list[tuple[str, str]],
    str,
]:
    """Run the v4 HPS extractor across an encoder job's events (ADR-070).

    Mirrors :func:`_extract_notes_v3` end-to-end; the only differences
    are the per-event ``extract_notes_v4`` call and the
    ``ExtractNotesV4Params`` shape (``hps`` instead of ``subharmonic``).
    """
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

    ridges_path = event_encoder_ridges_path(
        settings.storage_root, encoder.id, encoder.tokenizer_version
    )
    sidecar_by_event = _load_ridge_sidecar(ridges_path)
    ridges_status = str(ridges_path) if sidecar_by_event is not None else "absent"

    @dataclass
    class _PendingNote:
        row: dict[str, Any]
        raw_magnitude: float

    pending: list[_PendingNote] = []
    contour_rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

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
            extract_params = ExtractNotesV4Params(
                job_id=job_id,
                event_id=event.event_id,
                event_start_utc=region_offset + float(event.start_sec),
                pad_seconds=params.audio.pad_seconds,
                cqt=params.cqt,
                stft=params.stft,
                hps=params.hps,
                segmentation=params.segmentation,
                harmonic=params.harmonic_v3,
                midi=params.midi_v3,
            )
            sidecar = (
                sidecar_by_event.get(event.event_id)
                if sidecar_by_event is not None
                else None
            )
            result = extract_notes_v4(
                audio,
                params.cqt.target_sample_rate,
                params=extract_params,
                ridge_sidecar_rows=sidecar,
            )
            if not result.notes:
                continue

            token_id = int(token_map.get(event.event_id, _UNKNOWN_TOKEN))
            for note in result.notes:
                pending.append(
                    _PendingNote(
                        row=_note_v3_row(note, event.event_id, token_id),
                        raw_magnitude=float(note.peak_magnitude),
                    )
                )
            contour_rows.extend(_contour_v3_row(c) for c in result.contours)
        except Exception as exc:  # noqa: BLE001
            failures.append((event.event_id, _truncate(str(exc), limit=200)))
            logger.warning(
                "piano_roll_notes_v4 | event=%s failed",
                event.event_id,
                exc_info=True,
            )

    if not pending and failures:
        raise RuntimeError(
            f"piano roll notes v4 had no successful events ({len(failures)} failures)"
        )

    velocity_map = _velocity_mapper(
        [entry.raw_magnitude for entry in pending], params.velocity
    )
    for entry in pending:
        entry.row["velocity"] = velocity_map(entry.raw_magnitude)

    rows = [entry.row for entry in pending]
    rows.sort(key=lambda r: (r["start_utc"], r["midi_pitch"]))
    contour_rows.sort(key=lambda r: (r["note_uid"], r["frame_index"]))
    return rows, contour_rows, len(events), failures, ridges_status


async def _extract_notes_v5(
    session: AsyncSession,
    encoder: EventEncoderJob,
    job_id: str,
    params: _ResolvedParams,
    settings: Settings,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    list[tuple[str, str]],
    str,
]:
    """Run the v5 harmonic-Viterbi extractor across an encoder job's events (ADR-071).

    Mirrors :func:`_extract_notes_v4` end-to-end; the only differences
    are the per-event ``extract_notes_v5`` call and the
    ``ExtractNotesV5Params`` shape (``harmonic_viterbi`` instead of
    ``hps``). The ridge sidecar is loaded for parity with v3/v4 but
    not consumed by the v5 extractor.
    """
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

    ridges_path = event_encoder_ridges_path(
        settings.storage_root, encoder.id, encoder.tokenizer_version
    )
    sidecar_by_event = _load_ridge_sidecar(ridges_path)
    ridges_status = str(ridges_path) if sidecar_by_event is not None else "absent"

    @dataclass
    class _PendingNote:
        row: dict[str, Any]
        raw_magnitude: float

    pending: list[_PendingNote] = []
    contour_rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

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
            extract_params = ExtractNotesV5Params(
                job_id=job_id,
                event_id=event.event_id,
                event_start_utc=region_offset + float(event.start_sec),
                pad_seconds=params.audio.pad_seconds,
                cqt=params.cqt,
                stft=params.stft,
                harmonic_viterbi=params.harmonic_viterbi,
                segmentation=params.segmentation,
                harmonic=params.harmonic_v3,
                midi=params.midi_v3,
            )
            sidecar = (
                sidecar_by_event.get(event.event_id)
                if sidecar_by_event is not None
                else None
            )
            result = extract_notes_v5(
                audio,
                params.cqt.target_sample_rate,
                params=extract_params,
                ridge_sidecar_rows=sidecar,
            )
            if not result.notes:
                continue

            token_id = int(token_map.get(event.event_id, _UNKNOWN_TOKEN))
            for note in result.notes:
                pending.append(
                    _PendingNote(
                        row=_note_v3_row(note, event.event_id, token_id),
                        raw_magnitude=float(note.peak_magnitude),
                    )
                )
            contour_rows.extend(_contour_v3_row(c) for c in result.contours)
        except Exception as exc:  # noqa: BLE001
            failures.append((event.event_id, _truncate(str(exc), limit=200)))
            logger.warning(
                "piano_roll_notes_v5 | event=%s failed",
                event.event_id,
                exc_info=True,
            )

    if not pending and failures:
        raise RuntimeError(
            f"piano roll notes v5 had no successful events ({len(failures)} failures)"
        )

    velocity_map = _velocity_mapper(
        [entry.raw_magnitude for entry in pending], params.velocity
    )
    for entry in pending:
        entry.row["velocity"] = velocity_map(entry.raw_magnitude)

    rows = [entry.row for entry in pending]
    rows.sort(key=lambda r: (r["start_utc"], r["midi_pitch"]))
    contour_rows.sort(key=lambda r: (r["note_uid"], r["frame_index"]))
    return rows, contour_rows, len(events), failures, ridges_status


async def _extract_notes_v6(
    session: AsyncSession,
    encoder: EventEncoderJob,
    job_id: str,
    params: _ResolvedParams,
    settings: Settings,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    list[tuple[str, str]],
    str,
]:
    """Run the v6 de-spiked harmonic-Viterbi extractor across a job's events (ADR-072).

    Mirrors :func:`_extract_notes_v5` end-to-end; the only differences are
    the per-event ``extract_notes_v6`` call and the ``ExtractNotesV6Params``
    shape (a ``despike`` sub-param on top of the v5 fields). The ridge
    sidecar is loaded for parity but not consumed.
    """
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

    ridges_path = event_encoder_ridges_path(
        settings.storage_root, encoder.id, encoder.tokenizer_version
    )
    sidecar_by_event = _load_ridge_sidecar(ridges_path)
    ridges_status = str(ridges_path) if sidecar_by_event is not None else "absent"

    @dataclass
    class _PendingNote:
        row: dict[str, Any]
        raw_magnitude: float

    pending: list[_PendingNote] = []
    contour_rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []

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
            extract_params = ExtractNotesV6Params(
                job_id=job_id,
                event_id=event.event_id,
                event_start_utc=region_offset + float(event.start_sec),
                pad_seconds=params.audio.pad_seconds,
                cqt=params.cqt,
                stft=params.stft,
                harmonic_viterbi=params.harmonic_viterbi,
                segmentation=params.segmentation,
                harmonic=params.harmonic_v3,
                midi=params.midi_v3,
                despike=params.despike,
            )
            sidecar = (
                sidecar_by_event.get(event.event_id)
                if sidecar_by_event is not None
                else None
            )
            result = extract_notes_v6(
                audio,
                params.cqt.target_sample_rate,
                params=extract_params,
                ridge_sidecar_rows=sidecar,
            )
            if not result.notes:
                continue

            token_id = int(token_map.get(event.event_id, _UNKNOWN_TOKEN))
            for note in result.notes:
                pending.append(
                    _PendingNote(
                        row=_note_v3_row(note, event.event_id, token_id),
                        raw_magnitude=float(note.peak_magnitude),
                    )
                )
            contour_rows.extend(_contour_v3_row(c) for c in result.contours)
        except Exception as exc:  # noqa: BLE001
            failures.append((event.event_id, _truncate(str(exc), limit=200)))
            logger.warning(
                "piano_roll_notes_v6 | event=%s failed",
                event.event_id,
                exc_info=True,
            )

    if not pending and failures:
        raise RuntimeError(
            f"piano roll notes v6 had no successful events ({len(failures)} failures)"
        )

    velocity_map = _velocity_mapper(
        [entry.raw_magnitude for entry in pending], params.velocity
    )
    for entry in pending:
        entry.row["velocity"] = velocity_map(entry.raw_magnitude)

    rows = [entry.row for entry in pending]
    rows.sort(key=lambda r: (r["start_utc"], r["midi_pitch"]))
    contour_rows.sort(key=lambda r: (r["note_uid"], r["frame_index"]))
    return rows, contour_rows, len(events), failures, ridges_status


_RIDGE_AWARE_EXTRACTORS = {
    _V3_EXTRACTOR_VERSION: _extract_notes_v3,
    _V4_EXTRACTOR_VERSION: _extract_notes_v4,
    _V5_EXTRACTOR_VERSION: _extract_notes_v5,
    _V6_EXTRACTOR_VERSION: _extract_notes_v6,
}


def _note_v3_row(note: NoteV3, event_id: str, token_id: int) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_token": token_id,
        "partial_index": int(note.partial_index),
        "midi_pitch": int(note.midi_pitch),
        "start_utc": float(note.start_utc),
        "start_offset_s": float(note.start_offset_s),
        "duration_s": float(note.duration_s),
        "velocity": 0,  # filled in after job-level calibration
        "peak_magnitude": float(note.peak_magnitude),
        "track_id": int(note.track_id),
        "note_uid": note.note_uid,
        "f0_track_id": int(note.f0_track_id),
        "contour_frame_count": int(note.contour_frame_count),
    }


def _contour_v3_row(contour: ContourFrame) -> dict[str, Any]:
    return {
        "note_uid": contour.note_uid,
        "frame_index": int(contour.frame_index),
        "time_offset_s": float(contour.time_offset_s),
        "cents_from_pitch": float(contour.cents_from_pitch),
        "harmonic_strength": float(contour.harmonic_strength),
        "subharmonic_octave": int(contour.subharmonic_octave),
    }


def _load_ridge_sidecar(
    ridges_path: Path,
) -> Optional[dict[str, list[dict[str, Any]]]]:
    """Group ``event_ridges_*.parquet`` rows by ``event_id``.

    Returns ``None`` when the sidecar is missing so callers can pass
    ``ridge_sidecar_rows=None`` to ``extract_notes_v3`` and let the
    extractor recompute the ridge in-process.
    """
    if not ridges_path.exists():
        return None
    table = pq.read_table(
        ridges_path,
        columns=[
            "event_id",
            "frame_index",
            "frame_time_offset_s",
            "log_frequency",
            "strength",
            "energy_ratio",
        ],
    )
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in table.to_pylist():
        by_event.setdefault(str(row["event_id"]), []).append(
            {
                "frame_index": int(row["frame_index"]),
                "frame_time_offset_s": float(row["frame_time_offset_s"]),
                "log_frequency": float(row["log_frequency"]),
                "strength": float(row["strength"]),
                "energy_ratio": float(row["energy_ratio"]),
            }
        )
    for rows in by_event.values():
        rows.sort(key=lambda r: r["frame_index"])
    return by_event


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
    k_values = pq.read_table(tokens_path, columns=["k"]).column("k").to_pylist()
    if not k_values:
        return {}
    target_k = max(int(k) for k in k_values)
    filtered = pq.read_table(
        tokens_path,
        columns=["event_id", "token_id"],
        filters=[("k", "==", target_k)],
    )
    out: dict[str, int] = {}
    for event_id, token_id in zip(
        filtered.column("event_id").to_pylist(),
        filtered.column("token_id").to_pylist(),
    ):
        out.setdefault(str(event_id), int(token_id))
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


__all__ = [
    "run_piano_roll_notes_job",
    "NOTES_SCHEMA",
    "NOTES_V3_SCHEMA",
    "NOTE_CONTOURS_V3_SCHEMA",
]
