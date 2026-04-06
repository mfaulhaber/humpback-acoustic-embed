"""Detect and correct abrupt mic gain changes in hydrophone recordings.

Walks audio in 1-second RMS windows, identifies sustained high-gain segments,
and provides an attenuation function to normalize them back to the median level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_SILENCE_FLOOR_DB = -80.0
_CROSSFADE_SEC = 0.05  # 50ms cosine crossfade at segment boundaries
_PROFILE_SR = 4000  # Coarse sample rate for gain analysis


@dataclass
class GainSegment:
    """A contiguous region where the recording gain is elevated."""

    start_sec: float
    end_sec: float
    attenuation_db: float


@dataclass
class GainProfile:
    """Per-job gain normalization profile."""

    segments: list[GainSegment] = field(default_factory=list)
    global_median_rms_db: float = _SILENCE_FLOOR_DB

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [
                {
                    "start_sec": s.start_sec,
                    "end_sec": s.end_sec,
                    "attenuation_db": s.attenuation_db,
                }
                for s in self.segments
            ],
            "global_median_rms_db": self.global_median_rms_db,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GainProfile:
        segments = [
            GainSegment(
                start_sec=s["start_sec"],
                end_sec=s["end_sec"],
                attenuation_db=s["attenuation_db"],
            )
            for s in data.get("segments", [])
        ]
        return cls(
            segments=segments,
            global_median_rms_db=data.get("global_median_rms_db", _SILENCE_FLOOR_DB),
        )


_SCAN_MAX_PROBES = 200  # Max probes in coarse scan pass
_PROBE_DURATION = 10.0  # Seconds of audio per probe
_REFINE_STEP_SEC = 5.0  # Resolution for boundary refinement
_SPLIT_DROP_DB = 6.0  # Internal RMS drop to split a segment


def _rms_db_for_chunk(audio: np.ndarray) -> float:
    """Compute RMS in dB for an audio chunk."""
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms > 0:
        return 20.0 * np.log10(rms)
    return _SILENCE_FLOOR_DB


def _resolve_rms_db(
    audio_resolver: Callable[[float, float, int], np.ndarray],
    start_epoch: float,
    duration_sec: float,
) -> float:
    """Resolve audio and return its RMS in dB, or silence floor on failure."""
    try:
        audio = audio_resolver(start_epoch, duration_sec, _PROFILE_SR)
        if len(audio) == 0:
            return _SILENCE_FLOOR_DB
        return _rms_db_for_chunk(audio)
    except Exception:
        return _SILENCE_FLOOR_DB


def _split_on_internal_drops(
    rms_values: np.ndarray,
    flagged_start: int,
    flagged_end: int,
    drop_db: float = _SPLIT_DROP_DB,
) -> list[tuple[int, int]]:
    """Split a flagged run into sub-segments when the RMS drops internally.

    Detects points where the RMS drops by >= drop_db from the running level
    of the current sub-segment, creating a split.
    """
    if flagged_end - flagged_start <= 1:
        return [(flagged_start, flagged_end)]

    sub_segments: list[tuple[int, int]] = []
    sub_start = flagged_start
    # Running median for the current sub-segment
    sub_values: list[float] = [rms_values[flagged_start]]

    for i in range(flagged_start + 1, flagged_end):
        current_rms = rms_values[i]
        sub_median = float(np.median(sub_values))

        # If the current value drops significantly below the sub-segment level,
        # close the current sub-segment and start a new one
        if sub_median - current_rms >= drop_db:
            sub_segments.append((sub_start, i))
            sub_start = i
            sub_values = [current_rms]
        # Also split if the current value jumps significantly above
        elif current_rms - sub_median >= drop_db and len(sub_values) >= 2:
            sub_segments.append((sub_start, i))
            sub_start = i
            sub_values = [current_rms]
        else:
            sub_values.append(current_rms)

    sub_segments.append((sub_start, flagged_end))
    return sub_segments


def compute_gain_profile(
    audio_resolver: Callable[[float, float, int], np.ndarray],
    job_start: float,
    job_end: float,
    threshold_db: float = 6.0,
    min_duration_sec: float = 5.0,
) -> GainProfile:
    """Analyze a job's audio and build a gain normalization profile.

    Uses a two-pass approach for efficiency on long jobs:
    1. Coarse scan: sample up to _SCAN_MAX_PROBES evenly-spaced probes
    2. Refine: for detected transitions, probe at finer resolution to
       find precise segment boundaries

    Parameters
    ----------
    audio_resolver:
        Callable(start_epoch, duration_sec, target_sr) -> np.ndarray float32.
    job_start:
        Job start as epoch seconds.
    job_end:
        Job end as epoch seconds.
    threshold_db:
        RMS jump above median to flag as a gain change.
    min_duration_sec:
        Minimum sustained duration to count as a gain segment (filters transients).

    Returns
    -------
    GainProfile with detected segments and the global median RMS.
    """
    duration = job_end - job_start
    if duration <= 0:
        return GainProfile()

    # For short jobs (< 10 min), use direct 1-second scan
    if duration <= 600:
        return _compute_gain_profile_direct(
            audio_resolver, job_start, job_end, threshold_db, min_duration_sec
        )

    # --- Pass 1: coarse scan with evenly-spaced probes ---
    n_probes = min(_SCAN_MAX_PROBES, int(duration / _PROBE_DURATION))
    if n_probes < 2:
        return GainProfile()

    probe_step = duration / n_probes
    probe_times = [job_start + i * probe_step for i in range(n_probes)]
    probe_rms = np.array(
        [_resolve_rms_db(audio_resolver, t, _PROBE_DURATION) for t in probe_times]
    )

    non_silent = probe_rms > _SILENCE_FLOOR_DB
    if not np.any(non_silent):
        return GainProfile()

    global_median = float(np.median(probe_rms[non_silent]))

    # Flag probes above threshold
    flagged = (probe_rms > global_median + threshold_db) & non_silent

    if not np.any(flagged):
        return GainProfile(global_median_rms_db=global_median)

    # Group consecutive flagged probes into coarse segments, splitting on
    # large internal RMS drops so different gain levels get separate segments
    coarse_runs: list[tuple[int, int]] = []
    i = 0
    while i < n_probes:
        if not flagged[i]:
            i += 1
            continue
        seg_start = i
        while i < n_probes and flagged[i]:
            i += 1
        coarse_runs.append((seg_start, i))

    # --- Pass 2: refine outer boundaries and emit segments ---
    # For each coarse run, split on internal drops then refine only the
    # leading edge of the first sub-segment and trailing edge of the last.
    # Internal boundaries use probe times directly (no refinement).
    segments: list[GainSegment] = []
    for run_start, run_end in coarse_runs:
        sub_segs = _split_on_internal_drops(probe_rms, run_start, run_end)

        # Refine leading edge of the first sub-segment
        first_rough_start = probe_times[sub_segs[0][0]]
        refined_run_start = first_rough_start
        t = first_rough_start - _REFINE_STEP_SEC
        while t >= job_start:
            rms = _resolve_rms_db(audio_resolver, t, _REFINE_STEP_SEC)
            if rms <= global_median + threshold_db or rms <= _SILENCE_FLOOR_DB:
                break
            refined_run_start = t
            t -= _REFINE_STEP_SEC

        # Refine trailing edge of the last sub-segment
        last_end_idx = min(sub_segs[-1][1], n_probes - 1)
        last_rough_end = probe_times[last_end_idx] + _PROBE_DURATION
        refined_run_end = last_rough_end
        t = last_rough_end
        while t < job_end:
            rms = _resolve_rms_db(audio_resolver, t, _REFINE_STEP_SEC)
            if rms <= global_median + threshold_db or rms <= _SILENCE_FLOOR_DB:
                break
            refined_run_end = t + _REFINE_STEP_SEC
            t += _REFINE_STEP_SEC
        refined_run_end = min(refined_run_end, job_end)

        for sub_idx, (sub_start_idx, sub_end_idx) in enumerate(sub_segs):
            # Use refined boundary for first/last, probe boundary for internal
            if sub_idx == 0:
                seg_start_sec = refined_run_start
            else:
                seg_start_sec = probe_times[sub_start_idx]

            if sub_idx == len(sub_segs) - 1:
                seg_end_sec = refined_run_end
            else:
                seg_end_sec = probe_times[sub_segs[sub_idx + 1][0]]

            seg_duration = seg_end_sec - seg_start_sec
            if seg_duration < min_duration_sec:
                continue

            seg_rms_values = probe_rms[sub_start_idx:sub_end_idx]
            seg_non_silent = seg_rms_values[seg_rms_values > _SILENCE_FLOOR_DB]
            if len(seg_non_silent) == 0:
                continue
            seg_median_rms = float(np.median(seg_non_silent))
            attenuation_db = seg_median_rms - global_median

            segments.append(
                GainSegment(
                    start_sec=seg_start_sec,
                    end_sec=seg_end_sec,
                    attenuation_db=attenuation_db,
                )
            )

    # --- Pass 3: tighten boundaries at 1-second resolution ---
    segments = _tighten_segment_boundaries(
        segments,
        audio_resolver,
        global_median + threshold_db,
        min_duration_sec,
        global_median,
    )

    profile = GainProfile(segments=segments, global_median_rms_db=global_median)
    if segments:
        logger.info(
            "Gain profile: %d segment(s), median=%.1f dB, max_atten=%.1f dB",
            len(segments),
            global_median,
            max(s.attenuation_db for s in segments),
        )
    return profile


def _tighten_segment_boundaries(
    segments: list[GainSegment],
    audio_resolver: Callable[[float, float, int], np.ndarray],
    rms_threshold_db: float,
    min_duration_sec: float,
    global_median_db: float,
) -> list[GainSegment]:
    """Tighten segment boundaries at 1-second resolution.

    Advances each segment's start and retreats its end until the audio is
    actually elevated enough to warrant the segment's attenuation, removing
    quiet tails from imprecise coarse boundaries.
    """
    tightened: list[GainSegment] = []
    for seg in segments:
        # The audio must be at least halfway (in dB) between the median and
        # the segment's expected level to be considered part of this segment.
        # This prevents a 33 dB-attenuation segment from covering audio that's
        # only 7 dB above median.
        seg_level_threshold = global_median_db + seg.attenuation_db / 2.0

        # Advance start until audio is above the segment's own level
        start = seg.start_sec
        while start < seg.end_sec - min_duration_sec:
            rms = _resolve_rms_db(audio_resolver, start, 1.0)
            if rms > seg_level_threshold:
                break
            start += 1.0

        # Retreat end until audio is above the segment's own level
        end = seg.end_sec
        while end > start + min_duration_sec:
            rms = _resolve_rms_db(audio_resolver, end - 1.0, 1.0)
            if rms > seg_level_threshold:
                break
            end -= 1.0

        if end - start >= min_duration_sec:
            tightened.append(
                GainSegment(
                    start_sec=start,
                    end_sec=end,
                    attenuation_db=seg.attenuation_db,
                )
            )
    return tightened


def _compute_gain_profile_direct(
    audio_resolver: Callable[[float, float, int], np.ndarray],
    job_start: float,
    job_end: float,
    threshold_db: float,
    min_duration_sec: float,
) -> GainProfile:
    """Direct 1-second scan for short jobs (under 10 minutes)."""
    duration = job_end - job_start
    window_sec = 1.0
    n_windows = int(duration / window_sec)
    if n_windows == 0:
        return GainProfile()

    rms_db_values = np.full(n_windows, _SILENCE_FLOOR_DB, dtype=np.float64)

    chunk_windows = 60
    for chunk_start_idx in range(0, n_windows, chunk_windows):
        chunk_end_idx = min(chunk_start_idx + chunk_windows, n_windows)
        chunk_n = chunk_end_idx - chunk_start_idx
        start_epoch = job_start + chunk_start_idx * window_sec
        chunk_duration = chunk_n * window_sec

        try:
            audio = audio_resolver(start_epoch, chunk_duration, _PROFILE_SR)
        except Exception:
            logger.warning(
                "Failed to resolve audio at %.1f for gain profile, skipping chunk",
                start_epoch,
            )
            continue

        samples_per_window = int(_PROFILE_SR * window_sec)
        for i in range(chunk_n):
            start_sample = i * samples_per_window
            end_sample = start_sample + samples_per_window
            if end_sample > len(audio):
                break
            rms_db_values[chunk_start_idx + i] = _rms_db_for_chunk(
                audio[start_sample:end_sample]
            )

    non_silent_mask = rms_db_values > _SILENCE_FLOOR_DB
    if not np.any(non_silent_mask):
        return GainProfile()

    global_median = float(np.median(rms_db_values[non_silent_mask]))
    flagged = (rms_db_values > global_median + threshold_db) & non_silent_mask

    # Group consecutive flagged windows, splitting on internal RMS drops
    raw_runs: list[tuple[int, int]] = []
    i = 0
    while i < n_windows:
        if not flagged[i]:
            i += 1
            continue
        seg_start = i
        while i < n_windows and flagged[i]:
            i += 1
        raw_runs.append((seg_start, i))

    sub_segments: list[tuple[int, int]] = []
    for run_start, run_end in raw_runs:
        sub_segments.extend(_split_on_internal_drops(rms_db_values, run_start, run_end))

    segments: list[GainSegment] = []
    for seg_start, seg_end in sub_segments:
        seg_duration = (seg_end - seg_start) * window_sec
        if seg_duration < min_duration_sec:
            continue

        seg_median_rms = float(np.median(rms_db_values[seg_start:seg_end]))
        attenuation_db = seg_median_rms - global_median

        segments.append(
            GainSegment(
                start_sec=job_start + seg_start * window_sec,
                end_sec=job_start + seg_end * window_sec,
                attenuation_db=attenuation_db,
            )
        )

    profile = GainProfile(segments=segments, global_median_rms_db=global_median)
    if segments:
        logger.info(
            "Gain profile: %d segment(s), median=%.1f dB, max_atten=%.1f dB",
            len(segments),
            global_median,
            max(s.attenuation_db for s in segments),
        )
    return profile


def apply_gain_profile(
    audio: np.ndarray,
    sample_rate: int,
    start_epoch: float,
    gain_profile: GainProfile,
) -> np.ndarray:
    """Apply gain correction to an audio chunk.

    Parameters
    ----------
    audio:
        Float32 audio array.
    sample_rate:
        Sample rate of the audio.
    start_epoch:
        Absolute start time of this audio chunk (epoch seconds).
    gain_profile:
        The job's gain profile.

    Returns
    -------
    Corrected audio array. Returns the original array (same object) if no
    corrections are needed.
    """
    if not gain_profile.segments:
        return audio

    audio_end = start_epoch + len(audio) / sample_rate
    crossfade_samples = max(1, int(_CROSSFADE_SEC * sample_rate))

    # Check if any segment overlaps this audio
    needs_correction = False
    for seg in gain_profile.segments:
        if seg.start_sec < audio_end and seg.end_sec > start_epoch:
            needs_correction = True
            break

    if not needs_correction:
        return audio

    corrected = audio.copy()

    for seg in gain_profile.segments:
        if seg.start_sec >= audio_end or seg.end_sec <= start_epoch:
            continue

        # Convert to sample indices within this audio chunk
        seg_start_sample = max(0, int((seg.start_sec - start_epoch) * sample_rate))
        seg_end_sample = min(
            len(corrected), int((seg.end_sec - start_epoch) * sample_rate)
        )

        if seg_start_sample >= seg_end_sample:
            continue

        # Linear gain factor from dB attenuation
        gain_factor = 10.0 ** (-seg.attenuation_db / 20.0)

        # Build per-sample gain array with crossfade at boundaries
        n_seg = seg_end_sample - seg_start_sample
        gains = np.full(n_seg, gain_factor, dtype=np.float32)

        # Fade in at segment start (only if segment starts within this chunk)
        if seg.start_sec > start_epoch:
            fade_len = min(crossfade_samples, n_seg)
            fade_in = np.linspace(1.0, gain_factor, fade_len, dtype=np.float32)
            gains[:fade_len] = fade_in

        # Fade out at segment end (only if segment ends within this chunk)
        if seg.end_sec < audio_end:
            fade_len = min(crossfade_samples, n_seg)
            fade_out = np.linspace(gain_factor, 1.0, fade_len, dtype=np.float32)
            gains[-fade_len:] = fade_out

        corrected[seg_start_sample:seg_end_sample] *= gains

    return corrected
