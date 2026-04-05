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


def compute_gain_profile(
    audio_resolver: Callable[[float, float, int], np.ndarray],
    job_start: float,
    job_end: float,
    threshold_db: float = 6.0,
    min_duration_sec: float = 5.0,
) -> GainProfile:
    """Analyze a job's audio and build a gain normalization profile.

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

    # Walk in 1-second windows at coarse sample rate
    window_sec = 1.0
    n_windows = int(duration / window_sec)
    if n_windows == 0:
        return GainProfile()

    rms_db_values = np.full(n_windows, _SILENCE_FLOOR_DB, dtype=np.float64)

    # Process in chunks to avoid resolving the entire job at once
    chunk_windows = 60  # 60 seconds per chunk
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
            window = audio[start_sample:end_sample]
            rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
            if rms > 0:
                rms_db = 20.0 * np.log10(rms)
            else:
                rms_db = _SILENCE_FLOOR_DB
            rms_db_values[chunk_start_idx + i] = rms_db

    # Compute median of non-silent windows
    non_silent_mask = rms_db_values > _SILENCE_FLOOR_DB
    if not np.any(non_silent_mask):
        return GainProfile()

    global_median = float(np.median(rms_db_values[non_silent_mask]))

    # Flag windows above threshold
    flagged = (rms_db_values > global_median + threshold_db) & non_silent_mask

    # Group consecutive flagged windows into segments
    segments: list[GainSegment] = []
    i = 0
    while i < n_windows:
        if not flagged[i]:
            i += 1
            continue
        seg_start = i
        while i < n_windows and flagged[i]:
            i += 1
        seg_end = i

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
