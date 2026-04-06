"""Tests for gain normalization detection and correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ---- compute_gain_profile tests ----


def _make_audio(
    duration_sec: float,
    sr: int,
    rms_level: float = 0.01,
) -> np.ndarray:
    """Generate white noise at a target RMS level."""
    rng = np.random.RandomState(42)
    n_samples = int(sr * duration_sec)
    audio = rng.randn(n_samples).astype(np.float32)
    # Normalize to target RMS
    current_rms = float(np.sqrt(np.mean(audio**2)))
    if current_rms > 0:
        audio = audio * (rms_level / current_rms)
    return audio


def test_detect_single_gain_step():
    """A sustained high-gain region should be detected as one segment."""
    from humpback.processing.gain_normalization import compute_gain_profile

    sr = 4000
    normal_rms = 0.01
    loud_rms = 0.1  # 20 dB above normal
    duration = 30.0

    # Build audio: 10s normal, 10s loud, 10s normal
    rng = np.random.RandomState(42)
    n_total = int(sr * duration)
    audio = rng.randn(n_total).astype(np.float32)

    # Normalize sections
    for start_s, end_s, target_rms in [
        (0, 10, normal_rms),
        (10, 20, loud_rms),
        (20, 30, normal_rms),
    ]:
        chunk = audio[start_s * sr : end_s * sr]
        current = float(np.sqrt(np.mean(chunk**2)))
        if current > 0:
            audio[start_s * sr : end_s * sr] = chunk * (target_rms / current)

    job_start = 1000.0
    job_end = job_start + duration

    def resolver(start_epoch: float, dur: float, target_sr: int) -> np.ndarray:
        offset = start_epoch - job_start
        start_sample = int(offset * sr)
        end_sample = start_sample + int(dur * sr)
        return audio[start_sample:end_sample]

    profile = compute_gain_profile(
        audio_resolver=resolver,
        job_start=job_start,
        job_end=job_end,
        threshold_db=6.0,
        min_duration_sec=5.0,
    )

    assert len(profile.segments) == 1
    seg = profile.segments[0]
    # Segment should cover roughly the 10-20s region
    assert seg.start_sec >= job_start + 9.0
    assert seg.start_sec <= job_start + 11.0
    assert seg.end_sec >= job_start + 19.0
    assert seg.end_sec <= job_start + 21.0
    # Attenuation should be roughly 20 dB
    assert seg.attenuation_db > 15.0
    assert seg.attenuation_db < 25.0


def test_no_gain_changes():
    """Uniform audio should produce an empty gain profile."""
    from humpback.processing.gain_normalization import compute_gain_profile

    sr = 4000
    audio = _make_audio(20.0, sr, rms_level=0.01)
    job_start = 0.0
    job_end = 20.0

    def resolver(start_epoch: float, dur: float, target_sr: int) -> np.ndarray:
        start_sample = int((start_epoch - job_start) * sr)
        end_sample = start_sample + int(dur * sr)
        return audio[start_sample:end_sample]

    profile = compute_gain_profile(
        audio_resolver=resolver,
        job_start=job_start,
        job_end=job_end,
    )
    assert len(profile.segments) == 0


def test_short_transient_filtered():
    """A spike shorter than min_duration_sec should be filtered out."""
    from humpback.processing.gain_normalization import compute_gain_profile

    sr = 4000
    duration = 30.0
    normal_rms = 0.01
    loud_rms = 0.1

    rng = np.random.RandomState(42)
    n_total = int(sr * duration)
    audio = rng.randn(n_total).astype(np.float32)

    # Normalize: all normal except a 3-second spike at 10-13s
    for start_s, end_s, target_rms in [
        (0, 10, normal_rms),
        (10, 13, loud_rms),
        (13, 30, normal_rms),
    ]:
        chunk = audio[start_s * sr : end_s * sr]
        current = float(np.sqrt(np.mean(chunk**2)))
        if current > 0:
            audio[start_s * sr : end_s * sr] = chunk * (target_rms / current)

    job_start = 0.0
    job_end = duration

    def resolver(start_epoch: float, dur: float, target_sr: int) -> np.ndarray:
        start_sample = int((start_epoch - job_start) * sr)
        end_sample = start_sample + int(dur * sr)
        return audio[start_sample:end_sample]

    profile = compute_gain_profile(
        audio_resolver=resolver,
        job_start=job_start,
        job_end=job_end,
        threshold_db=6.0,
        min_duration_sec=5.0,
    )
    assert len(profile.segments) == 0


def test_multiple_gain_regions():
    """Two separate gain regions should produce two segments."""
    from humpback.processing.gain_normalization import compute_gain_profile

    sr = 4000
    duration = 60.0
    normal_rms = 0.01
    loud_rms = 0.1

    rng = np.random.RandomState(42)
    n_total = int(sr * duration)
    audio = rng.randn(n_total).astype(np.float32)

    # Two loud regions: 10-20s and 40-50s
    sections = [
        (0, 10, normal_rms),
        (10, 20, loud_rms),
        (20, 40, normal_rms),
        (40, 50, loud_rms),
        (50, 60, normal_rms),
    ]
    for start_s, end_s, target_rms in sections:
        chunk = audio[start_s * sr : end_s * sr]
        current = float(np.sqrt(np.mean(chunk**2)))
        if current > 0:
            audio[start_s * sr : end_s * sr] = chunk * (target_rms / current)

    job_start = 0.0
    job_end = duration

    def resolver(start_epoch: float, dur: float, target_sr: int) -> np.ndarray:
        start_sample = int((start_epoch - job_start) * sr)
        end_sample = start_sample + int(dur * sr)
        return audio[start_sample:end_sample]

    profile = compute_gain_profile(
        audio_resolver=resolver,
        job_start=job_start,
        job_end=job_end,
        threshold_db=6.0,
        min_duration_sec=5.0,
    )
    assert len(profile.segments) == 2


def test_long_job_uses_sampling_path():
    """A job longer than 10 minutes should use the two-pass sampling approach."""
    from humpback.processing.gain_normalization import compute_gain_profile

    normal_rms = 0.01
    loud_rms = 0.1
    # 3600s = 1 hour, triggers the sampling path (> 600s)
    duration = 3600.0

    rng = np.random.RandomState(42)

    def resolver(start_epoch: float, dur: float, target_sr: int) -> np.ndarray:
        n_samples = int(target_sr * dur)
        audio = rng.randn(n_samples).astype(np.float32)
        current = float(np.sqrt(np.mean(audio**2)))
        if current == 0:
            return audio
        # High gain region from 600s to 1200s
        offset = start_epoch - job_start
        end_offset = offset + dur
        if offset >= 600.0 and end_offset <= 1200.0:
            return audio * (loud_rms / current)
        elif offset < 600.0 and end_offset > 600.0:
            # Partial overlap with loud region
            return audio * (normal_rms / current)
        elif offset < 1200.0 and end_offset > 1200.0:
            return audio * (normal_rms / current)
        return audio * (normal_rms / current)

    job_start = 0.0
    job_end = duration

    profile = compute_gain_profile(
        audio_resolver=resolver,
        job_start=job_start,
        job_end=job_end,
        threshold_db=6.0,
        min_duration_sec=5.0,
    )

    assert len(profile.segments) >= 1
    seg = profile.segments[0]
    # Segment should be roughly in the 600-1200s region
    assert seg.start_sec < 650.0
    assert seg.end_sec > 1150.0
    assert seg.attenuation_db > 10.0


def test_internal_drop_splits_segment():
    """A segment with two distinct gain levels should be split into two."""
    from humpback.processing.gain_normalization import compute_gain_profile

    sr = 4000
    normal_rms = 0.005  # ~-46 dB
    very_loud_rms = 0.5  # ~-6 dB, 40 dB above normal
    moderately_loud_rms = 0.05  # ~-26 dB, 20 dB above normal
    # Both levels are well above threshold (median + 6 dB) but differ by ~20 dB
    duration = 50.0

    # Build: 15s normal, 10s very loud, 10s moderately loud, 15s normal
    rng = np.random.RandomState(42)
    n_total = int(sr * duration)
    audio = rng.randn(n_total).astype(np.float32)

    sections = [
        (0, 15, normal_rms),
        (15, 25, very_loud_rms),
        (25, 35, moderately_loud_rms),
        (35, 50, normal_rms),
    ]
    for start_s, end_s, target_rms in sections:
        chunk = audio[start_s * sr : end_s * sr]
        current = float(np.sqrt(np.mean(chunk**2)))
        if current > 0:
            audio[start_s * sr : end_s * sr] = chunk * (target_rms / current)

    job_start = 0.0
    job_end = duration

    def resolver(start_epoch: float, dur: float, target_sr: int) -> np.ndarray:
        start_sample = int((start_epoch - job_start) * sr)
        end_sample = start_sample + int(dur * sr)
        return audio[start_sample:end_sample]

    profile = compute_gain_profile(
        audio_resolver=resolver,
        job_start=job_start,
        job_end=job_end,
        threshold_db=6.0,
        min_duration_sec=5.0,
    )

    # Should produce two segments (very loud and moderately loud) instead of one
    assert len(profile.segments) == 2
    # The very loud segment should have higher attenuation
    assert profile.segments[0].attenuation_db > profile.segments[1].attenuation_db


# ---- apply_gain_profile tests ----


def test_apply_attenuation():
    """Attenuation should reduce amplitude by the expected dB amount."""
    from humpback.processing.gain_normalization import (
        GainProfile,
        GainSegment,
        apply_gain_profile,
    )

    sr = 16000
    duration = 10.0
    audio = np.ones(int(sr * duration), dtype=np.float32) * 0.5
    start_epoch = 100.0

    profile = GainProfile(
        segments=[
            GainSegment(start_sec=103.0, end_sec=107.0, attenuation_db=20.0),
        ],
        global_median_rms_db=-40.0,
    )

    corrected = apply_gain_profile(audio, sr, start_epoch, profile)

    # Outside the segment, audio should be unchanged
    assert corrected[0] == pytest.approx(0.5)
    assert corrected[-1] == pytest.approx(0.5)

    # Inside the segment (well past crossfade), amplitude should be reduced by ~20 dB
    # 20 dB attenuation = factor of 0.1
    mid_sample = int((105.0 - start_epoch) * sr)
    assert corrected[mid_sample] == pytest.approx(0.5 * 0.1, abs=0.01)


def test_apply_empty_profile_returns_same_object():
    """An empty gain profile should return the exact same array (no copy)."""
    from humpback.processing.gain_normalization import GainProfile, apply_gain_profile

    audio = np.ones(1000, dtype=np.float32)
    profile = GainProfile()
    result = apply_gain_profile(audio, 16000, 0.0, profile)
    assert result is audio


def test_apply_no_overlap_returns_same_object():
    """A profile with segments that don't overlap the audio returns same array."""
    from humpback.processing.gain_normalization import (
        GainProfile,
        GainSegment,
        apply_gain_profile,
    )

    audio = np.ones(16000, dtype=np.float32)
    profile = GainProfile(
        segments=[GainSegment(start_sec=200.0, end_sec=210.0, attenuation_db=10.0)]
    )
    result = apply_gain_profile(audio, 16000, 100.0, profile)
    assert result is audio


def test_crossfade_smooth_transition():
    """The crossfade at segment boundaries should produce a smooth transition."""
    from humpback.processing.gain_normalization import (
        GainProfile,
        GainSegment,
        apply_gain_profile,
    )

    sr = 16000
    audio = np.ones(sr * 10, dtype=np.float32)
    start_epoch = 0.0

    profile = GainProfile(
        segments=[GainSegment(start_sec=3.0, end_sec=7.0, attenuation_db=20.0)]
    )

    corrected = apply_gain_profile(audio, sr, start_epoch, profile)

    # Check the fade-in region: samples near the start of the segment should
    # transition smoothly from 1.0 toward the attenuated level
    fade_start = int(3.0 * sr)
    fade_samples = int(0.05 * sr)  # 50ms crossfade
    fade_region = corrected[fade_start : fade_start + fade_samples]

    # Should be monotonically decreasing (from ~1.0 toward ~0.1)
    diffs = np.diff(fade_region)
    assert np.all(diffs <= 0), "Fade-in should be monotonically decreasing"


def test_partial_overlap():
    """Segment extending beyond the audio chunk should still work."""
    from humpback.processing.gain_normalization import (
        GainProfile,
        GainSegment,
        apply_gain_profile,
    )

    sr = 16000
    audio = np.ones(sr * 5, dtype=np.float32) * 0.5
    start_epoch = 10.0

    # Segment starts before and ends after the audio chunk
    profile = GainProfile(
        segments=[GainSegment(start_sec=8.0, end_sec=20.0, attenuation_db=20.0)]
    )

    corrected = apply_gain_profile(audio, sr, start_epoch, profile)

    # The entire chunk is within the segment, so all samples should be attenuated
    # (no crossfade since boundaries are outside the chunk)
    mid_sample = len(corrected) // 2
    assert corrected[mid_sample] == pytest.approx(0.5 * 0.1, abs=0.01)


def test_does_not_modify_input():
    """apply_gain_profile should not modify the input array."""
    from humpback.processing.gain_normalization import (
        GainProfile,
        GainSegment,
        apply_gain_profile,
    )

    sr = 16000
    audio = np.ones(sr * 5, dtype=np.float32) * 0.5
    original = audio.copy()

    profile = GainProfile(
        segments=[GainSegment(start_sec=1.0, end_sec=4.0, attenuation_db=10.0)]
    )

    apply_gain_profile(audio, sr, 0.0, profile)
    np.testing.assert_array_equal(audio, original)


# ---- GainProfile serialization tests ----


def test_profile_round_trip():
    """GainProfile should survive serialization to dict and back."""
    from humpback.processing.gain_normalization import GainProfile, GainSegment

    profile = GainProfile(
        segments=[
            GainSegment(start_sec=10.0, end_sec=20.0, attenuation_db=12.5),
            GainSegment(start_sec=40.0, end_sec=55.0, attenuation_db=8.3),
        ],
        global_median_rms_db=-35.0,
    )

    data = profile.to_dict()
    restored = GainProfile.from_dict(data)

    assert len(restored.segments) == 2
    assert restored.segments[0].start_sec == 10.0
    assert restored.segments[0].attenuation_db == 12.5
    assert restored.segments[1].end_sec == 55.0
    assert restored.global_median_rms_db == -35.0


def test_profile_json_round_trip():
    """GainProfile should survive JSON serialization."""
    from humpback.processing.gain_normalization import GainProfile, GainSegment

    profile = GainProfile(
        segments=[GainSegment(start_sec=5.0, end_sec=15.0, attenuation_db=10.0)],
        global_median_rms_db=-40.0,
    )

    json_str = json.dumps(profile.to_dict())
    restored = GainProfile.from_dict(json.loads(json_str))
    assert len(restored.segments) == 1
    assert restored.segments[0].attenuation_db == 10.0


# ---- Cache integration tests ----


def test_cache_gain_profile_round_trip(tmp_path: Path):
    """Gain profile should be storable and retrievable from the tile cache."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(tmp_path / "cache", max_jobs=5)

    profile_data = {
        "segments": [{"start_sec": 10.0, "end_sec": 20.0, "attenuation_db": 12.0}],
        "global_median_rms_db": -35.0,
    }

    cache.put_gain_profile("job-123", profile_data)
    result = cache.get_gain_profile("job-123")

    assert result is not None
    assert len(result["segments"]) == 1
    assert result["segments"][0]["attenuation_db"] == 12.0
    assert result["global_median_rms_db"] == -35.0


def test_cache_gain_profile_missing(tmp_path: Path):
    """Missing gain profile should return None."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(tmp_path / "cache", max_jobs=5)
    assert cache.get_gain_profile("nonexistent") is None
