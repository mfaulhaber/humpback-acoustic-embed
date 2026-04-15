"""Tests for the shared audio loader module.

All 20 spec test cases plus the training audio loader.
Hydrophone tests mock ``resolve_timeline_audio`` to verify arguments
and call counts.  File-based tests use real temporary WAV files.
"""

from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from humpback.call_parsing.audio_loader import (
    CachedAudioSource,
    _compute_preload_span,
    _event_context,
    build_event_audio_loader,
    build_multi_source_event_audio_loader,
    build_region_audio_loader,
    build_training_audio_loader,
)
from humpback.config import Settings


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeEvent:
    start_sec: float
    end_sec: float


@dataclass
class _FakeRegion:
    padded_start_sec: float
    padded_end_sec: float


@dataclass
class _FakeTrainingSample:
    crop_start_sec: float
    crop_end_sec: float
    hydrophone_id: str
    start_timestamp: float
    end_timestamp: float


@dataclass
class _FakeEventSample:
    """Sample for multi-source event classification (feedback/bootstrap)."""

    start_sec: float
    end_sec: float
    hydrophone_id: str
    start_timestamp: float
    end_timestamp: float


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        storage_root=tmp_path / "storage",
        use_real_model=False,
    )


def _make_wav(path: Path, duration_sec: float = 10.0, sr: int = 16000) -> Path:
    """Write a sine-wave WAV and return its path."""
    n = int(sr * duration_sec)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *samples))
    return path


def _make_audio_file(path: Path) -> Any:
    """Return a minimal AudioFile-like object."""
    af = MagicMock()
    af.original_filename = path.name
    af.file_path = str(path)
    return af


def _mock_resolve(returned_audio: np.ndarray | None = None):
    """Patch resolve_timeline_audio and return the mock."""
    if returned_audio is None:
        returned_audio = np.zeros(16000, dtype=np.float32)

    def _side_effect(**kwargs: Any) -> np.ndarray:
        dur = kwargs.get("duration_sec", 1.0)
        sr = kwargs.get("target_sr", 16000)
        n = max(1, int(round(dur * sr)))
        return np.zeros(n, dtype=np.float32)

    return patch(
        "humpback.processing.timeline_audio.resolve_timeline_audio",
        side_effect=_side_effect,
    )


# ===================================================================
# Coordinate Conversion (spec tests 1-6)
# ===================================================================


class TestCoordinateConversion:
    """Verify that CachedAudioSource performs the job_start + offset conversion."""

    def test_1_relative_offset_becomes_absolute(self, tmp_path: Path) -> None:
        """Spec #1: rel offset + job_start = correct absolute ts."""
        settings = _make_settings(tmp_path)
        job_start = 1000.0
        rel_start = 5.0

        with _mock_resolve() as mock_rta:
            src = CachedAudioSource.from_hydrophone(
                hydrophone_id="h1",
                job_start_ts=job_start,
                job_end_ts=1100.0,
                target_sr=16000,
                settings=settings,
            )
            src.get_audio(rel_start, 2.0)

        call_kwargs = mock_rta.call_args.kwargs
        assert call_kwargs["start_sec"] == pytest.approx(job_start + rel_start)
        assert call_kwargs["duration_sec"] == pytest.approx(2.0)

    def test_2_event_at_job_start_no_negative(self) -> None:
        """Spec #2: event at start_sec=0 must not produce negative abs ts."""
        events = [_FakeEvent(start_sec=0.0, end_sec=3.0)]

        span = _compute_preload_span(events, job_duration=100.0)
        assert span[0] >= 0.0

    def test_3_event_at_job_end_no_exceed(self, tmp_path: Path) -> None:
        """Spec #3: context padding must not exceed job_end."""
        job_duration = 50.0
        events = [_FakeEvent(start_sec=48.0, end_sec=50.0)]

        span = _compute_preload_span(events, job_duration=job_duration)
        assert span[1] <= job_duration

    def test_4_multiple_events_span_covers_all(self) -> None:
        """Spec #4: preload span covers all events with context."""
        events = [
            _FakeEvent(start_sec=5.0, end_sec=8.0),
            _FakeEvent(start_sec=90.0, end_sec=95.0),
        ]
        span = _compute_preload_span(events, job_duration=100.0)
        # Span must cover both events
        assert span[0] <= 5.0
        assert span[1] >= 95.0

    def test_5_short_event_symmetric_padding(self) -> None:
        """Spec #5: event < 10s gets symmetric padding to 10s context."""
        events = [_FakeEvent(start_sec=50.0, end_sec=52.0)]  # 2s event
        span = _compute_preload_span(events, job_duration=200.0)

        # context = max(10, 2) = 10; pad = (10-2)/2 = 4
        assert span[0] == pytest.approx(46.0)
        assert span[1] == pytest.approx(56.0)

    def test_6_long_event_context_scales(self) -> None:
        """Spec #6: event > 10s gets context scaled with duration."""
        events = [_FakeEvent(start_sec=50.0, end_sec=65.0)]  # 15s event
        span = _compute_preload_span(events, job_duration=200.0)

        # context = max(10, 15) = 15; pad = (15-15)/2 = 0
        assert span[0] == pytest.approx(50.0)
        assert span[1] == pytest.approx(65.0)


# ===================================================================
# Pre-load Span Caching (spec tests 7-10)
# ===================================================================


class TestPreloadCaching:
    def test_7_preloaded_returns_same_buffer(self, tmp_path: Path) -> None:
        """Spec #7: pre-loaded source returns same buffer for different events."""
        settings = _make_settings(tmp_path)

        with _mock_resolve() as mock_rta:
            src = CachedAudioSource.from_hydrophone(
                hydrophone_id="h1",
                job_start_ts=0.0,
                job_end_ts=100.0,
                target_sr=16000,
                settings=settings,
                preload_span=(0.0, 100.0),
            )
            audio1, _ = src.get_audio(5.0, 2.0)
            audio2, _ = src.get_audio(50.0, 3.0)

        # Same buffer object (not a copy)
        assert audio1 is audio2
        # resolve_timeline_audio called exactly once (at construction)
        assert mock_rta.call_count == 1

    def test_8_degenerate_identical_start_end(self, tmp_path: Path) -> None:
        """Spec #8: events with identical start/end produce valid span."""
        events = [_FakeEvent(start_sec=10.0, end_sec=10.0)]
        span = _compute_preload_span(events, job_duration=100.0)

        # duration=0, context=10, pad=5
        assert span[0] == pytest.approx(5.0)
        assert span[1] == pytest.approx(15.0)

    def test_9_empty_events_no_preload(self, tmp_path: Path) -> None:
        """Spec #9: empty event list → no preload, falls back to per-sample."""
        settings = _make_settings(tmp_path)

        with _mock_resolve() as mock_rta:
            loader = build_event_audio_loader(
                target_sr=16000,
                settings=settings,
                hydrophone_id="h1",
                job_start_ts=0.0,
                job_end_ts=100.0,
                preload_events=[],
            )
            # No preload call yet
            assert mock_rta.call_count == 0

            # Each call triggers resolve
            loader(_FakeEvent(start_sec=5.0, end_sec=8.0))
            assert mock_rta.call_count == 1

            loader(_FakeEvent(start_sec=20.0, end_sec=25.0))
            assert mock_rta.call_count == 2

    def test_10_events_from_different_positions(self, tmp_path: Path) -> None:
        """Spec #10: events from different positions within one job."""
        events = [
            _FakeEvent(start_sec=2.0, end_sec=4.0),
            _FakeEvent(start_sec=80.0, end_sec=82.0),
        ]
        span = _compute_preload_span(events, job_duration=100.0)
        # Should cover both with context
        assert span[0] < 2.0
        assert span[1] > 82.0


# ===================================================================
# Boundary / Degenerate Inputs (spec tests 11-14)
# ===================================================================


class TestBoundaryInputs:
    def test_11_zero_duration_event(self, tmp_path: Path) -> None:
        """Spec #11: zero-duration event doesn't crash."""
        settings = _make_settings(tmp_path)

        with _mock_resolve():
            src = CachedAudioSource.from_hydrophone(
                hydrophone_id="h1",
                job_start_ts=0.0,
                job_end_ts=100.0,
                target_sr=16000,
                settings=settings,
            )
            audio, offset = src.get_audio(10.0, 0.0)

        assert isinstance(audio, np.ndarray)
        assert isinstance(offset, float)

    def test_12_very_short_event(self, tmp_path: Path) -> None:
        """Spec #12: event < 1 sample returns non-empty array."""
        settings = _make_settings(tmp_path)
        # At 16kHz, 1 sample = 0.0000625s. Request 0.00001s.
        with _mock_resolve():
            src = CachedAudioSource.from_hydrophone(
                hydrophone_id="h1",
                job_start_ts=0.0,
                job_end_ts=100.0,
                target_sr=16000,
                settings=settings,
            )
            audio, _ = src.get_audio(10.0, 0.00001)

        assert isinstance(audio, np.ndarray)
        assert len(audio) >= 1

    def test_13_padding_clamped_both_sides(self) -> None:
        """Spec #13: short job clamps padding on both sides."""
        # 6s job, 2s event in the middle → context = 10s, exceeds both ends
        events = [_FakeEvent(start_sec=2.0, end_sec=4.0)]
        span = _compute_preload_span(events, job_duration=6.0)

        assert span[0] == pytest.approx(0.0)
        assert span[1] == pytest.approx(6.0)

    def test_14_file_short_audio(self, tmp_path: Path) -> None:
        """Spec #14: file-based source with very short audio."""
        wav_path = _make_wav(tmp_path / "short.wav", duration_sec=0.5, sr=16000)
        af = _make_audio_file(wav_path)

        with patch(
            "humpback.call_parsing.audio_loader.resolve_audio_path",
            return_value=wav_path,
        ):
            src = CachedAudioSource.from_file(af, 16000, tmp_path)
            audio, offset = src.get_audio(0.0, 0.0)

        assert offset == 0.0
        # 0.5s at 16kHz = 8000 samples
        assert len(audio) == 8000


# ===================================================================
# Protocol Contracts (spec tests 15-18)
# ===================================================================


class TestProtocolContracts:
    def test_15_event_factory_returns_tuple(self, tmp_path: Path) -> None:
        """Spec #15: event factory always returns tuple[ndarray, float]."""
        settings = _make_settings(tmp_path)

        with _mock_resolve():
            loader = build_event_audio_loader(
                target_sr=16000,
                settings=settings,
                hydrophone_id="h1",
                job_start_ts=0.0,
                job_end_ts=100.0,
            )
            result = loader(_FakeEvent(start_sec=5.0, end_sec=8.0))

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], float)

    def test_16_region_factory_returns_ndarray(self, tmp_path: Path) -> None:
        """Spec #16: region factory always returns np.ndarray, not tuple."""
        settings = _make_settings(tmp_path)

        with _mock_resolve():
            loader = build_region_audio_loader(
                target_sr=16000,
                settings=settings,
                hydrophone_id="h1",
                job_start_ts=0.0,
                job_end_ts=100.0,
            )
            result = loader(_FakeRegion(padded_start_sec=5.0, padded_end_sec=8.0))

        assert isinstance(result, np.ndarray)
        assert not isinstance(result, tuple)

    def test_17_region_factory_slice_length(self, tmp_path: Path) -> None:
        """Spec #17: region slice length matches expected samples."""
        wav_path = _make_wav(tmp_path / "test.wav", duration_sec=10.0, sr=16000)
        af = _make_audio_file(wav_path)

        with patch(
            "humpback.call_parsing.audio_loader.resolve_audio_path",
            return_value=wav_path,
        ):
            loader = build_region_audio_loader(
                target_sr=16000,
                settings=_make_settings(tmp_path),
                audio_file=af,
                storage_root=tmp_path,
            )
            result = loader(_FakeRegion(padded_start_sec=2.0, padded_end_sec=5.0))

        expected_samples = int(round(3.0 * 16000))
        assert abs(len(result) - expected_samples) <= 1  # rounding tolerance

    def test_18_file_event_loader_offset_zero(self, tmp_path: Path) -> None:
        """Spec #18: file-based event loader always returns offset 0.0."""
        wav_path = _make_wav(tmp_path / "test.wav", duration_sec=10.0, sr=16000)
        af = _make_audio_file(wav_path)

        with patch(
            "humpback.call_parsing.audio_loader.resolve_audio_path",
            return_value=wav_path,
        ):
            loader = build_event_audio_loader(
                target_sr=16000,
                settings=_make_settings(tmp_path),
                audio_file=af,
                storage_root=tmp_path,
            )
            _, offset = loader(_FakeEvent(start_sec=3.0, end_sec=6.0))

        assert offset == 0.0


# ===================================================================
# Integration — resolve_timeline_audio call counts (spec tests 19-20)
# ===================================================================


class TestIntegrationCallCounts:
    def test_19_preloaded_calls_resolve_once(self, tmp_path: Path) -> None:
        """Spec #19: resolve called once regardless of event count."""
        settings = _make_settings(tmp_path)
        events = [
            _FakeEvent(start_sec=5.0, end_sec=8.0),
            _FakeEvent(start_sec=20.0, end_sec=25.0),
            _FakeEvent(start_sec=50.0, end_sec=55.0),
        ]

        with _mock_resolve() as mock_rta:
            loader = build_event_audio_loader(
                target_sr=16000,
                settings=settings,
                hydrophone_id="h1",
                job_start_ts=0.0,
                job_end_ts=100.0,
                preload_events=events,
            )
            # Construction called once
            assert mock_rta.call_count == 1

            # Three loads, no additional calls
            for ev in events:
                loader(ev)
            assert mock_rta.call_count == 1

    def test_20_per_sample_calls_resolve_n_times(self, tmp_path: Path) -> None:
        """Spec #20: without preload, resolve called N times for N samples."""
        settings = _make_settings(tmp_path)

        with _mock_resolve() as mock_rta:
            loader = build_event_audio_loader(
                target_sr=16000,
                settings=settings,
                hydrophone_id="h1",
                job_start_ts=0.0,
                job_end_ts=100.0,
            )
            assert mock_rta.call_count == 0

            for i in range(5):
                loader(_FakeEvent(start_sec=float(i * 10), end_sec=float(i * 10 + 3)))
            assert mock_rta.call_count == 5


# ===================================================================
# Training audio loader (extension beyond original 20 spec tests)
# ===================================================================


class TestTrainingAudioLoader:
    def test_multi_source_dispatch(self, tmp_path: Path) -> None:
        """Samples from different hydrophones use different sources."""
        settings = _make_settings(tmp_path)
        samples = [
            _FakeTrainingSample(
                crop_start_sec=0.0,
                crop_end_sec=2.0,
                hydrophone_id="h1",
                start_timestamp=1000.0,
                end_timestamp=1100.0,
            ),
            _FakeTrainingSample(
                crop_start_sec=0.0,
                crop_end_sec=2.0,
                hydrophone_id="h2",
                start_timestamp=2000.0,
                end_timestamp=2100.0,
            ),
        ]

        with _mock_resolve() as mock_rta:
            loader = build_training_audio_loader(
                target_sr=16000,
                settings=settings,
                samples=samples,
            )
            # 2 groups → 2 preload calls
            assert mock_rta.call_count == 2

            r1 = loader(samples[0])
            r2 = loader(samples[1])

        assert isinstance(r1, np.ndarray)
        assert isinstance(r2, np.ndarray)

    def test_preloads_per_group(self, tmp_path: Path) -> None:
        """Samples sharing the same source are grouped and pre-loaded once."""
        settings = _make_settings(tmp_path)
        samples = [
            _FakeTrainingSample(
                crop_start_sec=1.0,
                crop_end_sec=3.0,
                hydrophone_id="h1",
                start_timestamp=1000.0,
                end_timestamp=1100.0,
            ),
            _FakeTrainingSample(
                crop_start_sec=10.0,
                crop_end_sec=12.0,
                hydrophone_id="h1",
                start_timestamp=1000.0,
                end_timestamp=1100.0,
            ),
        ]

        with _mock_resolve() as mock_rta:
            loader = build_training_audio_loader(
                target_sr=16000,
                settings=settings,
                samples=samples,
            )
            # 1 group → 1 preload call
            assert mock_rta.call_count == 1

            loader(samples[0])
            loader(samples[1])
            # Still only 1 call
            assert mock_rta.call_count == 1

    def test_returns_ndarray(self, tmp_path: Path) -> None:
        """Training loader returns np.ndarray, not tuple."""
        settings = _make_settings(tmp_path)
        sample = _FakeTrainingSample(
            crop_start_sec=1.0,
            crop_end_sec=3.0,
            hydrophone_id="h1",
            start_timestamp=1000.0,
            end_timestamp=1100.0,
        )

        with _mock_resolve():
            loader = build_training_audio_loader(
                target_sr=16000,
                settings=settings,
            )
            result = loader(sample)

        assert isinstance(result, np.ndarray)
        assert not isinstance(result, tuple)


# ===================================================================
# Multi-source event audio loader
# ===================================================================


class TestMultiSourceEventAudioLoader:
    def test_returns_tuple(self, tmp_path: Path) -> None:
        """Multi-source event loader returns tuple[ndarray, float], not bare ndarray."""
        settings = _make_settings(tmp_path)
        sample = _FakeEventSample(
            start_sec=5.0,
            end_sec=8.0,
            hydrophone_id="h1",
            start_timestamp=1000.0,
            end_timestamp=1100.0,
        )

        with _mock_resolve():
            loader = build_multi_source_event_audio_loader(
                target_sr=16000,
                settings=settings,
            )
            result = loader(sample)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], float)

    def test_multi_source_dispatch(self, tmp_path: Path) -> None:
        """Samples from different hydrophones use different sources."""
        settings = _make_settings(tmp_path)
        samples = [
            _FakeEventSample(
                start_sec=5.0,
                end_sec=8.0,
                hydrophone_id="h1",
                start_timestamp=1000.0,
                end_timestamp=1100.0,
            ),
            _FakeEventSample(
                start_sec=5.0,
                end_sec=8.0,
                hydrophone_id="h2",
                start_timestamp=2000.0,
                end_timestamp=2100.0,
            ),
        ]

        with _mock_resolve() as mock_rta:
            loader = build_multi_source_event_audio_loader(
                target_sr=16000,
                settings=settings,
                samples=samples,
            )
            # 2 groups → 2 preload calls at construction
            assert mock_rta.call_count == 2

            r1 = loader(samples[0])
            r2 = loader(samples[1])
            # No additional calls — both served from cache
            assert mock_rta.call_count == 2

        assert isinstance(r1, tuple)
        assert isinstance(r2, tuple)

    def test_preloads_per_group(self, tmp_path: Path) -> None:
        """Samples sharing the same source are grouped and pre-loaded once."""
        settings = _make_settings(tmp_path)
        samples = [
            _FakeEventSample(
                start_sec=1.0,
                end_sec=3.0,
                hydrophone_id="h1",
                start_timestamp=1000.0,
                end_timestamp=1100.0,
            ),
            _FakeEventSample(
                start_sec=20.0,
                end_sec=25.0,
                hydrophone_id="h1",
                start_timestamp=1000.0,
                end_timestamp=1100.0,
            ),
        ]

        with _mock_resolve() as mock_rta:
            loader = build_multi_source_event_audio_loader(
                target_sr=16000,
                settings=settings,
                samples=samples,
            )
            # 1 group → 1 preload call
            assert mock_rta.call_count == 1

            loader(samples[0])
            loader(samples[1])
            # Still 1 call
            assert mock_rta.call_count == 1

    def test_per_request_fallback(self, tmp_path: Path) -> None:
        """Unknown source triggers per-request resolve with context padding."""
        settings = _make_settings(tmp_path)
        known = _FakeEventSample(
            start_sec=5.0,
            end_sec=8.0,
            hydrophone_id="h1",
            start_timestamp=1000.0,
            end_timestamp=1100.0,
        )
        unknown = _FakeEventSample(
            start_sec=5.0,
            end_sec=8.0,
            hydrophone_id="h_new",
            start_timestamp=3000.0,
            end_timestamp=3100.0,
        )

        with _mock_resolve() as mock_rta:
            loader = build_multi_source_event_audio_loader(
                target_sr=16000,
                settings=settings,
                samples=[known],
            )
            # 1 preload call for known
            assert mock_rta.call_count == 1

            # Load unknown → triggers per-request resolve
            result = loader(unknown)
            assert mock_rta.call_count == 2

        assert isinstance(result, tuple)
        # Verify context padding was applied (start_sec used in call)
        call_kwargs = mock_rta.call_args_list[1].kwargs
        # Event is 3s (5-8), context = max(10, 3) = 10, pad = 3.5
        # ctx_start = max(0, 5-3.5) = 1.5 → abs = 3000 + 1.5 = 3001.5
        assert call_kwargs["start_sec"] == pytest.approx(3001.5)

    def test_preloaded_returns_same_buffer_for_group(self, tmp_path: Path) -> None:
        """Pre-loaded samples from same group share buffer reference."""
        settings = _make_settings(tmp_path)
        s1 = _FakeEventSample(
            start_sec=1.0,
            end_sec=3.0,
            hydrophone_id="h1",
            start_timestamp=1000.0,
            end_timestamp=1100.0,
        )
        s2 = _FakeEventSample(
            start_sec=50.0,
            end_sec=55.0,
            hydrophone_id="h1",
            start_timestamp=1000.0,
            end_timestamp=1100.0,
        )

        with _mock_resolve():
            loader = build_multi_source_event_audio_loader(
                target_sr=16000,
                settings=settings,
                samples=[s1, s2],
            )
            audio1, _ = loader(s1)
            audio2, _ = loader(s2)

        # Same pre-loaded buffer
        assert audio1 is audio2


# ===================================================================
# _event_context helper
# ===================================================================


class TestEventContext:
    def test_short_event_symmetric(self) -> None:
        """Short event gets symmetric 10s context."""
        ctx_start, ctx_dur = _event_context(50.0, 52.0, 200.0)
        assert ctx_start == pytest.approx(46.0)
        assert ctx_dur == pytest.approx(10.0)

    def test_long_event_no_extra(self) -> None:
        """Event >= 10s gets no extra padding."""
        ctx_start, ctx_dur = _event_context(50.0, 65.0, 200.0)
        assert ctx_start == pytest.approx(50.0)
        assert ctx_dur == pytest.approx(15.0)

    def test_clamped_to_bounds(self) -> None:
        """Context clamped to [0, job_duration]."""
        ctx_start, ctx_dur = _event_context(1.0, 2.0, 5.0)
        assert ctx_start >= 0.0
        assert ctx_start + ctx_dur <= 5.0

    def test_zero_job_duration(self) -> None:
        """Zero job_duration doesn't crash."""
        ctx_start, ctx_dur = _event_context(5.0, 8.0, 0.0)
        assert ctx_start >= 0.0
        assert ctx_dur >= 0.0
