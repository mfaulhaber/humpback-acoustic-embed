"""Tests for label_processor: scoring, peak detection, extraction, synthesis."""

import numpy as np
import pytest

from humpback.classifier.label_processor import (
    ScorePeak,
    ScoreTimeSeries,
    classify_overlap,
    detect_peaks,
    extract_background_regions,
    extract_clean_window,
    isolate_call_segment,
    map_annotations_to_peaks,
    smooth_scores,
    synthesize_clean_window,
    synthesize_variants,
)
from humpback.classifier.raven_parser import RavenAnnotation


class TestSmoothScores:
    def test_empty_list(self):
        assert smooth_scores([]) == []

    def test_single_value(self):
        assert smooth_scores([0.5]) == [0.5]

    def test_window_1_is_identity(self):
        scores = [0.1, 0.5, 0.9]
        assert smooth_scores(scores, window_size=1) == pytest.approx(scores)

    def test_moving_average(self):
        scores = [0.0, 0.0, 1.0, 0.0, 0.0]
        result = smooth_scores(scores, window_size=3)
        assert len(result) == 5
        # Center value should be averaged with neighbors
        assert result[2] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_edge_padding(self):
        scores = [1.0, 0.0, 0.0]
        result = smooth_scores(scores, window_size=3)
        assert len(result) == 3
        # First value should be padded with itself
        assert result[0] > result[1]


class TestDetectPeaks:
    def test_single_peak(self):
        scores = [0.1, 0.3, 0.8, 0.3, 0.1]
        offsets = [0.0, 1.0, 2.0, 3.0, 4.0]
        peaks = detect_peaks(scores, offsets, threshold_high=0.7)
        assert len(peaks) == 1
        assert peaks[0].index == 2
        assert peaks[0].time_sec == 2.0
        assert peaks[0].score == pytest.approx(0.8)

    def test_multiple_peaks(self):
        scores = [0.1, 0.9, 0.3, 0.1, 0.85, 0.2]
        offsets = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        peaks = detect_peaks(scores, offsets, threshold_high=0.7)
        assert len(peaks) == 2
        assert peaks[0].index == 1
        assert peaks[1].index == 4

    def test_no_peaks_below_threshold(self):
        scores = [0.1, 0.3, 0.5, 0.3, 0.1]
        offsets = [0.0, 1.0, 2.0, 3.0, 4.0]
        peaks = detect_peaks(scores, offsets, threshold_high=0.7)
        assert len(peaks) == 0

    def test_plateau_not_detected(self):
        # Flat plateau should not be detected as peak
        scores = [0.1, 0.8, 0.8, 0.8, 0.1]
        offsets = [0.0, 1.0, 2.0, 3.0, 4.0]
        peaks = detect_peaks(scores, offsets, threshold_high=0.7)
        # Only edges can be peaks (index 1 has equal neighbor at 2)
        assert len(peaks) == 0

    def test_onset_offset_estimation(self):
        scores = [0.1, 0.3, 0.5, 0.9, 0.5, 0.3, 0.1]
        offsets = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        peaks = detect_peaks(
            scores, offsets, threshold_high=0.7, onset_offset_alpha=0.4
        )
        assert len(peaks) == 1
        # Onset should walk back until score < 0.4 * 0.9 = 0.36
        assert peaks[0].onset_sec <= 2.0
        # Offset should walk forward until score < 0.36
        assert peaks[0].offset_sec >= 9.0  # offset_idx=4, offset=4+5=9


class TestClassifyOverlap:
    def _make_peak(self, index, time_sec, score):
        return ScorePeak(
            index=index,
            time_sec=time_sec,
            score=score,
            onset_sec=time_sec - 2.0,
            offset_sec=time_sec + 7.0,
        )

    def test_clean_single_peak(self):
        peak = self._make_peak(5, 5.0, 0.9)
        all_peaks = [peak]
        assert classify_overlap(peak, all_peaks) == "clean"

    def test_mild_overlap_one_neighbor(self):
        peak = self._make_peak(5, 5.0, 0.9)
        neighbor = self._make_peak(7, 7.0, 0.8)
        all_peaks = [peak, neighbor]
        result = classify_overlap(peak, all_peaks, proximity_sec=3.0)
        assert result == "mild_overlap"

    def test_heavy_overlap_two_neighbors(self):
        peak = self._make_peak(5, 5.0, 0.9)
        neighbor1 = self._make_peak(7, 7.0, 0.8)
        neighbor2 = self._make_peak(3, 3.0, 0.7)
        all_peaks = [peak, neighbor1, neighbor2]
        result = classify_overlap(peak, all_peaks, proximity_sec=3.0)
        assert result == "heavy_overlap"

    def test_distant_neighbor_is_clean(self):
        peak = self._make_peak(5, 5.0, 0.9)
        distant = self._make_peak(20, 20.0, 0.9)
        all_peaks = [peak, distant]
        result = classify_overlap(peak, all_peaks, proximity_sec=3.0)
        assert result == "clean"

    def test_weak_neighbor_is_clean(self):
        peak = self._make_peak(5, 5.0, 0.9)
        weak = self._make_peak(7, 7.0, 0.2)  # Below 0.5 * 0.9
        all_peaks = [peak, weak]
        result = classify_overlap(peak, all_peaks, proximity_sec=3.0)
        assert result == "clean"


class TestMapAnnotationsToPeaks:
    def _make_annotation(self, begin, end, call_type="Moan"):
        return RavenAnnotation(
            selection=1,
            begin_time=begin,
            end_time=end,
            low_freq=100.0,
            high_freq=1000.0,
            call_type=call_type,
        )

    def test_annotation_matched_to_peak(self):
        annotations = [self._make_annotation(10.0, 12.0)]
        peaks = [
            ScorePeak(
                index=10, time_sec=10.5, score=0.9, onset_sec=9.0, offset_sec=15.5
            )
        ]
        results = map_annotations_to_peaks(annotations, peaks)
        assert len(results) == 1
        assert results[0].peak is not None
        assert results[0].peak.time_sec == 10.5
        assert results[0].treatment == "clean"

    def test_annotation_no_matching_peak(self):
        annotations = [self._make_annotation(50.0, 52.0)]
        peaks = [
            ScorePeak(
                index=10, time_sec=10.0, score=0.9, onset_sec=9.0, offset_sec=15.0
            )
        ]
        results = map_annotations_to_peaks(annotations, peaks, tolerance_sec=5.0)
        assert len(results) == 1
        assert results[0].peak is None
        assert results[0].treatment == "fallback"

    def test_mild_overlap_routes_to_synthesized(self):
        """Mild overlap annotations should be routed to synthesized treatment."""
        annotations = [self._make_annotation(10.0, 12.0)]
        # Two nearby peaks → mild overlap
        peaks = [
            ScorePeak(
                index=10, time_sec=10.5, score=0.9, onset_sec=9.0, offset_sec=15.5
            ),
            ScorePeak(
                index=12, time_sec=12.5, score=0.8, onset_sec=11.0, offset_sec=17.5
            ),
        ]
        results = map_annotations_to_peaks(
            annotations, peaks, proximity_sec=3.0, relative_threshold=0.5
        )
        assert len(results) == 1
        assert results[0].overlap_status == "mild_overlap"
        assert results[0].treatment == "synthesized"

    def test_annotation_with_tolerance(self):
        annotations = [self._make_annotation(10.0, 12.0)]
        # Peak just outside annotation but within tolerance
        peaks = [
            ScorePeak(
                index=14, time_sec=14.0, score=0.9, onset_sec=12.0, offset_sec=19.0
            )
        ]
        results = map_annotations_to_peaks(annotations, peaks, tolerance_sec=5.0)
        assert len(results) == 1
        assert results[0].peak is not None


class TestExtractCleanWindow:
    def test_basic_extraction(self):
        sr = 16000
        audio = np.random.randn(sr * 30).astype(np.float32)  # 30 seconds
        peak = ScorePeak(
            index=10, time_sec=10.0, score=0.9, onset_sec=8.0, offset_sec=15.0
        )
        sample = extract_clean_window(peak, audio, sr, window_size=5.0)
        assert sample is not None
        assert len(sample.audio_segment) == sr * 5
        assert sample.start_sec == 10.0
        assert sample.end_sec == 15.0

    def test_clamps_to_end(self):
        sr = 16000
        audio = np.random.randn(sr * 12).astype(np.float32)  # 12 seconds
        peak = ScorePeak(
            index=10, time_sec=10.0, score=0.9, onset_sec=8.0, offset_sec=15.0
        )
        sample = extract_clean_window(peak, audio, sr, window_size=5.0)
        assert sample is not None
        assert sample.end_sec <= 12.0
        assert sample.start_sec >= 7.0

    def test_too_short_returns_none(self):
        sr = 16000
        audio = np.random.randn(sr * 3).astype(np.float32)  # 3 seconds
        peak = ScorePeak(
            index=0, time_sec=0.0, score=0.9, onset_sec=0.0, offset_sec=5.0
        )
        sample = extract_clean_window(peak, audio, sr, window_size=5.0)
        assert sample is None

    def test_extraction_preserves_shape(self):
        sr = 32000
        audio = np.ones(sr * 20, dtype=np.float32)
        peak = ScorePeak(
            index=5, time_sec=5.0, score=0.8, onset_sec=3.0, offset_sec=10.0
        )
        sample = extract_clean_window(peak, audio, sr, window_size=5.0)
        assert sample is not None
        assert sample.audio_segment.shape == (sr * 5,)
        assert sample.sr == sr


class TestExtractFallbackWindow:
    """Tests for extract_fallback_window (annotation-center extraction)."""

    def test_basic_fallback(self):
        from humpback.classifier.label_processor import extract_fallback_window

        sr = 16000
        audio = np.zeros(sr * 30, dtype=np.float32)
        ann = RavenAnnotation(
            selection=1,
            begin_time=10.0,
            end_time=12.0,
            low_freq=200,
            high_freq=3000,
            call_type="Moan",
        )
        sample = extract_fallback_window(ann, audio, sr, window_size=5.0)
        assert sample is not None
        assert sample.treatment == "fallback"
        assert sample.audio_segment.shape == (sr * 5,)
        # Window should be centered on midpoint (11.0s) → start ~8.5s
        assert abs(sample.start_sec - 8.5) < 0.01

    def test_fallback_clamps_to_audio_end(self):
        from humpback.classifier.label_processor import extract_fallback_window

        sr = 16000
        audio = np.zeros(sr * 10, dtype=np.float32)
        ann = RavenAnnotation(
            selection=1,
            begin_time=8.0,
            end_time=9.5,
            low_freq=200,
            high_freq=3000,
            call_type="Chirp",
        )
        sample = extract_fallback_window(ann, audio, sr, window_size=5.0)
        assert sample is not None
        assert sample.start_sec == 5.0  # clamped: 10.0 - 5.0
        assert sample.end_sec == 10.0

    def test_fallback_audio_too_short(self):
        from humpback.classifier.label_processor import extract_fallback_window

        sr = 16000
        audio = np.zeros(sr * 3, dtype=np.float32)  # 3s < 5s * 0.9
        ann = RavenAnnotation(
            selection=1,
            begin_time=1.0,
            end_time=2.0,
            low_freq=200,
            high_freq=3000,
            call_type="Moan",
        )
        sample = extract_fallback_window(ann, audio, sr, window_size=5.0)
        assert sample is None


# ---------------------------------------------------------------------------
# Background extraction and synthesis tests
# ---------------------------------------------------------------------------


class TestExtractBackgroundRegions:
    def test_finds_low_score_regions(self):
        sr = 16000
        duration = 30.0
        audio = np.random.randn(int(sr * duration)).astype(np.float32)
        # Scores: low for first 10s, high in middle, low at end
        n_scores = int(duration)  # 1 Hz hop
        scores = [0.02] * 10 + [0.9] * 10 + [0.02] * 10
        offsets = [float(i) for i in range(n_scores)]
        series = ScoreTimeSeries(
            offsets=offsets,
            raw_scores=scores,
            smoothed_scores=scores,
            hop_seconds=1.0,
            window_size=5.0,
        )
        regions = extract_background_regions(
            series, audio, sr, threshold=0.1, min_duration=5.0, window_size=5.0
        )
        assert len(regions) >= 2  # at least one from each low run
        for r in regions:
            assert len(r) == sr * 5

    def test_no_regions_when_all_high(self):
        sr = 16000
        audio = np.random.randn(sr * 10).astype(np.float32)
        scores = [0.8] * 10
        offsets = [float(i) for i in range(10)]
        series = ScoreTimeSeries(
            offsets=offsets,
            raw_scores=scores,
            smoothed_scores=scores,
            hop_seconds=1.0,
            window_size=5.0,
        )
        regions = extract_background_regions(series, audio, sr)
        assert regions == []

    def test_short_run_ignored(self):
        sr = 16000
        audio = np.random.randn(sr * 10).astype(np.float32)
        # Only 3s of low scores → below min_duration
        scores = [0.02, 0.02, 0.02, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        offsets = [float(i) for i in range(10)]
        series = ScoreTimeSeries(
            offsets=offsets,
            raw_scores=scores,
            smoothed_scores=scores,
            hop_seconds=1.0,
            window_size=5.0,
        )
        regions = extract_background_regions(series, audio, sr, min_duration=5.0)
        assert regions == []


class TestIsolateCallSegment:
    def test_basic_isolation(self):
        sr = 16000
        audio = np.random.randn(sr * 30).astype(np.float32)
        peak = ScorePeak(
            index=10, time_sec=10.0, score=0.9, onset_sec=9.5, offset_sec=16.5
        )
        series = ScoreTimeSeries(
            offsets=[float(i) for i in range(30)],
            raw_scores=[0.1] * 30,
            smoothed_scores=[0.1] * 30,
            hop_seconds=1.0,
            window_size=5.0,
        )
        result = isolate_call_segment(peak, series, audio, sr)
        assert result is not None
        seg, dur = result
        assert 1.0 <= dur <= 3.0
        assert len(seg) == int(dur * sr)

    def test_very_short_audio_returns_none(self):
        sr = 16000
        audio = np.random.randn(sr).astype(np.float32)  # 1 second
        peak = ScorePeak(
            index=0, time_sec=0.0, score=0.9, onset_sec=0.0, offset_sec=5.0
        )
        series = ScoreTimeSeries(
            offsets=[0.0],
            raw_scores=[0.9],
            smoothed_scores=[0.9],
            hop_seconds=1.0,
            window_size=5.0,
        )
        result = isolate_call_segment(peak, series, audio, sr)
        # Should still produce a segment (audio is 1s, min dur is 0.5s)
        assert result is not None
        _, dur = result
        assert dur >= 0.5


class TestSynthesizeCleanWindow:
    def test_basic_synthesis(self):
        sr = 16000
        call = np.random.randn(sr * 2).astype(np.float32)  # 2s call
        bg = np.random.randn(sr * 5).astype(np.float32)  # 5s background
        sample = synthesize_clean_window(call, bg, sr, window_size=5.0)
        assert sample is not None
        assert len(sample.audio_segment) == sr * 5
        assert sample.treatment == "synthesized"

    def test_placement_centre(self):
        sr = 16000
        # Use a distinctive call (ones) vs background (zeros)
        call = np.ones(sr * 1, dtype=np.float32)
        bg = np.zeros(sr * 5, dtype=np.float32)
        sample = synthesize_clean_window(
            call, bg, sr, window_size=5.0, placement_sec=2.0
        )
        assert sample is not None
        # Call energy should appear around sample 2*sr
        mid = sample.audio_segment[int(2.0 * sr) : int(3.0 * sr)]
        assert float(np.mean(np.abs(mid))) > 0.5

    def test_short_background_returns_none(self):
        sr = 16000
        call = np.random.randn(sr * 2).astype(np.float32)
        bg = np.random.randn(sr * 3).astype(np.float32)  # too short for 5s
        sample = synthesize_clean_window(call, bg, sr, window_size=5.0)
        assert sample is None


class TestSynthesizeVariants:
    def test_generates_three_variants(self):
        sr = 16000
        call = np.random.randn(sr * 2).astype(np.float32)
        bgs = [np.random.randn(sr * 5).astype(np.float32) for _ in range(3)]
        variants = synthesize_variants(call, bgs, sr, window_size=5.0, n_variants=3)
        assert len(variants) == 3
        for v in variants:
            assert len(v.audio_segment) == sr * 5
            assert v.treatment == "synthesized"

    def test_reuses_backgrounds_when_fewer_available(self):
        sr = 16000
        call = np.random.randn(sr * 2).astype(np.float32)
        bgs = [np.random.randn(sr * 5).astype(np.float32)]  # only 1 bg
        variants = synthesize_variants(call, bgs, sr, window_size=5.0, n_variants=3)
        assert len(variants) == 3  # should still produce 3 via reuse

    def test_no_backgrounds_returns_empty(self):
        sr = 16000
        call = np.random.randn(sr * 2).astype(np.float32)
        variants = synthesize_variants(call, [], sr, window_size=5.0, n_variants=3)
        assert variants == []

    def test_variants_differ(self):
        sr = 16000
        call = np.ones(sr, dtype=np.float32) * 0.5
        bgs = [np.random.randn(sr * 5).astype(np.float32) for _ in range(3)]
        variants = synthesize_variants(call, bgs, sr, window_size=5.0, n_variants=3)
        # Each variant should have different content due to different placement + bg
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                assert not np.array_equal(
                    variants[i].audio_segment, variants[j].audio_segment
                )
