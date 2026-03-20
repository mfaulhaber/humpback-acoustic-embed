"""Score-based segmentation: score recordings, detect peaks, extract clean 5s windows."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from humpback.classifier.raven_parser import RavenAnnotation
from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.features import extract_logmel_batch
from humpback.processing.inference import EmbeddingModel
from humpback.processing.windowing import slice_windows_with_metadata

logger = logging.getLogger(__name__)


@dataclass
class ScoreTimeSeries:
    """Per-window classifier scores for a recording."""

    offsets: list[float]  # window start times (seconds)
    raw_scores: list[float]  # classifier P(whale) per window
    smoothed_scores: list[float]
    hop_seconds: float
    window_size: float


@dataclass
class ScorePeak:
    """A detected score peak with onset/offset estimates."""

    index: int  # index into score array
    time_sec: float  # peak start time (= offset[index])
    score: float  # smoothed score at peak
    onset_sec: float  # estimated call onset time
    offset_sec: float  # estimated call offset time (+ window_size)


@dataclass
class AnnotatedPeak:
    """An annotation matched to a score peak with overlap classification."""

    annotation: RavenAnnotation
    peak: ScorePeak | None
    overlap_status: str  # "clean", "mild_overlap", "heavy_overlap"
    treatment: str  # "clean", "synthesized", "fallback", "skipped"


@dataclass
class ExtractedSample:
    """A 5-second audio clip extracted from a recording."""

    audio_segment: np.ndarray  # float32 mono at target SR
    sr: int
    call_type: str
    treatment: str
    source_filename: str
    start_sec: float
    end_sec: float
    peak_score: float


@dataclass
class ProcessingResult:
    """Summary of processing one recording."""

    filename: str
    annotations_total: int
    annotated_peaks: list[AnnotatedPeak] = field(default_factory=list)
    extracted_samples: list[ExtractedSample] = field(default_factory=list)
    skipped_no_peak: int = 0


def smooth_scores(scores: list[float], window_size: int = 3) -> list[float]:
    """Smooth scores with centered moving average and edge padding.

    Replicates the algorithm from classifier/extractor.py:_smooth_scores().
    """
    if not scores:
        return []
    if window_size <= 1 or len(scores) == 1:
        return [float(v) for v in scores]
    arr = np.asarray(scores, dtype=np.float32)
    pad = window_size // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return [float(v) for v in smoothed]


def score_recording(
    audio_path: Path,
    pipeline: Pipeline,
    model: EmbeddingModel,
    window_size: float = 5.0,
    target_sr: int = 32000,
    hop_seconds: float = 1.0,
    input_format: str = "spectrogram",
    feature_config: dict | None = None,
    smoothing_window: int = 3,
) -> ScoreTimeSeries:
    """Score all windows in a recording using the classifier pipeline.

    Returns a ScoreTimeSeries with raw and smoothed classifier scores.
    Reuses the same decode->window->features->embed->classify pattern
    as detector.py:run_detection().
    """
    feature_config = feature_config or {}
    normalization = feature_config.get("normalization", "per_window_max")

    audio, sr = decode_audio(audio_path)
    audio = resample(audio, sr, target_sr)

    window_samples = int(target_sr * window_size)
    if len(audio) < window_samples:
        logger.warning(
            "Audio too short for windowing: %.3fs < %.1fs",
            len(audio) / target_sr,
            window_size,
        )
        return ScoreTimeSeries(
            offsets=[],
            raw_scores=[],
            smoothed_scores=[],
            hop_seconds=hop_seconds,
            window_size=window_size,
        )

    # Collect all windows
    raw_windows: list[np.ndarray] = []
    offsets: list[float] = []
    for window, meta in slice_windows_with_metadata(
        audio, target_sr, window_size, hop_seconds=hop_seconds
    ):
        raw_windows.append(window)
        offsets.append(meta.offset_sec)

    if not raw_windows:
        return ScoreTimeSeries(
            offsets=[],
            raw_scores=[],
            smoothed_scores=[],
            hop_seconds=hop_seconds,
            window_size=window_size,
        )

    # Feature extraction
    if input_format == "waveform":
        batch_items: list[np.ndarray] = raw_windows
    else:
        batch_items = extract_logmel_batch(
            raw_windows,
            target_sr,
            n_mels=128,
            hop_length=1252,
            target_frames=128,
            normalization=normalization,
        )

    # Batch embed (groups of 64)
    batch_size = 64
    file_embeddings: list[np.ndarray] = []
    for i in range(0, len(batch_items), batch_size):
        batch = np.stack(batch_items[i : i + batch_size])
        embeddings = model.embed(batch)
        file_embeddings.append(embeddings)

    all_emb = np.vstack(file_embeddings)

    # Classify
    proba = pipeline.predict_proba(all_emb)[:, 1]
    raw_scores = proba.tolist()

    smoothed = smooth_scores(raw_scores, smoothing_window)

    return ScoreTimeSeries(
        offsets=[float(o) for o in offsets],
        raw_scores=raw_scores,
        smoothed_scores=smoothed,
        hop_seconds=hop_seconds,
        window_size=window_size,
    )


def detect_peaks(
    smoothed_scores: list[float],
    offsets: list[float],
    threshold_high: float = 0.7,
    window_size: float = 5.0,
    onset_offset_alpha: float = 0.4,
) -> list[ScorePeak]:
    """Detect local maxima in smoothed scores above threshold.

    For each peak, estimates onset/offset using relative threshold descent.
    """
    if len(smoothed_scores) < 2:
        return []

    peaks: list[ScorePeak] = []
    n = len(smoothed_scores)

    for i in range(n):
        s = smoothed_scores[i]
        if s < threshold_high:
            continue

        # Local maximum check
        if i > 0 and smoothed_scores[i - 1] >= s:
            continue
        if i < n - 1 and smoothed_scores[i + 1] >= s:
            continue

        # Estimate onset/offset
        onset_sec, offset_sec = _estimate_onset_offset(
            smoothed_scores, offsets, i, onset_offset_alpha, window_size
        )

        peaks.append(
            ScorePeak(
                index=i,
                time_sec=offsets[i],
                score=s,
                onset_sec=onset_sec,
                offset_sec=offset_sec,
            )
        )

    return peaks


def _estimate_onset_offset(
    smoothed_scores: list[float],
    offsets: list[float],
    peak_idx: int,
    alpha: float,
    window_size: float,
) -> tuple[float, float]:
    """Estimate onset and offset times from a score peak using relative threshold.

    Walks backward/forward from the peak until score drops below alpha * peak_score.
    Returns (onset_sec, offset_sec) where offset includes the window duration.
    """
    peak_score = smoothed_scores[peak_idx]
    threshold = alpha * peak_score
    n = len(smoothed_scores)

    # Walk backward for onset
    onset_idx = peak_idx
    for j in range(peak_idx - 1, -1, -1):
        if smoothed_scores[j] < threshold:
            break
        onset_idx = j

    # Walk forward for offset
    offset_idx = peak_idx
    for j in range(peak_idx + 1, n):
        if smoothed_scores[j] < threshold:
            break
        offset_idx = j

    onset_sec = offsets[onset_idx]
    offset_sec = offsets[offset_idx] + window_size
    return onset_sec, offset_sec


def classify_overlap(
    peak: ScorePeak,
    all_peaks: list[ScorePeak],
    window_size: float = 5.0,
    proximity_sec: float = 3.0,
    relative_threshold: float = 0.5,
) -> str:
    """Classify overlap status for a peak.

    Returns "clean", "mild_overlap", or "heavy_overlap".
    """
    nearby_count = 0
    for other in all_peaks:
        if other.index == peak.index:
            continue
        distance = abs(other.time_sec - peak.time_sec)
        if distance <= proximity_sec and other.score >= relative_threshold * peak.score:
            nearby_count += 1

    if nearby_count == 0:
        # Also check for wide plateau (score region > 2x window)
        region_duration = peak.offset_sec - peak.onset_sec
        if region_duration > 2 * window_size:
            return "mild_overlap"
        return "clean"
    elif nearby_count == 1:
        return "mild_overlap"
    else:
        return "heavy_overlap"


def map_annotations_to_peaks(
    annotations: list[RavenAnnotation],
    peaks: list[ScorePeak],
    window_size: float = 5.0,
    tolerance_sec: float = 5.0,
    proximity_sec: float = 3.0,
    relative_threshold: float = 0.5,
) -> list[AnnotatedPeak]:
    """Map each annotation to its best-matching score peak.

    For each annotation, finds the highest-scoring peak whose time falls
    within the annotation bounds (expanded by tolerance). Classifies overlap
    status and assigns treatment.
    """
    results: list[AnnotatedPeak] = []

    for ann in annotations:
        # Find peaks within annotation time range (+ tolerance)
        search_start = ann.begin_time - tolerance_sec
        search_end = ann.end_time + tolerance_sec

        candidates = [p for p in peaks if search_start <= p.time_sec <= search_end]

        if not candidates:
            results.append(
                AnnotatedPeak(
                    annotation=ann,
                    peak=None,
                    overlap_status="clean",
                    treatment="fallback",
                )
            )
            continue

        # Select best peak (highest score, preferring those inside annotation bounds)
        best = _select_best_peak(candidates, ann)

        overlap_status = classify_overlap(
            best,
            peaks,
            window_size=window_size,
            proximity_sec=proximity_sec,
            relative_threshold=relative_threshold,
        )

        # Assign treatment based on overlap
        if overlap_status == "clean":
            treatment = "clean"
        else:
            treatment = "synthesized"

        results.append(
            AnnotatedPeak(
                annotation=ann,
                peak=best,
                overlap_status=overlap_status,
                treatment=treatment,
            )
        )

    return results


def _select_best_peak(
    candidates: list[ScorePeak],
    annotation: RavenAnnotation,
) -> ScorePeak:
    """Select the best peak from candidates for an annotation.

    Prefers peaks inside annotation bounds; ties broken by score.
    """
    inside = [
        p
        for p in candidates
        if annotation.begin_time <= p.time_sec <= annotation.end_time
    ]
    pool = inside if inside else candidates
    return max(pool, key=lambda p: p.score)


def extract_clean_window(
    peak: ScorePeak,
    full_audio: np.ndarray,
    sr: int,
    window_size: float = 5.0,
) -> ExtractedSample | None:
    """Extract a 5-second window centered on the peak.

    The window starts at the peak time (already a window start offset).
    Returns None if the audio is too short.
    """
    total_duration = len(full_audio) / sr
    start_sec = peak.time_sec
    end_sec = start_sec + window_size

    # Clamp to audio bounds
    if end_sec > total_duration:
        start_sec = max(0.0, total_duration - window_size)
        end_sec = start_sec + window_size
    if start_sec < 0:
        start_sec = 0.0
        end_sec = min(window_size, total_duration)

    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    segment = full_audio[start_sample:end_sample]

    expected_samples = int(window_size * sr)
    if len(segment) < expected_samples * 0.9:
        return None

    # Pad if slightly short
    if len(segment) < expected_samples:
        segment = np.pad(segment, (0, expected_samples - len(segment)))

    return ExtractedSample(
        audio_segment=segment[:expected_samples],
        sr=sr,
        call_type="",  # caller sets this
        treatment="clean",
        source_filename="",  # caller sets this
        start_sec=start_sec,
        end_sec=end_sec,
        peak_score=peak.score,
    )


def extract_fallback_window(
    annotation: RavenAnnotation,
    full_audio: np.ndarray,
    sr: int,
    window_size: float = 5.0,
) -> ExtractedSample | None:
    """Extract a window centered on the annotation midpoint (fallback when no peak found).

    Used when no classifier peak exceeds the threshold within the annotation bounds.
    """
    total_duration = len(full_audio) / sr
    if total_duration < window_size * 0.9:
        return None

    midpoint = (annotation.begin_time + annotation.end_time) / 2.0
    start_sec = midpoint - window_size / 2.0

    # Clamp to audio bounds
    start_sec = max(0.0, min(start_sec, total_duration - window_size))
    end_sec = start_sec + window_size

    start_sample = int(start_sec * sr)
    expected_samples = int(window_size * sr)
    segment = full_audio[start_sample : start_sample + expected_samples]

    if len(segment) < int(expected_samples * 0.9):
        return None

    if len(segment) < expected_samples:
        segment = np.pad(segment, (0, expected_samples - len(segment)))

    return ExtractedSample(
        audio_segment=segment[:expected_samples],
        sr=sr,
        call_type="",  # caller sets this
        treatment="fallback",
        source_filename="",  # caller sets this
        start_sec=start_sec,
        end_sec=end_sec,
        peak_score=0.0,
    )


# ---------------------------------------------------------------------------
# Background extraction and synthesis
# ---------------------------------------------------------------------------


def extract_background_regions(
    score_series: ScoreTimeSeries,
    full_audio: np.ndarray,
    sr: int,
    threshold: float = 0.1,
    min_duration: float = 5.0,
    window_size: float = 5.0,
) -> list[np.ndarray]:
    """Extract background (low-score) audio regions for synthesis use.

    Finds contiguous stretches where the smoothed score stays below *threshold*
    for at least *min_duration* seconds, then extracts non-overlapping
    *window_size*-second segments from each stretch.
    """
    if not score_series.smoothed_scores:
        return []

    hop = score_series.hop_seconds
    total_duration = len(full_audio) / sr
    regions: list[np.ndarray] = []

    # Walk through scores and find contiguous low-score runs
    run_start_idx: int | None = None
    for i, s in enumerate(score_series.smoothed_scores):
        if s < threshold:
            if run_start_idx is None:
                run_start_idx = i
        else:
            if run_start_idx is not None:
                _extract_from_run(
                    run_start_idx,
                    i,
                    score_series.offsets,
                    full_audio,
                    sr,
                    hop,
                    min_duration,
                    window_size,
                    total_duration,
                    regions,
                )
                run_start_idx = None

    # Handle run extending to end
    if run_start_idx is not None:
        _extract_from_run(
            run_start_idx,
            len(score_series.smoothed_scores),
            score_series.offsets,
            full_audio,
            sr,
            hop,
            min_duration,
            window_size,
            total_duration,
            regions,
        )

    return regions


def _extract_from_run(
    start_idx: int,
    end_idx: int,
    offsets: list[float],
    full_audio: np.ndarray,
    sr: int,
    hop: float,
    min_duration: float,
    window_size: float,
    total_duration: float,
    out: list[np.ndarray],
) -> None:
    """Extract non-overlapping windows from a single low-score run."""
    run_start_sec = offsets[start_idx]
    run_end_sec = offsets[min(end_idx, len(offsets) - 1)] + hop

    if run_end_sec - run_start_sec < min_duration:
        return

    pos = run_start_sec
    expected_samples = int(window_size * sr)
    while pos + window_size <= min(run_end_sec, total_duration):
        s = int(pos * sr)
        seg = full_audio[s : s + expected_samples]
        if len(seg) >= int(expected_samples * 0.9):
            if len(seg) < expected_samples:
                seg = np.pad(seg, (0, expected_samples - len(seg)))
            out.append(seg[:expected_samples])
        pos += window_size  # non-overlapping stride


def isolate_call_segment(
    peak: ScorePeak,
    score_series: ScoreTimeSeries,
    full_audio: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, float] | None:
    """Isolate the cleanest 1–3 s call segment around a peak using onset/offset.

    Returns ``(segment_audio, duration_sec)`` or ``None`` if the audio is too
    short or the onset/offset span is degenerate.
    """
    onset = peak.onset_sec
    offset = peak.offset_sec - score_series.window_size  # strip trailing window pad

    # Clamp to reasonable call duration (1–3 s centred on peak)
    call_centre = peak.time_sec + score_series.window_size / 2.0
    raw_dur = max(offset - onset, 0.0)
    dur = min(max(raw_dur, 1.0), 3.0)

    start = call_centre - dur / 2.0
    start = max(0.0, start)
    end = start + dur

    total_dur = len(full_audio) / sr
    if end > total_dur:
        end = total_dur
        start = max(0.0, end - dur)
        dur = end - start

    if dur < 0.5:
        return None

    seg = full_audio[int(start * sr) : int(end * sr)]
    return seg, dur


def _raised_cosine_fade(length: int) -> np.ndarray:
    """Return a half-cosine ramp from 0 → 1 of *length* samples."""
    if length <= 0:
        return np.array([], dtype=np.float32)
    return (0.5 * (1.0 - np.cos(np.pi * np.arange(length) / length))).astype(np.float32)


def synthesize_clean_window(
    call_segment: np.ndarray,
    background_segment: np.ndarray,
    sr: int,
    window_size: float = 5.0,
    placement_sec: float | None = None,
    crossfade_ms: float = 50.0,
) -> ExtractedSample | None:
    """Place a call segment into a background window with crossfade splicing.

    *placement_sec* is the desired call start position within the window;
    if ``None``, the call is centred.  Background RMS is roughly matched and
    a short raised-cosine crossfade is applied at splice boundaries.
    """
    expected_samples = int(window_size * sr)
    if len(background_segment) < expected_samples:
        return None

    bg = background_segment[:expected_samples].copy().astype(np.float32)
    call = call_segment.astype(np.float32)

    call_samples = len(call)
    if call_samples >= expected_samples:
        # Call fills the whole window — just return it
        return ExtractedSample(
            audio_segment=call[:expected_samples],
            sr=sr,
            call_type="",
            treatment="synthesized",
            source_filename="",
            start_sec=0.0,
            end_sec=window_size,
            peak_score=0.0,
        )

    # Determine placement
    if placement_sec is None:
        placement_sec = (window_size - call_samples / sr) / 2.0
    insert_sample = int(max(0.0, placement_sec) * sr)
    insert_sample = min(insert_sample, expected_samples - call_samples)

    # Match background RMS to call neighbourhood (avoid dead silence)
    bg_rms = float(np.sqrt(np.mean(bg**2))) + 1e-10
    call_rms = float(np.sqrt(np.mean(call**2))) + 1e-10
    # Keep background quieter than call (target ~20% of call energy)
    target_bg_rms = 0.2 * call_rms
    bg *= target_bg_rms / bg_rms

    # Cap transient spikes in the scaled background to prevent them from
    # dominating peak normalization (which happens in write_flac_file).
    # A crest factor >6 indicates impulsive noise; clamp peaks to 6× RMS.
    bg_peak = float(np.max(np.abs(bg)))
    bg_peak_limit = 6.0 * target_bg_rms
    if bg_peak > bg_peak_limit:
        np.clip(bg, -bg_peak_limit, bg_peak_limit, out=bg)

    # Insert call with crossfade (overlap-add with complementary weights)
    fade_len = max(1, int(crossfade_ms / 1000.0 * sr))
    end_sample = insert_sample + call_samples

    # Crossfade lengths (limited by available space on each side)
    pre_fade = min(fade_len, insert_sample, call_samples)
    post_fade = min(fade_len, expected_samples - end_sample, call_samples)

    # Ensure crossfade regions don't overlap within the call
    if pre_fade + post_fade > call_samples:
        half = call_samples // 2
        pre_fade = min(pre_fade, half)
        post_fade = min(post_fade, call_samples - pre_fade)

    result = bg.copy()

    # Entry crossfade: background → call (overlap-add blend)
    if pre_fade > 0:
        ramp = _raised_cosine_fade(pre_fade)  # 0 → 1
        idx = slice(insert_sample, insert_sample + pre_fade)
        result[idx] = bg[idx] * (1.0 - ramp) + call[:pre_fade] * ramp

    # Pure call region (between crossfades)
    pure_start = insert_sample + pre_fade
    pure_end = end_sample - post_fade
    if pure_start < pure_end:
        result[pure_start:pure_end] = call[pre_fade : call_samples - post_fade]

    # Exit crossfade: call → background (overlap-add blend)
    if post_fade > 0:
        ramp = _raised_cosine_fade(post_fade)  # 0 → 1
        idx = slice(end_sample - post_fade, end_sample)
        result[idx] = call[-post_fade:] * (1.0 - ramp) + bg[idx] * ramp

    return ExtractedSample(
        audio_segment=result[:expected_samples],
        sr=sr,
        call_type="",
        treatment="synthesized",
        source_filename="",
        start_sec=0.0,
        end_sec=window_size,
        peak_score=0.0,
    )


def synthesize_variants(
    call_segment: np.ndarray,
    backgrounds: list[np.ndarray],
    sr: int,
    window_size: float = 5.0,
    crossfade_ms: float = 50.0,
    n_variants: int = 3,
) -> list[ExtractedSample]:
    """Generate up to *n_variants* placement variants of a call in different backgrounds.

    Variant placements: early (~0.5–1.0 s), centre, late (call ends ~4.0–4.5 s).
    Each variant uses a different background when available.
    """
    call_dur = len(call_segment) / sr
    placements = [
        0.75,  # early
        (window_size - call_dur) / 2.0,  # centre
        window_size - call_dur - 0.75,  # late
    ]
    # Clamp placements
    placements = [max(0.0, min(p, window_size - call_dur)) for p in placements]

    results: list[ExtractedSample] = []
    for i in range(min(n_variants, len(placements))):
        bg = backgrounds[i % len(backgrounds)] if backgrounds else None
        if bg is None:
            continue
        sample = synthesize_clean_window(
            call_segment,
            bg,
            sr,
            window_size=window_size,
            placement_sec=placements[i],
            crossfade_ms=crossfade_ms,
        )
        if sample is not None:
            results.append(sample)

    return results


def process_recording(
    audio_path: Path,
    annotations: list[RavenAnnotation],
    pipeline: Pipeline,
    model: EmbeddingModel,
    window_size: float = 5.0,
    target_sr: int = 32000,
    hop_seconds: float = 1.0,
    input_format: str = "spectrogram",
    feature_config: dict | None = None,
    threshold_high: float = 0.7,
    smoothing_window: int = 3,
    onset_offset_alpha: float = 0.4,
    overlap_proximity_sec: float = 3.0,
    overlap_relative_threshold: float = 0.5,
    enable_synthesized: bool = True,
    background_threshold: float = 0.1,
    synthesis_crossfade_ms: float = 50.0,
    synthesis_variants: int = 3,
) -> ProcessingResult:
    """Process a single recording: score, detect peaks, map annotations, extract.

    Handles clean and synthesised treatments.  All annotations with a peak
    get a synthesis attempt (including clean ones, as additional variants).
    """
    filename = audio_path.name
    result = ProcessingResult(
        filename=filename,
        annotations_total=len(annotations),
    )

    # Score the recording
    scores = score_recording(
        audio_path,
        pipeline,
        model,
        window_size=window_size,
        target_sr=target_sr,
        hop_seconds=hop_seconds,
        input_format=input_format,
        feature_config=feature_config,
        smoothing_window=smoothing_window,
    )

    if not scores.smoothed_scores:
        logger.warning("No scores generated for %s", filename)
        result.skipped_no_peak = len(annotations)
        for ann in annotations:
            result.annotated_peaks.append(
                AnnotatedPeak(
                    annotation=ann,
                    peak=None,
                    overlap_status="clean",
                    treatment="skipped",
                )
            )
        return result

    # Detect peaks
    peaks = detect_peaks(
        scores.smoothed_scores,
        scores.offsets,
        threshold_high=threshold_high,
        window_size=window_size,
        onset_offset_alpha=onset_offset_alpha,
    )

    # Map annotations to peaks
    annotated_peaks = map_annotations_to_peaks(
        annotations,
        peaks,
        window_size=window_size,
        proximity_sec=overlap_proximity_sec,
        relative_threshold=overlap_relative_threshold,
    )
    result.annotated_peaks = annotated_peaks

    # Load full audio for extraction
    audio, sr = decode_audio(audio_path)
    audio = resample(audio, sr, target_sr)

    # --- Pass 0: fallback extraction (no peak found) -------------------
    for ap in annotated_peaks:
        if ap.treatment != "fallback":
            continue
        sample = extract_fallback_window(
            ap.annotation, audio, target_sr, window_size=window_size
        )
        if sample is None:
            ap.treatment = "skipped"
            result.skipped_no_peak += 1
            continue
        sample.call_type = ap.annotation.call_type
        sample.source_filename = filename
        result.extracted_samples.append(sample)

    # --- Pass 1: extract clean windows ---------------------------------
    for ap in annotated_peaks:
        if ap.treatment == "skipped" or ap.peak is None:
            result.skipped_no_peak += 1
            continue

        if ap.treatment != "clean":
            continue

        sample = extract_clean_window(
            ap.peak, audio, target_sr, window_size=window_size
        )
        if sample is None:
            ap.treatment = "skipped"
            result.skipped_no_peak += 1
            continue

        sample.call_type = ap.annotation.call_type
        sample.source_filename = filename
        result.extracted_samples.append(sample)

    # --- Pass 2: synthesise all annotations with a peak -----------------
    # This runs for every annotation that has a matched peak (including clean
    # ones), producing additional synthesis variants alongside the clean
    # extraction from Pass 1.  Annotations that already got a clean extraction
    # are NOT marked skipped if synthesis fails.
    if enable_synthesized:
        # Extract background regions once per recording
        backgrounds = extract_background_regions(
            scores,
            audio,
            target_sr,
            threshold=background_threshold,
            min_duration=window_size,
            window_size=window_size,
        )

        # Track which annotations already have a clean extraction
        has_clean = {
            id(ap)
            for ap in annotated_peaks
            if ap.peak is not None
            and ap.treatment == "clean"
            and any(
                s.source_filename == filename
                and s.treatment == "clean"
                and s.start_sec == ap.peak.time_sec
                for s in result.extracted_samples
            )
        }

        for ap in annotated_peaks:
            if ap.peak is None:
                continue

            if not backgrounds:
                logger.warning(
                    "No background regions for synthesis in %s — skipping", filename
                )
                if id(ap) not in has_clean:
                    ap.treatment = "skipped"
                    result.skipped_no_peak += 1
                continue

            seg_result = isolate_call_segment(ap.peak, scores, audio, target_sr)
            if seg_result is None:
                if id(ap) not in has_clean:
                    ap.treatment = "skipped"
                    result.skipped_no_peak += 1
                continue

            call_seg, _ = seg_result
            variants = synthesize_variants(
                call_seg,
                backgrounds,
                target_sr,
                window_size=window_size,
                crossfade_ms=synthesis_crossfade_ms,
                n_variants=synthesis_variants,
            )
            for sample in variants:
                sample.call_type = ap.annotation.call_type
                sample.source_filename = filename
                sample.peak_score = ap.peak.score
                sample.start_sec = ap.peak.time_sec
                sample.end_sec = ap.peak.time_sec + window_size
                result.extracted_samples.append(sample)

            if not variants and id(ap) not in has_clean:
                ap.treatment = "skipped"
                result.skipped_no_peak += 1

    return result
