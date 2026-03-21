"""Stage 10: Orchestrate the full sample-builder pipeline for a recording."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from humpback.classifier.raven_parser import RavenAnnotation
from humpback.processing.audio_io import decode_audio, resample
from humpback.sample_builder.construct import construct_sample
from humpback.sample_builder.contamination import (
    ContaminationConfig,
    _band_limited_rms,
    screen_fragment,
)
from humpback.sample_builder.discover import discover_background_fragments
from humpback.sample_builder.exclusion import build_exclusion_map
from humpback.sample_builder.normalize import normalize_annotations
from humpback.sample_builder.planner import plan_assembly
from humpback.sample_builder.similarity import SimilarityConfig, score_similarity
from humpback.sample_builder.smooth import smooth_joins
from humpback.sample_builder.types import (
    REASON_ASSEMBLY_FAILURE,
    REASON_CONTAMINATION_DETECTED,
    REASON_INSUFFICIENT_BACKGROUND,
    REASON_INVALID_ANNOTATION,
    BackgroundFragment,
    ExclusionMap,
    NormalizedAnnotation,
    SampleMetadata,
    SampleResult,
)
from humpback.sample_builder.validate import ValidationConfig, validate_sample

logger = logging.getLogger(__name__)


@dataclass
class SampleBuilderConfig:
    """All parameters for the sample-builder pipeline."""

    # Stage 1: Annotation normalization
    min_annotation_duration: float = 0.1
    max_annotation_duration: float = 10.0

    # Stage 2: Exclusion map
    guard_band_sec: float = 1.0

    # Stage 3: Fragment discovery
    min_fragment_sec: float = 0.5
    max_search_radius_sec: float = 60.0

    # Stage 4: Contamination screening
    contamination_config: ContaminationConfig = field(
        default_factory=ContaminationConfig
    )

    # Stage 5: Similarity scoring
    similarity_config: SimilarityConfig = field(default_factory=SimilarityConfig)

    # Stage 6: Assembly planning
    window_size: float = 5.0
    min_fill_fraction: float = 0.9

    # Stage 8: Join smoothing
    crossfade_ms: float = 50.0

    # Stage 9: Validation
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)

    # Target sample rate
    target_sr: int = 32000


def build_samples_for_recording(
    audio_path: Path,
    annotations: list[RavenAnnotation],
    *,
    sr: int = 32000,
    config: SampleBuilderConfig | None = None,
) -> list[SampleResult]:
    """Build 5-second samples for all valid annotations in a recording.

    Parameters
    ----------
    audio_path:
        Path to the audio file.
    annotations:
        Raven annotations for this recording.
    sr:
        Target sample rate (overrides config.target_sr if provided).
    config:
        Pipeline configuration.  Uses defaults when ``None``.

    Returns
    -------
    List of SampleResult — one per annotation (accepted or rejected).
    """
    if config is None:
        config = SampleBuilderConfig()

    # Decode and resample audio
    audio, orig_sr = decode_audio(audio_path)
    if orig_sr != sr:
        audio = resample(audio, orig_sr, sr)

    audio_duration_sec = len(audio) / sr
    source_filename = audio_path.stem

    # Stage 1: Normalize annotations
    normalized = normalize_annotations(
        annotations,
        min_duration=config.min_annotation_duration,
        max_duration=config.max_annotation_duration,
    )

    # Stage 2: Build exclusion map
    exclusion_map = build_exclusion_map(
        normalized, guard_band_sec=config.guard_band_sec
    )

    # Compute reference noise floor (median band-limited RMS across recording)
    reference_noise_floor = _compute_noise_floor(audio, sr, config)

    results: list[SampleResult] = []

    for norm_ann in normalized:
        if not norm_ann.valid:
            results.append(
                SampleResult(
                    accepted=False,
                    audio=None,
                    sr=sr,
                    call_type=norm_ann.original.call_type,
                    source_filename=source_filename,
                    annotation=norm_ann.original,
                    rejection_reason=REASON_INVALID_ANNOTATION,
                )
            )
            continue

        result = _process_single_annotation(
            norm_ann=norm_ann,
            audio=audio,
            sr=sr,
            audio_duration_sec=audio_duration_sec,
            exclusion_map=exclusion_map,
            reference_noise_floor=reference_noise_floor,
            source_filename=source_filename,
            config=config,
        )
        results.append(result)

    return results


def _process_single_annotation(
    norm_ann: NormalizedAnnotation,
    audio: NDArray[np.floating],
    sr: int,
    audio_duration_sec: float,
    exclusion_map: ExclusionMap,
    reference_noise_floor: float,
    source_filename: str,
    config: SampleBuilderConfig,
) -> SampleResult:
    """Process a single annotation through stages 3-9."""
    ann = norm_ann

    # Stage 3: Discover background fragments
    fragments = discover_background_fragments(
        ann,
        exclusion_map,
        audio_duration_sec,
        min_fragment_sec=config.min_fragment_sec,
        max_search_radius_sec=config.max_search_radius_sec,
    )

    if not fragments:
        return SampleResult(
            accepted=False,
            audio=None,
            sr=sr,
            call_type=ann.original.call_type,
            source_filename=source_filename,
            annotation=ann.original,
            rejection_reason=REASON_INSUFFICIENT_BACKGROUND,
        )

    # Stage 4: Screen each fragment for contamination
    passing_fragments: list[BackgroundFragment] = []
    for frag in fragments:
        start_sample = int(frag.start_sec * sr)
        end_sample = min(int(frag.end_sec * sr), len(audio))
        if end_sample <= start_sample:
            continue
        frag_audio = audio[start_sample:end_sample]
        contam_result = screen_fragment(
            frag_audio, sr, reference_noise_floor, config.contamination_config
        )
        if contam_result.passed:
            passing_fragments.append(frag)

    if not passing_fragments:
        return SampleResult(
            accepted=False,
            audio=None,
            sr=sr,
            call_type=ann.original.call_type,
            source_filename=source_filename,
            annotation=ann.original,
            rejection_reason=REASON_CONTAMINATION_DETECTED,
        )

    # Stage 5: Score similarity of passing fragments
    reference_audio = _extract_reference_background(
        ann, exclusion_map, audio, sr, config
    )
    scored = []
    for frag in passing_fragments:
        start_sample = int(frag.start_sec * sr)
        end_sample = min(int(frag.end_sec * sr), len(audio))
        frag_audio = audio[start_sample:end_sample]
        if reference_audio is not None and len(frag_audio) > 0:
            sim = score_similarity(
                frag_audio, reference_audio, sr, config.similarity_config
            )
            scored.append((frag, sim.score))
        else:
            scored.append((frag, 0.5))

    # Sort by similarity score (highest first)
    scored.sort(key=lambda x: x[1], reverse=True)
    ranked_fragments = [f for f, _ in scored]
    similarity_scores = [s for _, s in scored]

    # Stage 6: Plan assembly
    plan = plan_assembly(
        ann,
        ranked_fragments,
        window_size=config.window_size,
        min_fill_fraction=config.min_fill_fraction,
    )

    if not plan.can_assemble:
        return SampleResult(
            accepted=False,
            audio=None,
            sr=sr,
            call_type=ann.original.call_type,
            source_filename=source_filename,
            annotation=ann.original,
            rejection_reason=plan.rejection_reason or REASON_ASSEMBLY_FAILURE,
        )

    # Stage 7: Construct sample
    assembled, splice_points = construct_sample(plan, audio, sr)

    # Stage 8: Smooth joins
    smoothed = smooth_joins(assembled, splice_points, sr, config.crossfade_ms)

    # Stage 9: Validate
    passed, rejection_reason = validate_sample(
        smoothed,
        sr,
        plan,
        splice_points,
        reference_noise_floor,
        config.validation_config,
    )

    metadata = SampleMetadata(
        fragment_starts=[
            f.start_sec for f in plan.left_fragments + plan.right_fragments
        ],
        fragment_ends=[f.end_sec for f in plan.left_fragments + plan.right_fragments],
        fragment_durations=[
            f.duration_sec for f in plan.left_fragments + plan.right_fragments
        ],
        similarity_scores=similarity_scores[
            : len(plan.left_fragments) + len(plan.right_fragments)
        ],
        splice_points=splice_points,
        target_start_sec=ann.original.begin_time,
        target_end_sec=ann.original.end_time,
        target_duration_sec=ann.duration_sec,
        window_size_sec=config.window_size,
    )

    if not passed:
        return SampleResult(
            accepted=False,
            audio=smoothed,
            sr=sr,
            call_type=ann.original.call_type,
            source_filename=source_filename,
            annotation=ann.original,
            metadata=metadata,
            rejection_reason=rejection_reason,
        )

    return SampleResult(
        accepted=True,
        audio=smoothed,
        sr=sr,
        call_type=ann.original.call_type,
        source_filename=source_filename,
        annotation=ann.original,
        metadata=metadata,
    )


def _compute_noise_floor(
    audio: NDArray[np.floating],
    sr: int,
    config: SampleBuilderConfig,
) -> float:
    """Compute reference noise floor as median band-limited RMS across 1s chunks."""
    chunk_samples = sr  # 1 second chunks
    n_chunks = max(1, len(audio) // chunk_samples)
    rms_values = []
    for i in range(n_chunks):
        chunk = audio[i * chunk_samples : (i + 1) * chunk_samples]
        rms = _band_limited_rms(
            chunk,
            sr,
            config.contamination_config.rms_low_hz,
            config.contamination_config.rms_high_hz,
        )
        rms_values.append(rms)
    return float(np.median(rms_values)) if rms_values else 0.0


def _extract_reference_background(
    target: NormalizedAnnotation,
    exclusion_map: ExclusionMap,
    audio: NDArray[np.floating],
    sr: int,
    config: SampleBuilderConfig,
) -> NDArray[np.floating] | None:
    """Extract non-protected audio near the target for similarity reference."""
    # Search for clean audio within 5 seconds of target
    radius = 5.0
    search_start = max(0.0, target.midpoint_sec - radius)
    search_end = min(len(audio) / sr, target.midpoint_sec + radius)

    # Collect non-protected samples
    ref_samples: list[NDArray[np.floating]] = []
    step = 0.5  # 0.5s steps
    t = search_start
    while t < search_end:
        t_end = min(t + step, search_end)
        if not exclusion_map.overlaps(t, t_end):
            s = int(t * sr)
            e = min(int(t_end * sr), len(audio))
            if e > s:
                ref_samples.append(audio[s:e])
        t += step

    if not ref_samples:
        return None
    return np.concatenate(ref_samples)
