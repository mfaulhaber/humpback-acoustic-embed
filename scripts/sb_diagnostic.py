"""Diagnostic script for sample builder pipeline.

Analyzes real recordings to identify which contamination features are causing
rejection and at what thresholds they would pass.  Helps tune parameters for
specific datasets.

Usage:
    uv run python scripts/sb_diagnostic.py \
        --annotation-folder /path/to/annotations \
        --audio-folder /path/to/audio \
        --max-files 3
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from humpback.classifier.raven_parser import pair_annotations_with_recordings
from humpback.processing.audio_io import decode_audio, resample
from humpback.sample_builder.contamination import (
    ContaminationConfig,
    _band_limited_rms,
    _spectral_occupancy,
    _tonal_persistence,
    _transient_energy,
    screen_fragment,
)
from humpback.sample_builder.discover import discover_background_fragments
from humpback.sample_builder.exclusion import build_exclusion_map
from humpback.sample_builder.normalize import normalize_annotations
from humpback.sample_builder.pipeline import SampleBuilderConfig, _compute_noise_floor

logger = logging.getLogger(__name__)

SR = 32000


def analyze_recording(
    audio_path: Path,
    annotations: list,
    config: SampleBuilderConfig,
) -> dict:
    """Analyze a single recording and return per-feature diagnostic data."""
    audio, orig_sr = decode_audio(audio_path)
    if orig_sr != SR:
        audio = resample(audio, orig_sr, SR)

    audio_duration_sec = len(audio) / SR

    # Stage 1: Normalize
    normalized = normalize_annotations(
        annotations,
        min_duration=config.min_annotation_duration,
        max_duration=config.max_annotation_duration,
    )

    valid_count = sum(1 for n in normalized if n.valid)
    invalid_count = sum(1 for n in normalized if not n.valid)

    # Annotation duration distribution
    durations = [n.duration_sec for n in normalized]

    # Stage 2: Exclusion map
    exclusion_map = build_exclusion_map(
        normalized, guard_band_sec=config.guard_band_sec
    )

    # Noise floor
    noise_floor = _compute_noise_floor(audio, SR, config)

    # Stage 3-4: Discover and screen fragments for each valid annotation
    feature_scores: dict[str, list[float]] = {
        "rms_ratio": [],
        "spectral_occupancy": [],
        "tonal_persistence": [],
        "transient_energy": [],
    }
    rejection_reasons: Counter[str] = Counter()
    fragment_count = 0
    passing_count = 0

    for norm_ann in normalized:
        if not norm_ann.valid:
            rejection_reasons["invalid_annotation"] += 1
            continue

        fragments = discover_background_fragments(
            norm_ann,
            exclusion_map,
            audio_duration_sec,
            min_fragment_sec=config.min_fragment_sec,
            max_search_radius_sec=config.max_search_radius_sec,
        )

        if not fragments:
            rejection_reasons["insufficient_background"] += 1
            continue

        ann_has_passing = False
        for frag in fragments:
            start_sample = int(frag.start_sec * SR)
            end_sample = min(int(frag.end_sec * SR), len(audio))
            if end_sample <= start_sample:
                continue
            frag_audio = audio[start_sample:end_sample]
            fragment_count += 1

            # Compute individual feature scores
            cc = config.contamination_config
            rms = _band_limited_rms(frag_audio, SR, cc.rms_low_hz, cc.rms_high_hz)
            rms_ratio = rms / noise_floor if noise_floor > 0 else 0.0
            feature_scores["rms_ratio"].append(rms_ratio)

            occ = _spectral_occupancy(
                frag_audio, SR, cc.occupancy_n_fft, cc.occupancy_noise_floor_db
            )
            feature_scores["spectral_occupancy"].append(occ)

            pers = _tonal_persistence(
                frag_audio, SR, cc.persistence_n_fft, cc.persistence_margin_db
            )
            feature_scores["tonal_persistence"].append(pers)

            trans = _transient_energy(frag_audio, SR, cc.transient_frame_length)
            feature_scores["transient_energy"].append(trans)

            # Full screen
            result = screen_fragment(frag_audio, SR, noise_floor, cc)
            if result.passed:
                passing_count += 1
                ann_has_passing = True

        if not ann_has_passing:
            rejection_reasons["contamination_detected"] += 1

    return {
        "file": audio_path.name,
        "duration_sec": audio_duration_sec,
        "noise_floor": noise_floor,
        "total_annotations": len(normalized),
        "valid_annotations": valid_count,
        "invalid_annotations": invalid_count,
        "annotation_durations": durations,
        "total_fragments": fragment_count,
        "passing_fragments": passing_count,
        "feature_scores": feature_scores,
        "rejection_reasons": dict(rejection_reasons),
    }


def print_report(results: list[dict]) -> None:
    """Print a diagnostic report from analyzed recordings."""
    print("\n" + "=" * 70)
    print("SAMPLE BUILDER DIAGNOSTIC REPORT")
    print("=" * 70)

    # Aggregate
    all_durations: list[float] = []
    all_features: dict[str, list[float]] = {
        "rms_ratio": [],
        "spectral_occupancy": [],
        "tonal_persistence": [],
        "transient_energy": [],
    }
    total_rejection: Counter[str] = Counter()
    total_annotations = 0
    total_valid = 0
    total_fragments = 0
    total_passing = 0

    for r in results:
        all_durations.extend(r["annotation_durations"])
        for key in all_features:
            all_features[key].extend(r["feature_scores"][key])
        for reason, count in r["rejection_reasons"].items():
            total_rejection[reason] += count
        total_annotations += r["total_annotations"]
        total_valid += r["valid_annotations"]
        total_fragments += r["total_fragments"]
        total_passing += r["passing_fragments"]

    # Per-recording summary
    print(f"\nRecordings analyzed: {len(results)}")
    for r in results:
        frag_pass = r["passing_fragments"]
        frag_total = r["total_fragments"]
        pct = (frag_pass / frag_total * 100) if frag_total > 0 else 0
        print(
            f"  {r['file']}: {r['duration_sec']:.0f}s, "
            f"noise_floor={r['noise_floor']:.6f}, "
            f"{r['valid_annotations']}/{r['total_annotations']} valid, "
            f"{frag_pass}/{frag_total} fragments pass ({pct:.0f}%)"
        )

    # Annotation duration histogram
    if all_durations:
        arr = np.array(all_durations)
        print(f"\n--- Annotation Duration Distribution (n={len(arr)}) ---")
        print(
            f"  min={arr.min():.3f}s  max={arr.max():.3f}s  "
            f"mean={arr.mean():.3f}s  median={np.median(arr):.3f}s"
        )
        bins = [0, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 10.0, float("inf")]
        labels = [
            "<0.1",
            "0.1-0.3",
            "0.3-0.5",
            "0.5-1.0",
            "1.0-2.0",
            "2.0-4.0",
            "4.0-10.0",
            ">10.0",
        ]
        hist, _ = np.histogram(arr, bins=bins)
        for label, count in zip(labels, hist):
            bar = "#" * min(count, 50)
            print(f"  {label:>8s}: {count:4d} {bar}")

    # Rejection summary
    print(f"\n--- Rejection Breakdown (n={total_annotations}) ---")
    for reason, count in sorted(total_rejection.items(), key=lambda x: -x[1]):
        pct = count / total_annotations * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    accepted = total_annotations - sum(total_rejection.values())
    print(f"  accepted: {accepted} ({accepted / total_annotations * 100:.1f}%)")

    # Per-feature score distributions
    print(f"\n--- Per-Feature Score Distributions (n={total_fragments} fragments) ---")
    config = ContaminationConfig()
    thresholds = {
        "rms_ratio": config.rms_threshold_factor,
        "spectral_occupancy": config.occupancy_threshold,
        "tonal_persistence": config.persistence_threshold,
        "transient_energy": config.transient_threshold,
    }
    for feature, scores in all_features.items():
        if not scores:
            continue
        arr = np.array(scores)
        threshold = thresholds[feature]
        exceeds = np.sum(arr > threshold)
        pct_exceed = exceeds / len(arr) * 100
        print(f"\n  {feature} (threshold={threshold}):")
        print(
            f"    min={arr.min():.4f}  p25={np.percentile(arr, 25):.4f}  "
            f"median={np.median(arr):.4f}  p75={np.percentile(arr, 75):.4f}  "
            f"max={arr.max():.4f}"
        )
        print(f"    exceeds threshold: {exceeds}/{len(arr)} ({pct_exceed:.1f}%)")

    # Fragment pass rate
    if total_fragments > 0:
        print("\n--- Fragment Pass Rate ---")
        print(
            f"  {total_passing}/{total_fragments} fragments pass all 4 features "
            f"({total_passing / total_fragments * 100:.1f}%)"
        )

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose sample builder pipeline rejection rates"
    )
    parser.add_argument(
        "--annotation-folder",
        type=Path,
        required=True,
        help="Path to Raven annotation .txt files",
    )
    parser.add_argument(
        "--audio-folder",
        type=Path,
        required=True,
        help="Path to audio files (.wav/.flac/.mp3)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Max recordings to analyze (0 = all)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pairs = pair_annotations_with_recordings(args.annotation_folder, args.audio_folder)
    if args.max_files > 0:
        pairs = pairs[: args.max_files]

    config = SampleBuilderConfig()
    results = []
    for i, recording in enumerate(pairs):
        logger.info(
            "Analyzing %s (%d/%d, %d annotations)",
            recording.audio_path.name,
            i + 1,
            len(pairs),
            len(recording.annotations),
        )
        result = analyze_recording(recording.audio_path, recording.annotations, config)
        results.append(result)

    print_report(results)


if __name__ == "__main__":
    main()
