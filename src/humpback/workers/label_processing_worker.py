"""Worker for label processing jobs: score recordings and extract clean samples."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.extractor import write_flac_file, write_spectrogram_png_file
from humpback.classifier.label_processor import (
    ExtractedSample,
    process_recording,
)
from humpback.classifier.raven_parser import pair_annotations_with_recordings
from humpback.config import Settings
from humpback.models.classifier import ClassifierModel
from humpback.models.label_processing import LabelProcessingJob
from humpback.workers.model_cache import get_model_by_version
from humpback.workers.queue import (
    complete_label_processing_job,
    fail_label_processing_job,
)

if TYPE_CHECKING:
    from humpback.sample_builder.types import SampleResult

logger = logging.getLogger(__name__)

# Default parameters for label processing
DEFAULT_PARAMS: dict[str, float | int | bool] = {
    "threshold_high": 0.7,
    "smoothing_window": 3,
    "onset_offset_alpha": 0.4,
    "overlap_proximity_sec": 3.0,
    "overlap_relative_threshold": 0.5,
    "enable_synthesized": True,
    "background_threshold": 0.1,
    "background_threshold_auto": True,
    "background_min_duration": 1.0,
    "synthesis_crossfade_ms": 50.0,
    "synthesis_variants": 3,
    "cleanup_score_cache": True,
}


async def run_label_processing_job(
    session: AsyncSession,
    job: LabelProcessingJob,
    settings: Settings,
) -> None:
    """Execute a label processing job: dispatch by workflow type."""
    workflow = job.workflow or "score_based"
    try:
        if workflow == "sample_builder":
            await _run_sample_builder(session, job, settings)
        else:
            await _run_score_based(session, job, settings)
    except Exception as e:
        logger.error("Label processing job %s failed: %s", job.id, e, exc_info=True)
        await fail_label_processing_job(session, job.id, str(e))


async def _run_score_based(
    session: AsyncSession,
    job: LabelProcessingJob,
    settings: Settings,
) -> None:
    """Execute the score-based label processing workflow (original path)."""
    # Parse parameters
    params = dict(DEFAULT_PARAMS)
    if job.parameters:
        params.update(json.loads(job.parameters))

    # Load classifier model metadata
    from sqlalchemy import select

    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == job.classifier_model_id)
    )
    cm = result.scalar_one()

    # Load sklearn pipeline
    pipeline = joblib.load(cm.model_path)

    # Load embedding model
    model, input_format = await get_model_by_version(
        session, cm.model_version, settings
    )

    feature_config = json.loads(cm.feature_config) if cm.feature_config else None

    # Parse and pair annotations with recordings
    ann_dir = Path(job.annotation_folder)
    aud_dir = Path(job.audio_folder)
    pairs = pair_annotations_with_recordings(ann_dir, aud_dir)

    output_root = Path(job.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Processing counters
    treatment_counts: Counter[str] = Counter()
    call_type_counts: Counter[str] = Counter()
    label_scores: dict[str, list[float]] = {}
    total_extracted = 0
    total_skipped = 0

    for file_idx, recording in enumerate(pairs):
        logger.info(
            "Processing %s (%d/%d, %d annotations)",
            recording.audio_path.name,
            file_idx + 1,
            len(pairs),
            len(recording.annotations),
        )

        proc_result = process_recording(
            audio_path=recording.audio_path,
            annotations=recording.annotations,
            pipeline=pipeline,
            model=model,
            window_size=cm.window_size_seconds,
            target_sr=cm.target_sample_rate,
            hop_seconds=1.0,
            input_format=input_format,
            feature_config=feature_config,
            threshold_high=float(params["threshold_high"]),
            smoothing_window=int(params["smoothing_window"]),
            onset_offset_alpha=float(params["onset_offset_alpha"]),
            overlap_proximity_sec=float(params["overlap_proximity_sec"]),
            overlap_relative_threshold=float(params["overlap_relative_threshold"]),
            enable_synthesized=bool(params["enable_synthesized"]),
            background_threshold=float(params["background_threshold"]),
            background_threshold_auto=bool(params["background_threshold_auto"]),
            background_min_duration=float(params["background_min_duration"]),
            synthesis_crossfade_ms=float(params["synthesis_crossfade_ms"]),
            synthesis_variants=int(params["synthesis_variants"]),
        )

        # Write extracted samples (track variant counters for unique filenames)
        variant_counters: dict[str, int] = {}
        for sample in proc_result.extracted_samples:
            _write_sample(sample, output_root, variant_counters)
            treatment_counts[sample.treatment] += 1
            call_type_counts[sample.call_type] += 1
            total_extracted += 1

        # Collect per-label peak scores for KPIs
        for ap in proc_result.annotated_peaks:
            if ap.peak is not None:
                ct = ap.annotation.call_type
                label_scores.setdefault(ct, []).append(ap.peak.score)

        # Count skipped annotations
        for ap in proc_result.annotated_peaks:
            if ap.treatment == "skipped":
                total_skipped += 1

        # Update progress
        await session.execute(
            update(LabelProcessingJob)
            .where(LabelProcessingJob.id == job.id)
            .values(
                files_processed=file_idx + 1,
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    # Compute per-label score statistics
    import numpy as np

    score_stats_by_label: dict[str, dict[str, float | int]] = {}
    for ct, scores_list in sorted(label_scores.items()):
        arr = np.array(scores_list)
        score_stats_by_label[ct] = {
            "count": len(scores_list),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    # Write result summary
    summary = {
        "total_extracted": total_extracted,
        "total_skipped": total_skipped,
        "treatment_counts": dict(treatment_counts),
        "call_type_counts": dict(sorted(call_type_counts.items(), key=lambda x: -x[1])),
        "score_stats_by_label": score_stats_by_label,
        "files_processed": len(pairs),
    }

    await session.execute(
        update(LabelProcessingJob)
        .where(LabelProcessingJob.id == job.id)
        .values(
            result_summary=json.dumps(summary),
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()

    # Write summary to disk
    summary_path = output_root / "job_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Label processing complete: %d extracted (%d clean, "
        "%d synthesized, %d fallback), %d skipped",
        total_extracted,
        treatment_counts.get("clean", 0),
        treatment_counts.get("synthesized", 0),
        treatment_counts.get("fallback", 0),
        total_skipped,
    )

    # Optionally clean up score cache
    if bool(params.get("cleanup_score_cache", True)):
        scores_dir = output_root / "scores"
        if scores_dir.is_dir():
            import shutil

            shutil.rmtree(scores_dir)
            logger.info("Cleaned up score cache: %s", scores_dir)

    await complete_label_processing_job(session, job.id)


async def _run_sample_builder(
    session: AsyncSession,
    job: LabelProcessingJob,
    settings: Settings,
) -> None:
    """Execute the sample-builder workflow (signal-processing-first)."""
    from humpback.sample_builder.pipeline import (
        SampleBuilderConfig,
        build_samples_for_recording,
    )

    # Parse and pair annotations with recordings
    ann_dir = Path(job.annotation_folder)
    aud_dir = Path(job.audio_folder)
    pairs = pair_annotations_with_recordings(ann_dir, aud_dir)

    output_root = Path(job.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Parse sample builder config from parameters
    sb_config = SampleBuilderConfig()
    if job.parameters:
        params = json.loads(job.parameters)
        if "window_size" in params:
            sb_config.window_size = float(params["window_size"])
        if "guard_band_sec" in params:
            sb_config.guard_band_sec = float(params["guard_band_sec"])
        if "min_fill_fraction" in params:
            sb_config.min_fill_fraction = float(params["min_fill_fraction"])
        if "crossfade_ms" in params:
            sb_config.crossfade_ms = float(params["crossfade_ms"])
        # Annotation duration bounds
        if "min_annotation_duration" in params:
            sb_config.min_annotation_duration = float(params["min_annotation_duration"])
        if "max_annotation_duration" in params:
            sb_config.max_annotation_duration = float(params["max_annotation_duration"])
        # Contamination config overrides
        contam = sb_config.contamination_config
        if "rms_threshold_factor" in params:
            contam.rms_threshold_factor = float(params["rms_threshold_factor"])
        if "occupancy_threshold" in params:
            contam.occupancy_threshold = float(params["occupancy_threshold"])
        if "occupancy_noise_floor_db" in params:
            contam.occupancy_noise_floor_db = float(params["occupancy_noise_floor_db"])
        if "persistence_threshold" in params:
            contam.persistence_threshold = float(params["persistence_threshold"])
        if "persistence_margin_db" in params:
            contam.persistence_margin_db = float(params["persistence_margin_db"])
        if "transient_threshold" in params:
            contam.transient_threshold = float(params["transient_threshold"])
        # Sync validation config so Stage 9 re-check uses same thresholds
        sb_config.validation_config.contamination_config = contam

    # Processing counters
    call_type_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    total_accepted = 0
    total_rejected = 0
    total_annotations = 0

    for file_idx, recording in enumerate(pairs):
        logger.info(
            "Sample builder: %s (%d/%d, %d annotations)",
            recording.audio_path.name,
            file_idx + 1,
            len(pairs),
            len(recording.annotations),
        )

        results = build_samples_for_recording(
            audio_path=recording.audio_path,
            annotations=recording.annotations,
            sr=sb_config.target_sr,
            config=sb_config,
        )

        for result in results:
            total_annotations += 1
            if result.accepted:
                total_accepted += 1
                call_type_counts[result.call_type] += 1
                _write_sample_builder_output(result, output_root)
            else:
                total_rejected += 1
                if result.rejection_reason:
                    rejection_counts[result.rejection_reason] += 1
                _write_rejection_metadata(result, output_root)

        # Update progress
        await session.execute(
            update(LabelProcessingJob)
            .where(LabelProcessingJob.id == job.id)
            .values(
                files_processed=file_idx + 1,
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    # Write result summary
    acceptance_rate = (
        total_accepted / total_annotations if total_annotations > 0 else 0.0
    )
    summary = {
        "workflow": "sample_builder",
        "total_annotations": total_annotations,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "acceptance_rate": round(acceptance_rate, 4),
        "rejection_counts": dict(rejection_counts),
        "call_type_counts": dict(sorted(call_type_counts.items(), key=lambda x: -x[1])),
        "files_processed": len(pairs),
    }

    await session.execute(
        update(LabelProcessingJob)
        .where(LabelProcessingJob.id == job.id)
        .values(
            result_summary=json.dumps(summary),
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()

    summary_path = output_root / "job_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Sample builder complete: %d accepted, %d rejected (%.0f%% acceptance)",
        total_accepted,
        total_rejected,
        acceptance_rate * 100,
    )

    await complete_label_processing_job(session, job.id)


def _write_sample_builder_output(
    result: SampleResult,
    output_root: Path,
) -> None:
    """Write an accepted sample as FLAC + PNG sidecar."""

    if result.audio is None:
        return

    midpoint = 0.0
    if result.metadata:
        midpoint = (
            result.metadata.target_start_sec + result.metadata.target_end_sec
        ) / 2.0

    stem = result.source_filename
    base = f"{stem}_{midpoint:.1f}s"
    out_dir = output_root / "accepted" / result.call_type
    out_path = out_dir / f"{base}.flac"

    write_flac_file(result.audio, result.sr, out_path)

    png_path = out_path.with_suffix(".png")
    write_spectrogram_png_file(result.audio, result.sr, png_path)


def _write_rejection_metadata(
    result: SampleResult,
    output_root: Path,
) -> None:
    """Write rejection metadata JSON for a rejected annotation."""
    reason = result.rejection_reason or "unknown"
    midpoint = 0.0
    if result.metadata:
        midpoint = (
            result.metadata.target_start_sec + result.metadata.target_end_sec
        ) / 2.0

    stem = result.source_filename
    base = f"{stem}_{midpoint:.1f}s"
    out_dir = output_root / "rejected" / reason / result.call_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base}.json"

    from typing import Any

    meta: dict[str, Any] = {
        "source_filename": result.source_filename,
        "call_type": result.call_type,
        "rejection_reason": reason,
    }
    if result.annotation:
        meta["annotation"] = {
            "begin_time": result.annotation.begin_time,
            "end_time": result.annotation.end_time,
        }

    out_path.write_text(json.dumps(meta, indent=2))


def _write_sample(
    sample: ExtractedSample,
    output_root: Path,
    variant_counters: dict[str, int] | None = None,
) -> None:
    """Write a FLAC clip and spectrogram PNG sidecar for an extracted sample."""
    stem = Path(sample.source_filename).stem
    base = f"{stem}_{sample.start_sec:.1f}s"

    # Synthesised variants get a _v1/_v2/… suffix for unique filenames
    if sample.treatment == "synthesized" and variant_counters is not None:
        key = f"{sample.treatment}/{sample.call_type}/{base}"
        idx = variant_counters.get(key, 0) + 1
        variant_counters[key] = idx
        base = f"{base}_v{idx}"

    filename = f"{base}.flac"
    out_dir = output_root / sample.treatment / sample.call_type
    out_path = out_dir / filename

    write_flac_file(sample.audio_segment, sample.sr, out_path)

    # Spectrogram sidecar
    png_path = out_path.with_suffix(".png")
    write_spectrogram_png_file(sample.audio_segment, sample.sr, png_path)
