"""Run detection: scan audio folder, classify windows, merge spans."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from humpback.classifier.detector_utils import (
    AUDIO_EXTENSIONS,
    EmbeddingDiffResult,
    TSV_FIELDNAMES,
    WINDOW_DIAGNOSTICS_SCHEMA,
    _build_file_timeline,
    _file_base_epoch,
    _smooth_scores,
    _window_diagnostics_table,
    append_detections_tsv,
    diff_row_store_vs_embeddings,
    match_embedding_records_to_row_store,
    merge_detection_events,
    merge_detection_spans,
    read_detection_embedding,
    read_detections_tsv,
    read_window_diagnostics_table,
    resolve_audio_for_window,
    resolve_audio_for_window_hydrophone,
    select_peak_windows_from_events,
    select_prominent_peaks_from_events,
    snap_and_merge_detection_events,
    snap_event_bounds,
    write_detection_embeddings,
    write_detections_tsv,
    write_window_diagnostics,
    write_window_diagnostics_shard,
)
from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.features import extract_logmel_batch
from humpback.processing.inference import EmbeddingModel
from humpback.processing.windowing import (
    WindowMetadata,
    format_short_audio_window_message,
    slice_windows_with_metadata,
    window_sample_count,
)

__all__ = [
    # Re-exported from detector_utils
    "EmbeddingDiffResult",
    "TSV_FIELDNAMES",
    "WINDOW_DIAGNOSTICS_SCHEMA",
    "_build_file_timeline",
    "_file_base_epoch",
    "_smooth_scores",
    "_window_diagnostics_table",
    "append_detections_tsv",
    "diff_row_store_vs_embeddings",
    "match_embedding_records_to_row_store",
    "merge_detection_events",
    "merge_detection_spans",
    "read_detection_embedding",
    "read_detections_tsv",
    "read_window_diagnostics_table",
    "resolve_audio_for_window",
    "resolve_audio_for_window_hydrophone",
    "select_peak_windows_from_events",
    "select_prominent_peaks_from_events",
    "snap_and_merge_detection_events",
    "snap_event_bounds",
    "write_detection_embeddings",
    "write_detections_tsv",
    "write_window_diagnostics",
    "write_window_diagnostics_shard",
    # Defined here
    "AUDIO_EXTENSIONS",
    "run_detection",
]

logger = logging.getLogger(__name__)


def run_detection(
    audio_folder: Path,
    pipeline: Pipeline,
    model: EmbeddingModel,
    window_size_seconds: float,
    target_sample_rate: int,
    confidence_threshold: float,
    input_format: str = "spectrogram",
    feature_config: dict | None = None,
    emit_diagnostics: bool = False,
    hop_seconds: float = 1.0,
    high_threshold: float = 0.70,
    low_threshold: float = 0.45,
    on_file_complete: Callable[[list[dict], int, int], None] | None = None,
    detection_mode: str | None = None,
    emit_embeddings: bool = False,
    window_selection: str | None = None,
    min_prominence: float | None = None,
) -> tuple[list[dict], dict, list[dict] | None, list[dict] | None]:
    """Scan audio folder, classify each window, merge events.

    Returns (detections_list, summary_dict, diagnostics_or_none, embeddings_or_none).
    Each detection: {start_utc, end_utc, raw_start_utc, raw_end_utc, avg_confidence, peak_confidence, n_windows}.
    When emit_diagnostics=True, diagnostics is a list of per-window records.
    When emit_embeddings=True, embeddings is a list of per-detection embedding records.

    When ``detection_mode="windowed"``, each merged event is reduced to
    non-overlapping peak windows of exactly ``window_size_seconds`` via NMS.
    """
    import time

    feature_config = feature_config or {}
    normalization = feature_config.get("normalization", "per_window_max")

    audio_files = sorted(
        p for p in audio_folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_folder}")

    all_detections: list[dict] = []
    all_confidences: list[float] = []
    diagnostics_records: list[dict] | None = [] if emit_diagnostics else None
    embedding_records: list[dict] | None = [] if emit_embeddings else None
    total_windows = 0
    total_positive = 0
    n_skipped_short = 0
    files_done = 0
    n_audio_files = len(audio_files)
    t_decode_total = 0.0
    t_features_total = 0.0
    t_inference_total = 0.0
    n_windows_total = 0

    for audio_path in audio_files:
        try:
            t0 = time.monotonic()
            audio, sr = decode_audio(audio_path)
            audio = resample(audio, sr, target_sample_rate)
            t_decode_total += time.monotonic() - t0

            window_samples = window_sample_count(
                target_sample_rate, window_size_seconds
            )
            if len(audio) < window_samples:
                logger.warning(
                    "Skipping %s: audio too short (%s)",
                    audio_path.name,
                    format_short_audio_window_message(
                        len(audio), target_sample_rate, window_size_seconds
                    ),
                )
                n_skipped_short += 1
                files_done += 1
                if on_file_complete is not None:
                    on_file_complete([], files_done, n_audio_files)
                continue

            # Phase 1: Collect all windows
            raw_windows: list[np.ndarray] = []
            window_metas: list[WindowMetadata] = []
            for window, meta in slice_windows_with_metadata(
                audio,
                target_sample_rate,
                window_size_seconds,
                hop_seconds=hop_seconds,
            ):
                window_metas.append(meta)
                raw_windows.append(window)

            if not raw_windows:
                files_done += 1
                if on_file_complete is not None:
                    on_file_complete([], files_done, n_audio_files)
                continue

            # Phase 2: Feature extraction (batch for spectrogram, pass-through for waveform)
            n_windows_total += len(raw_windows)
            if input_format == "waveform":
                batch_items: list[np.ndarray] = raw_windows
            else:
                t0 = time.monotonic()
                batch_items = extract_logmel_batch(
                    raw_windows,
                    target_sample_rate,
                    n_mels=128,
                    hop_length=1252,
                    target_frames=128,
                    normalization=normalization,
                )
                t_features_total += time.monotonic() - t0

            # Phase 3: Batch embed (groups of 64 — optimal for TFLite on M-series)
            batch_size = 64
            file_embeddings: list[np.ndarray] = []
            for i in range(0, len(batch_items), batch_size):
                batch = np.stack(batch_items[i : i + batch_size])
                t0 = time.monotonic()
                embeddings = model.embed(batch)
                t_inference_total += time.monotonic() - t0
                file_embeddings.append(embeddings)

            all_emb = np.vstack(file_embeddings)
            total_windows += len(all_emb)

            # Classify
            proba = pipeline.predict_proba(all_emb)[:, 1]  # P(whale)
            window_confidences = proba.tolist()

            rel_path = str(audio_path.relative_to(audio_folder))
            base_epoch = _file_base_epoch(audio_path)

            # Build window records for event merging
            window_records = [
                {
                    "offset_sec": meta.offset_sec,
                    "end_sec": meta.offset_sec + window_size_seconds,
                    "confidence": conf,
                }
                for meta, conf in zip(window_metas, window_confidences)
            ]

            # Collect per-window diagnostics
            if emit_diagnostics:
                for i_meta, (meta, conf) in enumerate(
                    zip(window_metas, window_confidences)
                ):
                    if meta.is_overlapped and i_meta > 0:
                        prev_end = (
                            window_metas[i_meta - 1].offset_sec + window_size_seconds
                        )
                        overlap_sec = prev_end - meta.offset_sec
                    else:
                        overlap_sec = 0.0
                    assert diagnostics_records is not None
                    diagnostics_records.append(
                        {
                            "filename": rel_path,
                            "window_index": meta.window_index,
                            "offset_sec": meta.offset_sec,
                            "end_sec": meta.offset_sec + window_size_seconds,
                            "confidence": conf,
                            "is_overlapped": meta.is_overlapped,
                            "overlap_sec": overlap_sec,
                        }
                    )

            # Merge events using hysteresis, then canonicalize snapped bounds.
            events = merge_detection_events(
                window_records, high_threshold, low_threshold
            )
            events = snap_and_merge_detection_events(events, window_size_seconds)

            # Windowed mode: reduce each event to peak windows.
            if detection_mode == "windowed":
                if window_selection == "prominence":
                    events = select_prominent_peaks_from_events(
                        events,
                        window_records,
                        window_size_seconds,
                        min_score=high_threshold,
                        min_prominence=min_prominence
                        if min_prominence is not None
                        else 1.0,
                    )
                else:
                    events = select_peak_windows_from_events(
                        events,
                        window_records,
                        window_size_seconds,
                        min_score=high_threshold,
                    )

            # Collect per-detection embeddings (peak-window vector)
            if emit_embeddings:
                assert embedding_records is not None
                for event in events:
                    ev_start = float(event["start_sec"])
                    ev_end = float(event["end_sec"])
                    # Find the peak-confidence window within event bounds
                    best_idx = -1
                    best_conf = -1.0
                    for w_idx, (meta, conf) in enumerate(
                        zip(window_metas, window_confidences)
                    ):
                        if (
                            meta.offset_sec + 1e-6 >= ev_start
                            and meta.offset_sec + window_size_seconds - 1e-6 <= ev_end
                            and conf > best_conf
                        ):
                            best_idx = w_idx
                            best_conf = conf
                    if best_idx >= 0:
                        embedding_records.append(
                            {
                                "filename": rel_path,
                                "start_sec": ev_start,
                                "end_sec": ev_end,
                                "embedding": all_emb[best_idx].tolist(),
                                "confidence": best_conf,
                            }
                        )

            # Convert file-relative events to UTC.
            for event in events:
                event["start_utc"] = base_epoch + float(event.pop("start_sec"))
                event["end_utc"] = base_epoch + float(event.pop("end_sec"))
                raw_s = event.pop("raw_start_sec", None)
                raw_e = event.pop("raw_end_sec", None)
                event["raw_start_utc"] = (
                    base_epoch + float(raw_s)
                    if raw_s is not None
                    else event["start_utc"]
                )
                event["raw_end_utc"] = (
                    base_epoch + float(raw_e) if raw_e is not None else event["end_utc"]
                )
                all_detections.append(event)

            total_positive += sum(
                1 for c in window_confidences if c >= confidence_threshold
            )
            all_confidences.extend(window_confidences)

            files_done += 1
            if on_file_complete is not None:
                # events already have filename set (line above)
                on_file_complete(list(events), files_done, n_audio_files)

        except Exception:
            logger.warning("Failed to process %s, skipping", audio_path, exc_info=True)
            files_done += 1
            if on_file_complete is not None:
                on_file_complete([], files_done, n_audio_files)
            continue

    summary: dict = {
        "n_files": len(audio_files),
        "n_windows": total_windows,
        "n_detections": total_positive,
        "n_spans": len(all_detections),
        "n_skipped_short": n_skipped_short,
        "hop_seconds": hop_seconds,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "detection_mode": detection_mode or "merged",
        "window_selection": window_selection or "nms",
    }

    if all_confidences:
        conf_arr = np.array(all_confidences)
        summary["confidence_stats"] = {
            "mean": float(np.mean(conf_arr)),
            "median": float(np.median(conf_arr)),
            "std": float(np.std(conf_arr)),
            "min": float(np.min(conf_arr)),
            "max": float(np.max(conf_arr)),
            "p10": float(np.percentile(conf_arr, 10)),
            "p25": float(np.percentile(conf_arr, 25)),
            "p75": float(np.percentile(conf_arr, 75)),
            "p90": float(np.percentile(conf_arr, 90)),
            "pct_above_threshold": float(np.mean(conf_arr >= confidence_threshold)),
        }

    logger.info(
        "Detection complete: %d files, %d windows, %d detections, %d spans, %d skipped (short)",
        summary["n_files"],
        summary["n_windows"],
        summary["n_detections"],
        summary["n_spans"],
        summary["n_skipped_short"],
    )
    logger.info(
        "Detection timing: decode=%.3fs, features=%.3fs (%d windows), inference=%.3fs",
        t_decode_total,
        t_features_total,
        n_windows_total,
        t_inference_total,
    )

    return all_detections, summary, diagnostics_records, embedding_records
