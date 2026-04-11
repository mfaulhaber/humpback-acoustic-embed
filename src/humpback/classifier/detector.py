"""Run detection: scan audio folder, classify windows, merge spans."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    select_tiled_windows_from_events,
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
    "select_tiled_windows_from_events",
    "snap_and_merge_detection_events",
    "snap_event_bounds",
    "write_detection_embeddings",
    "write_detections_tsv",
    "write_window_diagnostics",
    "write_window_diagnostics_shard",
    # Defined here
    "AUDIO_EXTENSIONS",
    "compute_hysteresis_events",
    "run_detection",
    "score_audio_windows",
]

logger = logging.getLogger(__name__)


@dataclass
class _DetectionPipelineState:
    """Shared intermediate state produced for one audio buffer.

    ``run_detection`` needs ``window_metas`` (for diagnostics and per-event
    embedding lookup) and ``embeddings`` (for per-event embedding export),
    while ``score_audio_windows`` / ``compute_hysteresis_events`` only need
    ``window_records``. The state dataclass lets both code paths share the
    same underlying computation without exposing internals as a public API.
    """

    window_metas: list[WindowMetadata] = field(default_factory=list)
    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    window_confidences: list[float] = field(default_factory=list)
    window_records: list[dict[str, Any]] = field(default_factory=list)
    t_features: float = 0.0
    t_inference: float = 0.0


def _run_window_pipeline(
    audio: np.ndarray,
    target_sample_rate: int,
    model: EmbeddingModel,
    pipeline: Pipeline,
    window_size_seconds: float,
    hop_seconds: float,
    input_format: str,
    normalization: str,
) -> _DetectionPipelineState:
    """Window → feature-extract → embed → classify.

    The shared backbone for ``run_detection`` and ``score_audio_windows``.
    Hysteresis merging lives in the callers so the scoring primitive stays
    pure and chunk-friendly. Empty audio (shorter than one window) produces
    an empty state; the caller is responsible for the short-audio log/skip.
    """
    state = _DetectionPipelineState()

    window_samples = window_sample_count(target_sample_rate, window_size_seconds)
    if len(audio) < window_samples:
        return state

    raw_windows: list[np.ndarray] = []
    for window, meta in slice_windows_with_metadata(
        audio,
        target_sample_rate,
        window_size_seconds,
        hop_seconds=hop_seconds,
    ):
        state.window_metas.append(meta)
        raw_windows.append(window)

    if not raw_windows:
        return state

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
        state.t_features = time.monotonic() - t0

    batch_size = 64
    file_embeddings: list[np.ndarray] = []
    for i in range(0, len(batch_items), batch_size):
        batch = np.stack(batch_items[i : i + batch_size])
        t0 = time.monotonic()
        embeddings = model.embed(batch)
        state.t_inference += time.monotonic() - t0
        file_embeddings.append(embeddings)

    state.embeddings = np.vstack(file_embeddings)

    proba = pipeline.predict_proba(state.embeddings)[:, 1]
    state.window_confidences = proba.tolist()

    state.window_records = [
        {
            "offset_sec": meta.offset_sec,
            "end_sec": meta.offset_sec + window_size_seconds,
            "confidence": conf,
        }
        for meta, conf in zip(state.window_metas, state.window_confidences)
    ]
    return state


def score_audio_windows(
    audio: np.ndarray,
    sample_rate: int,
    perch_model: EmbeddingModel,
    classifier: Pipeline,
    config: dict[str, Any],
    time_offset_sec: float = 0.0,
) -> list[dict[str, Any]]:
    """Pass 1 streaming primitive: audio → dense per-window score records.

    Returns window records with keys ``offset_sec``, ``end_sec``, and
    ``confidence``. ``offset_sec`` / ``end_sec`` are shifted by
    ``time_offset_sec`` so callers streaming audio chunk-by-chunk can set
    it to the chunk's absolute start and concatenate per-chunk records
    into a single absolute-time trace. ``config`` must contain
    ``window_size_seconds``, ``hop_seconds``; ``input_format`` and
    ``normalization`` are optional (defaults mirror ``run_detection``).
    Returns an empty list when the audio is shorter than one window.
    """
    state = _run_window_pipeline(
        audio=audio,
        target_sample_rate=sample_rate,
        model=perch_model,
        pipeline=classifier,
        window_size_seconds=float(config["window_size_seconds"]),
        hop_seconds=float(config.get("hop_seconds", 1.0)),
        input_format=str(config.get("input_format", "spectrogram")),
        normalization=str(config.get("normalization", "per_window_max")),
    )
    if time_offset_sec == 0.0:
        return state.window_records
    return [
        {
            "offset_sec": r["offset_sec"] + time_offset_sec,
            "end_sec": r["end_sec"] + time_offset_sec,
            "confidence": r["confidence"],
        }
        for r in state.window_records
    ]


def compute_hysteresis_events(
    audio: np.ndarray,
    sample_rate: int,
    perch_model: EmbeddingModel,
    classifier: Pipeline,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the Pass 1 backbone on one audio buffer.

    ``audio`` is a mono waveform already resampled to ``sample_rate``.
    ``config`` must contain ``window_size_seconds``, ``hop_seconds``,
    ``high_threshold``, and ``low_threshold``; ``input_format`` and
    ``normalization`` are optional (defaults mirror ``run_detection``).

    Re-implemented as a two-call composition of ``score_audio_windows``
    and ``merge_detection_events``. Returns ``(window_records, events)``.
    Returns two empty lists when the audio is shorter than one window.
    """
    window_records = score_audio_windows(
        audio=audio,
        sample_rate=sample_rate,
        perch_model=perch_model,
        classifier=classifier,
        config=config,
    )
    events = merge_detection_events(
        window_records,
        float(config["high_threshold"]),
        float(config["low_threshold"]),
    )
    return window_records, events


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
    max_logit_drop: float | None = None,
) -> tuple[list[dict], dict, list[dict] | None, list[dict] | None]:
    """Scan audio folder, classify each window, merge events.

    Returns (detections_list, summary_dict, diagnostics_or_none, embeddings_or_none).
    Each detection: {start_utc, end_utc, raw_start_utc, raw_end_utc, avg_confidence, peak_confidence, n_windows}.
    When emit_diagnostics=True, diagnostics is a list of per-window records.
    When emit_embeddings=True, embeddings is a list of per-detection embedding records.

    When ``detection_mode="windowed"``, each merged event is reduced to
    non-overlapping peak windows of exactly ``window_size_seconds`` via NMS.
    """

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

            state = _run_window_pipeline(
                audio=audio,
                target_sample_rate=target_sample_rate,
                model=model,
                pipeline=pipeline,
                window_size_seconds=window_size_seconds,
                hop_seconds=hop_seconds,
                input_format=input_format,
                normalization=normalization,
            )
            if not state.window_metas:
                files_done += 1
                if on_file_complete is not None:
                    on_file_complete([], files_done, n_audio_files)
                continue

            window_metas = state.window_metas
            all_emb = state.embeddings
            window_confidences = state.window_confidences
            window_records = state.window_records
            events = merge_detection_events(
                window_records, high_threshold, low_threshold
            )

            n_windows_total += len(window_metas)
            t_features_total += state.t_features
            t_inference_total += state.t_inference
            total_windows += len(all_emb)

            rel_path = str(audio_path.relative_to(audio_folder))
            base_epoch = _file_base_epoch(audio_path)

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

            # Canonicalize hysteresis-merged bounds (hysteresis merge itself
            # already happened inside ``_run_window_pipeline``).
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
                elif window_selection == "tiling":
                    events = select_tiled_windows_from_events(
                        events,
                        window_records,
                        window_size_seconds,
                        min_score=high_threshold,
                        max_logit_drop=max_logit_drop
                        if max_logit_drop is not None
                        else 2.0,
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
