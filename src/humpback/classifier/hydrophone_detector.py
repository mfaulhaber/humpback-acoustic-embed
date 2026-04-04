"""Streaming detection pipeline for Orcasound hydrophone audio."""

import logging
import threading
import time
from collections.abc import Callable

import numpy as np
from sklearn.pipeline import Pipeline

from humpback.classifier.archive import ArchiveProvider
from humpback.classifier.detector import (
    merge_detection_events,
    select_peak_windows_from_events,
    select_prominent_peaks_from_events,
    snap_and_merge_detection_events,
)
from humpback.classifier.s3_stream import (
    DEFAULT_HYDROPHONE_PREFETCH_INFLIGHT_SEGMENTS,
    DEFAULT_HYDROPHONE_PREFETCH_WORKERS,
    iter_audio_chunks,
    provider_supports_segment_prefetch,
)
from humpback.processing.features import extract_logmel_batch
from humpback.processing.inference import EmbeddingModel
from humpback.processing.windowing import slice_windows_with_metadata

logger = logging.getLogger(__name__)


def run_hydrophone_detection(
    provider: ArchiveProvider,
    start_timestamp: float,
    end_timestamp: float,
    pipeline: Pipeline,
    model: EmbeddingModel,
    window_size_seconds: float,
    target_sample_rate: int,
    confidence_threshold: float,
    input_format: str = "spectrogram",
    feature_config: dict | None = None,
    hop_seconds: float = 1.0,
    high_threshold: float = 0.70,
    low_threshold: float = 0.45,
    on_chunk_complete: Callable | None = None,
    on_chunk_diagnostics: Callable[[list[dict], int], None] | None = None,
    on_chunk_embeddings: Callable[[list[dict]], None] | None = None,
    on_alert: Callable | None = None,
    cancel_check: Callable[[], bool] | None = None,
    pause_gate: "threading.Event | None" = None,
    skip_segments: int = 0,
    prior_detections: list[dict] | None = None,
    on_resume_invalidation: Callable[[], None] | None = None,
    prefetch_enabled: bool = True,
    prefetch_workers: int = DEFAULT_HYDROPHONE_PREFETCH_WORKERS,
    prefetch_inflight_segments: int = DEFAULT_HYDROPHONE_PREFETCH_INFLIGHT_SEGMENTS,
    detection_mode: str | None = None,
    window_selection: str | None = None,
    min_prominence: float | None = None,
) -> tuple[list[dict], dict]:
    """Run detection on streamed hydrophone audio.

    Resume support:
    - skip_segments: number of timeline segments to skip (already processed)
    - prior_detections: detections from previous run to preserve

    When ``detection_mode="windowed"``, each merged event is reduced to
    non-overlapping peak windows of exactly ``window_size_seconds`` via NMS.

    Returns (all_detections, summary).
    """
    feature_config = feature_config or {}
    normalization = feature_config.get("normalization", "per_window_max")
    use_prefetch = (
        prefetch_enabled
        and prefetch_workers > 1
        and prefetch_inflight_segments > 1
        and provider_supports_segment_prefetch(provider)
    )

    all_detections: list[dict] = list(prior_detections) if prior_detections else []
    skip_invalidated = False
    all_confidences: list[float] = []
    total_windows = 0
    total_positive = 0
    t_pipeline_total = 0.0
    t_fetch_total = 0.0
    t_audio_decode_total = 0.0
    t_features_total = 0.0
    t_inference_total = 0.0
    time_covered = 0.0
    is_first_chunk = True

    def _on_segment_timing(fetch_sec: float, decode_sec: float) -> None:
        nonlocal t_fetch_total, t_audio_decode_total
        t_fetch_total += fetch_sec
        t_audio_decode_total += decode_sec

    for chunk_audio, chunk_start_utc, segs_done, segs_total in iter_audio_chunks(
        provider,
        start_timestamp,
        end_timestamp,
        chunk_seconds=60.0,
        target_sr=target_sample_rate,
        on_error=on_alert,
        skip_segments=skip_segments,
        prefetch_enabled=use_prefetch,
        prefetch_workers=prefetch_workers,
        prefetch_inflight_segments=prefetch_inflight_segments,
        on_segment_timing=_on_segment_timing,
    ):
        # On first yielded chunk, detect if skip was invalidated (timeline changed)
        if is_first_chunk and skip_segments > 0:
            is_first_chunk = False
            if segs_done < skip_segments:
                # Timeline shrank — skip was reset; clear prior detections
                logger.warning(
                    "Skip invalidated (segs_done=%d < skip_segments=%d); "
                    "clearing prior detections",
                    segs_done,
                    skip_segments,
                )
                all_detections.clear()
                skip_invalidated = True
                if on_resume_invalidation is not None:
                    on_resume_invalidation()
        else:
            is_first_chunk = False
        # Wait if paused (blocks until resumed or canceled)
        if pause_gate is not None:
            pause_gate.wait()

        if cancel_check and cancel_check():
            logger.info("Hydrophone detection canceled")
            break

        t0 = time.monotonic()
        window_samples = int(target_sample_rate * window_size_seconds)
        if len(chunk_audio) < window_samples:
            time_covered += len(chunk_audio) / target_sample_rate
            if on_chunk_complete:
                on_chunk_complete([], segs_done, segs_total, time_covered)
            continue

        # Slice windows
        raw_windows = []
        window_metas = []
        for window, meta in slice_windows_with_metadata(
            chunk_audio,
            target_sample_rate,
            window_size_seconds,
            hop_seconds=hop_seconds,
        ):
            window_metas.append(meta)
            raw_windows.append(window)

        if not raw_windows:
            time_covered += len(chunk_audio) / target_sample_rate
            if on_chunk_complete:
                on_chunk_complete([], segs_done, segs_total, time_covered)
            continue

        # Feature extraction
        if input_format == "waveform":
            batch_items = raw_windows
        else:
            t1 = time.monotonic()
            batch_items = extract_logmel_batch(
                raw_windows,
                target_sample_rate,
                n_mels=128,
                hop_length=1252,
                target_frames=128,
                normalization=normalization,
            )
            t_features_total += time.monotonic() - t1

        # Batch embed
        batch_size = 64
        file_embeddings = []
        for i in range(0, len(batch_items), batch_size):
            batch = np.stack(batch_items[i : i + batch_size])
            t1 = time.monotonic()
            embeddings = model.embed(batch)
            t_inference_total += time.monotonic() - t1
            file_embeddings.append(embeddings)

        all_emb = np.vstack(file_embeddings)
        total_windows += len(all_emb)

        # Classify
        proba = pipeline.predict_proba(all_emb)[:, 1]
        window_confidences = proba.tolist()

        # Build window records for event merging
        window_records = [
            {
                "offset_sec": meta.offset_sec,
                "end_sec": meta.offset_sec + window_size_seconds,
                "confidence": conf,
            }
            for meta, conf in zip(window_metas, window_confidences)
        ]

        synthetic_filename = chunk_start_utc.strftime("%Y%m%dT%H%M%SZ") + ".wav"
        chunk_diagnostics = []
        for i_meta, (meta, conf) in enumerate(zip(window_metas, window_confidences)):
            if meta.is_overlapped and i_meta > 0:
                prev_end = window_metas[i_meta - 1].offset_sec + window_size_seconds
                overlap_sec = prev_end - meta.offset_sec
            else:
                overlap_sec = 0.0
            chunk_diagnostics.append(
                {
                    "filename": synthetic_filename,
                    "window_index": meta.window_index,
                    "offset_sec": meta.offset_sec,
                    "end_sec": meta.offset_sec + window_size_seconds,
                    "confidence": conf,
                    "is_overlapped": meta.is_overlapped,
                    "overlap_sec": overlap_sec,
                }
            )
        if on_chunk_diagnostics is not None:
            on_chunk_diagnostics(chunk_diagnostics, segs_done)

        # Merge events, then canonicalize to snapped ranges.
        events = merge_detection_events(window_records, high_threshold, low_threshold)
        events = snap_and_merge_detection_events(events, window_size_seconds)

        # Windowed mode: reduce each event to peak windows.
        if detection_mode == "windowed":
            if window_selection == "prominence":
                events = select_prominent_peaks_from_events(
                    events,
                    window_records,
                    window_size_seconds,
                    min_score=high_threshold,
                    min_prominence=(
                        min_prominence if min_prominence is not None else 2.0
                    ),
                )
            else:
                events = select_peak_windows_from_events(
                    events,
                    window_records,
                    window_size_seconds,
                    min_score=high_threshold,
                )

        # Collect per-detection embeddings (peak-window vector) for this chunk
        if on_chunk_embeddings is not None:
            chunk_embedding_records: list[dict] = []
            for event in events:
                ev_start = float(event["start_sec"])
                ev_end = float(event["end_sec"])
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
                    chunk_embedding_records.append(
                        {
                            "filename": synthetic_filename,
                            "start_sec": ev_start,
                            "end_sec": ev_end,
                            "embedding": all_emb[best_idx].tolist(),
                        }
                    )
            if chunk_embedding_records:
                on_chunk_embeddings(chunk_embedding_records)

        # Convert file-relative events to UTC using chunk epoch.
        chunk_epoch = chunk_start_utc.timestamp()
        for event in events:
            event["start_utc"] = chunk_epoch + float(event.pop("start_sec"))
            event["end_utc"] = chunk_epoch + float(event.pop("end_sec"))
            raw_s = event.pop("raw_start_sec", None)
            raw_e = event.pop("raw_end_sec", None)
            event["raw_start_utc"] = (
                chunk_epoch + float(raw_s) if raw_s is not None else event["start_utc"]
            )
            event["raw_end_utc"] = (
                chunk_epoch + float(raw_e) if raw_e is not None else event["end_utc"]
            )
            event["hydrophone_name"] = provider.source_id
            all_detections.append(event)

        total_positive += sum(
            1 for c in window_confidences if c >= confidence_threshold
        )
        all_confidences.extend(window_confidences)
        time_covered += len(chunk_audio) / target_sample_rate
        t_pipeline_total += time.monotonic() - t0

        if on_chunk_complete:
            on_chunk_complete(list(events), segs_done, segs_total, time_covered)

    summary: dict = {
        "n_windows": total_windows,
        "n_detections": total_positive,
        "n_spans": len(all_detections),
        "hop_seconds": hop_seconds,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "detection_mode": detection_mode or "merged",
        "window_selection": window_selection or "nms",
        "hydrophone_id": provider.source_id,
        "time_covered_sec": time_covered,
        "prefetch_enabled": use_prefetch,
        "fetch_sec": t_fetch_total,
        "decode_sec": t_audio_decode_total,
        "features_sec": t_features_total,
        "inference_sec": t_inference_total,
        "pipeline_total_sec": t_pipeline_total,
    }
    if use_prefetch:
        summary["prefetch_workers"] = int(prefetch_workers)
        summary["prefetch_inflight_segments"] = int(prefetch_inflight_segments)
    if skip_segments > 0 and not skip_invalidated:
        summary["resumed_from_segment"] = skip_segments

    if all_confidences:
        conf_arr = np.array(all_confidences)
        summary["confidence_stats"] = {
            "mean": float(np.mean(conf_arr)),
            "median": float(np.median(conf_arr)),
            "std": float(np.std(conf_arr)),
            "min": float(np.min(conf_arr)),
            "max": float(np.max(conf_arr)),
            "pct_above_threshold": float(np.mean(conf_arr >= confidence_threshold)),
        }

    logger.info(
        "Hydrophone detection complete: %d windows, %d spans, %.1fs audio",
        total_windows,
        len(all_detections),
        time_covered,
    )
    logger.info(
        "Timing: fetch=%.3fs, decode=%.3fs, features=%.3fs, inference=%.3fs, pipeline_total=%.3fs",
        t_fetch_total,
        t_audio_decode_total,
        t_features_total,
        t_inference_total,
        t_pipeline_total,
    )

    return all_detections, summary
