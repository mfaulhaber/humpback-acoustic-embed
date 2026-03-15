"""Run detection: scan audio folder, classify windows, merge spans."""

import csv
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

import pyarrow as pa
import pyarrow.parquet as pq

from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.features import extract_logmel_batch
from humpback.processing.inference import EmbeddingModel
from humpback.processing.windowing import WindowMetadata, slice_windows_with_metadata

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


def merge_detection_spans(
    window_confidences: list[float],
    confidence_threshold: float,
    window_size_seconds: float,
) -> list[dict]:
    """Merge consecutive positive windows into detection spans.

    Returns list of dicts with: start_sec, end_sec, avg_confidence, peak_confidence.
    """
    spans: list[dict] = []
    in_span = False
    span_start = 0
    span_confidences: list[float] = []

    for i, conf in enumerate(window_confidences):
        if conf >= confidence_threshold:
            if not in_span:
                in_span = True
                span_start = i
                span_confidences = []
            span_confidences.append(conf)
        else:
            if in_span:
                spans.append(
                    {
                        "start_sec": span_start * window_size_seconds,
                        "end_sec": (i) * window_size_seconds,
                        "avg_confidence": float(np.mean(span_confidences)),
                        "peak_confidence": float(np.max(span_confidences)),
                    }
                )
                in_span = False

    # Close final span
    if in_span:
        spans.append(
            {
                "start_sec": span_start * window_size_seconds,
                "end_sec": len(window_confidences) * window_size_seconds,
                "avg_confidence": float(np.mean(span_confidences)),
                "peak_confidence": float(np.max(span_confidences)),
            }
        )

    return spans


def merge_detection_events(
    window_records: list[dict],
    high_threshold: float,
    low_threshold: float,
) -> list[dict]:
    """Merge windows into detection events using hysteresis thresholds.

    When not in event: confidence >= high_threshold opens a new event.
    When in event: confidence >= low_threshold continues; below closes.

    Returns list of dicts: {start_sec, end_sec, avg_confidence, peak_confidence, n_windows}.
    """
    events: list[dict] = []
    in_event = False
    event_start = 0.0
    event_end = 0.0
    event_confidences: list[float] = []

    for rec in window_records:
        conf = rec["confidence"]
        if not in_event:
            if conf >= high_threshold:
                in_event = True
                event_start = rec["offset_sec"]
                event_end = rec["end_sec"]
                event_confidences = [conf]
        else:
            if conf >= low_threshold:
                event_end = rec["end_sec"]
                event_confidences.append(conf)
            else:
                events.append(
                    {
                        "start_sec": event_start,
                        "end_sec": event_end,
                        "avg_confidence": float(np.mean(event_confidences)),
                        "peak_confidence": float(np.max(event_confidences)),
                        "n_windows": len(event_confidences),
                    }
                )
                in_event = False

    # Close final event
    if in_event:
        events.append(
            {
                "start_sec": event_start,
                "end_sec": event_end,
                "avg_confidence": float(np.mean(event_confidences)),
                "peak_confidence": float(np.max(event_confidences)),
                "n_windows": len(event_confidences),
            }
        )

    return events


def snap_event_bounds(
    start_sec: float,
    end_sec: float,
    window_size_seconds: float,
) -> tuple[float, float]:
    """Snap event bounds outward to window-size multiples."""
    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be > 0")
    if end_sec <= start_sec:
        return start_sec, end_sec
    snapped_start = math.floor(start_sec / window_size_seconds) * window_size_seconds
    snapped_end = math.ceil(end_sec / window_size_seconds) * window_size_seconds
    if snapped_end <= snapped_start:
        snapped_end = snapped_start + window_size_seconds
    return float(snapped_start), float(snapped_end)


def snap_and_merge_detection_events(
    events: list[dict],
    window_size_seconds: float,
) -> list[dict]:
    """Snap events to canonical bounds and merge snapped-range collisions."""
    if not events:
        return []

    grouped: dict[tuple[float, float], dict] = {}
    for event in events:
        raw_start = float(event["start_sec"])
        raw_end = float(event["end_sec"])
        snap_start, snap_end = snap_event_bounds(
            raw_start, raw_end, window_size_seconds
        )
        key = (snap_start, snap_end)

        n_windows = int(event.get("n_windows", 1) or 1)
        avg_conf = float(event.get("avg_confidence", 0.0))
        peak_conf = float(event.get("peak_confidence", 0.0))

        existing = grouped.get(key)
        if existing is None:
            grouped[key] = {
                "start_sec": snap_start,
                "end_sec": snap_end,
                "avg_confidence": avg_conf,
                "peak_confidence": peak_conf,
                "n_windows": n_windows,
                "raw_start_sec": raw_start,
                "raw_end_sec": raw_end,
                "merged_event_count": 1,
                "_weighted_sum": avg_conf * n_windows,
                "_weight": n_windows,
            }
            continue

        existing["_weighted_sum"] += avg_conf * n_windows
        existing["_weight"] += n_windows
        existing["avg_confidence"] = existing["_weighted_sum"] / existing["_weight"]
        existing["peak_confidence"] = max(existing["peak_confidence"], peak_conf)
        existing["n_windows"] += n_windows
        existing["raw_start_sec"] = min(existing["raw_start_sec"], raw_start)
        existing["raw_end_sec"] = max(existing["raw_end_sec"], raw_end)
        existing["merged_event_count"] += 1

    merged = sorted(grouped.values(), key=lambda e: (e["start_sec"], e["end_sec"]))
    for event in merged:
        event.pop("_weighted_sum", None)
        event.pop("_weight", None)
    return merged


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
) -> tuple[list[dict], dict, list[dict] | None]:
    """Scan audio folder, classify each window, merge events.

    Returns (detections_list, summary_dict, diagnostics_or_none).
    Each detection: {filename, start_sec, end_sec, avg_confidence, peak_confidence, n_windows}.
    When emit_diagnostics=True, diagnostics is a list of per-window records.
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

            window_samples = int(target_sample_rate * window_size_seconds)
            if len(audio) < window_samples:
                logger.warning(
                    "Skipping %s: audio too short (%.3fs < %.1fs window)",
                    audio_path.name,
                    len(audio) / target_sample_rate,
                    window_size_seconds,
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

            for event in events:
                event["filename"] = rel_path
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

    return all_detections, summary, diagnostics_records


TSV_FIELDNAMES = [
    "filename",
    "start_sec",
    "end_sec",
    "avg_confidence",
    "peak_confidence",
    "n_windows",
    "raw_start_sec",
    "raw_end_sec",
    "merged_event_count",
]

WINDOW_DIAGNOSTICS_SCHEMA = pa.schema(
    [
        ("filename", pa.string()),
        ("window_index", pa.int32()),
        ("offset_sec", pa.float32()),
        ("end_sec", pa.float32()),
        ("confidence", pa.float32()),
        ("is_overlapped", pa.bool_()),
        ("overlap_sec", pa.float32()),
    ]
)


def _window_diagnostics_table(records: list[dict]) -> pa.Table:
    """Build a Parquet table for persisted window diagnostics."""
    return pa.table(
        {
            "filename": [r["filename"] for r in records],
            "window_index": [r["window_index"] for r in records],
            "offset_sec": [r["offset_sec"] for r in records],
            "end_sec": [r["end_sec"] for r in records],
            "confidence": [r["confidence"] for r in records],
            "is_overlapped": [r["is_overlapped"] for r in records],
            "overlap_sec": [r["overlap_sec"] for r in records],
        },
        schema=WINDOW_DIAGNOSTICS_SCHEMA,
    )


def read_detections_tsv(path: Path, fieldnames: list[str] | None = None) -> list[dict]:
    """Read detections from a TSV file. Returns empty list if file missing/empty."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def write_detections_tsv(
    detections: list[dict],
    path: Path,
    fieldnames: list[str] | None = None,
) -> None:
    """Write detections to a TSV file (overwrites)."""
    tsv_fieldnames = fieldnames or TSV_FIELDNAMES
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tsv_fieldnames, delimiter="\t")
        writer.writeheader()
        for det in detections:
            writer.writerow({k: det.get(k, "") for k in tsv_fieldnames})


def append_detections_tsv(
    detections: list[dict],
    path: Path,
    fieldnames: list[str] | None = None,
) -> None:
    """Append detections to an existing TSV file (creates with header if needed)."""
    if not detections:
        return
    tsv_fieldnames = fieldnames or TSV_FIELDNAMES
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tsv_fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        for det in detections:
            writer.writerow({k: det.get(k, "") for k in tsv_fieldnames})


def write_window_diagnostics(records: list[dict], path: Path) -> None:
    """Write per-window diagnostic records to a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_window_diagnostics_table(records), path)


def write_window_diagnostics_shard(
    records: list[dict],
    directory: Path,
    shard_name: str,
) -> Path | None:
    """Write one Parquet shard for incremental diagnostics persistence."""
    if not records:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    shard_path = directory / shard_name
    pq.write_table(_window_diagnostics_table(records), shard_path)
    return shard_path


def read_window_diagnostics_table(
    path: Path,
    *,
    filename: str | None = None,
) -> pa.Table:
    """Read window diagnostics from a single file or a shard directory."""
    read_kwargs: dict[str, Any] = {}
    if filename is not None:
        read_kwargs["filters"] = [("filename", "=", filename)]
    return pq.read_table(str(path), **read_kwargs)
