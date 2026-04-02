"""Run detection: scan audio folder, classify windows, merge spans."""

import csv
import logging
import math
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

import pyarrow as pa
import pyarrow.parquet as pq

from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.features import extract_logmel_batch
from humpback.processing.inference import EmbeddingModel
from humpback.processing.windowing import (
    WindowMetadata,
    format_short_audio_window_message,
    slice_windows_with_metadata,
    window_sample_count,
)

logger = logging.getLogger(__name__)

_TS_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(?:\.(\d+))?Z")


def _file_base_epoch(filepath: Path) -> float:
    """Return the UTC epoch for a file: from timestamp in name, mtime, or 0.0."""
    match = _TS_PATTERN.search(filepath.name)
    if match is not None:
        year, month, day, hour, minute, second = (int(g) for g in match.groups()[:6])
        frac_str = match.group(7)
        microsecond = 0
        if frac_str:
            frac_str = frac_str[:6].ljust(6, "0")
            microsecond = int(frac_str)
        dt = datetime(
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond,
            tzinfo=timezone.utc,
        )
        return dt.timestamp()
    try:
        return os.path.getmtime(filepath)
    except OSError:
        return 0.0


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


def _smooth_scores(scores: list[float], window_size: int) -> list[float]:
    """Smooth confidence scores with a moving-average kernel."""
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


def select_peak_windows_from_events(
    events: list[dict],
    window_records: list[dict],
    window_size_seconds: float,
    min_score: float,
    smoothing_window: int = 3,
) -> list[dict]:
    """Reduce merged events to non-overlapping peak windows via NMS.

    For each merged event, finds overlapping window records, smooths scores,
    then iteratively selects the highest-scoring window and suppresses
    neighbors within ``window_size_seconds``.

    Returns detection dicts each spanning exactly ``window_size_seconds``.
    Audit fields (``raw_start_sec``, ``raw_end_sec``, ``merged_event_count``)
    are preserved from the parent event.
    """
    if not events or not window_records:
        return []

    result: list[dict] = []

    for event in events:
        ev_start = float(event["start_sec"])
        ev_end = float(event["end_sec"])

        # Filter window records overlapping event bounds (with tolerance).
        candidates = sorted(
            (
                rec
                for rec in window_records
                if float(rec["offset_sec"]) + 1e-6 >= ev_start
                and float(rec["end_sec"]) - 1e-6 <= ev_end
            ),
            key=lambda r: float(r["offset_sec"]),
        )
        if not candidates:
            continue

        raw_scores = [float(r["confidence"]) for r in candidates]
        smoothed = _smooth_scores(raw_scores, smoothing_window)

        # NMS loop: greedily pick best window, suppress overlapping.
        active = list(range(len(candidates)))
        while active:
            best_idx = max(active, key=lambda i: smoothed[i])
            if smoothed[best_idx] < min_score:
                break

            best_rec = candidates[best_idx]
            best_offset = float(best_rec["offset_sec"])
            best_end = best_offset + window_size_seconds
            best_conf = smoothed[best_idx]

            peak_det: dict = {
                "start_sec": best_offset,
                "end_sec": best_end,
                "avg_confidence": best_conf,
                "peak_confidence": best_conf,
                "n_windows": int(event.get("n_windows", 1)),
                "raw_start_sec": float(event.get("raw_start_sec", ev_start)),
                "raw_end_sec": float(event.get("raw_end_sec", ev_end)),
                "merged_event_count": int(event.get("merged_event_count", 1)),
            }
            # Carry extra fields (e.g. filename) from the parent event.
            for key in event:
                if key not in peak_det:
                    peak_det[key] = event[key]
            result.append(peak_det)

            # Suppress windows overlapping the selected one.
            active = [
                i
                for i in active
                if abs(float(candidates[i]["offset_sec"]) - best_offset)
                >= window_size_seconds - 1e-3
            ]

    # Deduplicate: adjacent events can share window records and independently
    # select the same peak.  Keep the higher-confidence entry per (start, end).
    seen: dict[tuple[float, float], int] = {}
    deduped: list[dict] = []
    for det in result:
        key = (det["start_sec"], det["end_sec"])
        if key in seen:
            existing = deduped[seen[key]]
            if det["peak_confidence"] > existing["peak_confidence"]:
                deduped[seen[key]] = det
        else:
            seen[key] = len(deduped)
            deduped.append(det)

    deduped.sort(key=lambda d: (d.get("filename", ""), d["start_sec"]))
    return deduped


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

            # Windowed mode: reduce each event to peak 5-sec windows via NMS.
            if detection_mode == "windowed":
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


TSV_FIELDNAMES = [
    "start_utc",
    "end_utc",
    "avg_confidence",
    "peak_confidence",
    "n_windows",
    "raw_start_utc",
    "raw_end_utc",
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


def write_detection_embeddings(records: list[dict], path: Path) -> None:
    """Write per-detection embedding records to a Parquet file."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    vector_dim = len(records[0]["embedding"])
    schema = pa.schema(
        [
            ("filename", pa.string()),
            ("start_sec", pa.float32()),
            ("end_sec", pa.float32()),
            ("embedding", pa.list_(pa.float32(), vector_dim)),
            ("confidence", pa.float32()),
        ]
    )
    table = pa.table(
        {
            "filename": [r["filename"] for r in records],
            "start_sec": [r["start_sec"] for r in records],
            "end_sec": [r["end_sec"] for r in records],
            "embedding": [r["embedding"] for r in records],
            "confidence": [r.get("confidence") for r in records],
        },
        schema=schema,
    )
    pq.write_table(table, path)


def read_detection_embedding(
    path: Path,
    filename: str,
    start_sec: float,
    end_sec: float,
) -> list[float] | None:
    """Read a single detection embedding matching filename and time bounds.

    Returns the embedding vector as a list of floats, or None if not found.
    """
    if not path.exists():
        return None
    table = pq.read_table(
        str(path),
        filters=[("filename", "=", filename)],
    )
    for i in range(table.num_rows):
        row_start = float(table.column("start_sec")[i].as_py())
        row_end = float(table.column("end_sec")[i].as_py())
        if abs(row_start - start_sec) < 0.01 and abs(row_end - end_sec) < 0.01:
            return table.column("embedding")[i].as_py()
    return None


# ---------------------------------------------------------------------------
# Embedding sync: diff row store vs embeddings parquet
# ---------------------------------------------------------------------------

_SYNC_TOLERANCE_SEC = 0.5


class EmbeddingDiffResult:
    """Result of comparing a detection row store against its embeddings parquet."""

    __slots__ = ("missing", "orphaned_indices", "matched_count")

    def __init__(
        self,
        missing: list[dict[str, str]],
        orphaned_indices: list[int],
        matched_count: int,
    ) -> None:
        self.missing = missing
        self.orphaned_indices = orphaned_indices
        self.matched_count = matched_count


def _embedding_utc_pairs(
    emb_path: Path,
) -> list[tuple[float, float]]:
    """Return (start_utc, end_utc) for every row in an embeddings parquet.

    Each embedding stores (filename, start_sec, end_sec) relative to the
    file's base epoch.  This converts them to absolute UTC.
    """
    table = pq.read_table(str(emb_path), columns=["filename", "start_sec", "end_sec"])
    filenames = table.column("filename").to_pylist()
    start_secs = table.column("start_sec").to_pylist()
    end_secs = table.column("end_sec").to_pylist()

    # Cache base epoch per unique filename to avoid repeated parsing.
    epoch_cache: dict[str, float] = {}
    pairs: list[tuple[float, float]] = []
    for fname, ss, es in zip(filenames, start_secs, end_secs):
        if fname not in epoch_cache:
            epoch_cache[fname] = _file_base_epoch(Path(fname))
        base = epoch_cache[fname]
        pairs.append((base + float(ss), base + float(es)))
    return pairs


def diff_row_store_vs_embeddings(
    row_store_path: Path,
    embeddings_path: Path,
) -> EmbeddingDiffResult:
    """Compare a detection row store against its embeddings parquet.

    Returns an ``EmbeddingDiffResult`` with:
    * ``missing`` – row-store rows that have no matching embedding.
    * ``orphaned_indices`` – indices into the embeddings parquet with no
      matching row-store entry (should be removed).
    * ``matched_count`` – number of row-store rows with a matching embedding.

    Matching uses a tolerance of ``_SYNC_TOLERANCE_SEC`` seconds on both
    ``start_utc`` and ``end_utc``.
    """
    from humpback.classifier.detection_rows import read_detection_row_store

    _, rows = read_detection_row_store(row_store_path)
    emb_pairs = _embedding_utc_pairs(embeddings_path)

    # Build list of (start_utc, end_utc) from row store.
    row_pairs: list[tuple[float, float]] = []
    for r in rows:
        s = r.get("start_utc", "")
        e = r.get("end_utc", "")
        if s and e:
            row_pairs.append((float(s), float(e)))

    # For each row-store entry, find a matching embedding.
    tol = _SYNC_TOLERANCE_SEC
    emb_matched: set[int] = set()
    missing: list[dict[str, str]] = []
    matched_count = 0

    for row_idx, (rs, re_) in enumerate(row_pairs):
        found = False
        for emb_idx, (es, ee) in enumerate(emb_pairs):
            if emb_idx in emb_matched:
                continue
            if abs(rs - es) < tol and abs(re_ - ee) < tol:
                emb_matched.add(emb_idx)
                matched_count += 1
                found = True
                break
        if not found:
            missing.append(rows[row_idx])

    # Orphaned = embedding indices not matched to any row.
    orphaned_indices = [i for i in range(len(emb_pairs)) if i not in emb_matched]

    return EmbeddingDiffResult(
        missing=missing,
        orphaned_indices=orphaned_indices,
        matched_count=matched_count,
    )


# ---------------------------------------------------------------------------
# Audio resolution for embedding sync
# ---------------------------------------------------------------------------


def _build_file_timeline(
    audio_folder: Path,
    target_sample_rate: int,
) -> list[tuple[float, Path, float]]:
    """Build a sorted list of (base_epoch, file_path, duration_sec) for an audio folder.

    Only includes files with a parseable timestamp in the filename.
    Uses soundfile.info() for fast header-only duration reads.
    """
    import soundfile as sf

    entries: list[tuple[float, Path, float]] = []
    audio_files = sorted(
        p for p in audio_folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    for af in audio_files:
        base = _file_base_epoch(af)
        if base == 0.0:
            continue
        try:
            info = sf.info(str(af))
            duration = info.duration
        except Exception:
            logger.debug("Cannot read info for %s, skipping", af)
            continue
        entries.append((base, af, duration))
    entries.sort(key=lambda e: e[0])
    return entries


def resolve_audio_for_window(
    start_utc: float,
    end_utc: float,
    audio_folder: Path,
    target_sample_rate: int,
    *,
    _file_timeline: list[tuple[float, Path, float]] | None = None,
    _audio_cache: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """Load audio for a UTC window from a local audio folder.

    Returns ``(audio_array, None)`` on success, or ``(None, reason)``
    when the audio cannot be resolved.

    The optional ``_file_timeline`` parameter accepts a pre-built file
    timeline (from ``_build_file_timeline``) to avoid re-scanning the
    folder for every window.

    The optional ``_audio_cache`` dict caches decoded+resampled audio
    keyed by file path, avoiding repeated decoding when multiple windows
    come from the same file.
    """
    window_dur = end_utc - start_utc
    if window_dur <= 0:
        return None, "invalid window (end <= start)"

    timeline = _file_timeline or _build_file_timeline(audio_folder, target_sample_rate)
    if not timeline:
        return None, "no audio files with parseable timestamps"

    # Find the file that covers start_utc.
    covering: tuple[float, Path, float] | None = None
    for base_epoch, fpath, dur in timeline:
        file_end = base_epoch + dur
        if base_epoch <= start_utc + 0.01 and file_end >= end_utc - 0.01:
            covering = (base_epoch, fpath, dur)
            break

    if covering is None:
        return None, f"no file covers UTC range [{start_utc}, {end_utc}]"

    base_epoch, fpath, _ = covering
    offset_sec = start_utc - base_epoch
    cache_key = str(fpath)

    if _audio_cache is not None and cache_key in _audio_cache:
        audio_data = _audio_cache[cache_key]
    else:
        try:
            audio_data, sr = decode_audio(fpath)
            audio_data = resample(audio_data, sr, target_sample_rate)
        except Exception as exc:
            return None, f"audio decode failed: {exc}"
        if _audio_cache is not None:
            _audio_cache[cache_key] = audio_data

    start_sample = int(offset_sec * target_sample_rate)
    end_sample = int((offset_sec + window_dur) * target_sample_rate)

    if start_sample < 0:
        start_sample = 0
    if end_sample > len(audio_data):
        return None, (
            f"window extends past file end "
            f"(need sample {end_sample}, file has {len(audio_data)})"
        )

    window_audio = audio_data[start_sample:end_sample]
    return window_audio, None


def resolve_audio_for_window_hydrophone(
    start_utc: float,
    end_utc: float,
    provider: Any,
    target_sample_rate: int,
) -> tuple[np.ndarray | None, str | None]:
    """Load audio for a UTC window via a hydrophone ArchiveProvider.

    Uses ``iter_audio_chunks`` with a tight time range so only the
    needed segment(s) are fetched.  With warm caches this reads from
    local disk.

    Returns ``(audio_array, None)`` on success, or ``(None, reason)``
    when the audio cannot be resolved.
    """
    from humpback.classifier.s3_stream import iter_audio_chunks

    window_dur = end_utc - start_utc
    if window_dur <= 0:
        return None, "invalid window (end <= start)"

    # Request a chunk covering the window with a small margin.
    margin = 1.0
    try:
        chunks = list(
            iter_audio_chunks(
                provider,
                start_utc - margin,
                end_utc + margin,
                chunk_seconds=window_dur + 2 * margin,
                target_sr=target_sample_rate,
            )
        )
    except Exception as exc:
        return None, f"provider audio fetch failed: {exc}"

    if not chunks:
        return None, "provider returned no audio for time range"

    # chunks: list of (audio_ndarray, chunk_start_utc_dt, segs_done, segs_total)
    # Concatenate and extract the precise window.
    for audio_data, chunk_start_dt, _, _ in chunks:
        chunk_start = chunk_start_dt.timestamp()
        chunk_end = chunk_start + len(audio_data) / target_sample_rate
        if chunk_start <= start_utc + 0.01 and chunk_end >= end_utc - 0.01:
            offset = start_utc - chunk_start
            start_sample = int(offset * target_sample_rate)
            end_sample = start_sample + int(window_dur * target_sample_rate)
            if end_sample <= len(audio_data):
                return audio_data[start_sample:end_sample], None

    return None, "audio chunk does not fully cover the window"
