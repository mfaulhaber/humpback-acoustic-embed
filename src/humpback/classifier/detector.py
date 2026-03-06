"""Run detection: scan audio folder, classify windows, merge spans."""

import csv
import logging
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

import pyarrow as pa
import pyarrow.parquet as pq

from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.features import extract_logmel
from humpback.processing.inference import EmbeddingModel
from humpback.processing.windowing import WindowMetadata, slice_windows, slice_windows_with_metadata

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
                spans.append({
                    "start_sec": span_start * window_size_seconds,
                    "end_sec": (i) * window_size_seconds,
                    "avg_confidence": float(np.mean(span_confidences)),
                    "peak_confidence": float(np.max(span_confidences)),
                })
                in_span = False

    # Close final span
    if in_span:
        spans.append({
            "start_sec": span_start * window_size_seconds,
            "end_sec": len(window_confidences) * window_size_seconds,
            "avg_confidence": float(np.mean(span_confidences)),
            "peak_confidence": float(np.max(span_confidences)),
        })

    return spans


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
) -> tuple[list[dict], dict, list[dict] | None]:
    """Scan audio folder, classify each window, merge spans.

    Returns (detections_list, summary_dict, diagnostics_or_none).
    Each detection: {filename, start_sec, end_sec, avg_confidence, peak_confidence}.
    When emit_diagnostics=True, diagnostics is a list of per-window records.
    """
    feature_config = feature_config or {}
    normalization = feature_config.get("normalization", "per_window_max")

    audio_files = sorted(
        p for p in audio_folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_folder}")

    all_detections: list[dict] = []
    diagnostics_records: list[dict] = [] if emit_diagnostics else None
    total_windows = 0
    total_positive = 0
    n_skipped_short = 0

    for audio_path in audio_files:
        try:
            audio, sr = decode_audio(audio_path)
            audio = resample(audio, sr, target_sample_rate)

            window_samples = int(target_sample_rate * window_size_seconds)
            if len(audio) < window_samples:
                logger.warning(
                    "Skipping %s: audio too short (%.3fs < %.1fs window)",
                    audio_path.name, len(audio) / target_sample_rate, window_size_seconds,
                )
                n_skipped_short += 1
                continue

            # Embed all windows
            batch_items: list[np.ndarray] = []
            batch_size = 32
            file_embeddings: list[np.ndarray] = []
            window_metas: list[WindowMetadata] = [] if emit_diagnostics else None

            if emit_diagnostics:
                for window, meta in slice_windows_with_metadata(
                    audio, target_sample_rate, window_size_seconds
                ):
                    window_metas.append(meta)
                    if input_format == "waveform":
                        batch_items.append(window)
                    else:
                        spec = extract_logmel(
                            window, target_sample_rate,
                            n_mels=128, hop_length=1252, target_frames=128,
                            normalization=normalization,
                        )
                        batch_items.append(spec)

                    if len(batch_items) >= batch_size:
                        batch = np.stack(batch_items)
                        embeddings = model.embed(batch)
                        file_embeddings.append(embeddings)
                        batch_items.clear()
            else:
                for window in slice_windows(audio, target_sample_rate, window_size_seconds):
                    if input_format == "waveform":
                        batch_items.append(window)
                    else:
                        spec = extract_logmel(
                            window, target_sample_rate,
                            n_mels=128, hop_length=1252, target_frames=128,
                            normalization=normalization,
                        )
                        batch_items.append(spec)

                    if len(batch_items) >= batch_size:
                        batch = np.stack(batch_items)
                        embeddings = model.embed(batch)
                        file_embeddings.append(embeddings)
                        batch_items.clear()

            if batch_items:
                batch = np.stack(batch_items)
                embeddings = model.embed(batch)
                file_embeddings.append(embeddings)

            if not file_embeddings:
                continue

            all_emb = np.vstack(file_embeddings)
            total_windows += len(all_emb)

            # Classify
            proba = pipeline.predict_proba(all_emb)[:, 1]  # P(whale)
            window_confidences = proba.tolist()

            rel_path = str(audio_path.relative_to(audio_folder))

            # Collect per-window diagnostics
            if emit_diagnostics and window_metas is not None:
                for i_meta, (meta, conf) in enumerate(zip(window_metas, window_confidences)):
                    if meta.is_overlapped and i_meta > 0:
                        prev_end = window_metas[i_meta - 1].offset_sec + window_size_seconds
                        overlap_sec = prev_end - meta.offset_sec
                    else:
                        overlap_sec = 0.0
                    diagnostics_records.append({
                        "filename": rel_path,
                        "window_index": meta.window_index,
                        "offset_sec": meta.offset_sec,
                        "confidence": conf,
                        "is_overlapped": meta.is_overlapped,
                        "overlap_sec": overlap_sec,
                    })

            # Merge spans
            spans = merge_detection_spans(
                window_confidences, confidence_threshold, window_size_seconds
            )

            for span in spans:
                span["filename"] = rel_path
                all_detections.append(span)

            total_positive += sum(1 for c in window_confidences if c >= confidence_threshold)

        except Exception:
            logger.warning("Failed to process %s, skipping", audio_path, exc_info=True)
            continue

    summary = {
        "n_files": len(audio_files),
        "n_windows": total_windows,
        "n_detections": total_positive,
        "n_spans": len(all_detections),
        "n_skipped_short": n_skipped_short,
    }

    logger.info(
        "Detection complete: %d files, %d windows, %d detections, %d spans, %d skipped (short)",
        summary["n_files"], summary["n_windows"],
        summary["n_detections"], summary["n_spans"], summary["n_skipped_short"],
    )

    return all_detections, summary, diagnostics_records


def write_detections_tsv(detections: list[dict], path: Path) -> None:
    """Write detections to a TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "start_sec", "end_sec", "avg_confidence", "peak_confidence"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for det in detections:
            writer.writerow({k: det[k] for k in fieldnames})


def write_window_diagnostics(records: list[dict], path: Path) -> None:
    """Write per-window diagnostic records to a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        ("filename", pa.string()),
        ("window_index", pa.int32()),
        ("offset_sec", pa.float32()),
        ("confidence", pa.float32()),
        ("is_overlapped", pa.bool_()),
        ("overlap_sec", pa.float32()),
    ])
    table = pa.table({
        "filename": [r["filename"] for r in records],
        "window_index": [r["window_index"] for r in records],
        "offset_sec": [r["offset_sec"] for r in records],
        "confidence": [r["confidence"] for r in records],
        "is_overlapped": [r["is_overlapped"] for r in records],
        "overlap_sec": [r["overlap_sec"] for r in records],
    }, schema=schema)
    pq.write_table(table, path)
