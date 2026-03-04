"""Run detection: scan audio folder, classify windows, merge spans."""

import csv
import logging
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.features import extract_logmel
from humpback.processing.inference import EmbeddingModel
from humpback.processing.windowing import slice_windows

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
) -> tuple[list[dict], dict]:
    """Scan audio folder, classify each window, merge spans.

    Returns (detections_list, summary_dict).
    Each detection: {filename, start_sec, end_sec, avg_confidence, peak_confidence}.
    """
    feature_config = feature_config or {}
    normalization = feature_config.get("normalization", "per_window_max")

    audio_files = sorted(
        p for p in audio_folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_folder}")

    all_detections: list[dict] = []
    total_windows = 0
    total_positive = 0

    for audio_path in audio_files:
        try:
            audio, sr = decode_audio(audio_path)
            audio = resample(audio, sr, target_sample_rate)

            # Embed all windows
            batch_items: list[np.ndarray] = []
            batch_size = 32
            file_embeddings: list[np.ndarray] = []

            for window in slice_windows(audio, target_sample_rate, window_size_seconds):
                if input_format == "waveform":
                    batch_items.append(window)
                else:
                    spec = extract_logmel(
                        window,
                        target_sample_rate,
                        n_mels=128,
                        hop_length=1252,
                        target_frames=128,
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

            # Merge spans
            spans = merge_detection_spans(
                window_confidences, confidence_threshold, window_size_seconds
            )

            rel_path = str(audio_path.relative_to(audio_folder))
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
    }

    logger.info(
        "Detection complete: %d files, %d windows, %d detections, %d spans",
        summary["n_files"], summary["n_windows"],
        summary["n_detections"], summary["n_spans"],
    )

    return all_detections, summary


def write_detections_tsv(detections: list[dict], path: Path) -> None:
    """Write detections to a TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "start_sec", "end_sec", "avg_confidence", "peak_confidence"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for det in detections:
            writer.writerow({k: det[k] for k in fieldnames})
