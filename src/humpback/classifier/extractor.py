"""Extract labeled audio samples from detection TSV files."""

import csv
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

from humpback.classifier.archive import ArchiveProvider
from humpback.classifier.detector import read_window_diagnostics_table
from humpback.classifier.s3_stream import build_stream_timeline, resolve_audio_slice
from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.features import extract_logmel_batch
from humpback.processing.inference import EmbeddingModel
from humpback.processing.windowing import slice_windows_with_metadata

logger = logging.getLogger(__name__)

# Regex patterns for parsing timestamps from filenames
# e.g. "20250115T143022Z_..." or "20250115T143022.123456Z_..."
_TS_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(?:\.(\d+))?Z")
_COMPACT_TS_FORMAT = "%Y%m%dT%H%M%SZ"
_KNOWN_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
_OUTPUT_AUDIO_EXTENSION = ".flac"
DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW = 3
DEFAULT_POSITIVE_SELECTION_MIN_SCORE = 0.70
DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE = 0.60
POSITIVE_SELECTION_SCORE_SOURCE_STORED = "stored_diagnostics"
POSITIVE_SELECTION_SCORE_SOURCE_FALLBACK = "rescored_fallback"
POSITIVE_SELECTION_FIELDNAMES = [
    "positive_selection_score_source",
    "positive_selection_decision",
    "positive_selection_offsets",
    "positive_selection_raw_scores",
    "positive_selection_smoothed_scores",
    "positive_selection_start_sec",
    "positive_selection_end_sec",
    "positive_selection_peak_score",
    "positive_extract_filename",
]


@dataclass(slots=True)
class PositiveSelectionResult:
    score_source: str
    decision: Literal["positive", "skip"]
    offsets: list[float]
    raw_scores: list[float]
    smoothed_scores: list[float]
    start_sec: float | None
    end_sec: float | None
    peak_score: float | None


def parse_recording_timestamp(filename: str) -> datetime | None:
    """Extract a recording start timestamp from a filename.

    Looks for ISO-like pattern YYYYMMDDTHHMMSSZ or YYYYMMDDTHHMMSS.ffffffZ.
    Returns a timezone-aware UTC datetime, or None if no pattern found.
    """
    m = _TS_PATTERN.search(filename)
    if m is None:
        return None
    year, month, day, hour, minute, second = (int(g) for g in m.groups()[:6])
    frac_str = m.group(7)
    microsecond = 0
    if frac_str:
        # Pad or truncate to 6 digits
        frac_str = frac_str[:6].ljust(6, "0")
        microsecond = int(frac_str)
    return datetime(
        year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc
    )


def _format_ts(dt: datetime) -> str:
    """Format datetime as YYYYMMDDTHHMMss.ffffffZ."""
    return dt.strftime("%Y%m%dT%H%M%S.%fZ")


def _date_folder(dt: datetime) -> str:
    """Return YYYY/MM/dd path component."""
    return dt.strftime("%Y/%m/%d")


def _strip_known_audio_extension(filename: str) -> str:
    """Strip a supported audio extension from a filename."""
    suffix = Path(filename).suffix.lower()
    if suffix in _KNOWN_AUDIO_EXTENSIONS:
        return filename[: -len(suffix)]
    return filename


def _with_output_audio_extension(filename: str) -> str:
    """Return filename with the configured extracted-audio extension."""
    return f"{_strip_known_audio_extension(filename)}{_OUTPUT_AUDIO_EXTENSION}"


def _validate_positive_selection_config(
    smoothing_window: int,
    min_score: float,
    extend_min_score: float,
) -> None:
    """Validate positive-window selection parameters."""
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError(
            "positive_selection_smoothing_window must be an odd integer >= 1"
        )
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("positive_selection_min_score must be between 0.0 and 1.0")
    if not 0.0 <= extend_min_score <= 1.0:
        raise ValueError(
            "positive_selection_extend_min_score must be between 0.0 and 1.0"
        )


def _read_tsv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read all TSV rows while preserving the original field order."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _write_tsv_rows(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]]
) -> None:
    """Atomically rewrite a TSV file with updated rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tsv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _ensure_fieldnames(fieldnames: list[str], required: list[str]) -> list[str]:
    """Append required fieldnames without disturbing existing order."""
    for field in required:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def _is_positive_row(row: dict[str, str]) -> bool:
    return row.get("humpback", "").strip() == "1" or row.get("orca", "").strip() == "1"


def _is_negative_row(row: dict[str, str]) -> bool:
    return (
        row.get("ship", "").strip() == "1" or row.get("background", "").strip() == "1"
    )


def _positive_labels_for_row(row: dict[str, str]) -> list[str]:
    labels: list[str] = []
    if row.get("humpback", "").strip() == "1":
        labels.append("humpback")
    if row.get("orca", "").strip() == "1":
        labels.append("orca")
    return labels


def _negative_labels_for_row(row: dict[str, str]) -> list[str]:
    labels: list[str] = []
    if row.get("ship", "").strip() == "1":
        labels.append("ship")
    if row.get("background", "").strip() == "1":
        labels.append("background")
    return labels


def _serialize_json_list(values: list[float]) -> str:
    """Serialize float lists compactly for TSV storage."""
    rounded = [round(float(v), 6) for v in values]
    return json.dumps(rounded, separators=(",", ":"))


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _blank_positive_selection_fields() -> dict[str, str]:
    return {field: "" for field in POSITIVE_SELECTION_FIELDNAMES}


def _selection_result_to_row_update(
    result: PositiveSelectionResult,
    *,
    positive_extract_filename: str | None,
) -> dict[str, str]:
    """Convert a positive-selection result into TSV row fields."""
    return {
        "positive_selection_score_source": result.score_source,
        "positive_selection_decision": result.decision,
        "positive_selection_offsets": _serialize_json_list(result.offsets),
        "positive_selection_raw_scores": _serialize_json_list(result.raw_scores),
        "positive_selection_smoothed_scores": _serialize_json_list(
            result.smoothed_scores
        ),
        "positive_selection_start_sec": (
            f"{result.start_sec:.6f}" if result.start_sec is not None else ""
        ),
        "positive_selection_end_sec": (
            f"{result.end_sec:.6f}" if result.end_sec is not None else ""
        ),
        "positive_selection_peak_score": (
            f"{result.peak_score:.6f}" if result.peak_score is not None else ""
        ),
        "positive_extract_filename": positive_extract_filename or "",
    }


def _smooth_scores(scores: list[float], window_size: int) -> list[float]:
    """Smooth scores with centered moving average and edge padding."""
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


def _candidate_offset_key(offset_sec: float) -> float:
    """Normalize candidate offsets for exact-window lookup."""
    return round(float(offset_sec), 6)


def _load_window_records(
    diagnostics_path: Path | None,
    *,
    filename: str,
) -> list[dict[str, float]] | None:
    """Load stored diagnostics rows for a single source filename."""
    if diagnostics_path is None or not diagnostics_path.exists():
        return None
    try:
        table = read_window_diagnostics_table(diagnostics_path, filename=filename)
    except Exception:
        logger.debug(
            "Failed to read stored window diagnostics for %s from %s",
            filename,
            diagnostics_path,
            exc_info=True,
        )
        return None

    records: list[dict[str, float]] = []
    for i in range(table.num_rows):
        records.append(
            {
                "offset_sec": float(table.column("offset_sec")[i].as_py()),
                "end_sec": float(table.column("end_sec")[i].as_py()),
                "confidence": float(table.column("confidence")[i].as_py()),
            }
        )
    return records


def _score_segment_windows(
    audio_segment: np.ndarray,
    *,
    source_sr: int,
    row_start_sec: float,
    pipeline: Any,
    model: EmbeddingModel,
    window_size_seconds: float,
    target_sample_rate: int,
    input_format: str = "spectrogram",
    feature_config: dict | None = None,
    hop_seconds: float = 1.0,
) -> list[dict[str, float]]:
    """Fallback: score candidate windows by re-running the classifier on a clip."""
    feature_config = feature_config or {}
    normalization = feature_config.get("normalization", "per_window_max")
    audio = (
        resample(audio_segment, source_sr, target_sample_rate)
        if source_sr != target_sample_rate
        else audio_segment
    )

    raw_windows: list[np.ndarray] = []
    offsets: list[float] = []
    for window, meta in slice_windows_with_metadata(
        audio,
        target_sample_rate,
        window_size_seconds,
        hop_seconds=hop_seconds,
    ):
        raw_windows.append(window)
        offsets.append(row_start_sec + meta.offset_sec)

    if not raw_windows:
        return []

    if input_format == "waveform":
        batch_items: list[np.ndarray] = raw_windows
    else:
        batch_items = extract_logmel_batch(
            raw_windows,
            target_sample_rate,
            n_mels=128,
            hop_length=1252,
            target_frames=128,
            normalization=normalization,
        )

    batch_size = 64
    embeddings: list[np.ndarray] = []
    for i in range(0, len(batch_items), batch_size):
        batch = np.stack(batch_items[i : i + batch_size])
        embeddings.append(model.embed(batch))
    all_emb = np.vstack(embeddings)
    proba = pipeline.predict_proba(all_emb)[:, 1]

    return [
        {
            "offset_sec": offset_sec,
            "end_sec": offset_sec + window_size_seconds,
            "confidence": float(conf),
        }
        for offset_sec, conf in zip(offsets, proba.tolist())
    ]


def _select_positive_window(
    *,
    row_start_sec: float,
    row_end_sec: float,
    window_size_seconds: float,
    window_records: list[dict[str, float]],
    smoothing_window: int,
    min_score: float,
    extend_min_score: float,
    score_source: str,
) -> PositiveSelectionResult:
    """Select the best-scoring candidate window within a labeled positive row."""
    deduped: dict[tuple[float, float], dict[str, float]] = {}
    for rec in window_records:
        offset_sec = float(rec["offset_sec"])
        end_sec = float(rec["end_sec"])
        if offset_sec + 1e-6 < row_start_sec or end_sec - 1e-6 > row_end_sec:
            continue
        deduped[(offset_sec, end_sec)] = {
            "offset_sec": offset_sec,
            "end_sec": end_sec,
            "confidence": float(rec["confidence"]),
        }

    candidates = sorted(
        deduped.values(),
        key=lambda rec: (rec["offset_sec"], rec["end_sec"]),
    )
    if not candidates:
        return PositiveSelectionResult(
            score_source=score_source,
            decision="skip",
            offsets=[],
            raw_scores=[],
            smoothed_scores=[],
            start_sec=None,
            end_sec=None,
            peak_score=None,
        )

    offsets = [float(rec["offset_sec"]) for rec in candidates]
    raw_scores = [float(rec["confidence"]) for rec in candidates]
    smoothed_scores = _smooth_scores(raw_scores, smoothing_window)
    best_idx = int(np.argmax(np.asarray(smoothed_scores, dtype=np.float32)))
    peak_score = float(smoothed_scores[best_idx])
    best = candidates[best_idx]
    start_sec = float(best["offset_sec"])
    end_sec = float(best["end_sec"])

    if peak_score >= min_score:
        candidates_by_offset = {
            _candidate_offset_key(float(rec["offset_sec"])): (
                rec,
                float(smoothed_scores[idx]),
            )
            for idx, rec in enumerate(candidates)
        }

        while True:
            left_key = _candidate_offset_key(start_sec - window_size_seconds)
            right_key = _candidate_offset_key(end_sec)
            left_candidate = candidates_by_offset.get(left_key)
            right_candidate = candidates_by_offset.get(right_key)

            left_score = left_candidate[1] if left_candidate is not None else None
            right_score = right_candidate[1] if right_candidate is not None else None
            can_extend_left = left_score is not None and left_score >= extend_min_score
            can_extend_right = (
                right_score is not None and right_score >= extend_min_score
            )
            if not can_extend_left and not can_extend_right:
                break

            if can_extend_left and (not can_extend_right or left_score >= right_score):
                assert left_candidate is not None
                start_sec = float(left_candidate[0]["offset_sec"])
            else:
                assert right_candidate is not None
                end_sec = float(right_candidate[0]["end_sec"])

    return PositiveSelectionResult(
        score_source=score_source,
        decision="positive" if peak_score >= min_score else "skip",
        offsets=offsets,
        raw_scores=raw_scores,
        smoothed_scores=smoothed_scores,
        start_sec=start_sec,
        end_sec=end_sec,
        peak_score=peak_score,
    )


def _resolve_local_clip_name(
    source_filename: str,
    start_sec: float,
    end_sec: float,
) -> tuple[str, str]:
    """Build a local extracted filename and date folder for clip bounds."""
    recording_ts = parse_recording_timestamp(source_filename)
    if recording_ts:
        abs_start = recording_ts + timedelta(seconds=start_sec)
        abs_end = recording_ts + timedelta(seconds=end_sec)
        clip_name = (
            f"{_format_ts(abs_start)}_{_format_ts(abs_end)}{_OUTPUT_AUDIO_EXTENSION}"
        )
        date_folder = _date_folder(abs_start)
    else:
        stem = Path(source_filename).stem
        clip_name = f"{stem}_{start_sec}_{end_sec}{_OUTPUT_AUDIO_EXTENSION}"
        date_folder = "unknown_date"
    return clip_name, date_folder


def _resolve_positive_output_path(
    root: Path,
    *,
    label: str,
    clip_name: str,
    source_id: str | None = None,
) -> Path:
    """Resolve the final positive artifact path from its filename."""
    clip_start = parse_recording_timestamp(clip_name)
    date_folder = _date_folder(clip_start) if clip_start is not None else "unknown_date"
    if source_id:
        return root / label / source_id / date_folder / clip_name
    return root / label / date_folder / clip_name


def _delete_stale_positive_outputs(
    root: Path,
    *,
    clip_name: str,
    labels: list[str],
    source_id: str | None = None,
) -> None:
    """Remove previously extracted positive artifacts for one row."""
    if not clip_name:
        return
    for label in labels:
        path = _resolve_positive_output_path(
            root,
            label=label,
            clip_name=_with_output_audio_extension(clip_name),
            source_id=source_id,
        )
        if path.exists():
            path.unlink()


def write_flac_file(audio_segment: np.ndarray, sr: int, output_path: Path) -> None:
    """Write a float32 audio segment as 16-bit PCM FLAC."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Peak normalize
    peak = np.max(np.abs(audio_segment))
    if peak > 0:
        audio_segment = audio_segment / peak
    sf.write(
        str(output_path),
        audio_segment.astype(np.float32),
        sr,
        format="FLAC",
        subtype="PCM_16",
    )


def write_wav_file(audio_segment: np.ndarray, sr: int, output_path: Path) -> None:
    """Backward-compatible alias for callers that imported the old helper name."""
    write_flac_file(audio_segment, sr, output_path)


def extract_labeled_samples(
    tsv_path: str | Path,
    audio_folder: str | Path,
    positive_output_path: str | Path,
    negative_output_path: str | Path,
    window_size_seconds: float = 5.0,
    window_diagnostics_path: str | Path | None = None,
    positive_selection_smoothing_window: int = DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW,
    positive_selection_min_score: float = DEFAULT_POSITIVE_SELECTION_MIN_SCORE,
    positive_selection_extend_min_score: float = (
        DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE
    ),
    fallback_pipeline: Any | None = None,
    fallback_model: EmbeddingModel | None = None,
    fallback_target_sample_rate: int = 32000,
    fallback_input_format: str = "spectrogram",
    fallback_feature_config: dict | None = None,
) -> dict:
    """Extract labeled audio segments from a detection TSV.

    Reads the TSV, filters to rows with at least one label=1,
    slices audio, and writes WAV files to the appropriate directories.

    Returns a summary dict with counts per label.
    """
    tsv_path = Path(tsv_path)
    audio_folder = Path(audio_folder)
    positive_output_path = Path(positive_output_path)
    negative_output_path = Path(negative_output_path)
    diagnostics_path = (
        Path(window_diagnostics_path) if window_diagnostics_path is not None else None
    )
    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be > 0")
    _validate_positive_selection_config(
        positive_selection_smoothing_window,
        positive_selection_min_score,
        positive_selection_extend_min_score,
    )

    fieldnames, all_rows = _read_tsv_rows(tsv_path)
    _ensure_fieldnames(fieldnames, POSITIVE_SELECTION_FIELDNAMES)
    counts = {
        "n_humpback": 0,
        "n_orca": 0,
        "n_ship": 0,
        "n_background": 0,
        "n_skipped": 0,
        "n_positive_selected": 0,
        "n_positive_selection_skipped": 0,
    }
    if not all_rows:
        return counts

    process_rows = [
        row
        for row in all_rows
        if _is_positive_row(row)
        or _is_negative_row(row)
        or bool(row.get("positive_extract_filename", "").strip())
    ]
    if not process_rows:
        return counts

    by_file: dict[str, list[dict[str, str]]] = {}
    for row in process_rows:
        by_file.setdefault(row.get("filename", ""), []).append(row)

    for source_filename, file_rows in by_file.items():
        source_path = audio_folder / source_filename
        needs_audio = any(
            _is_positive_row(row) or _is_negative_row(row) for row in file_rows
        )
        audio: np.ndarray | None = None
        sr: int | None = None
        if needs_audio:
            if not source_path.is_file():
                logger.warning("Source audio not found: %s", source_path)
                continue
            audio, sr = decode_audio(source_path)

        stored_records = _load_window_records(
            diagnostics_path, filename=source_filename
        )

        for row in file_rows:
            row_start_sec = float(row.get("start_sec", 0))
            row_end_sec = float(row.get("end_sec", 0))
            positive_labels = _positive_labels_for_row(row)
            negative_labels = _negative_labels_for_row(row)
            old_positive_filename = row.get("positive_extract_filename", "").strip()

            if positive_labels:
                selection = (
                    _select_positive_window(
                        row_start_sec=row_start_sec,
                        row_end_sec=row_end_sec,
                        window_size_seconds=window_size_seconds,
                        window_records=stored_records,
                        smoothing_window=positive_selection_smoothing_window,
                        min_score=positive_selection_min_score,
                        extend_min_score=positive_selection_extend_min_score,
                        score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
                    )
                    if stored_records is not None
                    else None
                )
                if (selection is None or not selection.offsets) and (
                    fallback_pipeline is not None
                    and fallback_model is not None
                    and audio is not None
                    and sr is not None
                ):
                    row_start_sample = min(int(row_start_sec * sr), len(audio))
                    row_end_sample = min(int(row_end_sec * sr), len(audio))
                    row_segment = audio[row_start_sample:row_end_sample]
                    fallback_records = _score_segment_windows(
                        row_segment,
                        source_sr=sr,
                        row_start_sec=row_start_sec,
                        pipeline=fallback_pipeline,
                        model=fallback_model,
                        window_size_seconds=window_size_seconds,
                        target_sample_rate=fallback_target_sample_rate,
                        input_format=fallback_input_format,
                        feature_config=fallback_feature_config,
                    )
                    selection = _select_positive_window(
                        row_start_sec=row_start_sec,
                        row_end_sec=row_end_sec,
                        window_size_seconds=window_size_seconds,
                        window_records=fallback_records,
                        smoothing_window=positive_selection_smoothing_window,
                        min_score=positive_selection_min_score,
                        extend_min_score=positive_selection_extend_min_score,
                        score_source=POSITIVE_SELECTION_SCORE_SOURCE_FALLBACK,
                    )
                if selection is None:
                    selection = PositiveSelectionResult(
                        score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
                        decision="skip",
                        offsets=[],
                        raw_scores=[],
                        smoothed_scores=[],
                        start_sec=None,
                        end_sec=None,
                        peak_score=None,
                    )

                if (
                    selection.decision == "positive"
                    and selection.start_sec is not None
                    and selection.end_sec is not None
                    and audio is not None
                    and sr is not None
                ):
                    clip_name, date_folder = _resolve_local_clip_name(
                        source_filename,
                        selection.start_sec,
                        selection.end_sec,
                    )
                    if old_positive_filename and old_positive_filename != clip_name:
                        _delete_stale_positive_outputs(
                            positive_output_path,
                            clip_name=old_positive_filename,
                            labels=["humpback", "orca"],
                        )
                    selected_start = min(int(selection.start_sec * sr), len(audio))
                    selected_end = min(int(selection.end_sec * sr), len(audio))
                    selected_segment = audio[selected_start:selected_end]
                    if len(selected_segment) == 0:
                        selection = PositiveSelectionResult(
                            score_source=selection.score_source,
                            decision="skip",
                            offsets=selection.offsets,
                            raw_scores=selection.raw_scores,
                            smoothed_scores=selection.smoothed_scores,
                            start_sec=selection.start_sec,
                            end_sec=selection.end_sec,
                            peak_score=selection.peak_score,
                        )
                        _delete_stale_positive_outputs(
                            positive_output_path,
                            clip_name=clip_name,
                            labels=["humpback", "orca"],
                        )
                        row.update(
                            _selection_result_to_row_update(
                                selection,
                                positive_extract_filename=None,
                            )
                        )
                        counts["n_positive_selection_skipped"] += 1
                    else:
                        for label_name in positive_labels:
                            out_dir = positive_output_path / label_name / date_folder
                            out_path = out_dir / clip_name
                            if out_path.exists():
                                counts["n_skipped"] += 1
                                continue
                            write_flac_file(selected_segment, sr, out_path)
                            counts[f"n_{label_name}"] += 1
                        row.update(
                            _selection_result_to_row_update(
                                selection,
                                positive_extract_filename=clip_name,
                            )
                        )
                        counts["n_positive_selected"] += 1
                else:
                    if old_positive_filename:
                        _delete_stale_positive_outputs(
                            positive_output_path,
                            clip_name=old_positive_filename,
                            labels=["humpback", "orca"],
                        )
                    row.update(
                        _selection_result_to_row_update(
                            selection,
                            positive_extract_filename=None,
                        )
                    )
                    counts["n_positive_selection_skipped"] += 1
            else:
                if old_positive_filename:
                    _delete_stale_positive_outputs(
                        positive_output_path,
                        clip_name=old_positive_filename,
                        labels=["humpback", "orca"],
                    )
                row.update(_blank_positive_selection_fields())

            if negative_labels and audio is not None and sr is not None:
                start_sample = min(int(row_start_sec * sr), len(audio))
                end_sample = min(int(row_end_sec * sr), len(audio))
                segment = audio[start_sample:end_sample]
                if len(segment) == 0:
                    continue
                clip_name, date_folder = _resolve_local_clip_name(
                    source_filename,
                    row_start_sec,
                    row_end_sec,
                )
                for label_name in negative_labels:
                    out_dir = negative_output_path / label_name / date_folder
                    out_path = out_dir / clip_name
                    if out_path.exists():
                        counts["n_skipped"] += 1
                        continue
                    write_flac_file(segment, sr, out_path)
                    counts[f"n_{label_name}"] += 1

    _write_tsv_rows(tsv_path, fieldnames, all_rows)
    return counts


def _fetch_audio_range(
    provider: ArchiveProvider,
    abs_start_ts: float,
    abs_end_ts: float,
    target_sr: int,
) -> np.ndarray | None:
    """Fetch and decode provider audio covering a time range."""
    timeline = provider.build_timeline(abs_start_ts, abs_end_ts)
    all_audio: list[np.ndarray] = []
    for segment in timeline:
        try:
            seg_bytes = provider.fetch_segment(segment.key)
            audio = provider.decode_segment(seg_bytes, target_sr)
        except Exception:
            continue
        if len(audio) == 0:
            continue
        decoded_end_ts = segment.start_ts + (len(audio) / target_sr)
        start_ts = max(segment.start_ts, abs_start_ts)
        end_ts = min(decoded_end_ts, abs_end_ts)
        if end_ts <= start_ts:
            continue
        start_sample = max(0, int(round((start_ts - segment.start_ts) * target_sr)))
        end_sample = min(
            len(audio), int(round((end_ts - segment.start_ts) * target_sr))
        )
        if end_sample <= start_sample:
            continue
        all_audio.append(audio[start_sample:end_sample])
    if not all_audio:
        return None
    return np.concatenate(all_audio)


def _format_compact_ts(dt: datetime) -> str:
    """Format datetime as YYYYMMDDTHHMMSSz (compact, no microseconds)."""
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _parse_compact_range_filename(
    filename: str,
) -> tuple[datetime, datetime] | None:
    """Parse compact UTC range filename into (start, end) datetimes."""
    base = _strip_known_audio_extension(filename)
    parts = base.split("_")
    if len(parts) != 2:
        return None
    try:
        start = datetime.strptime(parts[0], _COMPACT_TS_FORMAT).replace(
            tzinfo=timezone.utc
        )
        end = datetime.strptime(parts[1], _COMPACT_TS_FORMAT).replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    if end <= start:
        return None
    return start, end


def _resolve_hydrophone_clip_name(
    source_filename: str,
    start_sec: float,
    end_sec: float,
) -> tuple[str, datetime, datetime] | None:
    """Build a compact UTC filename for a hydrophone clip."""
    recording_ts = parse_recording_timestamp(source_filename)
    if recording_ts is None:
        return None
    abs_start = recording_ts + timedelta(seconds=start_sec)
    abs_end = recording_ts + timedelta(seconds=end_sec)
    clip_name = (
        f"{_format_compact_ts(abs_start)}_{_format_compact_ts(abs_end)}"
        f"{_OUTPUT_AUDIO_EXTENSION}"
    )
    return clip_name, abs_start, abs_end


def _snap_range_for_window(
    start_sec: float,
    end_sec: float,
    window_size_seconds: float,
) -> tuple[float, float]:
    """Snap bounds outward to window-size multiples."""
    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be > 0")
    if end_sec <= start_sec:
        return start_sec, end_sec
    start_sec = math.floor(start_sec / window_size_seconds) * window_size_seconds
    end_sec = math.ceil(end_sec / window_size_seconds) * window_size_seconds
    if end_sec <= start_sec:
        end_sec = start_sec + window_size_seconds
    return float(start_sec), float(end_sec)


def extract_hydrophone_labeled_samples(
    tsv_path: str | Path,
    provider: ArchiveProvider,
    positive_output_path: str | Path,
    negative_output_path: str | Path,
    target_sample_rate: int = 32000,
    window_size_seconds: float = 5.0,
    stream_start_timestamp: float | None = None,
    stream_end_timestamp: float | None = None,
    window_diagnostics_path: str | Path | None = None,
    positive_selection_smoothing_window: int = DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW,
    positive_selection_min_score: float = DEFAULT_POSITIVE_SELECTION_MIN_SCORE,
    positive_selection_extend_min_score: float = (
        DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE
    ),
    fallback_pipeline: Any | None = None,
    fallback_model: EmbeddingModel | None = None,
    fallback_input_format: str = "spectrogram",
    fallback_feature_config: dict | None = None,
) -> dict:
    """Extract labeled audio segments from a hydrophone detection TSV.

    Similar to extract_labeled_samples but reconstructs audio from HLS segments
    instead of reading from a local audio folder.

    Returns a summary dict with counts per label.
    """
    tsv_path = Path(tsv_path)
    positive_output_path = Path(positive_output_path)
    negative_output_path = Path(negative_output_path)
    diagnostics_path = (
        Path(window_diagnostics_path) if window_diagnostics_path is not None else None
    )
    _validate_positive_selection_config(
        positive_selection_smoothing_window,
        positive_selection_min_score,
        positive_selection_extend_min_score,
    )

    fieldnames, all_rows = _read_tsv_rows(tsv_path)
    _ensure_fieldnames(fieldnames, POSITIVE_SELECTION_FIELDNAMES)
    counts = {
        "n_humpback": 0,
        "n_orca": 0,
        "n_ship": 0,
        "n_background": 0,
        "n_skipped": 0,
        "n_positive_selected": 0,
        "n_positive_selection_skipped": 0,
    }
    if not all_rows:
        return counts

    process_rows = [
        row
        for row in all_rows
        if _is_positive_row(row)
        or _is_negative_row(row)
        or bool(row.get("positive_extract_filename", "").strip())
    ]
    if not process_rows:
        return counts

    by_file: dict[str, list[dict[str, str]]] = {}
    for row in process_rows:
        fn = row.get("filename", "")
        by_file.setdefault(fn, []).append(row)
    use_stream_resolver = (
        stream_start_timestamp is not None and stream_end_timestamp is not None
    )
    stream_timeline = None
    processing_start_ts: float | None = None
    stream_start_ts_value: float | None = None
    stream_end_ts_value: float | None = None

    if use_stream_resolver:
        assert stream_start_timestamp is not None
        assert stream_end_timestamp is not None

        stream_start_ts_value = float(stream_start_timestamp)
        stream_end_ts_value = float(stream_end_timestamp)

        try:
            stream_timeline = build_stream_timeline(
                provider=provider,
                stream_start_ts=stream_start_ts_value,
                stream_end_ts=stream_end_ts_value,
            )
            processing_start_ts = max(
                stream_start_ts_value, stream_timeline[0].start_ts
            )
        except Exception as exc:
            logger.warning(
                "Hydrophone extraction timeline unavailable for %s [%.1f, %.1f]: %s",
                provider.source_id,
                stream_start_ts_value,
                stream_end_ts_value,
                exc,
            )
            stream_timeline = []

    for source_filename, file_rows in by_file.items():
        recording_ts = parse_recording_timestamp(source_filename)
        stored_records = _load_window_records(
            diagnostics_path, filename=source_filename
        )

        for row in file_rows:
            start_sec = float(row.get("start_sec", 0))
            end_sec = float(row.get("end_sec", 0))
            positive_labels = _positive_labels_for_row(row)
            negative_labels = _negative_labels_for_row(row)
            old_positive_filename = row.get("positive_extract_filename", "").strip()

            if recording_ts is None:
                logger.warning(
                    "Skipping hydrophone extraction row with non-timestamped filename: %s",
                    source_filename,
                )
                if old_positive_filename:
                    _delete_stale_positive_outputs(
                        positive_output_path,
                        clip_name=old_positive_filename,
                        labels=["humpback", "orca"],
                        source_id=provider.source_id,
                    )
                if positive_labels:
                    row.update(
                        _selection_result_to_row_update(
                            PositiveSelectionResult(
                                score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
                                decision="skip",
                                offsets=[],
                                raw_scores=[],
                                smoothed_scores=[],
                                start_sec=None,
                                end_sec=None,
                                peak_score=None,
                            ),
                            positive_extract_filename=None,
                        )
                    )
                    counts["n_positive_selection_skipped"] += 1
                else:
                    row.update(_blank_positive_selection_fields())
                if negative_labels:
                    counts["n_skipped"] += len(negative_labels)
                continue

            detection_filename = row.get("detection_filename", "").strip() or None
            extract_filename = row.get("extract_filename", "").strip() or None
            parsed_detection_range = (
                _parse_compact_range_filename(detection_filename)
                if detection_filename
                else None
            )
            parsed_extract_range = (
                _parse_compact_range_filename(extract_filename)
                if extract_filename
                else None
            )

            # Resolve bounds with precedence:
            # detection_filename -> extract_filename -> snapped start/end fallback.
            if parsed_detection_range is not None:
                abs_start, abs_end = parsed_detection_range
                start_sec = (abs_start - recording_ts).total_seconds()
                end_sec = (abs_end - recording_ts).total_seconds()
            elif parsed_extract_range is not None:
                abs_start, abs_end = parsed_extract_range
                start_sec = (abs_start - recording_ts).total_seconds()
                end_sec = (abs_end - recording_ts).total_seconds()
                detection_filename = (
                    f"{_format_compact_ts(abs_start)}_{_format_compact_ts(abs_end)}"
                    f"{_OUTPUT_AUDIO_EXTENSION}"
                )
            else:
                start_sec, end_sec = _snap_range_for_window(
                    start_sec, end_sec, window_size_seconds
                )
                abs_start = recording_ts + timedelta(seconds=start_sec)
                abs_end = recording_ts + timedelta(seconds=end_sec)
                detection_filename = (
                    f"{_format_compact_ts(abs_start)}_{_format_compact_ts(abs_end)}"
                    f"{_OUTPUT_AUDIO_EXTENSION}"
                )

            resolved_start_sec = start_sec
            resolved_end_sec = end_sec

            if positive_labels:
                selection = (
                    _select_positive_window(
                        row_start_sec=resolved_start_sec,
                        row_end_sec=resolved_end_sec,
                        window_size_seconds=window_size_seconds,
                        window_records=stored_records,
                        smoothing_window=positive_selection_smoothing_window,
                        min_score=positive_selection_min_score,
                        extend_min_score=positive_selection_extend_min_score,
                        score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
                    )
                    if stored_records is not None
                    else None
                )
                if (selection is None or not selection.offsets) and (
                    fallback_pipeline is not None and fallback_model is not None
                ):
                    row_duration = max(0.0, resolved_end_sec - resolved_start_sec)
                    fallback_segment: np.ndarray | None = None
                    if use_stream_resolver:
                        if stream_timeline:
                            assert stream_start_ts_value is not None
                            assert stream_end_ts_value is not None
                            try:
                                fallback_segment = resolve_audio_slice(
                                    provider=provider,
                                    stream_start_ts=stream_start_ts_value,
                                    stream_end_ts=stream_end_ts_value,
                                    filename=source_filename,
                                    row_start_sec=resolved_start_sec,
                                    duration_sec=row_duration,
                                    target_sr=target_sample_rate,
                                    legacy_anchor_start_ts=stream_start_ts_value,
                                    timeline=stream_timeline,
                                    processing_start_ts=processing_start_ts,
                                )
                            except Exception:
                                fallback_segment = None
                    else:
                        if recording_ts is not None:
                            abs_start_ts = (
                                recording_ts + timedelta(seconds=resolved_start_sec)
                            ).timestamp()
                            abs_end_ts = (
                                recording_ts + timedelta(seconds=resolved_end_sec)
                            ).timestamp()
                            fallback_segment = _fetch_audio_range(
                                provider,
                                abs_start_ts,
                                abs_end_ts,
                                target_sample_rate,
                            )
                    if fallback_segment is not None and len(fallback_segment) > 0:
                        fallback_records = _score_segment_windows(
                            fallback_segment,
                            source_sr=target_sample_rate,
                            row_start_sec=resolved_start_sec,
                            pipeline=fallback_pipeline,
                            model=fallback_model,
                            window_size_seconds=window_size_seconds,
                            target_sample_rate=target_sample_rate,
                            input_format=fallback_input_format,
                            feature_config=fallback_feature_config,
                        )
                        selection = _select_positive_window(
                            row_start_sec=resolved_start_sec,
                            row_end_sec=resolved_end_sec,
                            window_size_seconds=window_size_seconds,
                            window_records=fallback_records,
                            smoothing_window=positive_selection_smoothing_window,
                            min_score=positive_selection_min_score,
                            extend_min_score=positive_selection_extend_min_score,
                            score_source=POSITIVE_SELECTION_SCORE_SOURCE_FALLBACK,
                        )
                if selection is None:
                    selection = PositiveSelectionResult(
                        score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
                        decision="skip",
                        offsets=[],
                        raw_scores=[],
                        smoothed_scores=[],
                        start_sec=None,
                        end_sec=None,
                        peak_score=None,
                    )

                positive_written = False
                if (
                    selection.decision == "positive"
                    and selection.start_sec is not None
                    and selection.end_sec is not None
                ):
                    resolved_name = _resolve_hydrophone_clip_name(
                        source_filename,
                        selection.start_sec,
                        selection.end_sec,
                    )
                    if resolved_name is None:
                        selection = PositiveSelectionResult(
                            score_source=selection.score_source,
                            decision="skip",
                            offsets=selection.offsets,
                            raw_scores=selection.raw_scores,
                            smoothed_scores=selection.smoothed_scores,
                            start_sec=selection.start_sec,
                            end_sec=selection.end_sec,
                            peak_score=selection.peak_score,
                        )
                    else:
                        clip_name, _, _ = resolved_name
                        if old_positive_filename and old_positive_filename != clip_name:
                            _delete_stale_positive_outputs(
                                positive_output_path,
                                clip_name=old_positive_filename,
                                labels=["humpback", "orca"],
                                source_id=provider.source_id,
                            )
                        duration = selection.end_sec - selection.start_sec
                        segment: np.ndarray | None = None
                        if use_stream_resolver:
                            if stream_timeline:
                                assert stream_start_ts_value is not None
                                assert stream_end_ts_value is not None
                                try:
                                    segment = resolve_audio_slice(
                                        provider=provider,
                                        stream_start_ts=stream_start_ts_value,
                                        stream_end_ts=stream_end_ts_value,
                                        filename=source_filename,
                                        row_start_sec=selection.start_sec,
                                        duration_sec=duration,
                                        target_sr=target_sample_rate,
                                        legacy_anchor_start_ts=stream_start_ts_value,
                                        timeline=stream_timeline,
                                        processing_start_ts=processing_start_ts,
                                    )
                                except Exception:
                                    segment = None
                        else:
                            if recording_ts is not None:
                                segment = _fetch_audio_range(
                                    provider,
                                    (
                                        recording_ts
                                        + timedelta(seconds=selection.start_sec)
                                    ).timestamp(),
                                    (
                                        recording_ts
                                        + timedelta(seconds=selection.end_sec)
                                    ).timestamp(),
                                    target_sample_rate,
                                )

                        if segment is None or len(segment) == 0:
                            selection = PositiveSelectionResult(
                                score_source=selection.score_source,
                                decision="skip",
                                offsets=selection.offsets,
                                raw_scores=selection.raw_scores,
                                smoothed_scores=selection.smoothed_scores,
                                start_sec=selection.start_sec,
                                end_sec=selection.end_sec,
                                peak_score=selection.peak_score,
                            )
                        else:
                            for label_name in positive_labels:
                                out_path = _resolve_positive_output_path(
                                    positive_output_path,
                                    label=label_name,
                                    clip_name=clip_name,
                                    source_id=provider.source_id,
                                )
                                if out_path.exists():
                                    counts["n_skipped"] += 1
                                    continue
                                write_flac_file(segment, target_sample_rate, out_path)
                                counts[f"n_{label_name}"] += 1
                            row.update(
                                _selection_result_to_row_update(
                                    selection,
                                    positive_extract_filename=clip_name,
                                )
                            )
                            counts["n_positive_selected"] += 1
                            positive_written = True
                if not positive_written:
                    if old_positive_filename:
                        _delete_stale_positive_outputs(
                            positive_output_path,
                            clip_name=old_positive_filename,
                            labels=["humpback", "orca"],
                            source_id=provider.source_id,
                        )
                    row.update(
                        _selection_result_to_row_update(
                            selection,
                            positive_extract_filename=None,
                        )
                    )
                    counts["n_positive_selection_skipped"] += 1
            else:
                if old_positive_filename:
                    _delete_stale_positive_outputs(
                        positive_output_path,
                        clip_name=old_positive_filename,
                        labels=["humpback", "orca"],
                        source_id=provider.source_id,
                    )
                row.update(_blank_positive_selection_fields())

            if abs_end <= abs_start:
                counts["n_skipped"] += 1
                continue

            abs_start_ts = abs_start.timestamp()
            abs_end_ts = abs_end.timestamp()

            # Output filename and folder
            assert detection_filename is not None
            clip_name = _with_output_audio_extension(detection_filename)
            date_folder = _date_folder(abs_start)

            # Route to label-specific folders (species/category before hydrophone_id)
            labels_to_write: list[tuple[Path, str]] = []
            if row.get("ship", "").strip() == "1":
                out_dir = (
                    negative_output_path / "ship" / provider.source_id / date_folder
                )
                labels_to_write.append((out_dir, "ship"))
            if row.get("background", "").strip() == "1":
                out_dir = (
                    negative_output_path
                    / "background"
                    / provider.source_id
                    / date_folder
                )
                labels_to_write.append((out_dir, "background"))

            pending_writes: list[tuple[Path, str]] = []
            for out_dir, label_name in labels_to_write:
                out_path = out_dir / clip_name
                if out_path.exists():
                    counts["n_skipped"] += 1
                    continue
                pending_writes.append((out_path, label_name))

            if not pending_writes:
                continue

            duration = (abs_end - abs_start).total_seconds()
            segment: np.ndarray | None = None

            if use_stream_resolver:
                if not stream_timeline:
                    counts["n_skipped"] += len(pending_writes)
                    continue
                assert stream_start_ts_value is not None
                assert stream_end_ts_value is not None
                try:
                    segment = resolve_audio_slice(
                        provider=provider,
                        stream_start_ts=stream_start_ts_value,
                        stream_end_ts=stream_end_ts_value,
                        filename=source_filename,
                        row_start_sec=start_sec,
                        duration_sec=duration,
                        target_sr=target_sample_rate,
                        legacy_anchor_start_ts=stream_start_ts_value,
                        timeline=stream_timeline,
                        processing_start_ts=processing_start_ts,
                    )
                except Exception as exc:
                    logger.warning(
                        "No hydrophone audio for %s (%.1f-%.1f): %s",
                        source_filename,
                        start_sec,
                        end_sec,
                        exc,
                    )
            else:
                # Backward-compatible fallback for direct callers that do not
                # provide stream bounds.
                combined = _fetch_audio_range(
                    provider, abs_start_ts, abs_end_ts, target_sample_rate
                )
                if combined is not None:
                    end_sample = min(int(duration * target_sample_rate), len(combined))
                    segment = combined[:end_sample]

            if segment is None or len(segment) == 0:
                counts["n_skipped"] += len(pending_writes)
                continue

            for out_path, label_name in pending_writes:
                write_flac_file(segment, target_sample_rate, out_path)
                counts[f"n_{label_name}"] += 1

    _write_tsv_rows(tsv_path, fieldnames, all_rows)
    return counts
