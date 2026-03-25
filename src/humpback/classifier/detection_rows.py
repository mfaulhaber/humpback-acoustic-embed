"""Detection row normalization and Parquet-backed row-store helpers."""

from __future__ import annotations

import csv
import hashlib
import io
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
import pyarrow as pa
import pyarrow.parquet as pq

from humpback.classifier.detector import read_window_diagnostics_table

logger = logging.getLogger(__name__)

DETECTION_ROW_STORE_FILENAME = "detection_rows.parquet"

DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW = 3
DEFAULT_POSITIVE_SELECTION_MIN_SCORE = 0.70
DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE = 0.60

POSITIVE_SELECTION_SCORE_SOURCE_STORED = "stored_diagnostics"
POSITIVE_SELECTION_SCORE_SOURCE_FALLBACK = "rescored_fallback"
POSITIVE_SELECTION_SCORE_SOURCE_MANUAL = "manual_override"
POSITIVE_SELECTION_SCORE_SOURCE_FALLBACK_CLIP = "clip_bounds_fallback"

POSITIVE_SELECTION_ORIGIN_AUTO = "auto_selection"
POSITIVE_SELECTION_ORIGIN_MANUAL = "manual_override"
POSITIVE_SELECTION_ORIGIN_CLIP = "clip_bounds_fallback"

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

AUTO_POSITIVE_SELECTION_FIELDNAMES = [
    "auto_positive_selection_score_source",
    "auto_positive_selection_decision",
    "auto_positive_selection_offsets",
    "auto_positive_selection_raw_scores",
    "auto_positive_selection_smoothed_scores",
    "auto_positive_selection_start_sec",
    "auto_positive_selection_end_sec",
    "auto_positive_selection_peak_score",
]

MANUAL_POSITIVE_SELECTION_FIELDNAMES = [
    "manual_positive_selection_start_sec",
    "manual_positive_selection_end_sec",
]

LABEL_FIELDNAMES = ["humpback", "orca", "ship", "background"]

ROW_STORE_FIELDNAMES = [
    "row_id",
    "filename",
    "start_sec",
    "end_sec",
    "avg_confidence",
    "peak_confidence",
    "n_windows",
    "raw_start_sec",
    "raw_end_sec",
    "merged_event_count",
    "detection_filename",
    "extract_filename",
    "hydrophone_name",
    "humpback",
    "orca",
    "ship",
    "background",
    *AUTO_POSITIVE_SELECTION_FIELDNAMES,
    *MANUAL_POSITIVE_SELECTION_FIELDNAMES,
    "positive_selection_origin",
    *POSITIVE_SELECTION_FIELDNAMES,
]

ROW_STORE_SCHEMA = pa.schema([(field, pa.string()) for field in ROW_STORE_FIELDNAMES])

_TS_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(?:\.(\d+))?Z")
_COMPACT_TS_FORMAT = "%Y%m%dT%H%M%SZ"
_KNOWN_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".aif", ".aiff"}


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
    """Extract a recording start timestamp from a filename."""
    match = _TS_PATTERN.search(filename)
    if match is None:
        return None
    year, month, day, hour, minute, second = (
        int(group) for group in match.groups()[:6]
    )
    frac_str = match.group(7)
    microsecond = 0
    if frac_str:
        frac_str = frac_str[:6].ljust(6, "0")
        microsecond = int(frac_str)
    return datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
        tzinfo=timezone.utc,
    )


def strip_known_audio_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in _KNOWN_AUDIO_EXTENSIONS:
        return filename[: -len(suffix)]
    return filename


def safe_float(value: str | float | int | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def safe_int(value: str | int | None, default: int | None = None) -> int | None:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def safe_optional_float(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_float_list(value: str | list[float] | None) -> list[float] | None:
    if value in (None, ""):
        return None
    parsed: Any
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
    else:
        parsed = value
    if not isinstance(parsed, list):
        return None
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError):
        return None


def parse_label(value: str | int | None) -> int | None:
    if value in (None, ""):
        return None
    if value == 0 or value == "0":
        return 0
    if value == 1 or value == "1":
        return 1
    return None


def is_positive_row(row: dict[str, str]) -> bool:
    return row.get("humpback", "").strip() == "1" or row.get("orca", "").strip() == "1"


def is_negative_row(row: dict[str, str]) -> bool:
    return (
        row.get("ship", "").strip() == "1" or row.get("background", "").strip() == "1"
    )


def positive_labels_for_row(row: dict[str, str]) -> list[str]:
    labels: list[str] = []
    if row.get("humpback", "").strip() == "1":
        labels.append("humpback")
    if row.get("orca", "").strip() == "1":
        labels.append("orca")
    return labels


def negative_labels_for_row(row: dict[str, str]) -> list[str]:
    labels: list[str] = []
    if row.get("ship", "").strip() == "1":
        labels.append("ship")
    if row.get("background", "").strip() == "1":
        labels.append("background")
    return labels


def serialize_json_list(values: list[float]) -> str:
    rounded = [round(float(value), 6) for value in values]
    return json.dumps(rounded, separators=(",", ":"))


def format_optional_float(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else ""


def format_optional_int(value: int | None) -> str:
    return str(value) if value is not None else ""


def blank_positive_selection_fields() -> dict[str, str]:
    return {field: "" for field in POSITIVE_SELECTION_FIELDNAMES}


def blank_auto_positive_selection_fields() -> dict[str, str]:
    return {field: "" for field in AUTO_POSITIVE_SELECTION_FIELDNAMES}


def parse_compact_range_filename(
    filename: str | None,
) -> tuple[datetime, datetime] | None:
    if not filename:
        return None
    base = strip_known_audio_extension(filename)
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


def snap_bounds_to_window(
    start_sec: float,
    end_sec: float,
    window_size_seconds: float,
) -> tuple[float, float]:
    if end_sec <= start_sec or window_size_seconds <= 0:
        return start_sec, end_sec
    snap_start = math.floor(start_sec / window_size_seconds) * window_size_seconds
    snap_end = math.ceil(end_sec / window_size_seconds) * window_size_seconds
    if snap_end <= snap_start:
        snap_end = snap_start + window_size_seconds
    return float(snap_start), float(snap_end)


def derive_detection_filename(
    filename: str,
    start_sec: float,
    end_sec: float,
) -> str | None:
    if end_sec <= start_sec:
        return None
    base = strip_known_audio_extension(filename)
    try:
        chunk_start = datetime.strptime(base, _COMPACT_TS_FORMAT).replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    abs_start = chunk_start + timedelta(seconds=start_sec)
    abs_end = chunk_start + timedelta(seconds=end_sec)
    return (
        f"{abs_start.strftime(_COMPACT_TS_FORMAT)}"
        f"_{abs_end.strftime(_COMPACT_TS_FORMAT)}.flac"
    )


def normalize_detection_row(
    row: dict[str, str],
    *,
    is_hydrophone: bool,
    window_size_seconds: float,
) -> dict[str, Any]:
    filename = (row.get("filename", "") or "").strip()
    start_sec = safe_float(row.get("start_sec"), 0.0)
    end_sec = safe_float(row.get("end_sec"), 0.0)
    avg_confidence = safe_float(row.get("avg_confidence"), 0.0)
    peak_confidence = safe_float(row.get("peak_confidence"), 0.0)
    n_windows = safe_int(row.get("n_windows"), None)

    detection_filename = (row.get("detection_filename", "") or "").strip() or None
    extract_filename = (row.get("extract_filename", "") or "").strip() or None

    if detection_filename is None:
        if parse_compact_range_filename(extract_filename) is not None:
            detection_filename = extract_filename
        else:
            snap_start, snap_end = snap_bounds_to_window(
                start_sec, end_sec, window_size_seconds
            )
            detection_filename = derive_detection_filename(
                filename, snap_start, snap_end
            )

    if extract_filename is None and is_hydrophone:
        extract_filename = detection_filename

    raw_start_sec = safe_float(row.get("raw_start_sec"), start_sec)
    raw_end_sec = safe_float(row.get("raw_end_sec"), end_sec)
    merged_event_count = safe_int(row.get("merged_event_count"), 1)
    positive_extract_filename = row.get("positive_extract_filename", "").strip() or None

    return {
        "row_id": (row.get("row_id", "") or "").strip() or None,
        "filename": filename,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "avg_confidence": avg_confidence,
        "peak_confidence": peak_confidence,
        "n_windows": n_windows,
        "detection_filename": detection_filename,
        "extract_filename": extract_filename,
        "hydrophone_name": (row.get("hydrophone_name", "").strip() or None),
        "raw_start_sec": raw_start_sec,
        "raw_end_sec": raw_end_sec,
        "merged_event_count": merged_event_count,
        "humpback": parse_label(row.get("humpback")),
        "orca": parse_label(row.get("orca")),
        "ship": parse_label(row.get("ship")),
        "background": parse_label(row.get("background")),
        "auto_positive_selection_score_source": (
            row.get("auto_positive_selection_score_source", "").strip() or None
        ),
        "auto_positive_selection_decision": (
            row.get("auto_positive_selection_decision", "").strip() or None
        ),
        "auto_positive_selection_offsets": safe_float_list(
            row.get("auto_positive_selection_offsets")
        ),
        "auto_positive_selection_raw_scores": safe_float_list(
            row.get("auto_positive_selection_raw_scores")
        ),
        "auto_positive_selection_smoothed_scores": safe_float_list(
            row.get("auto_positive_selection_smoothed_scores")
        ),
        "auto_positive_selection_start_sec": safe_optional_float(
            row.get("auto_positive_selection_start_sec")
        ),
        "auto_positive_selection_end_sec": safe_optional_float(
            row.get("auto_positive_selection_end_sec")
        ),
        "auto_positive_selection_peak_score": safe_optional_float(
            row.get("auto_positive_selection_peak_score")
        ),
        "manual_positive_selection_start_sec": safe_optional_float(
            row.get("manual_positive_selection_start_sec")
        ),
        "manual_positive_selection_end_sec": safe_optional_float(
            row.get("manual_positive_selection_end_sec")
        ),
        "positive_selection_origin": (
            row.get("positive_selection_origin", "").strip() or None
        ),
        "positive_selection_score_source": (
            row.get("positive_selection_score_source", "").strip() or None
        ),
        "positive_selection_decision": (
            row.get("positive_selection_decision", "").strip() or None
        ),
        "positive_selection_offsets": safe_float_list(
            row.get("positive_selection_offsets")
        ),
        "positive_selection_raw_scores": safe_float_list(
            row.get("positive_selection_raw_scores")
        ),
        "positive_selection_smoothed_scores": safe_float_list(
            row.get("positive_selection_smoothed_scores")
        ),
        "positive_selection_start_sec": safe_optional_float(
            row.get("positive_selection_start_sec")
        ),
        "positive_selection_end_sec": safe_optional_float(
            row.get("positive_selection_end_sec")
        ),
        "positive_selection_peak_score": safe_optional_float(
            row.get("positive_selection_peak_score")
        ),
        "positive_extract_filename": positive_extract_filename,
    }


def build_detection_row_id(row: dict[str, Any]) -> str:
    detection_filename = row.get("detection_filename") or ""
    material = (
        f"{row.get('filename', '')}|{detection_filename}|"
        f"{safe_float(row.get('start_sec'), 0.0):.6f}|"
        f"{safe_float(row.get('end_sec'), 0.0):.6f}"
    )
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def resolve_clip_bounds(
    row: dict[str, Any],
    *,
    window_size_seconds: float,
) -> tuple[float, float]:
    filename = str(row.get("filename", "") or "")
    start_sec = safe_float(row.get("start_sec"), 0.0)
    end_sec = safe_float(row.get("end_sec"), 0.0)
    detection_filename = str(row.get("detection_filename", "") or "") or None
    extract_filename = str(row.get("extract_filename", "") or "") or None
    recording_ts = parse_recording_timestamp(filename)

    if recording_ts is not None:
        for candidate in (detection_filename, extract_filename):
            parsed = parse_compact_range_filename(candidate)
            if parsed is not None:
                return (
                    float((parsed[0] - recording_ts).total_seconds()),
                    float((parsed[1] - recording_ts).total_seconds()),
                )

    return snap_bounds_to_window(start_sec, end_sec, window_size_seconds)


def selection_result_to_row_update(
    result: PositiveSelectionResult,
    *,
    positive_extract_filename: str | None,
) -> dict[str, str]:
    return {
        "positive_selection_score_source": result.score_source,
        "positive_selection_decision": result.decision,
        "positive_selection_offsets": serialize_json_list(result.offsets),
        "positive_selection_raw_scores": serialize_json_list(result.raw_scores),
        "positive_selection_smoothed_scores": serialize_json_list(
            result.smoothed_scores
        ),
        "positive_selection_start_sec": format_optional_float(result.start_sec),
        "positive_selection_end_sec": format_optional_float(result.end_sec),
        "positive_selection_peak_score": format_optional_float(result.peak_score),
        "positive_extract_filename": positive_extract_filename or "",
    }


def prefixed_selection_result_to_row_update(
    result: PositiveSelectionResult,
    *,
    prefix: str,
) -> dict[str, str]:
    return {
        f"{prefix}score_source": result.score_source,
        f"{prefix}decision": result.decision,
        f"{prefix}offsets": serialize_json_list(result.offsets),
        f"{prefix}raw_scores": serialize_json_list(result.raw_scores),
        f"{prefix}smoothed_scores": serialize_json_list(result.smoothed_scores),
        f"{prefix}start_sec": format_optional_float(result.start_sec),
        f"{prefix}end_sec": format_optional_float(result.end_sec),
        f"{prefix}peak_score": format_optional_float(result.peak_score),
    }


def smooth_scores(scores: list[float], window_size: int) -> list[float]:
    if not scores:
        return []
    if window_size <= 1 or len(scores) == 1:
        return [float(value) for value in scores]
    arr = np.asarray(scores, dtype=np.float32)
    pad = window_size // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return [float(value) for value in smoothed]


def candidate_offset_key(offset_sec: float) -> float:
    return round(float(offset_sec), 6)


def select_positive_window(
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
    smoothed_scores = smooth_scores(raw_scores, smoothing_window)
    best_idx = int(np.argmax(np.asarray(smoothed_scores, dtype=np.float32)))
    peak_score = float(smoothed_scores[best_idx])
    best = candidates[best_idx]
    start_sec = float(best["offset_sec"])
    end_sec = float(best["end_sec"])

    if peak_score >= min_score:
        candidates_by_offset = {
            candidate_offset_key(float(rec["offset_sec"])): (
                rec,
                float(smoothed_scores[idx]),
            )
            for idx, rec in enumerate(candidates)
        }
        while True:
            left_key = candidate_offset_key(start_sec - window_size_seconds)
            right_key = candidate_offset_key(end_sec)
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


def read_tsv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def ensure_fieldnames(fieldnames: list[str], required: list[str]) -> list[str]:
    for field in required:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def read_detection_row_store(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    table = pq.read_table(str(path), schema=ROW_STORE_SCHEMA)
    fieldnames = list(table.column_names)
    rows: list[dict[str, str]] = []
    columns = table.to_pydict()
    for idx in range(table.num_rows):
        row: dict[str, str] = {}
        for field in fieldnames:
            value = columns[field][idx]
            row[field] = value if value is not None else ""
        rows.append(row)
    return fieldnames, rows


def write_detection_row_store(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {field: (row.get(field, "") or None) for field in ROW_STORE_FIELDNAMES}
        )
    table = pa.Table.from_pylist(normalized_rows, schema=ROW_STORE_SCHEMA)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".parquet")
    os.close(fd)
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def append_detection_row_store(
    path: Path,
    new_rows: list[dict[str, str]],
) -> None:
    """Append rows to an existing Parquet row store, creating if needed."""
    existing_rows: list[dict[str, str]] = []
    if path.is_file():
        _, existing_rows = read_detection_row_store(path)
    write_detection_row_store(path, existing_rows + new_rows)


def stream_detection_rows_as_tsv(
    rows: list[dict[str, str]],
    *,
    fieldnames: list[str],
) -> io.StringIO:
    export_fieldnames = [field for field in fieldnames if field != "row_id"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=export_fieldnames, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in export_fieldnames})
    buf.seek(0)
    return buf


def iter_detection_rows_as_tsv(
    rows: list[dict[str, str]],
    *,
    fieldnames: list[str],
):
    export_fieldnames = [field for field in fieldnames if field != "row_id"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=export_fieldnames, delimiter="\t")
    writer.writeheader()
    yield buf.getvalue()
    buf.seek(0)
    buf.truncate(0)
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in export_fieldnames})
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate(0)


def _load_all_window_records(
    diagnostics_path: Path | None,
) -> list[dict[str, Any]] | None:
    if diagnostics_path is None or not diagnostics_path.exists():
        return None
    try:
        table = read_window_diagnostics_table(diagnostics_path)
    except Exception:
        logger.debug(
            "Failed to read diagnostics from %s", diagnostics_path, exc_info=True
        )
        return None
    records: list[dict[str, Any]] = []
    for idx in range(table.num_rows):
        records.append(
            {
                "filename": str(table.column("filename")[idx].as_py()),
                "offset_sec": float(table.column("offset_sec")[idx].as_py()),
                "end_sec": float(table.column("end_sec")[idx].as_py()),
                "confidence": float(table.column("confidence")[idx].as_py()),
            }
        )
    return records


def _existing_selection_to_auto_update(row: dict[str, str]) -> dict[str, str]:
    return {
        "auto_positive_selection_score_source": row.get(
            "positive_selection_score_source", ""
        ),
        "auto_positive_selection_decision": row.get("positive_selection_decision", ""),
        "auto_positive_selection_offsets": row.get("positive_selection_offsets", ""),
        "auto_positive_selection_raw_scores": row.get(
            "positive_selection_raw_scores", ""
        ),
        "auto_positive_selection_smoothed_scores": row.get(
            "positive_selection_smoothed_scores", ""
        ),
        "auto_positive_selection_start_sec": row.get(
            "positive_selection_start_sec", ""
        ),
        "auto_positive_selection_end_sec": row.get("positive_selection_end_sec", ""),
        "auto_positive_selection_peak_score": row.get(
            "positive_selection_peak_score", ""
        ),
    }


def _build_absolute_window_records(
    diagnostics_records: list[dict[str, Any]],
) -> list[dict[str, float]]:
    absolute_records: list[dict[str, float]] = []
    for record in diagnostics_records:
        chunk_ts = parse_recording_timestamp(str(record["filename"]))
        if chunk_ts is None:
            continue
        base_ts = chunk_ts.timestamp()
        absolute_records.append(
            {
                "offset_sec": base_ts + float(record["offset_sec"]),
                "end_sec": base_ts + float(record["end_sec"]),
                "confidence": float(record["confidence"]),
            }
        )
    return absolute_records


def compute_auto_selection_update(
    row: dict[str, str],
    *,
    diagnostics_records: list[dict[str, Any]] | None,
    diagnostics_by_filename: dict[str, list[dict[str, Any]]],
    is_hydrophone: bool,
    window_size_seconds: float,
    smoothing_window: int = DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW,
    min_score: float = DEFAULT_POSITIVE_SELECTION_MIN_SCORE,
    extend_min_score: float = DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE,
    detection_mode: str | None = None,
) -> dict[str, str]:
    # Windowed mode: the detection IS the positive window — trivial selection.
    if detection_mode == "windowed":
        normalized = normalize_detection_row(
            row,
            is_hydrophone=is_hydrophone,
            window_size_seconds=window_size_seconds,
        )
        row_start, row_end = resolve_clip_bounds(
            normalized, window_size_seconds=window_size_seconds
        )
        conf = safe_optional_float(row.get("peak_confidence")) or 0.0
        selection = PositiveSelectionResult(
            score_source="windowed_peak",
            decision="positive",
            offsets=[row_start],
            raw_scores=[conf],
            smoothed_scores=[conf],
            start_sec=row_start,
            end_sec=row_end,
            peak_score=conf,
        )
        return prefixed_selection_result_to_row_update(
            selection, prefix="auto_positive_selection_"
        )

    if row.get("auto_positive_selection_start_sec", "").strip():
        return {
            field: row.get(field, "") for field in AUTO_POSITIVE_SELECTION_FIELDNAMES
        }

    if row.get("positive_selection_start_sec", "").strip():
        return _existing_selection_to_auto_update(row)

    normalized = normalize_detection_row(
        row,
        is_hydrophone=is_hydrophone,
        window_size_seconds=window_size_seconds,
    )
    row_start_sec, row_end_sec = resolve_clip_bounds(
        normalized, window_size_seconds=window_size_seconds
    )

    source_filename = normalized["filename"]
    exact_records = diagnostics_by_filename.get(source_filename, [])
    selection: PositiveSelectionResult | None = None
    if exact_records:
        selection = select_positive_window(
            row_start_sec=row_start_sec,
            row_end_sec=row_end_sec,
            window_size_seconds=window_size_seconds,
            window_records=exact_records,
            smoothing_window=smoothing_window,
            min_score=min_score,
            extend_min_score=extend_min_score,
            score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
        )

    if (
        (selection is None or not selection.offsets)
        and diagnostics_records
        and parse_recording_timestamp(source_filename) is not None
    ):
        recording_ts = parse_recording_timestamp(source_filename)
        assert recording_ts is not None
        absolute_records = _build_absolute_window_records(diagnostics_records)
        absolute_selection = select_positive_window(
            row_start_sec=recording_ts.timestamp() + row_start_sec,
            row_end_sec=recording_ts.timestamp() + row_end_sec,
            window_size_seconds=window_size_seconds,
            window_records=absolute_records,
            smoothing_window=smoothing_window,
            min_score=min_score,
            extend_min_score=extend_min_score,
            score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
        )
        if absolute_selection.offsets:
            selection = PositiveSelectionResult(
                score_source=absolute_selection.score_source,
                decision=absolute_selection.decision,
                offsets=[
                    float(offset - recording_ts.timestamp())
                    for offset in absolute_selection.offsets
                ],
                raw_scores=list(absolute_selection.raw_scores),
                smoothed_scores=list(absolute_selection.smoothed_scores),
                start_sec=(
                    absolute_selection.start_sec - recording_ts.timestamp()
                    if absolute_selection.start_sec is not None
                    else None
                ),
                end_sec=(
                    absolute_selection.end_sec - recording_ts.timestamp()
                    if absolute_selection.end_sec is not None
                    else None
                ),
                peak_score=absolute_selection.peak_score,
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

    return prefixed_selection_result_to_row_update(
        selection, prefix="auto_positive_selection_"
    )


def apply_effective_positive_selection(
    row: dict[str, str],
    *,
    window_size_seconds: float,
) -> None:
    if not is_positive_row(row):
        positive_extract_filename = row.get("positive_extract_filename", "")
        row["positive_selection_origin"] = ""
        row.update(blank_positive_selection_fields())
        row["positive_extract_filename"] = positive_extract_filename
        return

    manual_start = safe_optional_float(row.get("manual_positive_selection_start_sec"))
    manual_end = safe_optional_float(row.get("manual_positive_selection_end_sec"))
    if (
        manual_start is not None
        and manual_end is not None
        and manual_end > manual_start
    ):
        row["positive_selection_origin"] = POSITIVE_SELECTION_ORIGIN_MANUAL
        row["positive_selection_score_source"] = POSITIVE_SELECTION_SCORE_SOURCE_MANUAL
        row["positive_selection_decision"] = "positive"
        row["positive_selection_offsets"] = ""
        row["positive_selection_raw_scores"] = ""
        row["positive_selection_smoothed_scores"] = ""
        row["positive_selection_start_sec"] = format_optional_float(manual_start)
        row["positive_selection_end_sec"] = format_optional_float(manual_end)
        row["positive_selection_peak_score"] = ""
        return

    auto_start = safe_optional_float(row.get("auto_positive_selection_start_sec"))
    auto_end = safe_optional_float(row.get("auto_positive_selection_end_sec"))
    auto_decision = row.get("auto_positive_selection_decision", "").strip()
    if auto_decision == "positive" and auto_start is not None and auto_end is not None:
        row["positive_selection_origin"] = POSITIVE_SELECTION_ORIGIN_AUTO
        row["positive_selection_score_source"] = row.get(
            "auto_positive_selection_score_source", ""
        )
        row["positive_selection_decision"] = auto_decision
        row["positive_selection_offsets"] = row.get(
            "auto_positive_selection_offsets", ""
        )
        row["positive_selection_raw_scores"] = row.get(
            "auto_positive_selection_raw_scores", ""
        )
        row["positive_selection_smoothed_scores"] = row.get(
            "auto_positive_selection_smoothed_scores", ""
        )
        row["positive_selection_start_sec"] = row.get(
            "auto_positive_selection_start_sec", ""
        )
        row["positive_selection_end_sec"] = row.get(
            "auto_positive_selection_end_sec", ""
        )
        row["positive_selection_peak_score"] = row.get(
            "auto_positive_selection_peak_score", ""
        )
        return

    row["positive_selection_origin"] = POSITIVE_SELECTION_ORIGIN_CLIP
    row["positive_selection_score_source"] = (
        POSITIVE_SELECTION_SCORE_SOURCE_FALLBACK_CLIP
    )
    row["positive_selection_decision"] = "positive"
    row["positive_selection_offsets"] = ""
    row["positive_selection_raw_scores"] = ""
    row["positive_selection_smoothed_scores"] = ""
    clip_start, clip_end = resolve_clip_bounds(
        row, window_size_seconds=window_size_seconds
    )
    row["positive_selection_start_sec"] = format_optional_float(clip_start)
    row["positive_selection_end_sec"] = format_optional_float(clip_end)
    row["positive_selection_peak_score"] = ""


def build_detection_row_store_rows(
    rows: list[dict[str, str]],
    *,
    diagnostics_path: Path | None,
    is_hydrophone: bool,
    window_size_seconds: float,
    detection_mode: str | None = None,
) -> list[dict[str, str]]:
    diagnostics_records = _load_all_window_records(diagnostics_path)
    diagnostics_by_filename: dict[str, list[dict[str, Any]]] = {}
    if diagnostics_records:
        for record in diagnostics_records:
            diagnostics_by_filename.setdefault(str(record["filename"]), []).append(
                record
            )

    store_rows: list[dict[str, str]] = []
    for row in rows:
        normalized = normalize_detection_row(
            row,
            is_hydrophone=is_hydrophone,
            window_size_seconds=window_size_seconds,
        )
        out_row = {field: "" for field in ROW_STORE_FIELDNAMES}
        out_row["row_id"] = normalized["row_id"] or build_detection_row_id(normalized)
        out_row["filename"] = normalized["filename"]
        out_row["start_sec"] = format_optional_float(normalized["start_sec"])
        out_row["end_sec"] = format_optional_float(normalized["end_sec"])
        out_row["avg_confidence"] = format_optional_float(normalized["avg_confidence"])
        out_row["peak_confidence"] = format_optional_float(
            normalized["peak_confidence"]
        )
        out_row["n_windows"] = format_optional_int(normalized["n_windows"])
        out_row["raw_start_sec"] = format_optional_float(normalized["raw_start_sec"])
        out_row["raw_end_sec"] = format_optional_float(normalized["raw_end_sec"])
        out_row["merged_event_count"] = format_optional_int(
            normalized["merged_event_count"]
        )
        out_row["detection_filename"] = normalized["detection_filename"] or ""
        out_row["extract_filename"] = normalized["extract_filename"] or ""
        out_row["hydrophone_name"] = normalized["hydrophone_name"] or ""
        for label in ("humpback", "orca", "ship", "background"):
            value = normalized[label]
            out_row[label] = "" if value is None else str(value)

        out_row["manual_positive_selection_start_sec"] = row.get(
            "manual_positive_selection_start_sec", ""
        )
        out_row["manual_positive_selection_end_sec"] = row.get(
            "manual_positive_selection_end_sec", ""
        )
        out_row["positive_extract_filename"] = (
            normalized["positive_extract_filename"] or ""
        )

        out_row.update(
            compute_auto_selection_update(
                row,
                diagnostics_records=diagnostics_records,
                diagnostics_by_filename=diagnostics_by_filename,
                is_hydrophone=is_hydrophone,
                window_size_seconds=window_size_seconds,
                detection_mode=detection_mode,
            )
        )
        apply_effective_positive_selection(
            out_row,
            window_size_seconds=window_size_seconds,
        )
        store_rows.append(out_row)

    return store_rows


def merge_detection_row_store_state(
    refreshed_rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
    *,
    window_size_seconds: float,
) -> list[dict[str, str]]:
    """Overlay editable row-store state onto refreshed detection rows."""
    existing_by_row_id = {
        row.get("row_id", "").strip(): row
        for row in existing_rows
        if row.get("row_id", "").strip()
    }

    merged_rows: list[dict[str, str]] = []
    for row in refreshed_rows:
        existing = existing_by_row_id.get(row.get("row_id", "").strip())
        if existing is None:
            merged_rows.append(row)
            continue

        for field in LABEL_FIELDNAMES:
            row[field] = existing.get(field, "")

        for field in MANUAL_POSITIVE_SELECTION_FIELDNAMES:
            row[field] = existing.get(field, "")

        if not any(
            row.get(field, "").strip() for field in AUTO_POSITIVE_SELECTION_FIELDNAMES
        ):
            for field in AUTO_POSITIVE_SELECTION_FIELDNAMES:
                row[field] = existing.get(field, "")

        if existing.get("positive_extract_filename", "").strip():
            row["positive_extract_filename"] = existing["positive_extract_filename"]

        apply_effective_positive_selection(
            row,
            window_size_seconds=window_size_seconds,
        )
        merged_rows.append(row)

    return merged_rows


def ensure_detection_row_store(
    *,
    row_store_path: Path,
    diagnostics_path: Path | None,
    is_hydrophone: bool,
    window_size_seconds: float,
    refresh_existing: bool = False,
    detection_mode: str | None = None,
    tsv_path: Path | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    if row_store_path.is_file() and not refresh_existing:
        return read_detection_row_store(row_store_path)

    # Determine the source rows.  When a TSV is provided (legacy fallback)
    # prefer it as the source of truth for detection rows.  Otherwise use the
    # existing row store.
    source_rows: list[dict[str, str]] = []
    if tsv_path is not None and tsv_path.is_file():
        _, source_rows = read_tsv_rows(tsv_path)
    elif row_store_path.is_file():
        _, source_rows = read_detection_row_store(row_store_path)

    if not source_rows:
        write_detection_row_store(row_store_path, [])
        return ROW_STORE_FIELDNAMES, []

    store_rows = build_detection_row_store_rows(
        source_rows,
        diagnostics_path=diagnostics_path,
        is_hydrophone=is_hydrophone,
        window_size_seconds=window_size_seconds,
        detection_mode=detection_mode,
    )
    if row_store_path.is_file():
        _existing_fieldnames, existing_rows = read_detection_row_store(row_store_path)
        store_rows = merge_detection_row_store_state(
            store_rows,
            existing_rows,
            window_size_seconds=window_size_seconds,
        )
    write_detection_row_store(row_store_path, store_rows)
    return ROW_STORE_FIELDNAMES, store_rows
