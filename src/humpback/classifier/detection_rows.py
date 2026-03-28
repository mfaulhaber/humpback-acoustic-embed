"""Detection row normalization and Parquet-backed row-store helpers."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
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
    "positive_selection_start_utc",
    "positive_selection_end_utc",
    "positive_selection_peak_score",
    "positive_extract_filename",
]

AUTO_POSITIVE_SELECTION_FIELDNAMES = [
    "auto_positive_selection_score_source",
    "auto_positive_selection_decision",
    "auto_positive_selection_offsets",
    "auto_positive_selection_raw_scores",
    "auto_positive_selection_smoothed_scores",
    "auto_positive_selection_start_utc",
    "auto_positive_selection_end_utc",
    "auto_positive_selection_peak_score",
]

MANUAL_POSITIVE_SELECTION_FIELDNAMES = [
    "manual_positive_selection_start_utc",
    "manual_positive_selection_end_utc",
]

LABEL_FIELDNAMES = ["humpback", "orca", "ship", "background"]

ROW_STORE_FIELDNAMES = [
    "start_utc",
    "end_utc",
    "avg_confidence",
    "peak_confidence",
    "n_windows",
    "raw_start_utc",
    "raw_end_utc",
    "merged_event_count",
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
    start_utc: float | None
    end_utc: float | None
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
    start_utc: float,
    end_utc: float,
) -> str | None:
    """Format a detection filename from UTC epoch float pair."""
    if end_utc <= start_utc:
        return None
    abs_start = datetime.fromtimestamp(start_utc, tz=timezone.utc)
    abs_end = datetime.fromtimestamp(end_utc, tz=timezone.utc)
    return (
        f"{abs_start.strftime(_COMPACT_TS_FORMAT)}"
        f"_{abs_end.strftime(_COMPACT_TS_FORMAT)}.flac"
    )


def normalize_detection_row(
    row: dict[str, str],
) -> dict[str, Any]:
    """Parse and validate a detection row with UTC identity fields."""
    start_utc = safe_float(row.get("start_utc"), 0.0)
    end_utc = safe_float(row.get("end_utc"), 0.0)
    avg_confidence = safe_optional_float(row.get("avg_confidence"))
    peak_confidence = safe_optional_float(row.get("peak_confidence"))
    n_windows = safe_int(row.get("n_windows"), None)

    raw_start_utc = safe_float(row.get("raw_start_utc"), start_utc)
    raw_end_utc = safe_float(row.get("raw_end_utc"), end_utc)
    merged_event_count = safe_int(row.get("merged_event_count"), 1)
    positive_extract_filename = row.get("positive_extract_filename", "").strip() or None

    return {
        "start_utc": start_utc,
        "end_utc": end_utc,
        "avg_confidence": avg_confidence,
        "peak_confidence": peak_confidence,
        "n_windows": n_windows,
        "hydrophone_name": (row.get("hydrophone_name", "").strip() or None),
        "raw_start_utc": raw_start_utc,
        "raw_end_utc": raw_end_utc,
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
        "auto_positive_selection_start_utc": safe_optional_float(
            row.get("auto_positive_selection_start_utc")
        ),
        "auto_positive_selection_end_utc": safe_optional_float(
            row.get("auto_positive_selection_end_utc")
        ),
        "auto_positive_selection_peak_score": safe_optional_float(
            row.get("auto_positive_selection_peak_score")
        ),
        "manual_positive_selection_start_utc": safe_optional_float(
            row.get("manual_positive_selection_start_utc")
        ),
        "manual_positive_selection_end_utc": safe_optional_float(
            row.get("manual_positive_selection_end_utc")
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
        "positive_selection_start_utc": safe_optional_float(
            row.get("positive_selection_start_utc")
        ),
        "positive_selection_end_utc": safe_optional_float(
            row.get("positive_selection_end_utc")
        ),
        "positive_selection_peak_score": safe_optional_float(
            row.get("positive_selection_peak_score")
        ),
        "positive_extract_filename": positive_extract_filename,
    }


def _row_key(row: dict[str, str]) -> tuple[str, str]:
    """Return the (start_utc, end_utc) composite key for a detection row."""
    return (row.get("start_utc", ""), row.get("end_utc", ""))


def _is_row_labeled(row: dict[str, str]) -> bool:
    """Return True if any label field is set to '1'."""
    return any(row.get(f, "").strip() == "1" for f in LABEL_FIELDNAMES)


def _rows_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and a_end > b_start


def apply_label_edits(
    rows: list[dict[str, str]],
    edits: list[dict[str, Any]],
    *,
    job_duration: float,
) -> list[dict[str, str]]:
    """Apply a batch of label edits (add/move/delete/change_type) to detection rows.

    Rows are identified by ``(start_utc, end_utc)`` composite key.
    Returns an updated copy of the row list.  Raises ``ValueError`` on invalid
    operations (overlapping labeled rows, out-of-bounds moves, missing keys).
    """
    result: list[dict[str, str]] = [dict(r) for r in rows]

    # Build lookup by (start_utc, end_utc) composite key.
    row_index: dict[tuple[str, str], dict[str, str]] = {_row_key(r): r for r in result}

    new_rows: list[dict[str, str]] = []
    delete_keys: set[tuple[str, str]] = set()
    touched_keys: set[tuple[str, str]] = set()

    for edit in edits:
        action = edit.get("action")

        if action == "add":
            start = float(edit["start_utc"])
            end = float(edit["end_utc"])
            label = edit.get("label")

            new_row = {f: "" for f in ROW_STORE_FIELDNAMES}
            new_row["start_utc"] = str(start)
            new_row["end_utc"] = str(end)
            if label:
                new_row[label] = "1"
            new_rows.append(new_row)
            touched_keys.add(_row_key(new_row))

        elif action == "move":
            key = (str(edit.get("start_utc", "")), str(edit.get("end_utc", "")))
            target = row_index.get(key)
            if target is None:
                raise ValueError(f"Row with key ({key[0]}, {key[1]}) not found")
            new_start = float(edit["new_start_utc"])
            new_end = float(edit["new_end_utc"])
            target["start_utc"] = str(new_start)
            target["end_utc"] = str(new_end)
            touched_keys.add(_row_key(target))

        elif action == "delete":
            key = (str(edit.get("start_utc", "")), str(edit.get("end_utc", "")))
            if key not in row_index:
                raise ValueError(f"Row with key ({key[0]}, {key[1]}) not found")
            delete_keys.add(key)

        elif action == "change_type":
            key = (str(edit.get("start_utc", "")), str(edit.get("end_utc", "")))
            target = row_index.get(key)
            if target is None:
                raise ValueError(f"Row with key ({key[0]}, {key[1]}) not found")
            label = edit.get("label", "")
            for lf in LABEL_FIELDNAMES:
                target[lf] = ""
            if label:
                target[label] = "1"
            touched_keys.add(key)

        else:
            raise ValueError(f"Unknown edit action: {action!r}")

    # Apply deletes.
    result = [r for r in result if _row_key(r) not in delete_keys]

    # Unlabeled replacement: remove unlabeled rows that overlap with new adds.
    if new_rows:
        labeled_new = [r for r in new_rows if _is_row_labeled(r)]
        if labeled_new:
            keep: list[dict[str, str]] = []
            for r in result:
                if _is_row_labeled(r):
                    keep.append(r)
                    continue
                r_start = safe_float(r.get("start_utc"), 0.0)
                r_end = safe_float(r.get("end_utc"), 0.0)
                replaced = False
                for nr in labeled_new:
                    nr_start = safe_float(nr.get("start_utc"), 0.0)
                    nr_end = safe_float(nr.get("end_utc"), 0.0)
                    if _rows_overlap(r_start, r_end, nr_start, nr_end):
                        replaced = True
                        break
                if not replaced:
                    keep.append(r)
            result = keep

    result.extend(new_rows)

    # Overlap validation: only reject overlaps where BOTH rows were touched.
    if len(touched_keys) >= 2:
        touched_labeled = [
            (safe_float(r.get("start_utc"), 0.0), safe_float(r.get("end_utc"), 0.0), r)
            for r in result
            if _is_row_labeled(r) and _row_key(r) in touched_keys
        ]
        touched_labeled.sort(key=lambda t: t[0])
        for i in range(len(touched_labeled) - 1):
            a_start, a_end, _ = touched_labeled[i]
            b_start, b_end, _ = touched_labeled[i + 1]
            if _rows_overlap(a_start, a_end, b_start, b_end):
                raise ValueError(
                    f"Labeled rows overlap: [{a_start}, {a_end}] "
                    f"and [{b_start}, {b_end}]"
                )

    return result


def resolve_clip_bounds(
    row: dict[str, Any],
) -> tuple[float, float]:
    """Return the (start_utc, end_utc) clip bounds directly from the row."""
    return (
        safe_float(row.get("start_utc"), 0.0),
        safe_float(row.get("end_utc"), 0.0),
    )


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
        "positive_selection_start_utc": format_optional_float(result.start_utc),
        "positive_selection_end_utc": format_optional_float(result.end_utc),
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
        f"{prefix}start_utc": format_optional_float(result.start_utc),
        f"{prefix}end_utc": format_optional_float(result.end_utc),
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
    row_start_utc: float,
    row_end_utc: float,
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
        if offset_sec + 1e-6 < row_start_utc or end_sec - 1e-6 > row_end_utc:
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
            start_utc=None,
            end_utc=None,
            peak_score=None,
        )

    offsets = [float(rec["offset_sec"]) for rec in candidates]
    raw_scores = [float(rec["confidence"]) for rec in candidates]
    smoothed_scores = smooth_scores(raw_scores, smoothing_window)
    best_idx = int(np.argmax(np.asarray(smoothed_scores, dtype=np.float32)))
    peak_score = float(smoothed_scores[best_idx])
    best = candidates[best_idx]
    sel_start = float(best["offset_sec"])
    sel_end = float(best["end_sec"])

    if peak_score >= min_score:
        candidates_by_offset = {
            candidate_offset_key(float(rec["offset_sec"])): (
                rec,
                float(smoothed_scores[idx]),
            )
            for idx, rec in enumerate(candidates)
        }
        while True:
            left_key = candidate_offset_key(sel_start - window_size_seconds)
            right_key = candidate_offset_key(sel_end)
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
                sel_start = float(left_candidate[0]["offset_sec"])
            else:
                assert right_candidate is not None
                sel_end = float(right_candidate[0]["end_sec"])

    return PositiveSelectionResult(
        score_source=score_source,
        decision="positive" if peak_score >= min_score else "skip",
        offsets=offsets,
        raw_scores=raw_scores,
        smoothed_scores=smoothed_scores,
        start_utc=sel_start,
        end_utc=sel_end,
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


def _migrate_legacy_row(row: dict[str, str]) -> dict[str, str]:
    """Convert a single old-schema row to the new UTC schema."""
    new_row: dict[str, str] = {}

    # Derive start_utc/end_utc from detection_filename (primary) or
    # filename + offsets (fallback).
    start_utc: float | None = None
    end_utc: float | None = None

    detection_filename = (row.get("detection_filename", "") or "").strip()
    if detection_filename:
        parsed = parse_compact_range_filename(detection_filename)
        if parsed is not None:
            start_utc = parsed[0].timestamp()
            end_utc = parsed[1].timestamp()

    if start_utc is None:
        filename = (row.get("filename", "") or "").strip()
        recording_ts = parse_recording_timestamp(filename) if filename else None
        start_sec = safe_float(row.get("start_sec"), 0.0)
        end_sec = safe_float(row.get("end_sec"), 0.0)
        if recording_ts is not None:
            base = recording_ts.timestamp()
            start_utc = base + start_sec
            end_utc = base + end_sec
        else:
            start_utc = start_sec
            end_utc = end_sec

    new_row["start_utc"] = str(start_utc)
    new_row["end_utc"] = str(end_utc)

    # Derive raw_start_utc/raw_end_utc
    raw_start_sec = safe_float(row.get("raw_start_sec"), 0.0)
    raw_end_sec = safe_float(row.get("raw_end_sec"), 0.0)
    filename = (row.get("filename", "") or "").strip()
    recording_ts = parse_recording_timestamp(filename) if filename else None
    if recording_ts is not None:
        base = recording_ts.timestamp()
        new_row["raw_start_utc"] = str(base + raw_start_sec)
        new_row["raw_end_utc"] = str(base + raw_end_sec)
    else:
        new_row["raw_start_utc"] = str(raw_start_sec)
        new_row["raw_end_utc"] = str(raw_end_sec)

    # Copy retained fields directly
    for field in (
        "avg_confidence",
        "peak_confidence",
        "n_windows",
        "merged_event_count",
        "hydrophone_name",
        "humpback",
        "orca",
        "ship",
        "background",
        "positive_selection_origin",
        "positive_selection_score_source",
        "positive_selection_decision",
        "positive_selection_offsets",
        "positive_selection_raw_scores",
        "positive_selection_smoothed_scores",
        "positive_selection_peak_score",
        "positive_extract_filename",
        "auto_positive_selection_score_source",
        "auto_positive_selection_decision",
        "auto_positive_selection_offsets",
        "auto_positive_selection_raw_scores",
        "auto_positive_selection_smoothed_scores",
        "auto_positive_selection_peak_score",
    ):
        new_row[field] = row.get(field, "")

    # Migrate positive selection _sec -> _utc fields
    _OLD_TO_NEW_SELECTION = [
        ("positive_selection_start_sec", "positive_selection_start_utc"),
        ("positive_selection_end_sec", "positive_selection_end_utc"),
        ("auto_positive_selection_start_sec", "auto_positive_selection_start_utc"),
        ("auto_positive_selection_end_sec", "auto_positive_selection_end_utc"),
        ("manual_positive_selection_start_sec", "manual_positive_selection_start_utc"),
        ("manual_positive_selection_end_sec", "manual_positive_selection_end_utc"),
    ]
    for old_key, new_key in _OLD_TO_NEW_SELECTION:
        old_val = safe_optional_float(row.get(old_key))
        if old_val is not None and recording_ts is not None:
            new_row[new_key] = str(recording_ts.timestamp() + old_val)
        elif old_val is not None:
            new_row[new_key] = str(old_val)
        else:
            new_row[new_key] = ""

    return new_row


def read_detection_row_store(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a Parquet row store, lazily migrating old schema if needed."""
    # Read without schema enforcement to detect old vs new format.
    table = pq.read_table(str(path))
    col_names = set(table.column_names)

    if "start_utc" in col_names:
        # New schema — read with enforced schema.
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

    # Old schema — migrate.
    logger.info("Migrating legacy detection row store: %s", path)
    columns_dict = table.to_pydict()
    old_fieldnames = list(table.column_names)
    old_rows: list[dict[str, str]] = []
    for idx in range(table.num_rows):
        row_data: dict[str, str] = {}
        for field in old_fieldnames:
            value = columns_dict[field][idx]
            row_data[field] = value if value is not None else ""
        old_rows.append(row_data)

    migrated = [_migrate_legacy_row(r) for r in old_rows]
    # Atomically rewrite in new schema.
    write_detection_row_store(path, migrated)
    return ROW_STORE_FIELDNAMES, migrated


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


def _tsv_export_fieldnames(fieldnames: list[str]) -> list[str]:
    """Build TSV export columns: add derived detection_filename after end_utc."""
    export = list(fieldnames)
    try:
        idx = export.index("end_utc") + 1
    except ValueError:
        idx = 2
    export.insert(idx, "detection_filename")
    return export


def _row_for_tsv(row: dict[str, str], export_fieldnames: list[str]) -> dict[str, str]:
    """Produce a TSV row dict with derived detection_filename."""
    out = {field: row.get(field, "") for field in export_fieldnames}
    start = safe_float(row.get("start_utc"), 0.0)
    end = safe_float(row.get("end_utc"), 0.0)
    out["detection_filename"] = derive_detection_filename(start, end) or ""
    return out


def stream_detection_rows_as_tsv(
    rows: list[dict[str, str]],
    *,
    fieldnames: list[str],
) -> io.StringIO:
    export_fieldnames = _tsv_export_fieldnames(fieldnames)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=export_fieldnames, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow(_row_for_tsv(row, export_fieldnames))
    buf.seek(0)
    return buf


def iter_detection_rows_as_tsv(
    rows: list[dict[str, str]],
    *,
    fieldnames: list[str],
):
    export_fieldnames = _tsv_export_fieldnames(fieldnames)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=export_fieldnames, delimiter="\t")
    writer.writeheader()
    yield buf.getvalue()
    buf.seek(0)
    buf.truncate(0)
    for row in rows:
        writer.writerow(_row_for_tsv(row, export_fieldnames))
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
        "auto_positive_selection_start_utc": row.get(
            "positive_selection_start_utc", ""
        ),
        "auto_positive_selection_end_utc": row.get("positive_selection_end_utc", ""),
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
    window_size_seconds: float,
    smoothing_window: int = DEFAULT_POSITIVE_SELECTION_SMOOTHING_WINDOW,
    min_score: float = DEFAULT_POSITIVE_SELECTION_MIN_SCORE,
    extend_min_score: float = DEFAULT_POSITIVE_SELECTION_EXTEND_MIN_SCORE,
    detection_mode: str | None = None,
) -> dict[str, str]:
    normalized = normalize_detection_row(row)
    row_start_utc, row_end_utc = resolve_clip_bounds(normalized)

    # Windowed mode: the detection IS the positive window — trivial selection.
    if detection_mode == "windowed":
        conf = safe_optional_float(row.get("peak_confidence")) or 0.0
        selection = PositiveSelectionResult(
            score_source="windowed_peak",
            decision="positive",
            offsets=[row_start_utc],
            raw_scores=[conf],
            smoothed_scores=[conf],
            start_utc=row_start_utc,
            end_utc=row_end_utc,
            peak_score=conf,
        )
        return prefixed_selection_result_to_row_update(
            selection, prefix="auto_positive_selection_"
        )

    if row.get("auto_positive_selection_start_utc", "").strip():
        return {
            field: row.get(field, "") for field in AUTO_POSITIVE_SELECTION_FIELDNAMES
        }

    if row.get("positive_selection_start_utc", "").strip():
        return _existing_selection_to_auto_update(row)

    # Convert all diagnostics to absolute UTC and select against row UTC bounds.
    selection: PositiveSelectionResult | None = None
    if diagnostics_records:
        absolute_records = _build_absolute_window_records(diagnostics_records)
        if absolute_records:
            selection = select_positive_window(
                row_start_utc=row_start_utc,
                row_end_utc=row_end_utc,
                window_size_seconds=window_size_seconds,
                window_records=absolute_records,
                smoothing_window=smoothing_window,
                min_score=min_score,
                extend_min_score=extend_min_score,
                score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
            )

    if selection is None:
        selection = PositiveSelectionResult(
            score_source=POSITIVE_SELECTION_SCORE_SOURCE_STORED,
            decision="skip",
            offsets=[],
            raw_scores=[],
            smoothed_scores=[],
            start_utc=None,
            end_utc=None,
            peak_score=None,
        )

    return prefixed_selection_result_to_row_update(
        selection, prefix="auto_positive_selection_"
    )


def apply_effective_positive_selection(
    row: dict[str, str],
) -> None:
    if not is_positive_row(row):
        positive_extract_filename = row.get("positive_extract_filename", "")
        row["positive_selection_origin"] = ""
        row.update(blank_positive_selection_fields())
        row["positive_extract_filename"] = positive_extract_filename
        return

    manual_start = safe_optional_float(row.get("manual_positive_selection_start_utc"))
    manual_end = safe_optional_float(row.get("manual_positive_selection_end_utc"))
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
        row["positive_selection_start_utc"] = format_optional_float(manual_start)
        row["positive_selection_end_utc"] = format_optional_float(manual_end)
        row["positive_selection_peak_score"] = ""
        return

    auto_start = safe_optional_float(row.get("auto_positive_selection_start_utc"))
    auto_end = safe_optional_float(row.get("auto_positive_selection_end_utc"))
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
        row["positive_selection_start_utc"] = row.get(
            "auto_positive_selection_start_utc", ""
        )
        row["positive_selection_end_utc"] = row.get(
            "auto_positive_selection_end_utc", ""
        )
        row["positive_selection_peak_score"] = row.get(
            "auto_positive_selection_peak_score", ""
        )
        return

    # Clip-bounds fallback: use the row's own start_utc/end_utc.
    row["positive_selection_origin"] = POSITIVE_SELECTION_ORIGIN_CLIP
    row["positive_selection_score_source"] = (
        POSITIVE_SELECTION_SCORE_SOURCE_FALLBACK_CLIP
    )
    row["positive_selection_decision"] = "positive"
    row["positive_selection_offsets"] = ""
    row["positive_selection_raw_scores"] = ""
    row["positive_selection_smoothed_scores"] = ""
    clip_start, clip_end = resolve_clip_bounds(row)
    row["positive_selection_start_utc"] = format_optional_float(clip_start)
    row["positive_selection_end_utc"] = format_optional_float(clip_end)
    row["positive_selection_peak_score"] = ""


def build_detection_row_store_rows(
    rows: list[dict[str, str]],
    *,
    diagnostics_path: Path | None,
    window_size_seconds: float,
    detection_mode: str | None = None,
) -> list[dict[str, str]]:
    diagnostics_records = _load_all_window_records(diagnostics_path)

    store_rows: list[dict[str, str]] = []
    for row in rows:
        normalized = normalize_detection_row(row)
        out_row = {field: "" for field in ROW_STORE_FIELDNAMES}
        out_row["start_utc"] = format_optional_float(normalized["start_utc"])
        out_row["end_utc"] = format_optional_float(normalized["end_utc"])
        out_row["avg_confidence"] = format_optional_float(normalized["avg_confidence"])
        out_row["peak_confidence"] = format_optional_float(
            normalized["peak_confidence"]
        )
        out_row["n_windows"] = format_optional_int(normalized["n_windows"])
        out_row["raw_start_utc"] = format_optional_float(normalized["raw_start_utc"])
        out_row["raw_end_utc"] = format_optional_float(normalized["raw_end_utc"])
        out_row["merged_event_count"] = format_optional_int(
            normalized["merged_event_count"]
        )
        out_row["hydrophone_name"] = normalized["hydrophone_name"] or ""
        for label in ("humpback", "orca", "ship", "background"):
            value = normalized[label]
            out_row[label] = "" if value is None else str(value)

        out_row["manual_positive_selection_start_utc"] = row.get(
            "manual_positive_selection_start_utc", ""
        )
        out_row["manual_positive_selection_end_utc"] = row.get(
            "manual_positive_selection_end_utc", ""
        )
        out_row["positive_extract_filename"] = (
            normalized["positive_extract_filename"] or ""
        )

        out_row.update(
            compute_auto_selection_update(
                row,
                diagnostics_records=diagnostics_records,
                window_size_seconds=window_size_seconds,
                detection_mode=detection_mode,
            )
        )
        apply_effective_positive_selection(out_row)
        store_rows.append(out_row)

    return store_rows


def merge_detection_row_store_state(
    refreshed_rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Overlay editable row-store state onto refreshed detection rows."""
    existing_by_key = {_row_key(row): row for row in existing_rows}

    merged_rows: list[dict[str, str]] = []
    for row in refreshed_rows:
        existing = existing_by_key.get(_row_key(row))
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

        apply_effective_positive_selection(row)
        merged_rows.append(row)

    return merged_rows


def ensure_detection_row_store(
    *,
    row_store_path: Path,
    diagnostics_path: Path | None,
    window_size_seconds: float,
    refresh_existing: bool = False,
    detection_mode: str | None = None,
    tsv_path: Path | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    if row_store_path.is_file() and not refresh_existing:
        return read_detection_row_store(row_store_path)

    source_rows: list[dict[str, str]] = []
    is_legacy_tsv = False
    if tsv_path is not None and tsv_path.is_file():
        _, source_rows = read_tsv_rows(tsv_path)
        is_legacy_tsv = True
    elif row_store_path.is_file():
        _, source_rows = read_detection_row_store(row_store_path)

    if not source_rows:
        write_detection_row_store(row_store_path, [])
        return ROW_STORE_FIELDNAMES, []

    # Migrate legacy TSV rows to UTC schema before building row store.
    if is_legacy_tsv and source_rows and "start_utc" not in source_rows[0]:
        source_rows = [_migrate_legacy_row(r) for r in source_rows]

    store_rows = build_detection_row_store_rows(
        source_rows,
        diagnostics_path=diagnostics_path,
        window_size_seconds=window_size_seconds,
        detection_mode=detection_mode,
    )
    if row_store_path.is_file():
        _existing_fieldnames, existing_rows = read_detection_row_store(row_store_path)
        store_rows = merge_detection_row_store_state(
            store_rows,
            existing_rows,
        )
    write_detection_row_store(row_store_path, store_rows)
    return ROW_STORE_FIELDNAMES, store_rows
