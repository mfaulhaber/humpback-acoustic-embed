"""Detection utilities: window processing, I/O helpers, embedding I/O, audio resolution."""

import csv
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq

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


# ---------------------------------------------------------------------------
# Window processing
# ---------------------------------------------------------------------------


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


_LOGIT_EPS = 1e-7


def _to_logit(p: float) -> float:
    """Convert a probability to log-odds, clamping to avoid ±infinity."""
    p = max(_LOGIT_EPS, min(1.0 - _LOGIT_EPS, p))
    return math.log(p / (1.0 - p))


def _find_prominent_peaks(
    raw_scores: list[float],
    min_prominence: float,
    min_score: float,
) -> list[int]:
    """Find local peaks in *raw* scores filtered by prominence.

    A local peak is an index where the raw value is >= both neighbors
    (or >= its one neighbor at the edges).

    Prominence is computed on raw scores: the peak's value minus the highest
    valley between it and its nearest higher neighbor on each side.

    Returns sorted indices of peaks passing both ``min_prominence`` and
    ``min_score``.
    """
    n = len(raw_scores)
    if n == 0:
        return []

    # Step 1: find local maxima in raw scores.
    peak_indices: list[int] = []
    for i in range(n):
        if raw_scores[i] < min_score:
            continue
        left_ok = i == 0 or raw_scores[i] >= raw_scores[i - 1]
        right_ok = i == n - 1 or raw_scores[i] >= raw_scores[i + 1]
        if left_ok and right_ok:
            peak_indices.append(i)

    if not peak_indices:
        return []

    # Step 2: compute prominence for each peak.
    # For each peak, look left and right to find the highest valley (minimum
    # raw score) between the peak and the nearest higher-or-equal raw peak.
    surviving: list[int] = []
    for pi in peak_indices:
        raw_peak = raw_scores[pi]

        # Search left for the minimum raw score before a higher peak.
        # If the peak is at the left edge, this side imposes no constraint.
        if pi == 0:
            left_min = float("-inf")
        else:
            left_min = raw_peak
            for j in range(pi - 1, -1, -1):
                left_min = min(left_min, raw_scores[j])
                if raw_scores[j] >= raw_peak:
                    break

        # Search right for the minimum raw score before a higher peak.
        # If the peak is at the right edge, this side imposes no constraint.
        if pi == n - 1:
            right_min = float("-inf")
        else:
            right_min = raw_peak
            for j in range(pi + 1, n):
                right_min = min(right_min, raw_scores[j])
                if raw_scores[j] >= raw_peak:
                    break

        # Prominence = peak raw score - highest valley on either side.
        prominence = raw_peak - max(left_min, right_min)
        if prominence >= min_prominence:
            surviving.append(pi)

    return surviving


_DEFAULT_MIN_GAP_FILL = 3.0


def _fill_gaps_recursive(
    candidates: list[dict],
    raw_scores: list[float],
    selected_offsets: set[float],
    left: float,
    right: float,
    min_gap_fill: float,
    min_score: float,
) -> list[int]:
    """Find gap-fill windows between *left* and *right* boundaries.

    Scans candidate window records in the open interval ``(left, right)`` for
    the highest raw-probability window above ``min_score``.  If the gap exceeds
    ``min_gap_fill`` and a qualifying candidate exists, its index is emitted and
    the gap is recursively split into two sub-gaps.

    Returns indices into *candidates* for windows to add.
    """
    if (right - left) <= min_gap_fill:
        return []

    midpoint = (left + right) / 2.0
    best_idx = -1
    best_score = -1.0
    best_dist = float("inf")
    for i, rec in enumerate(candidates):
        offset = float(rec["offset_sec"])
        if offset <= left or offset >= right:
            continue
        if offset in selected_offsets:
            continue
        if raw_scores[i] >= min_score:
            dist = abs(offset - midpoint)
            if raw_scores[i] > best_score or (
                raw_scores[i] == best_score and dist < best_dist
            ):
                best_idx = i
                best_score = raw_scores[i]
                best_dist = dist

    if best_idx < 0:
        return []

    fill_offset = float(candidates[best_idx]["offset_sec"])
    selected_offsets.add(fill_offset)

    result = [best_idx]
    result.extend(
        _fill_gaps_recursive(
            candidates,
            raw_scores,
            selected_offsets,
            left,
            fill_offset,
            min_gap_fill,
            min_score,
        )
    )
    result.extend(
        _fill_gaps_recursive(
            candidates,
            raw_scores,
            selected_offsets,
            fill_offset,
            right,
            min_gap_fill,
            min_score,
        )
    )
    return result


def select_prominent_peaks_from_events(
    events: list[dict],
    window_records: list[dict],
    window_size_seconds: float,
    min_score: float,
    min_prominence: float = 1.0,
) -> list[dict]:
    """Select peak windows via prominence-based detection (overlapping allowed).

    For each merged event, finds overlapping window records, then identifies
    peaks in raw scores with sufficient prominence.  Each peak emits a
    ``window_size_seconds`` detection window.  Unlike NMS, neighboring peaks
    are NOT suppressed, so output windows may overlap.

    Scores are transformed to logit (log-odds) space before peak finding and
    prominence computation.  This amplifies meaningful dips in high-confidence
    regions where probability scores saturate near 1.0.

    ``min_prominence`` is in logit units (default 2.0).

    Returns detection dicts each spanning exactly ``window_size_seconds``.
    Audit fields (``raw_start_sec``, ``raw_end_sec``, ``merged_event_count``)
    are preserved from the parent event.
    """
    if not events or not window_records:
        return []

    logit_min_score = _to_logit(min_score)

    result: list[dict] = []

    for event in events:
        ev_start = float(event["start_sec"])
        ev_end = float(event["end_sec"])

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
        logit_scores = [_to_logit(s) for s in raw_scores]

        peak_indices = _find_prominent_peaks(
            logit_scores, min_prominence, logit_min_score
        )

        # Fallback: if no peaks pass the prominence filter but the event has
        # windows above min_score, emit the single highest-scoring window.
        # This ensures every detected event produces at least one detection
        # (e.g. a broad plateau vocalization with tiny internal dips).
        if not peak_indices:
            above = [i for i, s in enumerate(raw_scores) if s >= min_score]
            if above:
                peak_indices = [max(above, key=lambda i: raw_scores[i])]

        # Gap-fill pass: scan for uncovered regions between selected peaks
        # (and from event edges to nearest peak) and recursively fill them.
        if peak_indices:
            selected_offsets: set[float] = {
                float(candidates[i]["offset_sec"]) for i in peak_indices
            }
            sorted_offsets = sorted(selected_offsets)
            boundaries = (
                [(ev_start, sorted_offsets[0])]
                + [
                    (sorted_offsets[j], sorted_offsets[j + 1])
                    for j in range(len(sorted_offsets) - 1)
                ]
                + [(sorted_offsets[-1], ev_end)]
            )
            for left, right in boundaries:
                fill_indices = _fill_gaps_recursive(
                    candidates,
                    raw_scores,
                    selected_offsets,
                    left,
                    right,
                    _DEFAULT_MIN_GAP_FILL,
                    min_score,
                )
                peak_indices.extend(fill_indices)

        for idx in peak_indices:
            rec = candidates[idx]
            offset = float(rec["offset_sec"])
            conf = raw_scores[idx]

            peak_det: dict = {
                "start_sec": offset,
                "end_sec": offset + window_size_seconds,
                "avg_confidence": conf,
                "peak_confidence": conf,
                "n_windows": int(event.get("n_windows", 1)),
                "raw_start_sec": float(event.get("raw_start_sec", ev_start)),
                "raw_end_sec": float(event.get("raw_end_sec", ev_end)),
                "merged_event_count": int(event.get("merged_event_count", 1)),
            }
            for key in event:
                if key not in peak_det:
                    peak_det[key] = event[key]
            result.append(peak_det)

    # Deduplicate: adjacent events can share window records.
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


# ---------------------------------------------------------------------------
# TSV I/O
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Diagnostics I/O
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------


def match_embedding_records_to_row_store(
    records: list[dict],
    row_store_rows: list[dict[str, str]],
) -> list[dict]:
    """Match accumulated embedding records to row-store rows and add ``row_id``.

    Embedding records produced during detection have ``filename`` + ``start_sec``
    / ``end_sec`` but no ``row_id``.  The row store (finalized after detection)
    carries ``row_id`` + ``start_utc`` / ``end_utc``.  This function converts the
    filename-relative offsets to UTC, matches against the row store within 0.5 s
    tolerance, and returns records keyed by ``row_id``.
    """
    if not records or not row_store_rows:
        return []

    rs_utc_index: list[tuple[float, float, str]] = []
    for r in row_store_rows:
        s = r.get("start_utc", "")
        e = r.get("end_utc", "")
        rid = r.get("row_id", "")
        if s and e and rid:
            rs_utc_index.append((float(s), float(e), rid))

    matched: list[dict] = []
    for rec in records:
        fname = rec.get("filename", "")
        base = _file_base_epoch(Path(fname)) if fname else 0.0
        emb_start = base + float(rec["start_sec"])
        emb_end = base + float(rec["end_sec"])
        matched_rid = None
        for rs_start, rs_end, rid in rs_utc_index:
            if abs(emb_start - rs_start) <= 0.5 and abs(emb_end - rs_end) <= 0.5:
                matched_rid = rid
                break
        if matched_rid:
            matched.append(
                {
                    "row_id": matched_rid,
                    "embedding": rec["embedding"],
                    "confidence": rec.get("confidence"),
                }
            )
    return matched


def write_detection_embeddings(records: list[dict], path: Path) -> None:
    """Write per-detection embedding records to a Parquet file.

    Schema: ``(row_id, embedding, confidence)``.
    """
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    vector_dim = len(records[0]["embedding"])
    schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("embedding", pa.list_(pa.float32(), vector_dim)),
            ("confidence", pa.float32()),
        ]
    )
    table = pa.table(
        {
            "row_id": [r["row_id"] for r in records],
            "embedding": [r["embedding"] for r in records],
            "confidence": [r.get("confidence") for r in records],
        },
        schema=schema,
    )
    pq.write_table(table, path)


def read_detection_embedding(
    path: Path,
    row_id: str,
) -> list[float] | None:
    """Read a single detection embedding by row_id.

    Returns the embedding vector as a list of floats, or None if not found.
    """
    if not path.exists():
        return None
    table = pq.read_table(str(path))
    col_names = set(table.column_names)
    if "row_id" not in col_names:
        return None  # Legacy schema, no row_id
    row_ids = table.column("row_id").to_pylist()
    for i, rid in enumerate(row_ids):
        if rid == row_id:
            return table.column("embedding")[i].as_py()
    return None


# ---------------------------------------------------------------------------
# Embedding sync: diff row store vs embeddings parquet
# ---------------------------------------------------------------------------


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


def diff_row_store_vs_embeddings(
    row_store_path: Path,
    embeddings_path: Path,
) -> EmbeddingDiffResult:
    """Compare a detection row store against its embeddings parquet.

    Returns an ``EmbeddingDiffResult`` with:
    * ``missing`` -- row-store rows that have no matching embedding.
    * ``orphaned_indices`` -- indices into the embeddings parquet with no
      matching row-store entry (should be removed).
    * ``matched_count`` -- number of row-store rows with a matching embedding.

    Matching is by ``row_id`` -- exact set comparison with no tolerance.
    """
    from humpback.classifier.detection_rows import read_detection_row_store

    _, rows = read_detection_row_store(row_store_path)

    # Read embedding row_ids.
    table = pq.read_table(str(embeddings_path))
    col_names = set(table.column_names)
    if "row_id" not in col_names:
        # Legacy embedding schema — treat all rows as missing.
        return EmbeddingDiffResult(
            missing=rows,
            orphaned_indices=list(range(table.num_rows)),
            matched_count=0,
        )

    emb_row_ids = table.column("row_id").to_pylist()

    # Build set of row_ids from row store.
    rs_row_ids = {r.get("row_id", "") for r in rows}
    emb_row_id_set = set(emb_row_ids)

    # Missing: row store rows not in embeddings.
    missing = [r for r in rows if r.get("row_id", "") not in emb_row_id_set]

    # Orphaned: embedding indices not in row store.
    orphaned_indices = [i for i, rid in enumerate(emb_row_ids) if rid not in rs_row_ids]

    matched_count = len(rows) - len(missing)

    return EmbeddingDiffResult(
        missing=missing,
        orphaned_indices=orphaned_indices,
        matched_count=matched_count,
    )


# ---------------------------------------------------------------------------
# Audio resolution for embedding sync
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


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
    from humpback.processing.audio_io import decode_audio, resample

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

    return None, "no chunk covers the requested window"
