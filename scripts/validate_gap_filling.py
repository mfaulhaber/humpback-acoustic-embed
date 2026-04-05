"""Validate prominence gap-filling spacing for a detection job.

Reads window diagnostics from an existing job, re-runs the prominence peak
selection + gap-filling algorithm with the current code, and compares results
against the job's stored detection rows.

Usage:
    uv run python scripts/validate_gap_filling.py JOB_ID START_UTC END_UTC

    # Validate job 4ae10fc6 from 02:11:00 to 02:14:00
    uv run python scripts/validate_gap_filling.py \
        4ae10fc6-9083-4366-b5d7-02c64c5f3098 02:11:00 02:14:00

Times are HH:MM:SS within the detection job's date (inferred from data).
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from humpback.classifier.detector_utils import (  # noqa: E402
    merge_detection_events,
    select_prominent_peaks_from_events,
    snap_and_merge_detection_events,
)
from humpback.config import Settings  # noqa: E402

_TS_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(?:\.(\d+))?Z")


def _parse_file_epoch(filename: str) -> float:
    m = _TS_PATTERN.search(Path(filename).name)
    if not m:
        return 0.0
    y, mo, d, h, mi, s = (int(g) for g in m.groups()[:6])
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc).timestamp()


def _fmt_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%H:%M:%S")


def _to_logit(p: float) -> float:
    p = max(1e-7, min(1.0 - 1e-7, p))
    return math.log(p / (1.0 - p))


def _analyze_detections(
    label: str,
    detections: list[dict],
    windows: list[dict],
) -> None:
    """Print spacing analysis for a set of detections."""
    print(f"  Detections: {len(detections)}")

    if len(detections) < 2:
        print()
        return

    spacings = [
        detections[i + 1]["start"] - detections[i]["start"]
        for i in range(len(detections) - 1)
    ]
    min_sp = min(spacings)
    max_sp = max(spacings)
    avg_sp = sum(spacings) / len(spacings)

    print(f"  Min spacing:  {min_sp:.1f}s")
    print(f"  Max spacing:  {max_sp:.1f}s")
    print(f"  Avg spacing:  {avg_sp:.1f}s")

    dense = [(i, spacings[i]) for i in range(len(spacings)) if spacings[i] < 3.0]
    large = [(i, spacings[i]) for i in range(len(spacings)) if spacings[i] > 10.0]
    print(f"  Dense pairs (< 3s): {len(dense)}")
    print(f"  Large gaps (> 10s): {len(large)}")
    print()

    # Histogram
    buckets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 60, 999]
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        lbl = f"{lo}-{hi}s" if hi < 999 else f"{lo}s+"
        c = sum(1 for s in spacings if lo <= s < hi)
        if c > 0:
            print(f"    {lbl:>8s}  {c:3d}  {'█' * c}")
    print()


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    from dotenv import load_dotenv

    load_dotenv()

    job_id = sys.argv[1]
    start_hms = sys.argv[2]
    end_hms = sys.argv[3]

    settings = Settings()
    job_dir = settings.storage_root / "detections" / job_id

    if not job_dir.exists():
        print(f"Job directory not found: {job_dir}")
        sys.exit(1)

    # --- Read job parameters ---
    summary_path = job_dir / "run_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    high_threshold = float(summary.get("high_threshold", 0.9))
    low_threshold = float(summary.get("low_threshold", 0.8))
    window_selection = summary.get("window_selection", "prominence")
    min_prominence = float(summary.get("min_prominence", 1.0))
    window_size_seconds = 5.0  # standard for this project

    if window_selection != "prominence":
        print(
            f"Job uses window_selection='{window_selection}', not prominence. Nothing to validate."
        )
        sys.exit(0)

    # --- Read detection rows (stored results) ---
    rows_path = job_dir / "detection_rows.parquet"
    rows_table = pq.read_table(str(rows_path))
    rows = rows_table.to_pydict()
    n_rows = len(rows["start_utc"])

    # Infer date from first detection to resolve HH:MM:SS to epoch
    first_epoch = float(rows["start_utc"][0])
    base_date = datetime.fromtimestamp(first_epoch, tz=timezone.utc).date()

    def hms_to_epoch(hms: str) -> float:
        parts = hms.split(":")
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        dt = datetime(
            base_date.year, base_date.month, base_date.day, h, m, s, tzinfo=timezone.utc
        )
        return dt.timestamp()

    range_start = hms_to_epoch(start_hms)
    range_end = hms_to_epoch(end_hms)

    # Filter stored detections to time range
    stored_detections: list[dict] = []
    for i in range(n_rows):
        s = float(rows["start_utc"][i])
        e = float(rows["end_utc"][i])
        conf = float(rows["avg_confidence"][i])
        if s >= range_start and e <= range_end:
            stored_detections.append({"start": s, "end": e, "conf": conf})
    stored_detections.sort(key=lambda d: d["start"])

    # --- Read window diagnostics ---
    diag_path = job_dir / "window_diagnostics.parquet"
    diag_table = pq.read_table(str(diag_path))
    diag = diag_table.to_pydict()
    n_diag = len(diag["filename"])

    # Build per-file window records with file-relative offsets (for the algorithm)
    # and UTC times (for display). Group by file since events are per-file.
    file_epoch_cache: dict[str, float] = {}
    file_windows: dict[str, list[dict]] = {}
    all_utc_windows: list[dict] = []

    for i in range(n_diag):
        fname = diag["filename"][i]
        if fname not in file_epoch_cache:
            file_epoch_cache[fname] = _parse_file_epoch(fname)
        base = file_epoch_cache[fname]
        if base == 0.0:
            continue
        offset_sec = float(diag["offset_sec"][i])
        end_sec = float(diag["end_sec"][i])
        conf = float(diag["confidence"][i])
        w_start_utc = base + offset_sec

        # Include windows in extended range (events may start before range)
        if w_start_utc < range_start - 60 or w_start_utc > range_end + 60:
            continue

        rec = {
            "offset_sec": offset_sec,
            "end_sec": end_sec,
            "confidence": conf,
        }
        file_windows.setdefault(fname, []).append(rec)
        all_utc_windows.append({"start": w_start_utc, "conf": conf})

    all_utc_windows.sort(key=lambda w: w["start"])

    # --- Re-run prominence selection per file ---
    rerun_detections_utc: list[dict] = []

    for fname, w_recs in sorted(file_windows.items()):
        base = file_epoch_cache[fname]
        w_recs.sort(key=lambda r: r["offset_sec"])

        # Build events via hysteresis merge (same as detector pipeline)
        events = merge_detection_events(w_recs, high_threshold, low_threshold)
        events = snap_and_merge_detection_events(events, window_size_seconds)

        if not events:
            continue

        # Run prominence selection with current code (includes gap-filling)
        peak_dets = select_prominent_peaks_from_events(
            events,
            w_recs,
            window_size_seconds,
            min_score=high_threshold,
            min_prominence=min_prominence,
        )

        # Convert to UTC
        for det in peak_dets:
            det_start_utc = base + det["start_sec"]
            det_end_utc = base + det["end_sec"]
            if det_start_utc >= range_start and det_end_utc <= range_end:
                rerun_detections_utc.append(
                    {
                        "start": det_start_utc,
                        "end": det_end_utc,
                        "conf": det["avg_confidence"],
                    }
                )

    rerun_detections_utc.sort(key=lambda d: d["start"])

    # --- Output ---
    print(f"Job: {job_id}")
    print(f"Range: {start_hms} - {end_hms}")
    print(
        f"Params: high={high_threshold} low={low_threshold} min_prominence={min_prominence}"
    )
    print(f"Window scores in range: {len(all_utc_windows)}")
    print()

    print("=" * 72)
    print("STORED (original job results)")
    print("=" * 72)
    _analyze_detections("stored", stored_detections, all_utc_windows)

    print("=" * 72)
    print("RE-RUN (current code)")
    print("=" * 72)
    _analyze_detections("rerun", rerun_detections_utc, all_utc_windows)

    # --- Side-by-side detection list ---
    print("=" * 72)
    print("SIDE-BY-SIDE DETECTIONS")
    print("=" * 72)

    stored_set = {round(d["start"], 1) for d in stored_detections}
    rerun_set = {round(d["start"], 1) for d in rerun_detections_utc}

    all_times = sorted(stored_set | rerun_set)
    stored_map = {round(d["start"], 1): d for d in stored_detections}
    rerun_map = {round(d["start"], 1): d for d in rerun_detections_utc}

    for t in all_times:
        in_stored = "●" if t in stored_set else " "
        in_rerun = "●" if t in rerun_set else " "
        s_conf = f"{stored_map[t]['conf']:.3f}" if t in stored_map else "     "
        r_conf = f"{rerun_map[t]['conf']:.3f}" if t in rerun_map else "     "
        status = ""
        if t in stored_set and t not in rerun_set:
            status = " REMOVED"
        elif t not in stored_set and t in rerun_set:
            status = " ADDED"
        print(
            f"  {_fmt_utc(t)}  stored={in_stored} {s_conf}  rerun={in_rerun} {r_conf}{status}"
        )
    print()

    # Summary
    only_stored = stored_set - rerun_set
    only_rerun = rerun_set - stored_set
    both = stored_set & rerun_set
    print(f"  Kept:    {len(both)}")
    print(f"  Removed: {len(only_stored)}")
    print(f"  Added:   {len(only_rerun)}")


if __name__ == "__main__":
    main()
