#!/usr/bin/env python3
"""
Generate detection job payloads from NOAA SanctSound humpback detection metadata.

Fetches (or reads locally) a daily presence/absence CSV, filters for days with
confirmed humpback presence, groups consecutive presence days into job ranges,
and outputs a JSON file of payloads ready to POST to the hydrophone detection API.

Example — generate jobs from OC01 deployment 03 CSV (one per presence day):
    uv run python scripts/noaa_detection_metadata.py \\
        --csv-url https://storage.googleapis.com/noaa-passive-bioacoustic/sanctsound/products/detections/oc01/sanctsound_oc01_03_humpbackwhale_1d/data/SanctSound_OC01_03_humpbackwhale_1d.csv \\
        --hydrophone-id sanctsound_oc01 \\
        --classifier-model-id <uuid>

Example — consolidate up to 7 consecutive presence days per job:
    uv run python scripts/noaa_detection_metadata.py \\
        --csv-url <url> --hydrophone-id <id> \\
        --classifier-model-id <uuid> --days-per-job 7

Example — use a local CSV file:
    uv run python scripts/noaa_detection_metadata.py \\
        --csv-path /path/to/detections.csv \\
        --hydrophone-id sanctsound_ci01 \\
        --classifier-model-id <uuid>

Example — post job #3 from the generated file:
    uv run python scripts/noaa_detection_metadata.py \\
        --post --job-index 3 --output detection_jobs.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_MAX_DAYS_PER_JOB = 7  # API constraint


@dataclass(frozen=True)
class PresenceDay:
    date: date
    deployment: str
    source: str


def parse_noaa_detection_csv(
    content: str, deployment: str = "01", source: str = ""
) -> list[PresenceDay]:
    """Parse NOAA detection CSV and return days with Presence=1."""
    # Strip BOM and normalize line endings (NOAA CSVs may have \ufeff + \r\n)
    clean = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    reader = csv.DictReader(io.StringIO(clean))
    # Normalize column names to lowercase — NOAA CSVs vary:
    # CI01 uses "ISOStartTime", OC01 uses "IsoStartTime"
    if reader.fieldnames is not None:
        reader.fieldnames = [f.lower() for f in reader.fieldnames]
    days: list[PresenceDay] = []
    for row in reader:
        iso_time = row.get("isostarttime", "").strip()
        presence_raw = row.get("presence", "").strip()
        if not iso_time or not presence_raw:
            continue
        try:
            presence = int(presence_raw)
        except ValueError:
            continue
        if presence != 1:
            continue
        # Parse ISO timestamp to date (always UTC)
        dt_str = iso_time.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(dt_str)
        except ValueError:
            continue
        days.append(PresenceDay(date=dt.date(), deployment=deployment, source=source))
    days.sort(key=lambda d: d.date)
    return days


def group_consecutive_days(days: list[PresenceDay]) -> list[list[PresenceDay]]:
    """Group presence days into runs of consecutive dates."""
    if not days:
        return []
    sorted_days = sorted(days, key=lambda d: d.date)
    groups: list[list[PresenceDay]] = [[sorted_days[0]]]
    for day in sorted_days[1:]:
        prev = groups[-1][-1]
        if day.date == prev.date + timedelta(days=1):
            groups[-1].append(day)
        else:
            groups.append([day])
    return groups


def split_into_job_ranges(
    groups: list[list[PresenceDay]], days_per_job: int = 1
) -> list[tuple[date, date, list[PresenceDay]]]:
    """Split grouped runs into job ranges respecting days_per_job limit.

    Returns (start_date, end_date_inclusive, days_in_chunk) tuples.
    """
    capped = min(days_per_job, _MAX_DAYS_PER_JOB)
    if capped < 1:
        capped = 1
    ranges: list[tuple[date, date, list[PresenceDay]]] = []
    for group in groups:
        for i in range(0, len(group), capped):
            chunk = group[i : i + capped]
            ranges.append((chunk[0].date, chunk[-1].date, chunk))
    return ranges


def _date_to_epoch(d: date) -> float:
    """Convert a UTC date to epoch seconds at 00:00:00 UTC."""
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()


def build_job_payloads(
    ranges: list[tuple[date, date, list[PresenceDay]]],
    classifier_model_id: str,
    hydrophone_id: str = "sanctsound_ci01",
    hop_seconds: float = 1.0,
    high_threshold: float = 0.70,
    low_threshold: float = 0.45,
    detection_mode: str | None = "windowed",
) -> list[dict]:
    """Build API-ready job payloads from date ranges."""
    payloads: list[dict] = []
    for start_date, end_date, chunk_days in ranges:
        start_ts = _date_to_epoch(start_date)
        # end_timestamp is start of day AFTER last date (exclusive)
        end_ts = _date_to_epoch(end_date + timedelta(days=1))
        payload: dict = {
            "classifier_model_id": classifier_model_id,
            "hydrophone_id": hydrophone_id,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "hop_seconds": hop_seconds,
            "high_threshold": high_threshold,
            "low_threshold": low_threshold,
        }
        if detection_mode is not None:
            payload["detection_mode"] = detection_mode
        payload["_metadata"] = {
            "index": len(payloads),
            "presence_days": len(chunk_days),
            "deployment": chunk_days[0].deployment if chunk_days else "01",
            "date_range": (
                f"{start_date.isoformat()} to {end_date.isoformat()}"
                if start_date != end_date
                else start_date.isoformat()
            ),
        }
        payloads.append(payload)
    return payloads


def fetch_csv_from_gcs(url: str) -> str | None:
    """Fetch CSV content from a public GCS URL. Returns None on error."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception as exc:
        print(f"ERROR: Failed to fetch {url}: {exc}", file=sys.stderr)
        return None


def post_job_payload(api_url: str, payload: dict) -> dict:
    """POST a single job payload to the detection API."""
    url = f"{api_url.rstrip('/')}/classifier/hydrophone-detection-jobs"
    # Strip metadata before sending
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    data = json.dumps(clean).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc


def load_and_post_job(json_path: str, job_index: int, api_url: str) -> dict:
    """Load output JSON, extract job at index, and POST it."""
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Output file not found: {json_path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    jobs = data.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("No jobs found in output file")
    if job_index < 0 or job_index >= len(jobs):
        raise IndexError(f"job-index {job_index} out of range (0..{len(jobs) - 1})")
    payload = jobs[job_index]
    return post_job_payload(api_url, payload)


def resolve_model_id_by_name(api_url: str, name: str) -> str | None:
    """Look up a classifier model UUID by name via the API."""
    url = f"{api_url.rstrip('/')}/classifier/models"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            models = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"ERROR: Failed to fetch models from {url}: {exc}", file=sys.stderr)
        return None
    for m in models:
        if m.get("name") == name:
            return m["id"]
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate detection job payloads from NOAA SanctSound "
            "humpback whale detection metadata."
        ),
    )
    parser.add_argument(
        "--classifier-model-id",
        default=None,
        help="UUID of trained classifier model",
    )
    parser.add_argument(
        "--classifier-model-name",
        default=None,
        help="Name of trained classifier model (resolved to ID via API, e.g. lr-v17)",
    )
    parser.add_argument(
        "--hydrophone-id",
        default=None,
        help="Archive source ID (e.g. sanctsound_oc01)",
    )
    parser.add_argument(
        "--csv-url",
        default=None,
        help="URL of NOAA detection CSV (GCS or other HTTPS URL)",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Local CSV file path (alternative to --csv-url)",
    )
    parser.add_argument(
        "--deployment",
        default="01",
        help="Deployment identifier for metadata (default: 01)",
    )
    parser.add_argument(
        "--days-per-job",
        type=int,
        default=1,
        help=(
            "Max consecutive presence days per job "
            "(default: 1; max: 7 per API constraint)"
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=["consecutive", "daily", "full-range"],
        default="consecutive",
        help="Grouping strategy (default: consecutive)",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.70,
        help="Hysteresis high threshold (default: 0.70)",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.45,
        help="Hysteresis low threshold (default: 0.45)",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=1.0,
        help="Detection hop stride (default: 1.0)",
    )
    parser.add_argument(
        "--detection-mode",
        choices=["merged", "windowed"],
        default="windowed",
        help="Detection mode (default: windowed)",
    )
    parser.add_argument(
        "--output",
        default="detection_jobs.json",
        help="Output JSON file path (default: detection_jobs.json)",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="POST a single job to API (requires --job-index)",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="0-based index into the jobs array to POST (used with --post)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL for --post (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing output file",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # POST mode: read existing JSON and submit one job
    if args.post:
        if args.job_index is None:
            print("ERROR: --post requires --job-index", file=sys.stderr)
            return 2
        try:
            result = load_and_post_job(args.output, args.job_index, args.api_url)
        except (FileNotFoundError, ValueError, IndexError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"ERROR: POST failed: {exc}", file=sys.stderr)
            return 1
        print(f"Job created: {result.get('id', '(unknown)')}")
        print(f"Status: {result.get('status', '(unknown)')}")
        print(json.dumps(result, indent=2))
        return 0

    # Resolve classifier model ID
    classifier_model_id = args.classifier_model_id
    if args.classifier_model_name and not classifier_model_id:
        resolved = resolve_model_id_by_name(args.api_url, args.classifier_model_name)
        if resolved is None:
            print(
                f"ERROR: Classifier model not found by name: {args.classifier_model_name}",
                file=sys.stderr,
            )
            return 1
        classifier_model_id = resolved
        print(f"Resolved model '{args.classifier_model_name}' -> {classifier_model_id}")
    if not classifier_model_id:
        print(
            "ERROR: --classifier-model-id or --classifier-model-name is required",
            file=sys.stderr,
        )
        return 2

    # Validate CSV source and hydrophone-id
    if not args.csv_path and not args.csv_url:
        print(
            "ERROR: --csv-url or --csv-path is required",
            file=sys.stderr,
        )
        return 2
    if not args.hydrophone_id:
        print(
            "ERROR: --hydrophone-id is required",
            file=sys.stderr,
        )
        return 2

    # Fetch CSV
    if args.csv_path:
        csv_path = Path(args.csv_path)
        if not csv_path.is_file():
            print(f"ERROR: CSV file not found: {args.csv_path}", file=sys.stderr)
            return 1
        csv_content = csv_path.read_text(encoding="utf-8")
        source_label = str(csv_path)
    else:
        print(f"Fetching detection metadata from {args.csv_url} ...")
        csv_content = fetch_csv_from_gcs(args.csv_url)
        if csv_content is None:
            return 1
        source_label = args.csv_url

    # Parse
    presence_days = parse_noaa_detection_csv(
        csv_content, deployment=args.deployment, source=source_label
    )
    if not presence_days:
        print("No presence days found in CSV.")
        return 0

    # Group
    if args.strategy == "daily":
        # One job per presence day regardless of consecutiveness
        groups = [[d] for d in presence_days]
    elif args.strategy == "full-range":
        # Single group spanning all days
        groups = [presence_days]
    else:
        # consecutive (default)
        groups = group_consecutive_days(presence_days)

    # Split into job ranges
    ranges = split_into_job_ranges(groups, days_per_job=args.days_per_job)

    # Build payloads
    payloads = build_job_payloads(
        ranges,
        classifier_model_id=classifier_model_id,
        hydrophone_id=args.hydrophone_id,
        hop_seconds=args.hop_seconds,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        detection_mode=args.detection_mode,
    )

    # Summary
    date_min = presence_days[0].date.isoformat()
    date_max = presence_days[-1].date.isoformat()
    print(f"Source: {source_label}")
    print(f"Presence days: {len(presence_days)} ({date_min} to {date_max})")
    print(f"Strategy: {args.strategy}, days-per-job: {args.days_per_job}")
    print(f"Jobs generated: {len(payloads)}")
    for p in payloads:
        meta = p.get("_metadata", {})
        print(
            f"  [{meta.get('index', '?')}] {meta.get('date_range', '?')} "
            f"({meta.get('presence_days', '?')} day(s))"
        )

    if args.dry_run:
        print("Dry run — no output file written.")
        return 0

    # Write output
    output_path = Path(args.output)
    envelope = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "NOAA SanctSound humpback whale detection metadata",
        "hydrophone_id": args.hydrophone_id,
        "strategy": args.strategy,
        "days_per_job": args.days_per_job,
        "total_presence_days": len(presence_days),
        "total_jobs": len(payloads),
        "deployment": args.deployment,
        "jobs": payloads,
    }
    output_path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    print(f"Output written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
