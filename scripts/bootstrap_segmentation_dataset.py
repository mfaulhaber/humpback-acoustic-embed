"""Bootstrap a segmentation training dataset from vocalization-labeled detection rows.

Reads row IDs from a file **or** discovers single-label rows from
detection jobs, resolves each to its detection job's row store, checks
for vocalization labels, computes a UTC crop window centered on the
detection event, and inserts ``SegmentationTrainingSample`` rows with
hydrophone source fields.  Idempotent: rows with an existing
``(training_dataset_id, source_ref=row_id)`` pair are skipped.

Only hydrophone-sourced detection jobs are supported — file-based jobs
are skipped with a warning.

Usage::

    # From explicit row IDs:
    uv run python scripts/bootstrap_segmentation_dataset.py \\
        --row-ids-file rows.txt --dataset-name "bootstrap-v1" \\
        --crop-seconds 10.0 --dry-run

    # From detection job IDs (auto-discovers single-label rows):
    uv run python scripts/bootstrap_segmentation_dataset.py \\
        --detection-job-ids JOB1 JOB2 --dataset-name "bootstrap-v1" \\
        --crop-seconds 10.0 --dry-run

"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


from humpback.classifier.detection_rows import (
    read_detection_row_store,
)
from humpback.config import Settings
from humpback.database import create_engine, create_session_factory
from humpback.models.classifier import DetectionJob
from humpback.models.labeling import VocalizationLabel
from humpback.models.segmentation_training import (
    SegmentationTrainingDataset,
    SegmentationTrainingSample,
)
from humpback.storage import detection_row_store_path

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    dataset_id: str = ""
    inserted: int = 0
    skipped: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def read_row_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids.append(stripped)
    return ids


async def _discover_row_ids_from_jobs(
    session: AsyncSession,
    detection_job_ids: list[str],
) -> list[str]:
    """Discover single-label vocalization-labeled row IDs from detection jobs.

    Matches the event classifier bootstrap filter: only rows with exactly
    one distinct non-negative vocalization type label are included.
    """
    row_ids: list[str] = []
    for dj_id in detection_job_ids:
        dj = await session.get(DetectionJob, dj_id)
        if dj is None:
            logger.warning("detection job %s not found, skipping", dj_id)
            continue

        if not dj.hydrophone_id:
            logger.warning("detection job %s: not a hydrophone job, skipping", dj_id)
            continue

        label_result = await session.execute(
            select(VocalizationLabel).where(
                VocalizationLabel.detection_job_id == dj_id,
                VocalizationLabel.source == "manual",
            )
        )
        labels = list(label_result.scalars().all())

        labels_by_row: dict[str, list[str]] = defaultdict(list)
        for lb in labels:
            labels_by_row[lb.row_id].append(lb.label)

        for row_id, type_labels in labels_by_row.items():
            non_negative = [t for t in type_labels if t != "(Negative)"]
            if not non_negative:
                logger.info("row_id=%s: (Negative) only, skipping", row_id)
                continue
            if len(set(non_negative)) > 1:
                logger.info(
                    "row_id=%s: %d distinct type labels, skipping",
                    row_id,
                    len(set(non_negative)),
                )
                continue
            row_ids.append(row_id)

    logger.info(
        "Discovered %d single-label rows from %d detection jobs",
        len(row_ids),
        len(detection_job_ids),
    )
    return row_ids


async def _get_or_create_dataset(
    session: AsyncSession,
    *,
    dataset_name: str | None,
    dataset_id: str | None,
) -> SegmentationTrainingDataset:
    if dataset_id is not None:
        ds = await session.get(SegmentationTrainingDataset, dataset_id)
        if ds is None:
            raise SystemExit(f"ERROR: dataset {dataset_id!r} not found")
        return ds
    assert dataset_name is not None
    ds = SegmentationTrainingDataset(name=dataset_name)
    session.add(ds)
    await session.flush()
    return ds


async def run_bootstrap(
    session: AsyncSession,
    *,
    row_ids: list[str],
    dataset_name: str | None,
    dataset_id: str | None,
    crop_seconds: float,
    allow_multi_label: bool,
    dry_run: bool,
    storage_root: Path,
) -> BootstrapResult:
    """Core bootstrap logic — testable without argparse."""
    result = BootstrapResult()

    ds = await _get_or_create_dataset(
        session, dataset_name=dataset_name, dataset_id=dataset_id
    )
    result.dataset_id = ds.id

    detection_job_cache: dict[str, DetectionJob] = {}
    row_store_cache: dict[str, dict[str, dict[str, str]]] = {}

    for row_id in row_ids:
        skip_reason = await _process_one_row(
            session,
            row_id=row_id,
            dataset_id=ds.id,
            crop_seconds=crop_seconds,
            allow_multi_label=allow_multi_label,
            dry_run=dry_run,
            storage_root=storage_root,
            detection_job_cache=detection_job_cache,
            row_store_cache=row_store_cache,
        )
        if skip_reason is not None:
            result.skipped[skip_reason] += 1
        else:
            result.inserted += 1

    if dry_run:
        await session.rollback()
    else:
        await session.commit()

    return result


async def _process_one_row(
    session: AsyncSession,
    *,
    row_id: str,
    dataset_id: str,
    crop_seconds: float,
    allow_multi_label: bool,
    dry_run: bool,
    storage_root: Path,
    detection_job_cache: dict[str, DetectionJob],
    row_store_cache: dict[str, dict[str, dict[str, str]]],
) -> str | None:
    """Process one row_id. Returns skip reason or None on success."""

    # 1. Look up human-annotated vocalization labels (skip inference results)
    label_result = await session.execute(
        select(VocalizationLabel).where(
            VocalizationLabel.row_id == row_id,
            VocalizationLabel.source == "manual",
        )
    )
    labels = list(label_result.scalars().all())
    if not labels:
        logger.warning("row_id=%s: no vocalization labels, skipping", row_id)
        return "no vocalization label"

    distinct_types = {lb.label for lb in labels if lb.label != "(Negative)"}
    if len(distinct_types) > 1 and not allow_multi_label:
        logger.info(
            "row_id=%s: %d distinct type labels, skipping (use --allow-multi-label)",
            row_id,
            len(distinct_types),
        )
        return "multi-label"

    detection_job_id = labels[0].detection_job_id

    # 2. Load detection job and verify it is hydrophone-sourced
    if detection_job_id not in detection_job_cache:
        dj = await session.get(DetectionJob, detection_job_id)
        if dj is None:
            logger.warning(
                "row_id=%s: detection job %s not found, skipping",
                row_id,
                detection_job_id,
            )
            return "detection job not found"
        detection_job_cache[detection_job_id] = dj
    dj = detection_job_cache[detection_job_id]

    if not dj.hydrophone_id:
        logger.warning(
            "row_id=%s: detection job %s is not hydrophone-sourced, skipping",
            row_id,
            detection_job_id,
        )
        return "not hydrophone source"

    if dj.start_timestamp is None or dj.end_timestamp is None:
        logger.warning(
            "row_id=%s: detection job %s missing start/end timestamps, skipping",
            row_id,
            detection_job_id,
        )
        return "missing job timestamps"

    # 3. Load row store and find the row
    if detection_job_id not in row_store_cache:
        rs_path = Path(dj.output_row_store_path) if dj.output_row_store_path else None
        if rs_path is None or not rs_path.exists():
            rs_path = detection_row_store_path(storage_root, detection_job_id)
        if not rs_path.exists():
            logger.warning(
                "row_id=%s: row store not found for job %s, skipping",
                row_id,
                detection_job_id,
            )
            return "row store not found"
        _, rows = read_detection_row_store(rs_path)
        row_store_cache[detection_job_id] = {r.get("row_id", ""): r for r in rows}

    row_index = row_store_cache[detection_job_id]
    det_row = row_index.get(row_id)
    if det_row is None:
        logger.warning("row_id=%s: not found in row store, skipping", row_id)
        return "detection row not found"

    # 4. Compute crop bounds (absolute UTC)
    start_utc = float(det_row.get("start_utc") or 0)
    end_utc = float(det_row.get("end_utc") or 0)
    center = (start_utc + end_utc) / 2.0
    half = crop_seconds / 2.0
    crop_start = max(dj.start_timestamp, center - half)
    crop_end = min(dj.end_timestamp, center + half)
    crop_duration = crop_end - crop_start

    if crop_duration < crop_seconds * 0.5:
        logger.info(
            "row_id=%s: crop too short at boundary (%.2f s < %.2f s minimum), skipping",
            row_id,
            crop_duration,
            crop_seconds * 0.5,
        )
        return "crop too short at boundary"

    # 5. Check idempotency
    existing = await session.execute(
        select(SegmentationTrainingSample.id).where(
            SegmentationTrainingSample.training_dataset_id == dataset_id,
            SegmentationTrainingSample.source_ref == row_id,
        )
    )
    if existing.scalar_one_or_none() is not None:
        logger.info("row_id=%s: already present in dataset, skipping", row_id)
        return "already present"

    # 6. Build events_json (relative to crop start)
    event_start_rel = start_utc - crop_start
    event_end_rel = end_utc - crop_start
    events_json = json.dumps([{"start_sec": event_start_rel, "end_sec": event_end_rel}])

    # 7. Insert
    sample = SegmentationTrainingSample(
        training_dataset_id=dataset_id,
        hydrophone_id=dj.hydrophone_id,
        start_timestamp=crop_start,
        end_timestamp=crop_end,
        crop_start_sec=0.0,
        crop_end_sec=crop_duration,
        events_json=events_json,
        source="bootstrap_vocalization_row",
        source_ref=row_id,
    )
    if not dry_run:
        session.add(sample)
        await session.flush()
    else:
        logger.info(
            "row_id=%s: would insert (crop [%.2f, %.2f] UTC, hydrophone=%s)",
            row_id,
            crop_start,
            crop_end,
            dj.hydrophone_id,
        )

    return None


def _print_summary(result: BootstrapResult, dry_run: bool) -> None:
    mode = "[DRY RUN] " if dry_run else ""
    print(f"\n{mode}Bootstrap summary")
    print(f"  Dataset ID: {result.dataset_id}")
    print(f"  Inserted:   {result.inserted}")
    total_skipped = sum(result.skipped.values())
    print(f"  Skipped:    {total_skipped}")
    if result.skipped:
        for reason, count in sorted(result.skipped.items()):
            print(f"    - {reason}: {count}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a segmentation training dataset from detection rows."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--row-ids-file",
        type=Path,
        help="Text file with one row_id per line (blank lines and # comments skipped).",
    )
    source_group.add_argument(
        "--detection-job-ids",
        nargs="+",
        help="One or more detection job IDs to discover single-label rows from.",
    )
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset-name",
        type=str,
        help="Create a new dataset with this name.",
    )
    dataset_group.add_argument(
        "--dataset-id",
        type=str,
        help="Add samples to an existing dataset by ID.",
    )
    parser.add_argument(
        "--crop-seconds",
        type=float,
        default=10.0,
        help="Crop duration in seconds (default 10.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print summary without committing.",
    )
    parser.add_argument(
        "--allow-multi-label",
        action="store_true",
        help="Include rows with >1 distinct vocalization type label (only with --row-ids-file).",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> None:
    load_dotenv()
    settings = Settings()

    engine = create_engine(settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            if args.detection_job_ids is not None:
                row_ids = await _discover_row_ids_from_jobs(
                    session, args.detection_job_ids
                )
                if not row_ids:
                    raise SystemExit(
                        "ERROR: no single-label rows found in the given detection jobs"
                    )
                print(
                    f"Discovered {len(row_ids)} single-label rows "
                    f"from {len(args.detection_job_ids)} detection jobs"
                )
                allow_multi_label = False
            else:
                if not args.row_ids_file.exists():
                    raise SystemExit(
                        f"ERROR: row ids file not found: {args.row_ids_file}"
                    )
                row_ids = read_row_ids(args.row_ids_file)
                if not row_ids:
                    raise SystemExit("ERROR: no row IDs found in file")
                print(f"Loaded {len(row_ids)} row IDs from {args.row_ids_file}")
                allow_multi_label = args.allow_multi_label

            result = await run_bootstrap(
                session,
                row_ids=row_ids,
                dataset_name=args.dataset_name,
                dataset_id=args.dataset_id,
                crop_seconds=args.crop_seconds,
                allow_multi_label=allow_multi_label,
                dry_run=args.dry_run,
                storage_root=settings.storage_root,
            )
        _print_summary(result, dry_run=args.dry_run)
    finally:
        await engine.dispose()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
