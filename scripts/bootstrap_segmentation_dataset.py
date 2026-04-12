"""Bootstrap a segmentation training dataset from vocalization-labeled detection rows.

Reads row IDs from a file, resolves each to its detection job's row
store, checks for vocalization labels, computes an audio-relative crop
window centered on the detection event, and inserts
``SegmentationTrainingSample`` rows. Idempotent: rows with an existing
``(training_dataset_id, source_ref=row_id)`` pair are skipped.

Usage::

    uv run python scripts/bootstrap_segmentation_dataset.py \\
        --row-ids-file rows.txt --dataset-name "bootstrap-v1" \\
        --crop-seconds 10.0 --dry-run

"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.detection_rows import (
    parse_recording_timestamp,
    read_detection_row_store,
)
from humpback.config import Settings
from humpback.database import create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.classifier import DetectionJob
from humpback.models.labeling import VocalizationLabel
from humpback.models.segmentation_training import (
    SegmentationTrainingDataset,
    SegmentationTrainingSample,
)
from humpback.storage import detection_row_store_path

logger = logging.getLogger(__name__)

_KNOWN_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".aif", ".aiff"}


@dataclass
class BootstrapResult:
    dataset_id: str = ""
    inserted: int = 0
    skipped: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _file_base_epoch(filepath: Path) -> float:
    ts = parse_recording_timestamp(filepath.name)
    if ts is not None:
        return ts.timestamp()
    try:
        return os.path.getmtime(filepath)
    except OSError:
        return 0.0


def _build_audio_index(audio_folder: Path) -> list[tuple[float, Path]]:
    index: list[tuple[float, Path]] = []
    for child in sorted(audio_folder.iterdir()):
        if child.suffix.lower() in _KNOWN_AUDIO_EXTENSIONS and child.is_file():
            index.append((_file_base_epoch(child), child))
    index.sort(key=lambda x: x[0])
    return index


def _resolve_file_for_row(
    start_utc: float, audio_index: list[tuple[float, Path]]
) -> tuple[Path, float] | None:
    """Return ``(file_path, base_epoch)`` for the file covering ``start_utc``."""
    best: tuple[Path, float] | None = None
    for base_epoch, path in audio_index:
        if base_epoch <= start_utc + 1e-6:
            best = (path, base_epoch)
        else:
            break
    return best


def read_row_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids.append(stripped)
    return ids


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
    audio_index_cache: dict[str, list[tuple[float, Path]]] = {}
    audio_file_cache: dict[str, AudioFile | None] = {}

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
            audio_index_cache=audio_index_cache,
            audio_file_cache=audio_file_cache,
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
    audio_index_cache: dict[str, list[tuple[float, Path]]],
    audio_file_cache: dict[str, AudioFile | None],
) -> str | None:
    """Process one row_id. Returns skip reason or None on success."""

    # 1. Look up vocalization labels
    label_result = await session.execute(
        select(VocalizationLabel).where(VocalizationLabel.row_id == row_id)
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

    # 2. Load detection job
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

    if dj.hydrophone_id:
        logger.warning(
            "row_id=%s: hydrophone detection jobs not yet supported, skipping",
            row_id,
        )
        return "hydrophone source (not supported)"

    if not dj.audio_folder:
        logger.warning(
            "row_id=%s: detection job %s has no audio_folder, skipping",
            row_id,
            detection_job_id,
        )
        return "no audio source"

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

    # 4. Resolve audio file
    audio_folder = Path(dj.audio_folder)
    if str(audio_folder) not in audio_index_cache:
        if not audio_folder.exists():
            logger.warning(
                "row_id=%s: audio folder %s does not exist, skipping",
                row_id,
                audio_folder,
            )
            return "audio folder missing"
        audio_index_cache[str(audio_folder)] = _build_audio_index(audio_folder)

    audio_index = audio_index_cache[str(audio_folder)]
    start_utc = float(det_row.get("start_utc") or 0)
    end_utc = float(det_row.get("end_utc") or 0)

    resolved = _resolve_file_for_row(start_utc, audio_index)
    if resolved is None:
        logger.warning("row_id=%s: cannot resolve audio file, skipping", row_id)
        return "audio file not resolved"
    file_path, base_epoch = resolved

    # 5. Find AudioFile row
    af_cache_key = f"{audio_folder}:{file_path.name}"
    if af_cache_key not in audio_file_cache:
        af_result = await session.execute(
            select(AudioFile).where(
                AudioFile.source_folder == str(audio_folder),
                AudioFile.filename == file_path.name,
            )
        )
        audio_file_cache[af_cache_key] = af_result.scalar_one_or_none()
    af = audio_file_cache[af_cache_key]
    if af is None:
        logger.warning(
            "row_id=%s: no AudioFile for %s in %s, skipping",
            row_id,
            file_path.name,
            audio_folder,
        )
        return "audio file row not found"

    audio_duration = af.duration_seconds
    if audio_duration is None or audio_duration <= 0:
        logger.warning(
            "row_id=%s: AudioFile %s has no duration, skipping", row_id, af.id
        )
        return "audio duration unknown"

    # 6. Compute crop bounds
    start_sec = start_utc - base_epoch
    end_sec = end_utc - base_epoch
    center = (start_sec + end_sec) / 2.0
    half = crop_seconds / 2.0
    crop_start = max(0.0, center - half)
    crop_end = min(audio_duration, center + half)

    if (crop_end - crop_start) < crop_seconds * 0.5:
        logger.info(
            "row_id=%s: crop too short at boundary (%.2f s < %.2f s minimum), skipping",
            row_id,
            crop_end - crop_start,
            crop_seconds * 0.5,
        )
        return "crop too short at boundary"

    # 7. Check idempotency
    existing = await session.execute(
        select(SegmentationTrainingSample.id).where(
            SegmentationTrainingSample.training_dataset_id == dataset_id,
            SegmentationTrainingSample.source_ref == row_id,
        )
    )
    if existing.scalar_one_or_none() is not None:
        logger.info("row_id=%s: already present in dataset, skipping", row_id)
        return "already present"

    # 8. Build events_json (audio-relative)
    events_json = json.dumps([{"start_sec": start_sec, "end_sec": end_sec}])

    # 9. Insert
    sample = SegmentationTrainingSample(
        training_dataset_id=dataset_id,
        audio_file_id=af.id,
        crop_start_sec=crop_start,
        crop_end_sec=crop_end,
        events_json=events_json,
        source="bootstrap_vocalization_row",
        source_ref=row_id,
    )
    if not dry_run:
        session.add(sample)
        await session.flush()
    else:
        logger.info(
            "row_id=%s: would insert (crop [%.2f, %.2f], audio_file=%s)",
            row_id,
            crop_start,
            crop_end,
            af.id,
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
    parser.add_argument(
        "--row-ids-file",
        type=Path,
        required=True,
        help="Text file with one row_id per line (blank lines and # comments skipped).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset-name",
        type=str,
        help="Create a new dataset with this name.",
    )
    group.add_argument(
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
        help="Include rows with >1 distinct vocalization type label.",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> None:
    load_dotenv()
    settings = Settings()

    if not args.row_ids_file.exists():
        raise SystemExit(f"ERROR: row ids file not found: {args.row_ids_file}")

    row_ids = read_row_ids(args.row_ids_file)
    if not row_ids:
        raise SystemExit("ERROR: no row IDs found in file")

    print(f"Loaded {len(row_ids)} row IDs from {args.row_ids_file}")

    engine = create_engine(settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            result = await run_bootstrap(
                session,
                row_ids=row_ids,
                dataset_name=args.dataset_name,
                dataset_id=args.dataset_id,
                crop_seconds=args.crop_seconds,
                allow_multi_label=args.allow_multi_label,
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
