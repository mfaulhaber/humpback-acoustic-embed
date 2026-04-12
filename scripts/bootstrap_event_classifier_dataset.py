"""Bootstrap an event classifier training dataset from vocalization-labeled detection rows.

Reads detection job IDs, finds single-label vocalization-labeled windows,
runs Pass 2 segmentation within each window to discover event bounds,
and transfers the window's vocalization type label to every contained
event. Outputs a JSON file of training samples compatible with the event
classifier trainer (and the ``pytorch_event_cnn`` vocalization training
worker).

Usage::

    uv run python scripts/bootstrap_event_classifier_dataset.py \\
        --detection-job-ids JOB1 JOB2 \\
        --segmentation-model-id MODEL_ID \\
        --output samples.json \\
        --dry-run

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
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.segmentation.inference import run_inference
from humpback.call_parsing.segmentation.model import SegmentationCRNN
from humpback.call_parsing.types import Event
from humpback.classifier.detection_rows import (
    parse_recording_timestamp,
    read_detection_row_store,
)
from humpback.config import Settings
from humpback.database import create_engine, create_session_factory
from humpback.ml.checkpointing import load_checkpoint
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import SegmentationModel
from humpback.models.classifier import DetectionJob
from humpback.models.labeling import VocalizationLabel
from humpback.processing.audio_io import decode_audio, resample
from humpback.schemas.call_parsing import (
    SegmentationDecoderConfig,
    SegmentationFeatureConfig,
)
from humpback.storage import detection_row_store_path

logger = logging.getLogger(__name__)

_KNOWN_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".aif", ".aiff"}


@dataclass
class BootstrapResult:
    inserted: int = 0
    skipped: dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class _WindowInfo:
    row_id: str
    detection_job_id: str
    type_name: str
    start_utc: float
    end_utc: float


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
    best: tuple[Path, float] | None = None
    for base_epoch, path in audio_index:
        if base_epoch <= start_utc + 1e-6:
            best = (path, base_epoch)
        else:
            break
    return best


def _instantiate_segmentation_model(
    model_config: dict[str, Any],
) -> SegmentationCRNN:
    n_mels = int(model_config.get("n_mels", 64))
    conv_channels_raw = model_config.get("conv_channels", [32, 64, 96, 128])
    conv_channels = [int(c) for c in conv_channels_raw]
    gru_hidden = int(model_config.get("gru_hidden", 64))
    gru_layers = int(model_config.get("gru_layers", 2))
    return SegmentationCRNN(
        n_mels=n_mels,
        conv_channels=conv_channels,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
    )


@dataclass(frozen=True)
class _SyntheticRegion:
    """Minimal region-like object for segmentation inference."""

    region_id: str
    padded_start_sec: float
    padded_end_sec: float


def _run_segmentation_on_window(
    *,
    model: SegmentationCRNN,
    audio: np.ndarray,
    start_sec: float,
    end_sec: float,
    target_sr: int,
    feature_config: SegmentationFeatureConfig,
    decoder_config: SegmentationDecoderConfig,
) -> list[Event]:
    """Run segmentation on a window's audio slice, return events."""
    region = _SyntheticRegion(
        region_id="bootstrap",
        padded_start_sec=start_sec,
        padded_end_sec=end_sec,
    )

    def _load_region_audio(_region: Any) -> np.ndarray:
        s = max(0, int(round(start_sec * target_sr)))
        e = min(len(audio), int(round(end_sec * target_sr)))
        return audio[s:e].copy()

    return run_inference(
        model=model,
        region=region,
        audio_loader=_load_region_audio,
        feature_config=feature_config,
        decoder_config=decoder_config,
    )


async def _collect_windows(
    session: AsyncSession,
    detection_job_ids: list[str],
    storage_root: Path,
) -> tuple[list[_WindowInfo], dict[str, int]]:
    """Collect single-label vocalization-labeled windows from detection jobs.

    Returns ``(windows, skip_counts)`` where ``skip_counts`` maps skip
    reasons to counts.
    """
    skip_counts: dict[str, int] = defaultdict(int)
    windows: list[_WindowInfo] = []

    detection_job_cache: dict[str, DetectionJob] = {}
    row_store_cache: dict[str, dict[str, dict[str, str]]] = {}

    for dj_id in detection_job_ids:
        dj = await session.get(DetectionJob, dj_id)
        if dj is None:
            logger.warning("detection job %s not found, skipping", dj_id)
            skip_counts["detection job not found"] += 1
            continue
        detection_job_cache[dj_id] = dj

        if dj.hydrophone_id:
            logger.warning(
                "detection job %s: hydrophone source not supported, skipping", dj_id
            )
            skip_counts["hydrophone source (not supported)"] += 1
            continue

        if not dj.audio_folder:
            logger.warning("detection job %s: no audio_folder, skipping", dj_id)
            skip_counts["no audio source"] += 1
            continue

        # Load row store
        if dj_id not in row_store_cache:
            rs_path = (
                Path(dj.output_row_store_path) if dj.output_row_store_path else None
            )
            if rs_path is None or not rs_path.exists():
                rs_path = detection_row_store_path(storage_root, dj_id)
            if not rs_path.exists():
                logger.warning("detection job %s: row store not found, skipping", dj_id)
                skip_counts["row store not found"] += 1
                continue
            _, rows = read_detection_row_store(rs_path)
            row_store_cache[dj_id] = {r.get("row_id", ""): r for r in rows}

        # Load vocalization labels for this detection job
        label_result = await session.execute(
            select(VocalizationLabel).where(VocalizationLabel.detection_job_id == dj_id)
        )
        labels = list(label_result.scalars().all())

        labels_by_row: dict[str, list[str]] = defaultdict(list)
        for lb in labels:
            labels_by_row[lb.row_id].append(lb.label)

        row_index = row_store_cache[dj_id]

        for row_id, type_labels in labels_by_row.items():
            non_negative = [t for t in type_labels if t != "(Negative)"]
            if not non_negative:
                skip_counts["(Negative) only"] += 1
                continue
            if len(set(non_negative)) > 1:
                skip_counts["multi-label"] += 1
                continue

            det_row = row_index.get(row_id)
            if det_row is None:
                skip_counts["detection row not found"] += 1
                continue

            start_utc = float(det_row.get("start_utc") or 0)
            end_utc = float(det_row.get("end_utc") or 0)
            if end_utc <= start_utc:
                skip_counts["invalid time range"] += 1
                continue

            windows.append(
                _WindowInfo(
                    row_id=row_id,
                    detection_job_id=dj_id,
                    type_name=non_negative[0],
                    start_utc=start_utc,
                    end_utc=end_utc,
                )
            )

    return windows, dict(skip_counts)


async def run_bootstrap(
    session: AsyncSession,
    *,
    detection_job_ids: list[str],
    segmentation_model_id: str,
    dry_run: bool,
    output_path: Path,
    storage_root: Path,
) -> BootstrapResult:
    """Core bootstrap logic — testable without argparse."""
    result = BootstrapResult()

    # Load existing samples for idempotency
    existing_row_ids: set[str] = set()
    existing_samples: list[dict[str, Any]] = []
    if output_path.exists():
        existing_samples = json.loads(output_path.read_text())
        existing_row_ids = {s["source_row_id"] for s in existing_samples}
        logger.info(
            "Loaded %d existing samples from %s", len(existing_samples), output_path
        )

    # Load segmentation model
    seg_model = await session.get(SegmentationModel, segmentation_model_id)
    if seg_model is None:
        raise SystemExit(
            f"ERROR: segmentation model {segmentation_model_id!r} not found"
        )
    checkpoint_path = Path(seg_model.model_path)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"ERROR: segmentation model checkpoint not found at {checkpoint_path}"
        )
    model_config = json.loads(seg_model.config_json or "{}")
    feature_config_raw = model_config.get("feature_config") or {}
    feature_config = SegmentationFeatureConfig(**feature_config_raw)
    decoder_config = SegmentationDecoderConfig()

    crnn = _instantiate_segmentation_model(model_config)
    load_checkpoint(checkpoint_path, crnn)
    crnn.eval()

    # Collect qualifying windows
    windows, skip_counts = await _collect_windows(
        session, detection_job_ids, storage_root
    )
    for reason, count in skip_counts.items():
        result.skipped[reason] = count

    # Group windows by detection job for audio loading efficiency
    windows_by_job: dict[str, list[_WindowInfo]] = defaultdict(list)
    for w in windows:
        if w.row_id in existing_row_ids:
            result.skipped["already present"] = (
                result.skipped.get("already present", 0) + 1
            )
            continue
        windows_by_job[w.detection_job_id].append(w)

    new_samples: list[dict[str, Any]] = []
    target_sr = feature_config.sample_rate

    for dj_id, job_windows in windows_by_job.items():
        dj = await session.get(DetectionJob, dj_id)
        assert dj is not None
        audio_folder = Path(dj.audio_folder)  # type: ignore[arg-type]

        if not audio_folder.exists():
            for w in job_windows:
                result.skipped["audio folder missing"] = (
                    result.skipped.get("audio folder missing", 0) + 1
                )
            continue

        audio_index = _build_audio_index(audio_folder)
        audio_cache: dict[str, tuple[np.ndarray, float]] = {}

        for w in job_windows:
            resolved = _resolve_file_for_row(w.start_utc, audio_index)
            if resolved is None:
                result.skipped["audio file not resolved"] = (
                    result.skipped.get("audio file not resolved", 0) + 1
                )
                continue
            file_path, base_epoch = resolved

            # Find AudioFile row
            af_result = await session.execute(
                select(AudioFile).where(
                    AudioFile.source_folder == str(audio_folder),
                    AudioFile.filename == file_path.name,
                )
            )
            af = af_result.scalar_one_or_none()
            if af is None:
                result.skipped["audio file row not found"] = (
                    result.skipped.get("audio file row not found", 0) + 1
                )
                continue

            # Load audio (cached per file)
            cache_key = f"{audio_folder}:{file_path.name}"
            if cache_key not in audio_cache:
                raw, sr = decode_audio(file_path)
                resampled = np.asarray(resample(raw, sr, target_sr), dtype=np.float32)
                audio_cache[cache_key] = (resampled, base_epoch)
            audio, base_epoch = audio_cache[cache_key]

            # Audio-relative window bounds
            win_start = w.start_utc - base_epoch
            win_end = w.end_utc - base_epoch

            # Run segmentation on the window
            events = await asyncio.to_thread(
                _run_segmentation_on_window,
                model=crnn,
                audio=audio,
                start_sec=win_start,
                end_sec=win_end,
                target_sr=target_sr,
                feature_config=feature_config,
                decoder_config=decoder_config,
            )

            for event in events:
                if event.start_sec >= win_start and event.end_sec <= win_end:
                    new_samples.append(
                        {
                            "start_sec": round(event.start_sec, 4),
                            "end_sec": round(event.end_sec, 4),
                            "type_name": w.type_name,
                            "audio_file_id": af.id,
                            "source_row_id": w.row_id,
                        }
                    )
                    result.inserted += 1

    if not dry_run:
        all_samples = existing_samples + new_samples
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_samples, indent=2))
        logger.info("Wrote %d total samples to %s", len(all_samples), output_path)
    else:
        logger.info("[DRY RUN] Would write %d new samples", len(new_samples))

    return result


def _print_summary(result: BootstrapResult, dry_run: bool) -> None:
    mode = "[DRY RUN] " if dry_run else ""
    print(f"\n{mode}Bootstrap summary")
    print(f"  Inserted:   {result.inserted}")
    total_skipped = sum(result.skipped.values())
    print(f"  Skipped:    {total_skipped}")
    if result.skipped:
        for reason, count in sorted(result.skipped.items()):
            print(f"    - {reason}: {count}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap event classifier training samples from "
        "vocalization-labeled detection windows."
    )
    parser.add_argument(
        "--detection-job-ids",
        nargs="+",
        required=True,
        help="One or more detection job IDs to source windows from.",
    )
    parser.add_argument(
        "--segmentation-model-id",
        required=True,
        help="SegmentationModel ID for Pass 2 event discovery.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for training samples.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print summary without writing output.",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> None:
    load_dotenv()
    settings = Settings()

    engine = create_engine(settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            result = await run_bootstrap(
                session,
                detection_job_ids=args.detection_job_ids,
                segmentation_model_id=args.segmentation_model_id,
                dry_run=args.dry_run,
                output_path=args.output,
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
