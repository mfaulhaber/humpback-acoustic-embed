#!/usr/bin/env python3
"""Repair near-boundary hydrophone extracts that were written a few samples short."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

import soundfile as sf
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.extractor import (
    fetch_hydrophone_audio_range,
    parse_hydrophone_clip_range,
    write_flac_file,
    write_spectrogram_png_file,
)
from humpback.classifier.providers import build_archive_playback_provider
from humpback.classifier.s3_stream import expected_audio_samples
from humpback.config import ARCHIVE_SOURCE_IDS, Settings
from humpback.database import create_engine, create_session_factory
from humpback.models.audio import AudioFile


@dataclass(slots=True, frozen=True)
class RepairCandidate:
    audio_file_id: str
    file_path: Path
    filename: str
    folder_path: str
    label: str
    hydrophone_id: str
    clip_start_utc: datetime
    clip_end_utc: datetime
    sample_rate: int
    container_format: str
    actual_samples: int
    expected_samples: int
    missing_samples: int
    requires_suffix_fix: bool


@dataclass(slots=True, frozen=True)
class RepairResult:
    candidate: RepairCandidate
    status: Literal["dry-run", "repaired", "skipped", "failed"]
    detail: str
    checksum_sha256: str | None = None
    duration_seconds: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair imported hydrophone extracts that are within a small sample "
            "shortfall of the configured window length."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite files and update audio_files metadata (default: dry run)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=5.0,
        help="Nominal clip duration to repair (default: 5.0)",
    )
    parser.add_argument(
        "--max-missing-samples",
        type=int,
        default=64,
        help="Maximum shortfall to repair (default: 64)",
    )
    return parser


def _path_under_root(path: Path, root: Path) -> Path | None:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def candidate_from_audio_file(
    audio_file: AudioFile,
    *,
    positive_root: Path,
    negative_root: Path,
    window_seconds: float,
    max_missing_samples: int,
) -> RepairCandidate | None:
    if audio_file.source_folder is None:
        return None

    clip_range = parse_hydrophone_clip_range(audio_file.filename)
    if clip_range is None:
        return None
    clip_start_utc, clip_end_utc = clip_range
    clip_duration = (clip_end_utc - clip_start_utc).total_seconds()
    if abs(clip_duration - window_seconds) > 1e-6:
        return None

    source_folder = Path(audio_file.source_folder)
    relative_parent = _path_under_root(source_folder, positive_root)
    if relative_parent is None:
        relative_parent = _path_under_root(source_folder, negative_root)
    if relative_parent is None:
        return None

    if len(relative_parent.parts) != 5:
        return None
    label, hydrophone_id, year, month, day = relative_parent.parts
    if hydrophone_id not in ARCHIVE_SOURCE_IDS:
        return None
    if not (year.isdigit() and month.isdigit() and day.isdigit()):
        return None

    file_path = source_folder / audio_file.filename
    if not file_path.is_file():
        return None

    info = sf.info(str(file_path))
    if info.frames <= 0 or info.samplerate <= 0:
        return None

    expected_samples = expected_audio_samples(window_seconds, info.samplerate)
    missing_samples = expected_samples - info.frames
    container_format = str(info.format or "").upper()
    requires_suffix_fix = (
        file_path.suffix.lower() != ".flac" and container_format == "FLAC"
    )
    if not requires_suffix_fix and (
        missing_samples < 1 or missing_samples > max_missing_samples
    ):
        return None

    return RepairCandidate(
        audio_file_id=audio_file.id,
        file_path=file_path,
        filename=audio_file.filename,
        folder_path=audio_file.folder_path,
        label=label,
        hydrophone_id=hydrophone_id,
        clip_start_utc=clip_start_utc,
        clip_end_utc=clip_end_utc,
        sample_rate=info.samplerate,
        container_format=container_format,
        actual_samples=info.frames,
        expected_samples=expected_samples,
        missing_samples=missing_samples,
        requires_suffix_fix=requires_suffix_fix,
    )


async def discover_repair_candidates(
    session: AsyncSession,
    settings: Settings,
    *,
    window_seconds: float,
    max_missing_samples: int,
) -> list[RepairCandidate]:
    if settings.positive_sample_path is None or settings.negative_sample_path is None:
        raise ValueError("Positive and negative sample paths must be configured")

    positive_root = Path(settings.positive_sample_path).resolve()
    negative_root = Path(settings.negative_sample_path).resolve()

    result = await session.execute(
        select(AudioFile).where(
            AudioFile.source_folder.isnot(None),
            or_(
                AudioFile.source_folder.like(f"{positive_root}%"),
                AudioFile.source_folder.like(f"{negative_root}%"),
            ),
        )
    )

    candidates: list[RepairCandidate] = []
    for audio_file in result.scalars():
        candidate = candidate_from_audio_file(
            audio_file,
            positive_root=positive_root,
            negative_root=negative_root,
            window_seconds=window_seconds,
            max_missing_samples=max_missing_samples,
        )
        if candidate is not None:
            candidates.append(candidate)

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.hydrophone_id,
            candidate.label,
            str(candidate.file_path),
        ),
    )


def _stage_repaired_flac(
    candidate: RepairCandidate,
    audio_segment,
) -> tuple[Path, str, float]:
    fd, tmp_name = tempfile.mkstemp(dir=str(candidate.file_path.parent), suffix=".flac")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_flac_file(audio_segment, candidate.sample_rate, tmp_path)
        info = sf.info(str(tmp_path))
        checksum = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
        duration_seconds = info.frames / info.samplerate
        return tmp_path, checksum, duration_seconds
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _repair_target_path(candidate: RepairCandidate) -> Path:
    """Return the canonical repaired file path for a hydrophone extract."""
    return candidate.file_path.with_suffix(".flac")


async def repair_candidate(
    session: AsyncSession,
    settings: Settings,
    candidate: RepairCandidate,
    *,
    apply: bool,
) -> RepairResult:
    try:
        provider = build_archive_playback_provider(
            candidate.hydrophone_id,
            cache_path=settings.s3_cache_path,
            noaa_cache_path=settings.noaa_cache_path,
        )
        target_path = _repair_target_path(candidate)
        if target_path != candidate.file_path and target_path.exists():
            return RepairResult(
                candidate,
                "skipped",
                f"Target path already exists: {target_path}",
            )
        repaired_audio = fetch_hydrophone_audio_range(
            provider,
            candidate.clip_start_utc.timestamp(),
            candidate.clip_end_utc.timestamp(),
            candidate.sample_rate,
        )
        if repaired_audio is None:
            return RepairResult(candidate, "skipped", "No archive audio found")
        if len(repaired_audio) != candidate.expected_samples:
            return RepairResult(
                candidate,
                "skipped",
                "Archive audio still resolved short "
                f"({len(repaired_audio)} of {candidate.expected_samples} samples)",
            )

        staged_path, checksum, duration_seconds = _stage_repaired_flac(
            candidate, repaired_audio
        )
        try:
            conflict = await session.execute(
                select(AudioFile.id).where(
                    AudioFile.folder_path == candidate.folder_path,
                    AudioFile.checksum_sha256 == checksum,
                    AudioFile.id != candidate.audio_file_id,
                )
            )
            if conflict.scalar_one_or_none() is not None:
                return RepairResult(
                    candidate,
                    "skipped",
                    "Repaired checksum would collide with another audio_files row",
                    checksum_sha256=checksum,
                    duration_seconds=duration_seconds,
                )

            if not apply:
                detail = "Would rewrite clip and refresh audio_files metadata"
                if candidate.requires_suffix_fix:
                    detail += " (including filename suffix fix)"
                return RepairResult(
                    candidate,
                    "dry-run",
                    detail,
                    checksum_sha256=checksum,
                    duration_seconds=duration_seconds,
                )

            os.replace(staged_path, target_path)
            if candidate.file_path != target_path and candidate.file_path.exists():
                candidate.file_path.unlink()
            spectrogram_path = target_path.with_suffix(".png")
            write_spectrogram_png_file(
                repaired_audio,
                candidate.sample_rate,
                spectrogram_path,
                hop_length=settings.spectrogram_hop_length,
                dynamic_range_db=settings.spectrogram_dynamic_range_db,
                width_px=settings.spectrogram_width_px,
                height_px=settings.spectrogram_height_px,
            )
            if candidate.file_path != target_path:
                old_png_path = candidate.file_path.with_suffix(".png")
                if old_png_path != spectrogram_path and old_png_path.exists():
                    old_png_path.unlink()

            audio_file = await session.get(AudioFile, candidate.audio_file_id)
            if audio_file is None:
                raise ValueError(
                    f"AudioFile row disappeared during repair: {candidate.audio_file_id}"
                )
            audio_file.filename = target_path.name
            audio_file.duration_seconds = duration_seconds
            audio_file.sample_rate_original = candidate.sample_rate
            audio_file.checksum_sha256 = checksum
            audio_file.updated_at = datetime.now(timezone.utc)

            if candidate.missing_samples > 0:
                detail = f"Rewrote clip missing {candidate.missing_samples} samples"
            else:
                detail = "Rewrote clip"
            if candidate.requires_suffix_fix:
                detail += " and renamed output to .flac"
            return RepairResult(
                candidate,
                "repaired",
                detail,
                checksum_sha256=checksum,
                duration_seconds=duration_seconds,
            )
        finally:
            if staged_path.exists():
                staged_path.unlink()
    except Exception as exc:
        return RepairResult(candidate, "failed", str(exc))


async def run_repairs(
    settings: Settings,
    *,
    apply: bool,
    window_seconds: float,
    max_missing_samples: int,
) -> tuple[list[RepairCandidate], list[RepairResult]]:
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            candidates = await discover_repair_candidates(
                session,
                settings,
                window_seconds=window_seconds,
                max_missing_samples=max_missing_samples,
            )

            results: list[RepairResult] = []
            for candidate in candidates:
                result = await repair_candidate(
                    session, settings, candidate, apply=apply
                )
                results.append(result)
                if apply and result.status == "repaired":
                    await session.commit()
                else:
                    await session.rollback()
            return candidates, results
    finally:
        await engine.dispose()


def _print_report(
    candidates: list[RepairCandidate],
    results: list[RepairResult],
    *,
    apply: bool,
    window_seconds: float,
    max_missing_samples: int,
) -> None:
    print(
        f"Mode: {'APPLY' if apply else 'DRY RUN'} | "
        f"window={window_seconds:.1f}s | max_missing_samples={max_missing_samples}"
    )
    print(f"Candidates: {len(candidates)}")
    counts: dict[str, int] = {"dry-run": 0, "repaired": 0, "skipped": 0, "failed": 0}
    for result in results:
        counts[result.status] += 1
        print(
            f"[{result.status}] {result.candidate.file_path} "
            f"(missing {result.candidate.missing_samples} samples): {result.detail}"
        )
    if results:
        print(
            "Summary: "
            f"dry-run={counts['dry-run']} repaired={counts['repaired']} "
            f"skipped={counts['skipped']} failed={counts['failed']}"
        )


async def _async_main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    settings = Settings.from_repo_env()
    candidates, results = await run_repairs(
        settings,
        apply=args.apply,
        window_seconds=args.window_seconds,
        max_missing_samples=args.max_missing_samples,
    )
    _print_report(
        candidates,
        results,
        apply=args.apply,
        window_seconds=args.window_seconds,
        max_missing_samples=args.max_missing_samples,
    )
    return 0 if not any(result.status == "failed" for result in results) else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
