#!/usr/bin/env python3
"""
POC: NOAA GCS passive bioacoustic archive smoke test.

This script now reuses the production NOAA ArchiveProvider implementation rather
than carrying a separate client implementation.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from humpback.classifier.providers.noaa_gcs import (
    DEFAULT_NOAA_GCS_BUCKET,
    DEFAULT_NOAA_GCS_PREFIX,
    DateFolder,
    NoaaAudioFile,
    NoaaGCSProvider,
    decode_noaa_audio_bytes,
    group_noaa_files_by_date,
    list_noaa_objects,
)

TARGET_SR = 32000


def step_explore(files: list[NoaaAudioFile], folders: list[DateFolder]) -> None:
    """Step 1: Explore — list files, print date range, stats."""
    print("=" * 60)
    print("STEP 1: EXPLORE")
    print("=" * 60)
    print(f"Total .aif files found: {len(files)}")
    if not files:
        print("No files found.")
        return

    print(f"Date range: {files[0].timestamp} — {files[-1].timestamp}")
    print(f"Virtual date-folders: {len(folders)}")

    sizes = [audio_file.size for audio_file in files]
    print(
        f"File sizes: min={min(sizes):,} max={max(sizes):,} "
        f"avg={sum(sizes) // len(sizes):,} bytes"
    )

    if len(files) > 1:
        intervals = [
            files[i + 1].start_ts - files[i].start_ts
            for i in range(min(len(files) - 1, 100))
        ]
        avg_interval = sum(intervals) / len(intervals)
        print(
            f"Average interval (first {len(intervals)} pairs): "
            f"{avg_interval:.1f}s ({avg_interval / 60:.1f} min)"
        )

    print("\nSample filenames (first 5):")
    for audio_file in files[:5]:
        print(
            f"  {audio_file.filename}  ({audio_file.size:,} bytes)  "
            f"{audio_file.timestamp}"
        )
    print()


def step_filter(
    files: list[NoaaAudioFile],
    folders: list[DateFolder],
    start_ts: float,
    end_ts: float,
) -> list[DateFolder]:
    """Step 2: Filter — apply date range, show folder/segment mapping."""
    print("=" * 60)
    print("STEP 2: FILTER")
    print("=" * 60)
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    print(f"Requested range: {start_dt} — {end_dt}")

    matched: list[DateFolder] = []
    for folder in folders:
        if not folder.files:
            continue
        folder_start = folder.files[0].start_ts
        folder_end = folder.files[-1].start_ts + 300
        if folder_end >= start_ts and folder_start < end_ts:
            matched.append(folder)

    total_files = sum(len(folder.files) for folder in matched)
    print(f"Matched folders: {len(matched)}, total files: {total_files}")
    for folder in matched[:5]:
        in_range = [
            audio_file
            for audio_file in folder.files
            if start_ts <= audio_file.start_ts < end_ts
        ]
        print(
            f"  {folder.date_str} (epoch {int(folder.epoch)}): "
            f"{len(folder.files)} files, {len(in_range)} in exact range"
        )
    print()
    return matched


def step_download_decode(provider: NoaaGCSProvider, files: list[NoaaAudioFile]) -> None:
    """Step 3: Download + Decode — fetch one file, decode, print info."""
    print("=" * 60)
    print("STEP 3: DOWNLOAD + DECODE")
    print("=" * 60)
    target = files[0]
    print(f"Downloading: {target.filename} ({target.size:,} bytes)")

    raw = provider.fetch_segment(target.key)
    print(f"Downloaded {len(raw):,} bytes")

    audio = decode_noaa_audio_bytes(raw, TARGET_SR)
    duration = len(audio) / TARGET_SR
    print(f"Decoded: shape={audio.shape}, dtype={audio.dtype}")
    print(f"Target sample rate: {TARGET_SR} Hz")
    print(f"Duration: {duration:.2f}s ({duration / 60:.2f} min)")
    print(f"Amplitude range: [{audio.min():.4f}, {audio.max():.4f}]")
    print()


def step_interface_demo(
    provider: NoaaGCSProvider, start_ts: float, end_ts: float
) -> None:
    """Step 4: Interface demo — exercise the ArchiveProvider surface."""
    print("=" * 60)
    print("STEP 4: INTERFACE DEMO")
    print("=" * 60)

    timeline = provider.build_timeline(start_ts, end_ts)
    print(f"build_timeline({start_ts}, {end_ts}):")
    print(
        f"  -> {len(timeline)} segments: "
        f"{[segment.key for segment in timeline[:3]]}"
        f"{'...' if len(timeline) > 3 else ''}"
    )

    print(f"\ncount_segments({start_ts}, {end_ts}):")
    print(f"  -> {provider.count_segments(start_ts, end_ts)} total segments")

    if timeline:
        first_segment = timeline[0]
        print(f"\nfetch_segment({first_segment.key!r}):")
        raw = provider.fetch_segment(first_segment.key)
        print(f"  -> {len(raw):,} bytes")
        print("\ndecode_segment(...):")
        audio = provider.decode_segment(raw, TARGET_SR)
        print(f"  -> shape={audio.shape}, duration={len(audio) / TARGET_SR:.2f}s")

    print()
    print("ArchiveProvider interface exercised successfully.")
    print()


def parse_date(value: str) -> datetime:
    """Parse YYYY-MM-DD date strings to UTC datetimes."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date: {value!r} (expected YYYY-MM-DD)"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="POC: NOAA GCS Passive Bioacoustic Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_NOAA_GCS_PREFIX,
        help="GCS prefix under the NOAA bucket (default: Glacier Bay Jul-Oct 2015)",
    )
    parser.add_argument(
        "--start",
        type=parse_date,
        default=None,
        help="Start date YYYY-MM-DD (default: first available date)",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=None,
        help="End date YYYY-MM-DD (default: start + --hours)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=24,
        help="Hours from start date if --end not given (default: 24)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download+decode step (steps 1, 2, 4 only)",
    )
    args = parser.parse_args(argv)

    provider = NoaaGCSProvider(
        "noaa_glacier_bay",
        "NOAA Glacier Bay (Bartlett Cove)",
        bucket=DEFAULT_NOAA_GCS_BUCKET,
        prefix=args.prefix,
    )

    print(f"NOAA GCS POC — bucket: {DEFAULT_NOAA_GCS_BUCKET}")
    print(f"Prefix: {args.prefix}")
    print()

    print("Listing GCS objects (this may take a moment)...")
    files = list_noaa_objects(provider._get_bucket(), args.prefix)
    if not files:
        print("ERROR: No .aif files found under prefix.")
        return 1
    folders = group_noaa_files_by_date(files)

    if args.start:
        start_dt = args.start
    else:
        start_dt = files[0].timestamp.replace(hour=0, minute=0, second=0)

    if args.end:
        end_dt = args.end
    else:
        end_dt = start_dt + timedelta(hours=args.hours)

    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()

    step_explore(files, folders)
    matched = step_filter(files, folders, start_ts, end_ts)

    if not args.skip_download and matched:
        range_files = [
            audio_file
            for folder in matched
            for audio_file in folder.files
            if start_ts <= audio_file.start_ts < end_ts
        ]
        if range_files:
            step_download_decode(provider, range_files)
        else:
            print("No files in exact range for download test.\n")

    step_interface_demo(provider, start_ts, end_ts)

    print("=" * 60)
    print("POC COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
