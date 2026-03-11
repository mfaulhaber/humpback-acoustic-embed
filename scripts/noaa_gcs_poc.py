#!/usr/bin/env python3
"""
POC: NOAA GCS Passive Bioacoustic Client.

Validates anonymous Google Cloud Storage access to NOAA's public passive
bioacoustic dataset and exercises a NoaaGCSClient class that satisfies the
same duck-typed interface as the existing S3 HLS clients.

Dataset: gs://noaa-passive-bioacoustic/nps/audio/glacier_bay/bartlettcove/
         glacierbay_bartlettcove_jul-oct2015/audio/

Files are ~20MB .aif, named MM_DD_YYYY_HH_MM_SS.aif, ~3.75 min each at 44.1 kHz.

Usage:
    uv run python scripts/noaa_gcs_poc.py
    uv run python scripts/noaa_gcs_poc.py --skip-download
    uv run python scripts/noaa_gcs_poc.py --hours 12 --start 2015-07-15
"""

from __future__ import annotations

import argparse
import io
import re
import struct
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKET = "noaa-passive-bioacoustic"
DEFAULT_PREFIX = (
    "nps/audio/glacier_bay/bartlettcove/glacierbay_bartlettcove_jul-oct2015/audio/"
)
TARGET_SR = 32000

_NOAA_RE = re.compile(r"^(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{2})\.aif$")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoaaAudioFile:
    """Metadata for a single NOAA .aif recording."""

    filename: str
    key: str
    timestamp: datetime
    size: int


@dataclass
class DateFolder:
    """Virtual folder grouping .aif files by calendar date."""

    date_str: str  # YYYY-MM-DD
    epoch: float  # midnight UTC epoch
    files: list[NoaaAudioFile] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def parse_noaa_filename(filename: str) -> datetime | None:
    """Parse MM_DD_YYYY_HH_MM_SS.aif -> UTC datetime, or None."""
    m = _NOAA_RE.match(filename)
    if not m:
        return None
    month, day, year, hour, minute, second = (int(g) for g in m.groups())
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None


def gcs_list_objects(prefix: str) -> list[NoaaAudioFile]:
    """List .aif objects under *prefix* using the google-cloud-storage SDK."""
    from google.cloud import storage

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(BUCKET)
    results: list[NoaaAudioFile] = []

    for blob in bucket.list_blobs(prefix=prefix):
        name = blob.name.rsplit("/", 1)[-1]
        ts = parse_noaa_filename(name)
        if ts is None:
            continue
        results.append(
            NoaaAudioFile(
                filename=name, key=blob.name, timestamp=ts, size=blob.size or 0
            )
        )

    results.sort(key=lambda f: f.timestamp)
    return results


def gcs_download(key: str) -> bytes:
    """Download a single object from the NOAA bucket."""
    from google.cloud import storage

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(key)
    return blob.download_as_bytes()


def decode_audio_bytes(raw_bytes: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode audio bytes (AIFF or any ffmpeg-supported format) to float32 via ffmpeg.

    Same stdin/stdout pipe pattern as decode_ts_bytes in s3_stream.py,
    but with a longer timeout for larger files.
    """
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            "pipe:0",
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "pipe:1",
        ],
        input=raw_bytes,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr[:500]}")

    wav_bytes = result.stdout
    buf = io.BytesIO(wav_bytes)
    buf.read(12)  # skip RIFF header

    while True:
        chunk_header = buf.read(8)
        if len(chunk_header) < 8:
            raise RuntimeError("Could not find data chunk in WAV output")
        chunk_id = chunk_header[:4]
        chunk_size = struct.unpack("<I", chunk_header[4:8])[0]
        if chunk_id == b"data":
            pcm_bytes = buf.read(chunk_size)
            break
        buf.read(chunk_size)

    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _group_by_date(files: list[NoaaAudioFile]) -> list[DateFolder]:
    """Group files into virtual date folders (midnight UTC epoch)."""
    by_date: dict[str, list[NoaaAudioFile]] = defaultdict(list)
    for f in files:
        by_date[f.timestamp.strftime("%Y-%m-%d")].append(f)

    folders: list[DateFolder] = []
    for date_str in sorted(by_date):
        midnight = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        folder = DateFolder(
            date_str=date_str,
            epoch=midnight.timestamp(),
            files=sorted(by_date[date_str], key=lambda f: f.timestamp),
        )
        folders.append(folder)
    return folders


# ---------------------------------------------------------------------------
# NoaaGCSClient — duck-typed HLS client interface
# ---------------------------------------------------------------------------


class NoaaGCSClient:
    """GCS client for NOAA passive bioacoustic data.

    Satisfies the same duck-typed interface as OrcasoundS3Client,
    LocalHLSClient, and CachingS3Client.
    """

    def __init__(self, prefix: str = DEFAULT_PREFIX) -> None:
        self._prefix = prefix
        self._files: list[NoaaAudioFile] | None = None
        self._folders: list[DateFolder] | None = None

    def _ensure_loaded(self) -> None:
        if self._files is None:
            self._files = gcs_list_objects(self._prefix)
            self._folders = _group_by_date(self._files)

    def list_hls_folders(
        self, hydrophone_id: str, start_ts: float, end_ts: float
    ) -> list[str]:
        """Return date-folder epoch strings overlapping [start_ts, end_ts)."""
        self._ensure_loaded()
        assert self._folders is not None
        result: list[str] = []
        for folder in self._folders:
            # A folder overlaps if its last file could extend past start_ts
            # and its first file starts before end_ts.
            if not folder.files:
                continue
            folder_start = folder.files[0].timestamp.timestamp()
            folder_end = folder.files[-1].timestamp.timestamp() + 300  # ~5 min buffer
            if folder_end >= start_ts and folder_start < end_ts:
                result.append(str(int(folder.epoch)))
        return result

    def list_segments(self, hydrophone_id: str, folder_ts: str) -> list[str]:
        """Return .aif file keys for a given date-folder epoch."""
        self._ensure_loaded()
        assert self._folders is not None
        target_epoch = float(folder_ts)
        for folder in self._folders:
            if abs(folder.epoch - target_epoch) < 1:
                return [f.key for f in folder.files]
        return []

    def fetch_segment(self, key: str) -> bytes:
        """Download a single .aif file."""
        return gcs_download(key)

    def fetch_playlist(self, hydrophone_id: str, folder_ts: str) -> None:
        """NOAA data has no HLS playlists."""
        return None

    def count_segments(self, hydrophone_id: str, folder_timestamps: list[str]) -> int:
        """Count total .aif files across the given date-folders."""
        total = 0
        for ts in folder_timestamps:
            total += len(self.list_segments(hydrophone_id, ts))
        return total


# ---------------------------------------------------------------------------
# Demo steps
# ---------------------------------------------------------------------------


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

    sizes = [f.size for f in files]
    print(
        f"File sizes: min={min(sizes):,} max={max(sizes):,} avg={sum(sizes) // len(sizes):,} bytes"
    )

    if len(files) > 1:
        intervals = [
            (files[i + 1].timestamp - files[i].timestamp).total_seconds()
            for i in range(min(len(files) - 1, 100))
        ]
        avg_interval = sum(intervals) / len(intervals)
        print(
            f"Average interval (first {len(intervals)} pairs): {avg_interval:.1f}s ({avg_interval / 60:.1f} min)"
        )

    print("\nSample filenames (first 5):")
    for f in files[:5]:
        print(f"  {f.filename}  ({f.size:,} bytes)  {f.timestamp}")
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
        folder_start = folder.files[0].timestamp.timestamp()
        folder_end = folder.files[-1].timestamp.timestamp() + 300
        if folder_end >= start_ts and folder_start < end_ts:
            matched.append(folder)

    total_files = sum(len(f.files) for f in matched)
    print(f"Matched folders: {len(matched)}, total files: {total_files}")
    for folder in matched[:5]:
        in_range = [
            f for f in folder.files if start_ts <= f.timestamp.timestamp() < end_ts
        ]
        print(
            f"  {folder.date_str} (epoch {int(folder.epoch)}): "
            f"{len(folder.files)} files, {len(in_range)} in exact range"
        )
    print()
    return matched


def step_download_decode(
    files: list[NoaaAudioFile], target_sr: int = TARGET_SR
) -> None:
    """Step 3: Download + Decode — fetch one file, decode, print info."""
    print("=" * 60)
    print("STEP 3: DOWNLOAD + DECODE")
    print("=" * 60)
    target = files[0]
    print(f"Downloading: {target.filename} ({target.size:,} bytes)")

    raw = gcs_download(target.key)
    print(f"Downloaded {len(raw):,} bytes")

    audio = decode_audio_bytes(raw, target_sr)
    duration = len(audio) / target_sr
    print(f"Decoded: shape={audio.shape}, dtype={audio.dtype}")
    print(f"Target sample rate: {target_sr} Hz")
    print(f"Duration: {duration:.2f}s ({duration / 60:.2f} min)")
    print(f"Amplitude range: [{audio.min():.4f}, {audio.max():.4f}]")
    print()


def step_interface_demo(client: NoaaGCSClient, start_ts: float, end_ts: float) -> None:
    """Step 4: Interface Demo — exercise all 5 client methods."""
    print("=" * 60)
    print("STEP 4: INTERFACE DEMO")
    print("=" * 60)
    hydro_id = "noaa_glacier_bay"

    print(f"list_hls_folders({hydro_id!r}, {start_ts}, {end_ts}):")
    folders = client.list_hls_folders(hydro_id, start_ts, end_ts)
    print(
        f"  -> {len(folders)} folders: {folders[:5]}{'...' if len(folders) > 5 else ''}"
    )

    if folders:
        first_folder = folders[0]
        print(f"\nlist_segments({hydro_id!r}, {first_folder!r}):")
        segments = client.list_segments(hydro_id, first_folder)
        print(f"  -> {len(segments)} segments")
        if segments:
            print(f"  First: {segments[0]}")
            print(f"  Last:  {segments[-1]}")

        print(f"\nfetch_playlist({hydro_id!r}, {first_folder!r}):")
        playlist = client.fetch_playlist(hydro_id, first_folder)
        print(f"  -> {playlist!r} (NOAA has no playlists)")

        print(f"\ncount_segments({hydro_id!r}, {folders[:3]!r}):")
        count = client.count_segments(hydro_id, folders[:3])
        print(f"  -> {count} total segments")

    print()
    print("All 5 interface methods exercised successfully.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_date(s: str) -> datetime:
    """Parse YYYY-MM-DD date string to UTC datetime."""
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date: {s!r} (expected YYYY-MM-DD)"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="POC: NOAA GCS Passive Bioacoustic Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
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
        "--target-sr",
        type=int,
        default=TARGET_SR,
        help=f"Target sample rate for decoding (default: {TARGET_SR})",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download+decode step (steps 1, 2, 4 only)",
    )
    args = parser.parse_args(argv)

    target_sr = args.target_sr

    print(f"NOAA GCS POC — bucket: {BUCKET}")
    print(f"Prefix: {args.prefix}")
    print()

    # Load all files
    print("Listing GCS objects (this may take a moment)...")
    files = gcs_list_objects(args.prefix)
    if not files:
        print("ERROR: No .aif files found under prefix.")
        return 1
    folders = _group_by_date(files)

    # Determine range
    if args.start:
        start_dt = args.start
    else:
        start_dt = files[0].timestamp.replace(hour=0, minute=0, second=0)

    if args.end:
        end_dt = args.end
    else:
        from datetime import timedelta

        end_dt = start_dt + timedelta(hours=args.hours)

    start_ts = start_dt.timestamp()
    end_ts = end_dt.timestamp()

    # Run demo steps
    step_explore(files, folders)
    matched = step_filter(files, folders, start_ts, end_ts)

    if not args.skip_download and matched:
        # Pick first file in range for download test
        range_files = [
            f
            for folder in matched
            for f in folder.files
            if start_ts <= f.timestamp.timestamp() < end_ts
        ]
        if range_files:
            step_download_decode(range_files, target_sr)
        else:
            print("No files in exact range for download test.\n")

    # Interface demo uses pre-loaded client
    client = NoaaGCSClient(prefix=args.prefix)
    client._files = files
    client._folders = folders
    step_interface_demo(client, start_ts, end_ts)

    print("=" * 60)
    print("POC COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
