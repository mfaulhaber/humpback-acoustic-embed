"""NOAA GCS archive provider implementation."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import struct
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from humpback.classifier.archive import StreamSegment

DEFAULT_NOAA_GCS_BUCKET = "noaa-passive-bioacoustic"
DEFAULT_NOAA_GCS_PREFIX = (
    "nps/audio/glacier_bay/bartlettcove/glacierbay_bartlettcove_jul-oct2015/audio/"
)
DEFAULT_NOAA_SEGMENT_DURATION_SEC = 300.0
NOAA_FILENAME_RE = re.compile(r"^(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{2})\.aif$")


@dataclass(frozen=True)
class NoaaAudioFile:
    """Metadata for a single NOAA archive object."""

    filename: str
    key: str
    timestamp: datetime
    size: int

    @property
    def start_ts(self) -> float:
        return self.timestamp.timestamp()


@dataclass
class DateFolder:
    """Virtual grouping of NOAA files by UTC calendar day."""

    date_str: str
    epoch: float
    files: list[NoaaAudioFile] = field(default_factory=list)


def parse_noaa_filename(filename: str) -> datetime | None:
    """Parse NOAA filename timestamps in UTC."""
    match = NOAA_FILENAME_RE.match(filename)
    if match is None:
        return None

    month, day, year, hour, minute, second = (int(group) for group in match.groups())
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        return None


def decode_noaa_audio_bytes(raw_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode NOAA AIFF bytes to mono float32 audio via ffmpeg."""
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
    buf.read(12)

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


def list_noaa_objects(bucket: Any, prefix: str) -> list[NoaaAudioFile]:
    """List timestamped NOAA AIFF objects under a prefix."""
    files: list[NoaaAudioFile] = []
    for blob in bucket.list_blobs(prefix=prefix):
        name = blob.name.rsplit("/", 1)[-1]
        ts = parse_noaa_filename(name)
        if ts is None:
            continue
        files.append(
            NoaaAudioFile(
                filename=name,
                key=blob.name,
                timestamp=ts,
                size=blob.size or 0,
            )
        )

    files.sort(key=lambda file: file.timestamp)
    return files


def group_noaa_files_by_date(files: list[NoaaAudioFile]) -> list[DateFolder]:
    """Group NOAA files into virtual UTC date folders."""
    by_date: dict[str, list[NoaaAudioFile]] = defaultdict(list)
    for audio_file in files:
        by_date[audio_file.timestamp.strftime("%Y-%m-%d")].append(audio_file)

    folders: list[DateFolder] = []
    for date_str in sorted(by_date):
        midnight = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        folders.append(
            DateFolder(
                date_str=date_str,
                epoch=midnight.timestamp(),
                files=sorted(by_date[date_str], key=lambda file: file.timestamp),
            )
        )
    return folders


def estimate_noaa_interval_sec(files: list[NoaaAudioFile]) -> float:
    """Estimate nominal segment duration from adjacent object timestamps."""
    intervals = [
        files[idx + 1].start_ts - files[idx].start_ts
        for idx in range(len(files) - 1)
        if files[idx + 1].start_ts > files[idx].start_ts
    ]
    if not intervals:
        return DEFAULT_NOAA_SEGMENT_DURATION_SEC
    min_interval = min(intervals)
    nominal_intervals = [
        interval for interval in intervals if interval <= (min_interval * 1.1)
    ]
    return float(median(nominal_intervals or intervals))


class NoaaGCSProvider:
    """ArchiveProvider for NOAA's public passive bioacoustic GCS archive."""

    def __init__(
        self,
        source_id: str,
        name: str,
        *,
        bucket: str = DEFAULT_NOAA_GCS_BUCKET,
        prefix: str = DEFAULT_NOAA_GCS_PREFIX,
        bucket_obj: Any | None = None,
    ) -> None:
        self._source_id = source_id
        self._name = name
        self._bucket_name = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._bucket = bucket_obj
        self._files: list[NoaaAudioFile] | None = None
        self._default_interval_sec: float | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_id(self) -> str:
        return self._source_id

    def _get_bucket(self) -> Any:
        if self._bucket is None:
            from google.cloud import storage

            client = storage.Client.create_anonymous_client()
            self._bucket = client.bucket(self._bucket_name)
        return self._bucket

    def _ensure_loaded(self) -> None:
        if self._files is not None:
            return
        files = list_noaa_objects(self._get_bucket(), self._prefix)
        self._files = files
        self._default_interval_sec = estimate_noaa_interval_sec(files)

    def _build_segments(self) -> list[StreamSegment]:
        self._ensure_loaded()
        files = self._files or []
        default_interval = (
            self._default_interval_sec or DEFAULT_NOAA_SEGMENT_DURATION_SEC
        )

        segments: list[StreamSegment] = []
        for idx, audio_file in enumerate(files):
            duration_sec = default_interval
            if idx + 1 < len(files):
                next_gap = files[idx + 1].start_ts - audio_file.start_ts
                if 0 < next_gap < default_interval:
                    duration_sec = next_gap

            segments.append(
                StreamSegment(
                    key=audio_file.key,
                    start_ts=audio_file.start_ts,
                    duration_sec=duration_sec,
                )
            )
        return segments

    def build_timeline(self, start_ts: float, end_ts: float) -> list[StreamSegment]:
        timeline = [
            segment
            for segment in self._build_segments()
            if segment.end_ts > start_ts and segment.start_ts < end_ts
        ]
        if not (self._files or []):
            raise FileNotFoundError("No NOAA audio data found for this time range")
        if not timeline:
            raise FileNotFoundError("No NOAA stream segments found in requested range")
        return timeline

    def count_segments(self, start_ts: float, end_ts: float) -> int:
        self._ensure_loaded()
        return sum(
            1
            for segment in self._build_segments()
            if segment.end_ts > start_ts and segment.start_ts < end_ts
        )

    def fetch_segment(self, key: str) -> bytes:
        blob = self._get_bucket().blob(key)
        return blob.download_as_bytes()

    def decode_segment(self, raw_bytes: bytes, target_sr: int) -> np.ndarray:
        return decode_noaa_audio_bytes(raw_bytes, target_sr)

    def invalidate_cached_segment(self, key: str) -> bool:
        return False


# ---- Manifest cache ----

MANIFEST_FILENAME = "_noaa_manifest.json"

logger = logging.getLogger(__name__)


def _manifest_path(cache_root: str, bucket: str, prefix: str) -> Path:
    """Path for the metadata manifest for a given bucket/prefix."""
    return Path(cache_root) / bucket / prefix.strip("/") / MANIFEST_FILENAME


def write_noaa_manifest(
    path: Path,
    files: list[NoaaAudioFile],
    default_interval_sec: float,
) -> None:
    """Atomically write metadata manifest (tmp + os.replace)."""
    data = {
        "version": 1,
        "default_interval_sec": default_interval_sec,
        "cached_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "filename": f.filename,
                "key": f.key,
                "timestamp_iso": f.timestamp.isoformat(),
                "size": f.size,
            }
            for f in files
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_noaa_manifest(
    path: Path,
) -> tuple[list[NoaaAudioFile], float] | None:
    """Read cached manifest. Returns (files, default_interval_sec) or None if missing/corrupt."""
    if not path.is_file():
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
        files = [
            NoaaAudioFile(
                filename=entry["filename"],
                key=entry["key"],
                timestamp=datetime.fromisoformat(entry["timestamp_iso"]),
                size=entry["size"],
            )
            for entry in data["files"]
        ]
        return files, float(data["default_interval_sec"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Corrupt NOAA manifest at %s: %s", path, exc)
        return None


# ---- Caching provider ----


class CachingNoaaGCSProvider:
    """ArchiveProvider with local cache + GCS fallback for NOAA archives."""

    def __init__(
        self,
        source_id: str,
        name: str,
        cache_root: str,
        *,
        bucket: str = DEFAULT_NOAA_GCS_BUCKET,
        prefix: str = DEFAULT_NOAA_GCS_PREFIX,
        bucket_obj: Any | None = None,
    ) -> None:
        self._source_id = source_id
        self._name = name
        self._cache_root = cache_root
        self._bucket_name = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._gcs = NoaaGCSProvider(
            source_id, name, bucket=bucket, prefix=prefix, bucket_obj=bucket_obj
        )
        self._files: list[NoaaAudioFile] | None = None
        self._default_interval_sec: float | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_id(self) -> str:
        return self._source_id

    def _ensure_loaded(self) -> None:
        if self._files is not None:
            return
        manifest = _manifest_path(self._cache_root, self._bucket_name, self._prefix)
        cached = read_noaa_manifest(manifest)
        if cached is not None:
            self._files, self._default_interval_sec = cached
            return
        self._gcs._ensure_loaded()
        self._files = self._gcs._files
        self._default_interval_sec = self._gcs._default_interval_sec
        if self._files:
            write_noaa_manifest(
                manifest,
                self._files,
                self._default_interval_sec or DEFAULT_NOAA_SEGMENT_DURATION_SEC,
            )

    def _build_segments(self) -> list[StreamSegment]:
        self._ensure_loaded()
        files = self._files or []
        default_interval = (
            self._default_interval_sec or DEFAULT_NOAA_SEGMENT_DURATION_SEC
        )
        segments: list[StreamSegment] = []
        for idx, audio_file in enumerate(files):
            duration_sec = default_interval
            if idx + 1 < len(files):
                next_gap = files[idx + 1].start_ts - audio_file.start_ts
                if 0 < next_gap < default_interval:
                    duration_sec = next_gap
            segments.append(
                StreamSegment(
                    key=audio_file.key,
                    start_ts=audio_file.start_ts,
                    duration_sec=duration_sec,
                )
            )
        return segments

    def build_timeline(self, start_ts: float, end_ts: float) -> list[StreamSegment]:
        timeline = [
            segment
            for segment in self._build_segments()
            if segment.end_ts > start_ts and segment.start_ts < end_ts
        ]
        if not (self._files or []):
            raise FileNotFoundError("No NOAA audio data found for this time range")
        if not timeline:
            raise FileNotFoundError("No NOAA stream segments found in requested range")
        return timeline

    def count_segments(self, start_ts: float, end_ts: float) -> int:
        self._ensure_loaded()
        return sum(
            1
            for segment in self._build_segments()
            if segment.end_ts > start_ts and segment.start_ts < end_ts
        )

    def fetch_segment(self, key: str) -> bytes:
        local_path = Path(self._cache_root) / self._bucket_name / key
        if local_path.is_file():
            return local_path.read_bytes()
        raw = self._gcs.fetch_segment(key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(local_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
            os.replace(tmp, str(local_path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return raw

    def decode_segment(self, raw_bytes: bytes, target_sr: int) -> np.ndarray:
        return decode_noaa_audio_bytes(raw_bytes, target_sr)

    def invalidate_cached_segment(self, key: str) -> bool:
        local_path = Path(self._cache_root) / self._bucket_name / key
        if local_path.is_file():
            local_path.unlink()
            return True
        return False


# ---- Factory helpers ----


def build_noaa_detection_provider(
    source_id: str,
    name: str,
    *,
    noaa_cache_path: str | None,
    bucket: str,
    prefix: str,
) -> NoaaGCSProvider | CachingNoaaGCSProvider:
    if noaa_cache_path:
        return CachingNoaaGCSProvider(
            source_id, name, noaa_cache_path, bucket=bucket, prefix=prefix
        )
    return NoaaGCSProvider(source_id, name, bucket=bucket, prefix=prefix)


def build_noaa_playback_provider(
    source_id: str,
    name: str,
    *,
    noaa_cache_path: str | None,
    bucket: str,
    prefix: str,
) -> NoaaGCSProvider | CachingNoaaGCSProvider:
    if noaa_cache_path:
        return CachingNoaaGCSProvider(
            source_id, name, noaa_cache_path, bucket=bucket, prefix=prefix
        )
    return NoaaGCSProvider(source_id, name, bucket=bucket, prefix=prefix)
