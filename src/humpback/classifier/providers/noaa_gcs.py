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
import threading
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping

import numpy as np

from humpback.classifier.archive import StreamSegment

DEFAULT_NOAA_GCS_BUCKET = "noaa-passive-bioacoustic"
DEFAULT_NOAA_GCS_PREFIX = (
    "nps/audio/glacier_bay/bartlettcove/glacierbay_bartlettcove_jul-oct2015/audio/"
)
DEFAULT_NOAA_SEGMENT_DURATION_SEC = 300.0

NOAA_LEGACY_FILENAME_RE = re.compile(
    r"^(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{2})\.(?:aif|aiff|wav)$",
    re.IGNORECASE,
)
NOAA_YEAR_FIRST_FILENAME_RE = re.compile(
    r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})\.(?:aif|aiff|wav)$",
    re.IGNORECASE,
)
NOAA_ISO_DASH_FILENAME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})Z\.(?:aif|aiff|wav|flac)$",
    re.IGNORECASE,
)
SANCTSOUND_ISO_FILENAME_RE = re.compile(
    r".*_(\d{8}T\d{6}Z)\.flac$",
    re.IGNORECASE,
)
SANCTSOUND_COMPACT_FILENAME_RE = re.compile(
    r".*_(\d{12})(?:_o)?\.flac$",
    re.IGNORECASE,
)


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


@dataclass(frozen=True)
class NoaaChildFolderHint:
    """Known child folder under a NOAA archive root."""

    prefix: str
    start_utc: datetime | None = None
    end_utc: datetime | None = None

    def overlaps(self, start_ts: float, end_ts: float) -> bool:
        start_bound = (
            self.start_utc.timestamp() if self.start_utc is not None else float("-inf")
        )
        end_bound = (
            self.end_utc.timestamp() if self.end_utc is not None else float("inf")
        )
        return end_bound > start_ts and start_bound < end_ts


def _normalized_prefix(prefix: str) -> str:
    return prefix.strip("/") + "/"


def _join_prefix(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/")) + "/"


def _coerce_utc_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_child_folder_hints(
    raw_hints: object,
) -> list[NoaaChildFolderHint]:
    if not isinstance(raw_hints, list):
        return []

    hints: list[NoaaChildFolderHint] = []
    for entry in raw_hints:
        if not isinstance(entry, Mapping):
            continue
        prefix = entry.get("prefix")
        if not isinstance(prefix, str) or not prefix.strip():
            continue
        hints.append(
            NoaaChildFolderHint(
                prefix=_normalized_prefix(prefix),
                start_utc=_coerce_utc_datetime(entry.get("start_utc")),
                end_utc=_coerce_utc_datetime(entry.get("end_utc")),
            )
        )
    return hints


def parse_noaa_filename(filename: str) -> datetime | None:
    """Parse NOAA archive filename timestamps in UTC across known datasets."""
    match = NOAA_LEGACY_FILENAME_RE.match(filename)
    if match is not None:
        month, day, year, hour, minute, second = (
            int(group) for group in match.groups()
        )
        try:
            return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        except ValueError:
            return None

    match = NOAA_YEAR_FIRST_FILENAME_RE.match(filename)
    if match is not None:
        year, month, day, hour, minute, second = (
            int(group) for group in match.groups()
        )
        try:
            return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        except ValueError:
            return None

    match = NOAA_ISO_DASH_FILENAME_RE.match(filename)
    if match is not None:
        year, month, day, hour, minute, second = (
            int(group) for group in match.groups()
        )
        try:
            return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        except ValueError:
            return None

    match = SANCTSOUND_ISO_FILENAME_RE.match(filename)
    if match is not None:
        try:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return None

    match = SANCTSOUND_COMPACT_FILENAME_RE.match(filename)
    if match is not None:
        try:
            return datetime.strptime(match.group(1), "%y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return None

    return None


def decode_noaa_audio_bytes(raw_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode NOAA archive audio bytes to mono float32 audio via ffmpeg."""
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


def _iter_pcm_chunks_from_ffmpeg(
    command: list[str],
    *,
    chunk_samples: int,
    raw_bytes: bytes | None = None,
) -> Generator[np.ndarray, None, None]:
    """Stream ffmpeg PCM output as float32 chunks."""
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if raw_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    writer: threading.Thread | None = None

    if raw_bytes is not None:

        def _feed_stdin() -> None:
            assert proc.stdin is not None
            try:
                proc.stdin.write(raw_bytes)
            except BrokenPipeError:
                pass
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        writer = threading.Thread(target=_feed_stdin, daemon=True)
        writer.start()

    assert proc.stdout is not None
    assert proc.stderr is not None

    bytes_per_chunk = max(1, int(chunk_samples)) * 2
    pending = b""

    try:
        while True:
            chunk = proc.stdout.read(bytes_per_chunk)
            if not chunk:
                break
            pcm_bytes = pending + chunk
            usable_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
            pending = pcm_bytes[usable_len:]
            if usable_len == 0:
                continue
            pcm = np.frombuffer(pcm_bytes[:usable_len], dtype=np.int16)
            if len(pcm) == 0:
                continue
            yield pcm.astype(np.float32) / 32768.0
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass

    if writer is not None:
        writer.join()

    stderr = proc.stderr.read()
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {stderr[:500]}")


def iter_decode_noaa_audio_bytes(
    raw_bytes: bytes,
    target_sr: int,
    *,
    clip_start_sec: float,
    clip_end_sec: float,
    chunk_seconds: float,
) -> Generator[tuple[np.ndarray, float], None, None]:
    """Decode clipped NOAA bytes to sequential float32 chunks."""
    clip_duration_sec = max(0.0, clip_end_sec - clip_start_sec)
    if clip_duration_sec <= 0:
        return

    command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ss",
        str(clip_start_sec),
        "-t",
        str(clip_duration_sec),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    offset_samples = 0
    chunk_samples = max(1, int(round(chunk_seconds * target_sr)))
    for pcm in _iter_pcm_chunks_from_ffmpeg(
        command,
        chunk_samples=chunk_samples,
        raw_bytes=raw_bytes,
    ):
        yield pcm, clip_start_sec + (offset_samples / target_sr)
        offset_samples += len(pcm)


def iter_decode_noaa_audio_file(
    audio_path: Path,
    target_sr: int,
    *,
    clip_start_sec: float,
    clip_end_sec: float,
    chunk_seconds: float,
) -> Generator[tuple[np.ndarray, float], None, None]:
    """Decode a clipped NOAA file from disk to sequential float32 chunks."""
    clip_duration_sec = max(0.0, clip_end_sec - clip_start_sec)
    if clip_duration_sec <= 0:
        return

    command = [
        "ffmpeg",
        "-i",
        str(audio_path),
        "-ss",
        str(clip_start_sec),
        "-t",
        str(clip_duration_sec),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    offset_samples = 0
    chunk_samples = max(1, int(round(chunk_seconds * target_sr)))
    for pcm in _iter_pcm_chunks_from_ffmpeg(
        command,
        chunk_samples=chunk_samples,
        raw_bytes=None,
    ):
        yield pcm, clip_start_sec + (offset_samples / target_sr)
        offset_samples += len(pcm)


def list_noaa_objects(bucket: Any, prefix: str) -> list[NoaaAudioFile]:
    """List timestamped NOAA archive audio objects under a prefix."""
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
    """Estimate nominal segment duration from adjacent object timestamps.

    Uses the median of all inter-file gaps, which is robust to both small
    outliers (anomalous closely-spaced files) and large outliers (missing
    files creating big gaps).
    """
    intervals = [
        files[idx + 1].start_ts - files[idx].start_ts
        for idx in range(len(files) - 1)
        if files[idx + 1].start_ts > files[idx].start_ts
    ]
    if not intervals:
        return DEFAULT_NOAA_SEGMENT_DURATION_SEC
    return float(median(intervals))


def _load_noaa_prefix(bucket: Any, prefix: str) -> tuple[list[NoaaAudioFile], float]:
    files = list_noaa_objects(bucket, prefix)
    return files, estimate_noaa_interval_sec(files)


def _segments_from_files(
    files: list[NoaaAudioFile], default_interval_sec: float
) -> list[StreamSegment]:
    segments: list[StreamSegment] = []
    for idx, audio_file in enumerate(files):
        duration_sec = default_interval_sec
        if idx + 1 < len(files):
            next_gap = files[idx + 1].start_ts - audio_file.start_ts
            if 0 < next_gap < default_interval_sec:
                duration_sec = next_gap

        segments.append(
            StreamSegment(
                key=audio_file.key,
                start_ts=audio_file.start_ts,
                duration_sec=duration_sec,
            )
        )
    return segments


class NoaaGCSProvider:
    """ArchiveProvider for NOAA's public passive bioacoustic GCS archive."""

    def __init__(
        self,
        source_id: str,
        name: str,
        *,
        bucket: str = DEFAULT_NOAA_GCS_BUCKET,
        prefix: str = DEFAULT_NOAA_GCS_PREFIX,
        audio_subpath: str | None = None,
        child_folder_hints: list[Mapping[str, object]] | None = None,
        supports_segment_prefetch: bool = True,
        bucket_obj: Any | None = None,
    ) -> None:
        self._source_id = source_id
        self._name = name
        self._bucket_name = bucket
        self._prefix = _normalized_prefix(prefix)
        self._audio_subpath = (
            _normalized_prefix(audio_subpath) if audio_subpath is not None else None
        )
        self._child_folder_hints = _parse_child_folder_hints(child_folder_hints)
        self._supports_segment_prefetch = bool(supports_segment_prefetch)
        self._bucket = bucket_obj
        self._files_by_prefix: dict[str, list[NoaaAudioFile]] = {}
        self._default_interval_sec_by_prefix: dict[str, float] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def supports_segment_prefetch(self) -> bool:
        return self._supports_segment_prefetch

    @property
    def supports_chunked_segment_decode(self) -> bool:
        return True

    def _get_bucket(self) -> Any:
        if self._bucket is None:
            from google.cloud import storage

            client = storage.Client.create_anonymous_client()
            self._bucket = client.bucket(self._bucket_name)
        return self._bucket

    def _candidate_prefixes(
        self, start_ts: float | None = None, end_ts: float | None = None
    ) -> list[str]:
        if self._child_folder_hints:
            matching_hints = self._child_folder_hints
            if start_ts is not None and end_ts is not None:
                matching_hints = [
                    hint
                    for hint in self._child_folder_hints
                    if hint.overlaps(start_ts, end_ts)
                ]
            prefixes = [
                _join_prefix(self._prefix, hint.prefix, self._audio_subpath or "")
                for hint in matching_hints
            ]
            if prefixes:
                return list(dict.fromkeys(prefixes))
        return [self._prefix]

    def _load_prefix(self, prefix: str) -> tuple[list[NoaaAudioFile], float]:
        if prefix not in self._files_by_prefix:
            files, interval = _load_noaa_prefix(self._get_bucket(), prefix)
            self._files_by_prefix[prefix] = files
            self._default_interval_sec_by_prefix[prefix] = interval
        return (
            self._files_by_prefix[prefix],
            self._default_interval_sec_by_prefix[prefix],
        )

    def _build_segments(
        self, start_ts: float | None = None, end_ts: float | None = None
    ) -> tuple[list[StreamSegment], bool]:
        segments: list[StreamSegment] = []
        has_files = False
        for prefix in self._candidate_prefixes(start_ts, end_ts):
            files, default_interval = self._load_prefix(prefix)
            if files:
                has_files = True
            segments.extend(_segments_from_files(files, default_interval))
        segments.sort(key=lambda segment: segment.start_ts)
        return segments, has_files

    def build_timeline(self, start_ts: float, end_ts: float) -> list[StreamSegment]:
        segments, has_files = self._build_segments(start_ts, end_ts)
        timeline = [
            segment
            for segment in segments
            if segment.end_ts > start_ts and segment.start_ts < end_ts
        ]
        if not has_files:
            raise FileNotFoundError("No NOAA audio data found for this time range")
        if not timeline:
            raise FileNotFoundError("No NOAA stream segments found in requested range")
        return timeline

    def count_segments(self, start_ts: float, end_ts: float) -> int:
        segments, _ = self._build_segments(start_ts, end_ts)
        return sum(
            1
            for segment in segments
            if segment.end_ts > start_ts and segment.start_ts < end_ts
        )

    def fetch_segment(self, key: str) -> bytes:
        blob = self._get_bucket().blob(key)
        return blob.download_as_bytes()

    def decode_segment(self, raw_bytes: bytes, target_sr: int) -> np.ndarray:
        return decode_noaa_audio_bytes(raw_bytes, target_sr)

    def iter_decoded_segment_chunks(
        self,
        key: str,
        raw_bytes: bytes,
        target_sr: int,
        *,
        clip_start_sec: float,
        clip_end_sec: float,
        chunk_seconds: float,
    ) -> Generator[tuple[np.ndarray, float], None, None]:
        del key
        yield from iter_decode_noaa_audio_bytes(
            raw_bytes,
            target_sr,
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_end_sec,
            chunk_seconds=chunk_seconds,
        )

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
        interval = (
            estimate_noaa_interval_sec(files)
            if files
            else float(data["default_interval_sec"])
        )
        return files, interval
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
        audio_subpath: str | None = None,
        child_folder_hints: list[Mapping[str, object]] | None = None,
        supports_segment_prefetch: bool = True,
        bucket_obj: Any | None = None,
    ) -> None:
        self._source_id = source_id
        self._name = name
        self._cache_root = cache_root
        self._bucket_name = bucket
        self._prefix = _normalized_prefix(prefix)
        self._supports_segment_prefetch = bool(supports_segment_prefetch)
        self._gcs = NoaaGCSProvider(
            source_id,
            name,
            bucket=bucket,
            prefix=prefix,
            audio_subpath=audio_subpath,
            child_folder_hints=child_folder_hints,
            supports_segment_prefetch=supports_segment_prefetch,
            bucket_obj=bucket_obj,
        )
        self._files_by_prefix: dict[str, list[NoaaAudioFile]] = {}
        self._default_interval_sec_by_prefix: dict[str, float] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def supports_segment_prefetch(self) -> bool:
        return self._supports_segment_prefetch

    @property
    def supports_chunked_segment_decode(self) -> bool:
        return True

    def _load_prefix(self, prefix: str) -> tuple[list[NoaaAudioFile], float]:
        if prefix in self._files_by_prefix:
            return (
                self._files_by_prefix[prefix],
                self._default_interval_sec_by_prefix[prefix],
            )

        manifest = _manifest_path(self._cache_root, self._bucket_name, prefix)
        cached = read_noaa_manifest(manifest)
        if cached is None:
            files, interval = _load_noaa_prefix(self._gcs._get_bucket(), prefix)
            if files:
                write_noaa_manifest(manifest, files, interval)
        else:
            files, interval = cached

        self._files_by_prefix[prefix] = files
        self._default_interval_sec_by_prefix[prefix] = interval
        return files, interval

    def _build_segments(
        self, start_ts: float | None = None, end_ts: float | None = None
    ) -> tuple[list[StreamSegment], bool]:
        segments: list[StreamSegment] = []
        has_files = False
        for prefix in self._gcs._candidate_prefixes(start_ts, end_ts):
            files, default_interval = self._load_prefix(prefix)
            if files:
                has_files = True
            segments.extend(_segments_from_files(files, default_interval))
        segments.sort(key=lambda segment: segment.start_ts)
        return segments, has_files

    def build_timeline(self, start_ts: float, end_ts: float) -> list[StreamSegment]:
        segments, has_files = self._build_segments(start_ts, end_ts)
        timeline = [
            segment
            for segment in segments
            if segment.end_ts > start_ts and segment.start_ts < end_ts
        ]
        if not has_files:
            raise FileNotFoundError("No NOAA audio data found for this time range")
        if not timeline:
            raise FileNotFoundError("No NOAA stream segments found in requested range")
        return timeline

    def count_segments(self, start_ts: float, end_ts: float) -> int:
        segments, _ = self._build_segments(start_ts, end_ts)
        return sum(
            1
            for segment in segments
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

    def iter_decoded_segment_chunks(
        self,
        key: str,
        raw_bytes: bytes,
        target_sr: int,
        *,
        clip_start_sec: float,
        clip_end_sec: float,
        chunk_seconds: float,
    ) -> Generator[tuple[np.ndarray, float], None, None]:
        local_path = Path(self._cache_root) / self._bucket_name / key
        if local_path.is_file():
            yield from iter_decode_noaa_audio_file(
                local_path,
                target_sr,
                clip_start_sec=clip_start_sec,
                clip_end_sec=clip_end_sec,
                chunk_seconds=chunk_seconds,
            )
            return

        yield from iter_decode_noaa_audio_bytes(
            raw_bytes,
            target_sr,
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_end_sec,
            chunk_seconds=chunk_seconds,
        )

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
    audio_subpath: str | None = None,
    child_folder_hints: list[Mapping[str, object]] | None = None,
    supports_segment_prefetch: bool = True,
) -> NoaaGCSProvider | CachingNoaaGCSProvider:
    if noaa_cache_path:
        return CachingNoaaGCSProvider(
            source_id,
            name,
            noaa_cache_path,
            bucket=bucket,
            prefix=prefix,
            audio_subpath=audio_subpath,
            child_folder_hints=child_folder_hints,
            supports_segment_prefetch=supports_segment_prefetch,
        )
    return NoaaGCSProvider(
        source_id,
        name,
        bucket=bucket,
        prefix=prefix,
        audio_subpath=audio_subpath,
        child_folder_hints=child_folder_hints,
        supports_segment_prefetch=supports_segment_prefetch,
    )


def build_noaa_playback_provider(
    source_id: str,
    name: str,
    *,
    noaa_cache_path: str | None,
    bucket: str,
    prefix: str,
    audio_subpath: str | None = None,
    child_folder_hints: list[Mapping[str, object]] | None = None,
    supports_segment_prefetch: bool = True,
) -> NoaaGCSProvider | CachingNoaaGCSProvider:
    if noaa_cache_path:
        return CachingNoaaGCSProvider(
            source_id,
            name,
            noaa_cache_path,
            bucket=bucket,
            prefix=prefix,
            audio_subpath=audio_subpath,
            child_folder_hints=child_folder_hints,
            supports_segment_prefetch=supports_segment_prefetch,
        )
    return NoaaGCSProvider(
        source_id,
        name,
        bucket=bucket,
        prefix=prefix,
        audio_subpath=audio_subpath,
        child_folder_hints=child_folder_hints,
        supports_segment_prefetch=supports_segment_prefetch,
    )
