"""S3 and local HLS streaming clients for Orcasound hydrophone audio."""

import io
import json
import logging
import math
import os
import re
import struct
import subprocess
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from humpback.classifier.archive import ArchiveProvider, StreamSegment
from humpback.config import ORCASOUND_S3_BUCKET, Settings

logger = logging.getLogger(__name__)

DEFAULT_SEGMENT_DURATION_SEC = 10.0
STREAM_DISCONTINUITY_TOLERANCE_SEC = 0.25
DEFAULT_HYDROPHONE_TIMELINE_LOOKBACK_INCREMENT_HOURS = 4
DEFAULT_HYDROPHONE_TIMELINE_MAX_LOOKBACK_HOURS = 7 * 24
DEFAULT_HYDROPHONE_PREFETCH_WORKERS = 4
DEFAULT_HYDROPHONE_PREFETCH_INFLIGHT_SEGMENTS = 16
FOLDER_LOOKBACK_STEP_SEC = DEFAULT_HYDROPHONE_TIMELINE_LOOKBACK_INCREMENT_HOURS * 3600
MAX_HYDROPHONE_RANGE_SEC = DEFAULT_HYDROPHONE_TIMELINE_MAX_LOOKBACK_HOURS * 3600
_SEGMENT_FETCH_RETRIES = 3
_SEGMENT_FETCH_BACKOFF_BASE = 1.0  # seconds

_SEGMENT_INDEX_RE = re.compile(r"(\d+)(?=\.ts$)", re.IGNORECASE)
_SEGMENT_SUFFIX_RE = re.compile(r"^(.*?)(\d+)$")
_EXTINF_RE = re.compile(r"#EXTINF:([0-9.]+)")


class SegmentNotFoundError(Exception):
    """Raised when a segment is confirmed missing (404 cached)."""


class _LegacyHLSTimelineClient(Protocol):
    """Structural contract for legacy HLS timeline helpers."""

    def list_hls_folders(
        self, hydrophone_id: str, start_ts: float, end_ts: float, /
    ) -> Sequence[str]: ...

    def list_segments(self, hydrophone_id: str, folder_ts: str, /) -> Sequence[str]: ...

    def fetch_playlist(self, hydrophone_id: str, folder_ts: str, /) -> str | None: ...


# StreamSegment is defined in humpback.classifier.archive and re-exported here
# for backward compatibility.


def _segment_index_from_key(key: str) -> int | None:
    """Extract trailing numeric segment index from a .ts key name."""
    match = _SEGMENT_INDEX_RE.search(Path(key).name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _segment_sort_key(key: str) -> tuple[int, str, int, str]:
    """Sort key that prefers numeric segment suffix ordering when present."""
    name = Path(key).name
    stem = Path(key).stem
    match = _SEGMENT_SUFFIX_RE.match(stem)
    if not match:
        return (1, stem, 0, name)
    prefix, digits = match.groups()
    idx = int(digits)
    return (0, prefix, idx, name)


def _sort_segment_keys(keys: Iterable[str]) -> list[str]:
    """Sort segment keys chronologically by numeric suffix when available."""
    return sorted(keys, key=_segment_sort_key)


def _folder_in_candidate_range(folder_ts: int, start_ts: float, end_ts: float) -> bool:
    """Return True when folder timestamp is inside the requested folder range."""
    return start_ts <= folder_ts < end_ts


@lru_cache(maxsize=1)
def _hydrophone_timeline_lookback_seconds() -> tuple[int, int]:
    """Resolve hydrophone timeline lookback increment/max from runtime settings."""
    settings = Settings.from_repo_env()
    increment_sec = int(settings.hydrophone_timeline_lookback_increment_hours) * 3600
    max_sec = int(settings.hydrophone_timeline_max_lookback_hours) * 3600
    return increment_sec, max_sec


def _parse_playlist_segments(
    playlist_text: str,
    hydrophone_id: str,
    folder_ts: str,
    available_keys: set[str],
    default_duration_sec: float = DEFAULT_SEGMENT_DURATION_SEC,
) -> list[tuple[str, float, float]]:
    """Parse playlist into (segment_key, duration_sec, offset_from_folder_start_sec)."""
    ordered: list[tuple[str, float, float]] = []
    pending_duration: float | None = None
    offset_sec = 0.0

    for raw_line in playlist_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        duration_match = _EXTINF_RE.match(line)
        if duration_match:
            try:
                pending_duration = float(duration_match.group(1))
            except ValueError:
                pending_duration = None
            continue

        if line.startswith("#"):
            continue

        if not line.lower().endswith(".ts"):
            pending_duration = None
            continue

        segment_name = Path(line.split("?", 1)[0]).name
        key = f"{hydrophone_id}/hls/{folder_ts}/{segment_name}"
        if key not in available_keys and line in available_keys:
            key = line

        duration = (
            pending_duration
            if pending_duration and pending_duration > 0
            else default_duration_sec
        )
        if key in available_keys:
            ordered.append((key, duration, offset_sec))
        offset_sec += duration
        pending_duration = None

    return ordered


def _ordered_folder_segments(
    client: _LegacyHLSTimelineClient,
    hydrophone_id: str,
    folder_ts: str,
    default_duration_sec: float = DEFAULT_SEGMENT_DURATION_SEC,
) -> list[tuple[str, float, float | None]]:
    """Get ordered (segment_key, duration, optional offset) tuples for one folder."""
    segment_keys = _sort_segment_keys(client.list_segments(hydrophone_id, folder_ts))
    if not segment_keys:
        return []

    available_keys = set(segment_keys)
    playlist_text: str | None = None
    fetch_playlist = getattr(client, "fetch_playlist", None)
    if callable(fetch_playlist):
        try:
            fetched = fetch_playlist(hydrophone_id, folder_ts)
            if isinstance(fetched, str):
                playlist_text = fetched
        except Exception:
            logger.debug(
                "Failed to fetch playlist for %s/%s; falling back to key ordering",
                hydrophone_id,
                folder_ts,
                exc_info=True,
            )

    if playlist_text:
        playlist_entries = _parse_playlist_segments(
            playlist_text,
            hydrophone_id,
            folder_ts,
            available_keys,
            default_duration_sec=default_duration_sec,
        )
        if playlist_entries:
            # Keep numeric key ordering authoritative to avoid lexicographic
            # ordering artifacts from object listings/playlists.
            duration_by_key: dict[str, float] = {}
            offset_by_key: dict[str, float] = {}
            for key, duration, offset_sec in playlist_entries:
                if key not in duration_by_key:
                    duration_by_key[key] = duration
                    offset_by_key[key] = offset_sec
            return [
                (
                    key,
                    duration_by_key.get(key, default_duration_sec),
                    offset_by_key.get(key),
                )
                for key in segment_keys
            ]

    return [(key, default_duration_sec, None) for key in segment_keys]


class OrcasoundS3Client:
    """Anonymous S3 client for Orcasound public bucket."""

    def __init__(self):
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        self._client = boto3.client(
            "s3",
            config=Config(
                signature_version=UNSIGNED,
                retries={"max_attempts": 5, "mode": "adaptive"},
                connect_timeout=10,
                read_timeout=30,
            ),
        )
        self._bucket = ORCASOUND_S3_BUCKET

    def list_hls_folders(
        self, hydrophone_id: str, start_ts: float, end_ts: float
    ) -> list[str]:
        """List HLS folder prefixes (unix timestamps) within time range.

        Returns folder timestamps as strings, sorted chronologically.
        """
        prefix = f"{hydrophone_id}/hls/"
        paginator = self._client.get_paginator("list_objects_v2")
        folders: list[str] = []

        for page in paginator.paginate(
            Bucket=self._bucket, Prefix=prefix, Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                # e.g. "rpi_orcasound_lab/hls/1709312400/"
                folder_key = cp["Prefix"]
                ts_str = folder_key.rstrip("/").split("/")[-1]
                try:
                    ts = int(ts_str)
                except ValueError:
                    continue
                # Include folders that could contain audio in our range
                # using a bounded 7-day lookback from requested start.
                if _folder_in_candidate_range(ts, start_ts, end_ts):
                    folders.append(ts_str)

        return sorted(folders, key=int)

    def list_segments(self, hydrophone_id: str, folder_ts: str) -> list[str]:
        """List .ts segment keys in an HLS folder, sorted."""
        prefix = f"{hydrophone_id}/hls/{folder_ts}/"
        paginator = self._client.get_paginator("list_objects_v2")
        segments: list[str] = []

        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".ts"):
                    segments.append(key)

        return _sort_segment_keys(segments)

    def fetch_playlist(self, hydrophone_id: str, folder_ts: str) -> str | None:
        """Fetch live.m3u8 playlist text for an HLS folder."""
        key = f"{hydrophone_id}/hls/{folder_ts}/live.m3u8"
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
        except Exception:
            return None
        data = resp["Body"].read()
        return data.decode("utf-8", errors="replace")

    def fetch_segment(self, key: str) -> bytes:
        """Download a single .ts segment as bytes, with retry for transient errors."""
        from botocore.exceptions import (
            ClientError,
            ConnectionError as BotoConnectionError,
            EndpointConnectionError,
            ReadTimeoutError,
        )
        from urllib3.exceptions import IncompleteRead

        last_exc: Exception | None = None
        for attempt in range(_SEGMENT_FETCH_RETRIES):
            try:
                resp = self._client.get_object(Bucket=self._bucket, Key=key)
                return resp["Body"].read()
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("NoSuchKey", "404", "AccessDenied"):
                    raise
                last_exc = e
            except (
                IncompleteRead,
                ReadTimeoutError,
                BotoConnectionError,
                EndpointConnectionError,
                ConnectionResetError,
                OSError,
            ) as e:
                last_exc = e
            except Exception:
                raise

            if attempt < _SEGMENT_FETCH_RETRIES - 1:
                wait = _SEGMENT_FETCH_BACKOFF_BASE * (2**attempt)
                logger.warning(
                    "Retrying segment %s (attempt %d/%d) after error: %s",
                    key,
                    attempt + 1,
                    _SEGMENT_FETCH_RETRIES,
                    last_exc,
                )
                time.sleep(wait)

        assert last_exc is not None
        raise last_exc

    def count_segments(self, hydrophone_id: str, folder_timestamps: list[str]) -> int:
        """Count total .ts segments across folders (for progress estimation)."""
        total = 0
        for folder_ts in folder_timestamps:
            segs = self.list_segments(hydrophone_id, folder_ts)
            total += len(segs)
        return total


class LocalHLSClient:
    """Local filesystem client mirroring OrcasoundS3Client interface.

    Reads from a local cache directory that mirrors the S3 bucket structure:
    {cache_root}/{bucket}/{hydrophone_id}/hls/{timestamp}/{segments}.ts
    """

    def __init__(self, cache_root: str):
        self._root = Path(cache_root) / ORCASOUND_S3_BUCKET

    def list_hls_folders(
        self, hydrophone_id: str, start_ts: float, end_ts: float
    ) -> list[str]:
        """List HLS folder timestamps within time range from local cache.

        Skips folders that contain only .404.json marker files (no real
        .ts segments).  These markers indicate data that was missing on
        S3 at download time.
        """
        hls_dir = self._root / hydrophone_id / "hls"
        if not hls_dir.is_dir():
            return []
        folders: list[str] = []
        for entry in hls_dir.iterdir():
            if not entry.is_dir():
                continue
            try:
                ts = int(entry.name)
            except ValueError:
                continue
            if _folder_in_candidate_range(ts, start_ts, end_ts):
                # Quick check: skip folders with no .ts files
                has_ts = any(f.suffix == ".ts" for f in entry.iterdir())
                if has_ts:
                    folders.append(entry.name)
        return sorted(folders, key=int)

    def list_segments(self, hydrophone_id: str, folder_ts: str) -> list[str]:
        """List .ts segment keys in an HLS folder from local cache."""
        folder = self._root / hydrophone_id / "hls" / folder_ts
        if not folder.is_dir():
            return []
        segments: list[str] = []
        for entry in folder.iterdir():
            if entry.suffix == ".ts":
                # Return key in S3-style format for consistency
                segments.append(f"{hydrophone_id}/hls/{folder_ts}/{entry.name}")
        return _sort_segment_keys(segments)

    def fetch_playlist(self, hydrophone_id: str, folder_ts: str) -> str | None:
        """Read live.m3u8 playlist text from local filesystem cache."""
        path = self._root / hydrophone_id / "hls" / folder_ts / "live.m3u8"
        if not path.is_file():
            return None
        return path.read_text(errors="replace")

    def fetch_segment(self, key: str) -> bytes:
        """Read a .ts segment from local filesystem."""
        path = self._root / key
        return path.read_bytes()

    def invalidate_cached_segment(self, key: str) -> bool:
        """Delete a cached segment file. Returns True if file was deleted."""
        path = self._root / key
        if path.is_file():
            path.unlink()
            logger.info("Invalidated cached segment: %s", key)
            return True
        return False

    def count_segments(self, hydrophone_id: str, folder_timestamps: list[str]) -> int:
        """Count total .ts segments across folders."""
        total = 0
        for folder_ts in folder_timestamps:
            total += len(self.list_segments(hydrophone_id, folder_ts))
        return total


class CachingS3Client:
    """Write-through S3 cache: fetches from S3 and caches on local filesystem.

    Mirrors the S3 bucket directory structure under cache_root.
    Uses .404.json markers for confirmed-missing segments/folders.
    """

    def __init__(self, cache_root: str):
        self._s3 = OrcasoundS3Client()
        self._root = Path(cache_root) / ORCASOUND_S3_BUCKET

    def list_hls_folders(
        self, hydrophone_id: str, start_ts: float, end_ts: float
    ) -> list[str]:
        """Merge S3 folder list with locally cached folders."""
        # Query S3 for authoritative list
        try:
            s3_folders = self._s3.list_hls_folders(hydrophone_id, start_ts, end_ts)
        except Exception:
            logger.warning("S3 list_hls_folders failed, using cache only")
            s3_folders = []

        # Also check local cache for folders with .ts files in range
        hls_dir = self._root / hydrophone_id / "hls"
        local_folders: list[str] = []
        if hls_dir.is_dir():
            for entry in hls_dir.iterdir():
                if not entry.is_dir():
                    continue
                try:
                    ts = int(entry.name)
                except ValueError:
                    continue
                if _folder_in_candidate_range(ts, start_ts, end_ts):
                    has_ts = any(f.suffix == ".ts" for f in entry.iterdir())
                    if has_ts:
                        local_folders.append(entry.name)

        # Merge and deduplicate
        all_folders = set(s3_folders) | set(local_folders)
        return sorted(all_folders, key=int)

    def list_segments(self, hydrophone_id: str, folder_ts: str) -> list[str]:
        """List segments, merging local cache with S3."""
        folder_dir = self._root / hydrophone_id / "hls" / folder_ts
        marker_path = folder_dir / ".404.json"

        # Collect local .ts files
        local_keys: set[str] = set()
        if folder_dir.is_dir():
            for entry in folder_dir.iterdir():
                if entry.suffix == ".ts":
                    local_keys.add(f"{hydrophone_id}/hls/{folder_ts}/{entry.name}")

        # If folder marked as 404, return local-only
        if marker_path.is_file():
            return _sort_segment_keys(local_keys)

        # Query S3
        try:
            s3_keys = self._s3.list_segments(hydrophone_id, folder_ts)
        except Exception:
            logger.warning(
                "S3 list_segments failed for %s/%s, using cache only",
                hydrophone_id,
                folder_ts,
            )
            return _sort_segment_keys(local_keys)

        if not s3_keys and not local_keys:
            # Mark folder as empty
            folder_dir.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(
                json.dumps({"cached_at_utc": datetime.now(timezone.utc).isoformat()})
            )
            return []

        # Merge and deduplicate
        all_keys = local_keys | set(s3_keys)
        return _sort_segment_keys(all_keys)

    def fetch_playlist(self, hydrophone_id: str, folder_ts: str) -> str | None:
        """Fetch playlist text, preferring local cache then S3 with write-through."""
        local_path = self._root / hydrophone_id / "hls" / folder_ts / "live.m3u8"
        if local_path.is_file():
            return local_path.read_text(errors="replace")

        try:
            text = self._s3.fetch_playlist(hydrophone_id, folder_ts)
        except Exception:
            return None
        if text is None:
            return None

        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_suffix(".tmp")
        tmp_path.write_text(text)
        os.replace(str(tmp_path), str(local_path))
        return text

    def fetch_segment(self, key: str) -> bytes:
        """Fetch segment, using local cache or S3 with write-through."""
        local_path = self._root / key
        marker_path = local_path.parent / f"{local_path.name}.404.json"

        # Cache hit
        if local_path.is_file():
            return local_path.read_bytes()

        # Known missing
        if marker_path.is_file():
            raise SegmentNotFoundError(f"Segment confirmed missing: {key}")

        # Fetch from S3
        try:
            data = self._s3.fetch_segment(key)
        except Exception as e:
            # Check if it's a botocore ClientError with 404/NoSuchKey
            resp = getattr(e, "response", None)
            code = ""
            if isinstance(resp, dict):
                code = resp.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                local_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.write_text(
                    json.dumps(
                        {"cached_at_utc": datetime.now(timezone.utc).isoformat()}
                    )
                )
                raise SegmentNotFoundError(f"Segment not found on S3: {key}") from e
            raise

        # Atomic write to cache
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_suffix(".tmp")
        tmp_path.write_bytes(data)
        os.replace(str(tmp_path), str(local_path))

        return data

    def invalidate_cached_segment(self, key: str) -> bool:
        """Delete a cached segment file so it can be re-fetched from S3."""
        local_path = self._root / key
        if local_path.is_file():
            local_path.unlink()
            logger.info("Invalidated cached segment: %s", key)
            return True
        return False

    def count_segments(self, hydrophone_id: str, folder_timestamps: list[str]) -> int:
        """Count total .ts segments across folders."""
        total = 0
        for folder_ts in folder_timestamps:
            total += len(self.list_segments(hydrophone_id, folder_ts))
        return total


def decode_ts_bytes(ts_bytes: bytes, target_sr: int = 32000) -> np.ndarray:
    """Decode HLS .ts segment bytes to float32 audio array via ffmpeg.

    All processing in memory — no disk I/O.
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
        input=ts_bytes,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr[:500]}")

    wav_bytes = result.stdout
    # Parse WAV header to get to data
    # Minimal WAV parse: find "data" chunk
    buf = io.BytesIO(wav_bytes)
    # Skip RIFF header (12 bytes)
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


def _parse_chunk_start_timestamp(filename: str) -> float:
    """Parse synthetic chunk filename (YYYYMMDDTHHMMSSZ.wav) to unix timestamp."""
    basename = filename[:-4] if filename.endswith(".wav") else filename
    try:
        chunk_utc = datetime.strptime(basename, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"Invalid hydrophone filename format: {filename}") from exc
    return chunk_utc.timestamp()


def _build_stream_timeline(
    client: _LegacyHLSTimelineClient,
    hydrophone_id: str,
    stream_start_ts: float,
    stream_end_ts: float,
) -> list[StreamSegment]:
    """Build ordered stream segments for a hydrophone job range."""
    timeline: list[StreamSegment] = []
    seen_folders: set[str] = set()
    found_any_folders = False
    start_boundary_covered = False
    lookback_step_sec, max_lookback_sec = _hydrophone_timeline_lookback_seconds()
    max_lookback_steps = int(math.ceil(max_lookback_sec / lookback_step_sec))
    lookback_step = 0
    jumped_to_max_lookback = False
    while lookback_step <= max_lookback_steps:
        lookback_sec = min(lookback_step * lookback_step_sec, max_lookback_sec)
        window_start_ts = stream_start_ts - lookback_sec
        folders = client.list_hls_folders(hydrophone_id, window_start_ts, stream_end_ts)
        if not folders:
            lookback_step += 1
            continue
        found_any_folders = True

        for folder_ts in sorted(folders, key=int):
            if folder_ts in seen_folders:
                continue
            seen_folders.add(folder_ts)

            ordered = _ordered_folder_segments(client, hydrophone_id, folder_ts)
            if not ordered:
                continue
            folder_start_ts = float(folder_ts)
            cursor = folder_start_ts
            for key, duration_sec, offset_sec in ordered:
                duration = (
                    duration_sec if duration_sec > 0 else DEFAULT_SEGMENT_DURATION_SEC
                )
                if offset_sec is not None:
                    segment_start_ts = folder_start_ts + offset_sec
                    cursor = segment_start_ts + duration
                else:
                    segment_start_ts = cursor
                    cursor += duration
                segment = StreamSegment(
                    key=key,
                    start_ts=segment_start_ts,
                    duration_sec=duration,
                )
                if (
                    segment.end_ts <= stream_start_ts
                    or segment.start_ts >= stream_end_ts
                ):
                    continue
                timeline.append(segment)
                if segment.start_ts <= stream_start_ts < segment.end_ts:
                    start_boundary_covered = True

        if start_boundary_covered:
            break

        if (
            timeline
            and not jumped_to_max_lookback
            and lookback_step < max_lookback_steps
        ):
            # We found overlap in-range but not at the requested start boundary.
            # Jump straight to max lookback to avoid N incremental list calls.
            lookback_step = max_lookback_steps
            jumped_to_max_lookback = True
            continue

        lookback_step += 1

    if not found_any_folders:
        raise FileNotFoundError("No audio data found for this time range")

    if not timeline:
        raise FileNotFoundError("No stream segments found in requested range")

    timeline.sort(key=lambda seg: (seg.start_ts, _segment_sort_key(seg.key)))
    return timeline


def build_stream_timeline(
    provider: ArchiveProvider,
    stream_start_ts: float,
    stream_end_ts: float,
) -> list[StreamSegment]:
    """Build stream timeline via an ArchiveProvider."""
    return provider.build_timeline(stream_start_ts, stream_end_ts)


def _decode_and_clip_segment(
    provider: ArchiveProvider,
    segment: StreamSegment,
    target_sr: int,
    clip_start_ts: float,
    clip_end_ts: float,
    prefetched_bytes: bytes | None = None,
    prefetched_fetch_sec: float = 0.0,
    timing_callback: Callable[[float, float], None] | None = None,
) -> tuple[np.ndarray, float] | None:
    """Decode one segment and clip it to the requested absolute time interval."""
    fetch_sec = max(0.0, float(prefetched_fetch_sec))

    if prefetched_bytes is None:
        t_fetch = time.monotonic()
        try:
            seg_bytes = provider.fetch_segment(segment.key)
        except Exception:
            fetch_sec = time.monotonic() - t_fetch
            if timing_callback is not None:
                timing_callback(fetch_sec, 0.0)
            raise
        fetch_sec = time.monotonic() - t_fetch
    else:
        seg_bytes = prefetched_bytes

    t_decode = time.monotonic()
    try:
        audio = provider.decode_segment(seg_bytes, target_sr)
    except Exception:
        decode_sec = time.monotonic() - t_decode
        if timing_callback is not None:
            timing_callback(fetch_sec, decode_sec)
        raise
    decode_sec = time.monotonic() - t_decode
    if timing_callback is not None:
        timing_callback(fetch_sec, decode_sec)

    if len(audio) == 0:
        return None

    decoded_end_ts = segment.start_ts + (len(audio) / target_sr)
    start_ts = max(segment.start_ts, clip_start_ts)
    end_ts = min(decoded_end_ts, clip_end_ts)
    if end_ts <= start_ts:
        return None

    start_sample = max(0, int(round((start_ts - segment.start_ts) * target_sr)))
    end_sample = min(len(audio), int(round((end_ts - segment.start_ts) * target_sr)))
    if end_sample <= start_sample:
        return None

    return audio[start_sample:end_sample], start_ts


def provider_supports_segment_prefetch(provider: ArchiveProvider) -> bool:
    """Return whether raw-byte segment prefetch should be used for a provider."""
    return bool(getattr(provider, "supports_segment_prefetch", True))


def _provider_supports_chunked_segment_decode(provider: ArchiveProvider) -> bool:
    """Return whether a provider opts into chunk-wise segment decode."""
    instance_value = getattr(provider, "__dict__", {}).get(
        "supports_chunked_segment_decode"
    )
    if instance_value is not None:
        return bool(instance_value)

    class_value = getattr(type(provider), "supports_chunked_segment_decode", False)
    if isinstance(class_value, property):
        return bool(class_value.__get__(provider, type(provider)))
    return bool(class_value)


def _iter_segment_audio_chunks(
    provider: ArchiveProvider,
    segment: StreamSegment,
    target_sr: int,
    clip_start_ts: float,
    clip_end_ts: float,
    chunk_seconds: float,
    prefetched_bytes: bytes | None = None,
    prefetched_fetch_sec: float = 0.0,
    timing_callback: Callable[[float, float], None] | None = None,
) -> Generator[tuple[np.ndarray, float], None, None]:
    """Yield decoded audio for one timeline segment in chronological order."""
    if not _provider_supports_chunked_segment_decode(provider):
        clipped = _decode_and_clip_segment(
            provider=provider,
            segment=segment,
            target_sr=target_sr,
            clip_start_ts=clip_start_ts,
            clip_end_ts=clip_end_ts,
            prefetched_bytes=prefetched_bytes,
            prefetched_fetch_sec=prefetched_fetch_sec,
            timing_callback=timing_callback,
        )
        if clipped is not None:
            yield clipped
        return

    chunk_decoder = cast(Any, getattr(provider, "iter_decoded_segment_chunks"))

    fetch_sec = max(0.0, float(prefetched_fetch_sec))
    if prefetched_bytes is None:
        cache_check = getattr(provider, "is_segment_cached", None)
        if cache_check is not None and cache_check(segment.key):
            seg_bytes = b""
            fetch_sec = 0.0
        else:
            t_fetch = time.monotonic()
            try:
                seg_bytes = provider.fetch_segment(segment.key)
            except Exception:
                fetch_sec = time.monotonic() - t_fetch
                if timing_callback is not None:
                    timing_callback(fetch_sec, 0.0)
                raise
            fetch_sec = time.monotonic() - t_fetch
    else:
        seg_bytes = prefetched_bytes

    clipped_start_ts = max(segment.start_ts, clip_start_ts)
    clipped_end_ts = min(segment.end_ts, clip_end_ts)
    if clipped_end_ts <= clipped_start_ts:
        if timing_callback is not None:
            timing_callback(fetch_sec, 0.0)
        return

    t_decode = time.monotonic()
    try:
        for audio, offset_sec in chunk_decoder(
            segment.key,
            seg_bytes,
            target_sr,
            clip_start_sec=clipped_start_ts - segment.start_ts,
            clip_end_sec=clipped_end_ts - segment.start_ts,
            chunk_seconds=chunk_seconds,
        ):
            if len(audio) == 0:
                continue
            yield audio, segment.start_ts + offset_sec
    except Exception:
        decode_sec = time.monotonic() - t_decode
        if timing_callback is not None:
            timing_callback(fetch_sec, decode_sec)
        raise

    decode_sec = time.monotonic() - t_decode
    if timing_callback is not None:
        timing_callback(fetch_sec, decode_sec)


def _iter_prefetched_segments(
    provider: ArchiveProvider,
    segments: Iterable[StreamSegment],
    workers: int,
    inflight: int,
) -> Generator[tuple[StreamSegment, bytes | None, float, Exception | None], None, None]:
    """Yield segments with optionally prefetched bytes in timeline order."""
    max_workers = max(1, int(workers))
    max_inflight = max(1, int(inflight))
    pending_limit = max_inflight

    def _timed_fetch(
        segment: StreamSegment,
    ) -> tuple[bytes | None, float, Exception | None]:
        t_fetch = time.monotonic()
        try:
            data = provider.fetch_segment(segment.key)
            return data, time.monotonic() - t_fetch, None
        except Exception as exc:
            return None, time.monotonic() - t_fetch, exc

    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="hydrophone-prefetch",
    )
    pending: deque[
        tuple[StreamSegment, Future[tuple[bytes | None, float, Exception | None]]]
    ] = deque()
    segment_iter = iter(segments)

    def _schedule_one() -> bool:
        try:
            segment = next(segment_iter)
        except StopIteration:
            return False
        pending.append((segment, executor.submit(_timed_fetch, segment)))
        return True

    try:
        while len(pending) < pending_limit and _schedule_one():
            pass

        while pending:
            segment, future = pending.popleft()
            try:
                data, fetch_sec, fetch_exc = future.result()
            except Exception as exc:
                data, fetch_sec, fetch_exc = None, 0.0, exc

            yield segment, data, fetch_sec, fetch_exc

            while len(pending) < pending_limit and _schedule_one():
                pass
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def resolve_audio_slice(
    provider: ArchiveProvider,
    stream_start_ts: float,
    stream_end_ts: float,
    filename: str,
    row_start_sec: float,
    duration_sec: float,
    target_sr: int = 32000,
    legacy_anchor_start_ts: float | None = None,
    timeline: list[StreamSegment] | None = None,
    processing_start_ts: float | None = None,
) -> np.ndarray:
    """Resolve and decode an audio slice using stream-offset mapping.

    Anchor order:
    1. First available stream sample in job range (current behavior)
    2. legacy_anchor_start_ts (old jobs where filenames were anchored to job start)
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")

    if timeline is None:
        timeline = provider.build_timeline(stream_start_ts, stream_end_ts)
    if not timeline:
        raise FileNotFoundError("No stream segments found in requested range")

    if processing_start_ts is None:
        processing_start_ts = max(stream_start_ts, timeline[0].start_ts)
    else:
        processing_start_ts = float(processing_start_ts)
    chunk_start_ts = _parse_chunk_start_timestamp(filename)
    abs_start_ts = chunk_start_ts + row_start_sec

    anchors: list[float] = [processing_start_ts]
    if legacy_anchor_start_ts is not None:
        legacy_anchor = float(legacy_anchor_start_ts)
        if all(abs(legacy_anchor - a) > 1e-6 for a in anchors):
            anchors.append(legacy_anchor)

    last_reason = "No matching stream offset"
    max_timeline_ts = min(stream_end_ts, timeline[-1].end_ts)

    for anchor_ts in anchors:
        stream_offset = abs_start_ts - anchor_ts
        if stream_offset < 0:
            last_reason = "Computed negative stream offset"
            continue

        target_start_ts = processing_start_ts + stream_offset
        target_end_ts = target_start_ts + duration_sec

        if target_start_ts >= max_timeline_ts:
            last_reason = "Resolved start timestamp outside available stream range"
            continue

        candidates = [
            seg
            for seg in timeline
            if seg.end_ts > target_start_ts and seg.start_ts < target_end_ts
        ]
        if not candidates:
            last_reason = "No segments selected for requested timestamp window"
            continue

        decoded_parts: list[np.ndarray] = []
        for segment in candidates:
            try:
                for audio, _ts in _iter_segment_audio_chunks(
                    provider=provider,
                    segment=segment,
                    target_sr=target_sr,
                    clip_start_ts=target_start_ts,
                    clip_end_ts=target_end_ts,
                    chunk_seconds=duration_sec,
                ):
                    if len(audio) > 0:
                        decoded_parts.append(audio)
            except Exception:
                continue

        if not decoded_parts:
            last_reason = "Failed to decode all candidate segments"
            continue

        segment = np.concatenate(decoded_parts)
        max_samples = max(1, int(round(duration_sec * target_sr)))
        if len(segment) > max_samples:
            segment = segment[:max_samples]
        if len(segment) > 0:
            return segment

        last_reason = "Resolved segment was empty"

    raise FileNotFoundError(
        f"Could not resolve audio slice for {filename}: {last_reason}"
    )


def iter_audio_chunks(
    provider: ArchiveProvider,
    start_ts: float,
    end_ts: float,
    **kwargs,
) -> Generator:
    """Yield audio chunks from an ArchiveProvider."""
    yield from _iter_audio_chunks(provider, start_ts, end_ts, **kwargs)


def _iter_audio_chunks(
    provider: ArchiveProvider,
    start_ts: float,
    end_ts: float,
    chunk_seconds: float = 60.0,
    target_sr: int = 32000,
    on_error: Callable[[dict], None] | None = None,
    skip_segments: int = 0,
    prefetch_enabled: bool = False,
    prefetch_workers: int = DEFAULT_HYDROPHONE_PREFETCH_WORKERS,
    prefetch_inflight_segments: int = DEFAULT_HYDROPHONE_PREFETCH_INFLIGHT_SEGMENTS,
    on_segment_timing: Callable[[float, float], None] | None = None,
) -> Generator:
    """Core implementation for iter_audio_chunks."""
    timeline = provider.build_timeline(start_ts, end_ts)

    if skip_segments > 0:
        if skip_segments > len(timeline):
            logger.warning(
                "skip_segments=%d exceeds timeline length=%d; resetting to 0",
                skip_segments,
                len(timeline),
            )
            skip_segments = 0
        else:
            logger.info(
                "Resuming: skipping %d/%d segments",
                skip_segments,
                len(timeline),
            )

    chunk_samples = int(chunk_seconds * target_sr)
    if chunk_samples <= 0:
        raise ValueError("chunk_seconds must produce at least 1 sample")

    segments_total = len(timeline)
    accumulator = np.array([], dtype=np.float32)
    accumulator_start_ts: float | None = None
    segments_done = skip_segments
    use_prefetch = (
        prefetch_enabled
        and prefetch_workers > 1
        and prefetch_inflight_segments > 1
        and provider_supports_segment_prefetch(provider)
    )

    def _push_audio(
        audio: np.ndarray,
        clip_start_ts: float,
        progress_done: int,
    ) -> Generator[tuple[np.ndarray, datetime, int, int], None, None]:
        nonlocal accumulator, accumulator_start_ts

        if len(audio) == 0:
            return

        if len(accumulator) == 0:
            accumulator = audio
            accumulator_start_ts = clip_start_ts
        else:
            assert accumulator_start_ts is not None
            expected_next_ts = accumulator_start_ts + (len(accumulator) / target_sr)
            if (
                abs(clip_start_ts - expected_next_ts)
                > STREAM_DISCONTINUITY_TOLERANCE_SEC
            ):
                chunk_start_utc = datetime.fromtimestamp(
                    accumulator_start_ts, tz=timezone.utc
                )
                yield accumulator, chunk_start_utc, progress_done, segments_total
                accumulator = audio
                accumulator_start_ts = clip_start_ts
            else:
                accumulator = np.concatenate([accumulator, audio])

        while len(accumulator) >= chunk_samples:
            assert accumulator_start_ts is not None
            chunk = accumulator[:chunk_samples]
            chunk_start_utc = datetime.fromtimestamp(
                accumulator_start_ts, tz=timezone.utc
            )
            yield chunk, chunk_start_utc, progress_done, segments_total
            accumulator = accumulator[chunk_samples:]
            accumulator_start_ts += chunk_samples / target_sr

    if use_prefetch:
        segment_source = _iter_prefetched_segments(
            provider,
            timeline[skip_segments:],
            workers=prefetch_workers,
            inflight=prefetch_inflight_segments,
        )
    else:
        segment_source = (
            (segment, None, 0.0, None) for segment in timeline[skip_segments:]
        )

    for (
        segment,
        prefetched_bytes,
        prefetched_fetch_sec,
        prefetched_exc,
    ) in segment_source:
        progress_done = segments_done + 1
        try:
            if prefetched_exc is not None:
                if on_segment_timing is not None:
                    on_segment_timing(prefetched_fetch_sec, 0.0)
                raise prefetched_exc

            for audio, clip_start_ts in _iter_segment_audio_chunks(
                provider=provider,
                segment=segment,
                target_sr=target_sr,
                clip_start_ts=start_ts,
                clip_end_ts=end_ts,
                chunk_seconds=chunk_seconds,
                prefetched_bytes=prefetched_bytes,
                prefetched_fetch_sec=prefetched_fetch_sec,
                timing_callback=on_segment_timing,
            ):
                yield from _push_audio(audio, clip_start_ts, progress_done)
            segments_done = progress_done

        except Exception as e:
            # If provider supports cache invalidation, delete the cached segment
            # and retry once — the cached file may be corrupted/truncated.
            if provider.invalidate_cached_segment(segment.key):
                try:
                    for audio, clip_start_ts in _iter_segment_audio_chunks(
                        provider=provider,
                        segment=segment,
                        target_sr=target_sr,
                        clip_start_ts=start_ts,
                        clip_end_ts=end_ts,
                        chunk_seconds=chunk_seconds,
                        timing_callback=on_segment_timing,
                    ):
                        yield from _push_audio(audio, clip_start_ts, progress_done)
                    logger.info(
                        "Retry after cache invalidation succeeded for %s", segment.key
                    )
                    segments_done = progress_done
                    continue
                except Exception as e2:
                    logger.warning(
                        "Retry after cache invalidation also failed for %s: %s",
                        segment.key,
                        e2,
                    )

            segments_done = progress_done
            logger.warning("Failed to decode segment %s: %s", segment.key, e)
            if on_error:
                on_error(
                    {
                        "type": "warning",
                        "message": f"Failed to decode segment {segment.key}: {e}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

    # Yield remaining audio
    if len(accumulator) > 0 and accumulator_start_ts is not None:
        chunk_start_utc = datetime.fromtimestamp(accumulator_start_ts, tz=timezone.utc)
        yield accumulator, chunk_start_utc, segments_done, segments_total
