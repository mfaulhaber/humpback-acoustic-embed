"""S3 and local HLS streaming clients for Orcasound hydrophone audio."""

import io
import json
import logging
import os
import struct
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from humpback.config import ORCASOUND_S3_BUCKET

logger = logging.getLogger(__name__)


class SegmentNotFoundError(Exception):
    """Raised when a segment is confirmed missing (404 cached)."""


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
                # HLS folders typically span ~minutes, so include generously
                if ts <= end_ts and ts >= start_ts - 3600:
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

        return sorted(segments)

    def fetch_segment(self, key: str) -> bytes:
        """Download a single .ts segment as bytes."""
        resp = self._client.get_object(Bucket=self._bucket, Key=key)
        return resp["Body"].read()

    def count_segments(
        self, hydrophone_id: str, folder_timestamps: list[str]
    ) -> int:
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
            if ts <= end_ts and ts >= start_ts - 3600:
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
        for entry in sorted(folder.iterdir()):
            if entry.suffix == ".ts":
                # Return key in S3-style format for consistency
                segments.append(
                    f"{hydrophone_id}/hls/{folder_ts}/{entry.name}"
                )
        return segments

    def fetch_segment(self, key: str) -> bytes:
        """Read a .ts segment from local filesystem."""
        path = self._root / key
        return path.read_bytes()

    def count_segments(
        self, hydrophone_id: str, folder_timestamps: list[str]
    ) -> int:
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
                if ts <= end_ts and ts >= start_ts - 3600:
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
                    local_keys.add(
                        f"{hydrophone_id}/hls/{folder_ts}/{entry.name}"
                    )

        # If folder marked as 404, return local-only
        if marker_path.is_file():
            return sorted(local_keys)

        # Query S3
        try:
            s3_keys = self._s3.list_segments(hydrophone_id, folder_ts)
        except Exception:
            logger.warning(
                "S3 list_segments failed for %s/%s, using cache only",
                hydrophone_id, folder_ts,
            )
            return sorted(local_keys)

        if not s3_keys and not local_keys:
            # Mark folder as empty
            folder_dir.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(
                json.dumps({"cached_at_utc": datetime.now(timezone.utc).isoformat()})
            )
            return []

        # Merge and deduplicate
        all_keys = local_keys | set(s3_keys)
        return sorted(all_keys)

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
                    json.dumps({"cached_at_utc": datetime.now(timezone.utc).isoformat()})
                )
                raise SegmentNotFoundError(f"Segment not found on S3: {key}") from e
            raise

        # Atomic write to cache
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_suffix(".tmp")
        tmp_path.write_bytes(data)
        os.replace(str(tmp_path), str(local_path))

        return data

    def count_segments(
        self, hydrophone_id: str, folder_timestamps: list[str]
    ) -> int:
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
            "ffmpeg", "-i", "pipe:0",
            "-f", "wav", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(target_sr),
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


def _build_stream_segment_index(
    client: "OrcasoundS3Client | LocalHLSClient | CachingS3Client",
    hydrophone_id: str,
    stream_start_ts: float,
    stream_end_ts: float,
) -> tuple[list[str], list[str]]:
    """Build ordered folder and segment lists for a hydrophone job range."""
    folders = client.list_hls_folders(hydrophone_id, stream_start_ts, stream_end_ts)
    if not folders:
        raise FileNotFoundError("No audio data found for this time range")

    all_segments: list[str] = []
    for folder_ts in folders:
        all_segments.extend(client.list_segments(hydrophone_id, folder_ts))
    if not all_segments:
        raise FileNotFoundError("No cached segments found")

    return folders, all_segments


def resolve_hydrophone_audio_slice(
    client: "OrcasoundS3Client | LocalHLSClient | CachingS3Client",
    hydrophone_id: str,
    stream_start_ts: float,
    stream_end_ts: float,
    filename: str,
    row_start_sec: float,
    duration_sec: float,
    target_sr: int = 32000,
    legacy_anchor_start_ts: float | None = None,
    est_seg_dur: float = 10.0,
) -> np.ndarray:
    """Resolve and decode a hydrophone audio slice using stream-offset mapping.

    Anchor order:
    1. First available folder timestamp (current behavior)
    2. legacy_anchor_start_ts (old jobs where filenames were anchored to job start)
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")

    folders, all_segments = _build_stream_segment_index(
        client, hydrophone_id, stream_start_ts, stream_end_ts
    )
    chunk_start_ts = _parse_chunk_start_timestamp(filename)
    abs_start_ts = chunk_start_ts + row_start_sec

    anchors: list[float] = [float(folders[0])]
    if legacy_anchor_start_ts is not None:
        legacy_anchor = float(legacy_anchor_start_ts)
        if all(abs(legacy_anchor - a) > 1e-6 for a in anchors):
            anchors.append(legacy_anchor)

    n_needed = int((duration_sec + 4 * est_seg_dur) / est_seg_dur) + 4
    last_reason = "No matching stream offset"

    for anchor_ts in anchors:
        stream_offset = abs_start_ts - anchor_ts
        if stream_offset < 0:
            last_reason = "Computed negative stream offset"
            continue

        start_idx = max(0, int(stream_offset / est_seg_dur) - 2)
        if start_idx >= len(all_segments):
            last_reason = "Computed segment index outside available stream range"
            continue
        end_idx = min(len(all_segments), start_idx + n_needed)
        if end_idx <= start_idx:
            last_reason = "No segments selected for decode window"
            continue

        decoded_segments: list[np.ndarray] = []
        for seg_key in all_segments[start_idx:end_idx]:
            try:
                seg_bytes = client.fetch_segment(seg_key)
                decoded_segments.append(decode_ts_bytes(seg_bytes, target_sr))
            except Exception:
                continue

        if not decoded_segments:
            last_reason = "Failed to decode all candidate segments"
            continue

        combined = np.concatenate(decoded_segments)
        local_offset = max(0.0, stream_offset - start_idx * est_seg_dur)
        start_sample = max(0, int(local_offset * target_sr))
        end_sample = min(
            len(combined),
            start_sample + int(duration_sec * target_sr),
        )

        if end_sample <= start_sample:
            last_reason = "Resolved sample window had no overlap with decoded audio"
            continue

        segment = combined[start_sample:end_sample]
        if len(segment) > 0:
            return segment

        last_reason = "Resolved segment was empty"

    raise FileNotFoundError(
        f"Could not resolve hydrophone audio slice for {filename}: {last_reason}"
    )


def iter_audio_chunks(
    client: "OrcasoundS3Client | LocalHLSClient | CachingS3Client",
    hydrophone_id: str,
    start_ts: float,
    end_ts: float,
    chunk_seconds: float = 60.0,
    target_sr: int = 32000,
    on_error: Callable[[dict], None] | None = None,
) -> "Generator":
    """Yield audio chunks from HLS stream.

    Yields (chunk_audio, chunk_start_utc, segments_done, segments_total).
    """
    folders = client.list_hls_folders(hydrophone_id, start_ts, end_ts)
    if not folders:
        logger.warning("No HLS folders found for %s in range", hydrophone_id)
        return

    # Count total segments for progress
    segments_total = 0
    folder_segment_map: list[tuple[str, list[str]]] = []
    for folder_ts in folders:
        segs = client.list_segments(hydrophone_id, folder_ts)
        folder_segment_map.append((folder_ts, segs))
        segments_total += len(segs)

    if segments_total == 0:
        logger.warning("No .ts segments found")
        return

    chunk_samples = int(chunk_seconds * target_sr)
    accumulator = np.array([], dtype=np.float32)
    # Base chunk timestamps on the first folder's actual recording time,
    # not the job's start_ts (which may precede available data by hours).
    chunk_start_ts = float(folder_segment_map[0][0])
    segments_done = 0

    for folder_ts, segments in folder_segment_map:
        for seg_key in segments:
            try:
                seg_bytes = client.fetch_segment(seg_key)
                audio = decode_ts_bytes(seg_bytes, target_sr)
                accumulator = np.concatenate([accumulator, audio])
                segments_done += 1

                # Yield chunks when we have enough audio
                while len(accumulator) >= chunk_samples:
                    chunk = accumulator[:chunk_samples]
                    chunk_start_utc = datetime.fromtimestamp(
                        chunk_start_ts, tz=timezone.utc
                    )
                    yield chunk, chunk_start_utc, segments_done, segments_total
                    chunk_start_ts += chunk_seconds
                    accumulator = accumulator[chunk_samples:]

            except Exception as e:
                segments_done += 1
                logger.warning("Failed to decode segment %s: %s", seg_key, e)
                if on_error:
                    on_error({
                        "type": "warning",
                        "message": f"Failed to decode segment {seg_key}: {e}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

    # Yield remaining audio
    if len(accumulator) > 0:
        chunk_start_utc = datetime.fromtimestamp(chunk_start_ts, tz=timezone.utc)
        yield accumulator, chunk_start_utc, segments_done, segments_total
