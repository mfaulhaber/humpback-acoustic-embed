"""S3 HLS streaming client for Orcasound hydrophone audio."""

import io
import logging
import struct
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np

from humpback.config import ORCASOUND_S3_BUCKET

logger = logging.getLogger(__name__)


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


def iter_audio_chunks(
    client: OrcasoundS3Client,
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
    chunk_start_ts = start_ts
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
