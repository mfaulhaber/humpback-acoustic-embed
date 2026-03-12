"""Orcasound HLS archive providers wrapping existing S3/local clients."""

from __future__ import annotations

import numpy as np

from humpback.classifier.archive import StreamSegment
from humpback.classifier.s3_stream import (
    CachingS3Client,
    LocalHLSClient,
    OrcasoundS3Client,
    _build_stream_timeline,
    decode_ts_bytes,
)


class OrcasoundHLSProvider:
    """ArchiveProvider wrapping OrcasoundS3Client (direct S3, no cache)."""

    def __init__(self, source_id: str, name: str) -> None:
        self._source_id = source_id
        self._name = name
        self._client = OrcasoundS3Client()

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_id(self) -> str:
        return self._source_id

    def build_timeline(self, start_ts: float, end_ts: float) -> list[StreamSegment]:
        return _build_stream_timeline(self._client, self._source_id, start_ts, end_ts)

    def count_segments(self, start_ts: float, end_ts: float) -> int:
        folders = self._client.list_hls_folders(self._source_id, start_ts, end_ts)
        return self._client.count_segments(self._source_id, folders)

    def fetch_segment(self, key: str) -> bytes:
        return self._client.fetch_segment(key)

    def decode_segment(self, raw_bytes: bytes, target_sr: int) -> np.ndarray:
        return decode_ts_bytes(raw_bytes, target_sr)

    def invalidate_cached_segment(self, key: str) -> bool:
        return False


class CachingHLSProvider:
    """ArchiveProvider wrapping CachingS3Client (S3 with write-through cache)."""

    def __init__(self, cache_root: str, source_id: str, name: str) -> None:
        self._source_id = source_id
        self._name = name
        self._client = CachingS3Client(cache_root)

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_id(self) -> str:
        return self._source_id

    def build_timeline(self, start_ts: float, end_ts: float) -> list[StreamSegment]:
        return _build_stream_timeline(self._client, self._source_id, start_ts, end_ts)

    def count_segments(self, start_ts: float, end_ts: float) -> int:
        folders = self._client.list_hls_folders(self._source_id, start_ts, end_ts)
        return self._client.count_segments(self._source_id, folders)

    def fetch_segment(self, key: str) -> bytes:
        return self._client.fetch_segment(key)

    def decode_segment(self, raw_bytes: bytes, target_sr: int) -> np.ndarray:
        return decode_ts_bytes(raw_bytes, target_sr)

    def invalidate_cached_segment(self, key: str) -> bool:
        return self._client.invalidate_cached_segment(key)


class LocalHLSCacheProvider:
    """ArchiveProvider wrapping LocalHLSClient (filesystem-only cache)."""

    def __init__(self, cache_root: str, source_id: str, name: str) -> None:
        self._source_id = source_id
        self._name = name
        self._client = LocalHLSClient(cache_root)

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_id(self) -> str:
        return self._source_id

    def build_timeline(self, start_ts: float, end_ts: float) -> list[StreamSegment]:
        return _build_stream_timeline(self._client, self._source_id, start_ts, end_ts)

    def count_segments(self, start_ts: float, end_ts: float) -> int:
        folders = self._client.list_hls_folders(self._source_id, start_ts, end_ts)
        return self._client.count_segments(self._source_id, folders)

    def fetch_segment(self, key: str) -> bytes:
        return self._client.fetch_segment(key)

    def decode_segment(self, raw_bytes: bytes, target_sr: int) -> np.ndarray:
        return decode_ts_bytes(raw_bytes, target_sr)

    def invalidate_cached_segment(self, key: str) -> bool:
        return self._client.invalidate_cached_segment(key)
