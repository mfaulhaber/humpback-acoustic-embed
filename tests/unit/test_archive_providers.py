"""Tests for ArchiveProvider protocol and Orcasound HLS provider implementations."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from humpback.classifier.archive import ArchiveProvider, StreamSegment
from humpback.classifier.providers.orcasound_hls import (
    CachingHLSProvider,
    LocalHLSCacheProvider,
    OrcasoundHLSProvider,
)


# ---------------------------------------------------------------------------
# StreamSegment dataclass
# ---------------------------------------------------------------------------


class TestStreamSegment:
    def test_end_ts_computed(self):
        seg = StreamSegment(key="a/b.ts", start_ts=100.0, duration_sec=10.0)
        assert seg.end_ts == 110.0

    def test_frozen(self):
        seg = StreamSegment(key="a/b.ts", start_ts=0.0, duration_sec=5.0)
        with pytest.raises(AttributeError):
            seg.key = "other"  # type: ignore[misc]

    def test_re_export_from_s3_stream(self):
        """StreamSegment is still importable from s3_stream for backward compat."""
        from humpback.classifier.s3_stream import StreamSegment as SS

        assert SS is StreamSegment


# ---------------------------------------------------------------------------
# Protocol conformance — isinstance checks
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Each HLS provider must satisfy the ArchiveProvider protocol."""

    @patch("humpback.classifier.providers.orcasound_hls.OrcasoundS3Client")
    def test_orcasound_hls_is_archive_provider(self, mock_cls):
        provider = OrcasoundHLSProvider("rpi_lab", "Orcasound Lab")
        assert isinstance(provider, ArchiveProvider)

    @patch("humpback.classifier.providers.orcasound_hls.CachingS3Client")
    def test_caching_hls_is_archive_provider(self, mock_cls):
        provider = CachingHLSProvider("/tmp/cache", "rpi_lab", "Orcasound Lab")
        assert isinstance(provider, ArchiveProvider)

    @patch("humpback.classifier.providers.orcasound_hls.LocalHLSClient")
    def test_local_hls_cache_is_archive_provider(self, mock_cls):
        provider = LocalHLSCacheProvider("/tmp/cache", "rpi_lab", "Orcasound Lab")
        assert isinstance(provider, ArchiveProvider)


# ---------------------------------------------------------------------------
# Property accessors
# ---------------------------------------------------------------------------


class TestProviderProperties:
    @patch("humpback.classifier.providers.orcasound_hls.OrcasoundS3Client")
    def test_orcasound_name_and_source_id(self, mock_cls):
        p = OrcasoundHLSProvider("rpi_north_sjc", "North SJC")
        assert p.name == "North SJC"
        assert p.source_id == "rpi_north_sjc"

    @patch("humpback.classifier.providers.orcasound_hls.CachingS3Client")
    def test_caching_name_and_source_id(self, mock_cls):
        p = CachingHLSProvider("/cache", "rpi_lab", "Lab")
        assert p.name == "Lab"
        assert p.source_id == "rpi_lab"

    @patch("humpback.classifier.providers.orcasound_hls.LocalHLSClient")
    def test_local_name_and_source_id(self, mock_cls):
        p = LocalHLSCacheProvider("/cache", "rpi_pt", "Port Townsend")
        assert p.name == "Port Townsend"
        assert p.source_id == "rpi_pt"


# ---------------------------------------------------------------------------
# Delegation tests — verify providers delegate to wrapped clients
# ---------------------------------------------------------------------------


class TestOrcasoundHLSProviderDelegation:
    @patch("humpback.classifier.providers.orcasound_hls.OrcasoundS3Client")
    def _make(self, mock_cls):
        provider = OrcasoundHLSProvider("rpi_lab", "Lab")
        return provider, provider._client

    def test_fetch_segment_delegates(self):
        provider, mock_client = self._make()
        mock_client.fetch_segment.return_value = b"ts-data"
        result = provider.fetch_segment("hydro/hls/1000/live0.ts")
        mock_client.fetch_segment.assert_called_once_with("hydro/hls/1000/live0.ts")
        assert result == b"ts-data"

    @patch("humpback.classifier.providers.orcasound_hls.decode_ts_bytes")
    def test_decode_segment_delegates(self, mock_decode):
        provider, _ = self._make()
        mock_decode.return_value = np.zeros(160, dtype=np.float32)
        result = provider.decode_segment(b"raw", 16000)
        mock_decode.assert_called_once_with(b"raw", 16000)
        assert result.shape == (160,)

    def test_invalidate_returns_false(self):
        provider, _ = self._make()
        assert provider.invalidate_cached_segment("any/key") is False

    def test_count_segments_delegates(self):
        provider, mock_client = self._make()
        mock_client.list_hls_folders.return_value = ["1000", "2000"]
        mock_client.count_segments.return_value = 42
        result = provider.count_segments(1000.0, 3000.0)
        mock_client.list_hls_folders.assert_called_once_with("rpi_lab", 1000.0, 3000.0)
        mock_client.count_segments.assert_called_once_with("rpi_lab", ["1000", "2000"])
        assert result == 42

    @patch("humpback.classifier.providers.orcasound_hls._build_stream_timeline")
    def test_build_timeline_delegates(self, mock_build):
        provider, mock_client = self._make()
        expected = [StreamSegment("k", 1000.0, 10.0)]
        mock_build.return_value = expected
        result = provider.build_timeline(1000.0, 2000.0)
        mock_build.assert_called_once_with(mock_client, "rpi_lab", 1000.0, 2000.0)
        assert result == expected


class TestCachingHLSProviderDelegation:
    @patch("humpback.classifier.providers.orcasound_hls.CachingS3Client")
    def _make(self, mock_cls):
        provider = CachingHLSProvider("/cache", "rpi_lab", "Lab")
        return provider, provider._client

    def test_fetch_segment_delegates(self):
        provider, mock_client = self._make()
        mock_client.fetch_segment.return_value = b"cached-ts"
        result = provider.fetch_segment("key.ts")
        mock_client.fetch_segment.assert_called_once_with("key.ts")
        assert result == b"cached-ts"

    def test_invalidate_delegates(self):
        provider, mock_client = self._make()
        mock_client.invalidate_cached_segment.return_value = True
        assert provider.invalidate_cached_segment("key.ts") is True
        mock_client.invalidate_cached_segment.assert_called_once_with("key.ts")

    def test_count_segments_delegates(self):
        provider, mock_client = self._make()
        mock_client.list_hls_folders.return_value = ["1000"]
        mock_client.count_segments.return_value = 10
        assert provider.count_segments(1000.0, 2000.0) == 10

    @patch("humpback.classifier.providers.orcasound_hls._build_stream_timeline")
    def test_build_timeline_delegates(self, mock_build):
        provider, mock_client = self._make()
        mock_build.return_value = []
        provider.build_timeline(0.0, 100.0)
        mock_build.assert_called_once_with(mock_client, "rpi_lab", 0.0, 100.0)


class TestLocalHLSCacheProviderDelegation:
    @patch("humpback.classifier.providers.orcasound_hls.LocalHLSClient")
    def _make(self, mock_cls):
        provider = LocalHLSCacheProvider("/cache", "rpi_lab", "Lab")
        return provider, provider._client

    def test_fetch_segment_delegates(self):
        provider, mock_client = self._make()
        mock_client.fetch_segment.return_value = b"local-ts"
        result = provider.fetch_segment("key.ts")
        mock_client.fetch_segment.assert_called_once_with("key.ts")
        assert result == b"local-ts"

    def test_invalidate_delegates(self):
        provider, mock_client = self._make()
        mock_client.invalidate_cached_segment.return_value = True
        assert provider.invalidate_cached_segment("bad.ts") is True

    def test_count_segments_delegates(self):
        provider, mock_client = self._make()
        mock_client.list_hls_folders.return_value = ["5000"]
        mock_client.count_segments.return_value = 7
        assert provider.count_segments(5000.0, 6000.0) == 7

    @patch("humpback.classifier.providers.orcasound_hls._build_stream_timeline")
    def test_build_timeline_delegates(self, mock_build):
        provider, mock_client = self._make()
        mock_build.return_value = []
        provider.build_timeline(0.0, 100.0)
        mock_build.assert_called_once_with(mock_client, "rpi_lab", 0.0, 100.0)


# ---------------------------------------------------------------------------
# Non-conforming class should NOT satisfy Protocol
# ---------------------------------------------------------------------------


class TestNonConformance:
    def test_plain_object_is_not_archive_provider(self):
        assert not isinstance(object(), ArchiveProvider)

    def test_partial_implementation_fails(self):
        class Incomplete:
            @property
            def name(self) -> str:
                return "x"

            def fetch_segment(self, key: str) -> bytes:
                return b""

        assert not isinstance(Incomplete(), ArchiveProvider)
