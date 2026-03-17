"""Tests for ArchiveProvider protocol and provider implementations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from humpback.classifier.archive import ArchiveProvider, StreamSegment
from humpback.classifier.providers import (
    build_archive_detection_provider,
    build_archive_playback_provider,
    build_orcasound_detection_provider,
    build_orcasound_local_cache_provider,
)
from humpback.classifier.providers.noaa_gcs import (
    CachingNoaaGCSProvider,
    DEFAULT_NOAA_SEGMENT_DURATION_SEC,
    NoaaAudioFile,
    NoaaGCSProvider,
    estimate_noaa_interval_sec,
    parse_noaa_filename,
    read_noaa_manifest,
    write_noaa_manifest,
)
from humpback.classifier.providers.orcasound_hls import (
    CachingHLSProvider,
    LocalHLSCacheProvider,
    OrcasoundHLSProvider,
)


class _FakeBlob:
    def __init__(self, name: str, *, size: int = 0, data: bytes = b"blob-data") -> None:
        self.name = name
        self.size = size
        self._data = data

    def download_as_bytes(self) -> bytes:
        return self._data


class _FakeBucket:
    def __init__(self, blobs: list[_FakeBlob]) -> None:
        self._blobs = list(blobs)
        self.list_calls = 0
        self.blob_calls: list[str] = []

    def list_blobs(self, prefix: str):
        self.list_calls += 1
        return [blob for blob in self._blobs if blob.name.startswith(prefix)]

    def blob(self, key: str) -> _FakeBlob:
        self.blob_calls.append(key)
        for blob in self._blobs:
            if blob.name == key:
                return blob
        raise KeyError(key)


def _ts(year: int, month: int, day: int, hour: int, minute: int, second: int) -> float:
    return datetime(
        year, month, day, hour, minute, second, tzinfo=timezone.utc
    ).timestamp()


class TestStreamSegment:
    def test_end_ts_computed(self):
        seg = StreamSegment(key="a/b.ts", start_ts=100.0, duration_sec=10.0)
        assert seg.end_ts == 110.0

    def test_frozen(self):
        seg = StreamSegment(key="a/b.ts", start_ts=0.0, duration_sec=5.0)
        with pytest.raises(AttributeError):
            seg.key = "other"  # type: ignore[misc]

    def test_re_export_from_s3_stream(self):
        from humpback.classifier.s3_stream import StreamSegment as SS

        assert SS is StreamSegment


class TestProtocolConformance:
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

    def test_noaa_gcs_is_archive_provider(self):
        provider = NoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            bucket_obj=_FakeBucket([]),
        )
        assert isinstance(provider, ArchiveProvider)

    def test_caching_noaa_gcs_is_archive_provider(self):
        provider = CachingNoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            "/tmp/noaa-cache",
            bucket_obj=_FakeBucket([]),
        )
        assert isinstance(provider, ArchiveProvider)

    def test_caching_noaa_is_segment_cached(self, tmp_path):
        provider = CachingNoaaGCSProvider(
            "noaa_test",
            "NOAA Test",
            str(tmp_path),
            bucket="test-bucket",
            bucket_obj=_FakeBucket([]),
        )
        key = "audio/test_file.flac"
        assert not provider.is_segment_cached(key)

        # Create the cached file
        cached_path = tmp_path / "test-bucket" / key
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.write_bytes(b"fake-audio-data")
        assert provider.is_segment_cached(key)


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

    def test_noaa_name_and_source_id(self):
        p = NoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            bucket_obj=_FakeBucket([]),
        )
        assert p.name == "NOAA Glacier Bay"
        assert p.source_id == "noaa_glacier_bay"


class TestOrcasoundHLSProviderDelegation:
    @patch("humpback.classifier.providers.orcasound_hls.OrcasoundS3Client")
    def _make(self, mock_cls: MagicMock) -> tuple[OrcasoundHLSProvider, MagicMock]:
        provider = OrcasoundHLSProvider("rpi_lab", "Lab")
        return provider, cast(MagicMock, provider._client)

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
    def _make(self, mock_cls: MagicMock) -> tuple[CachingHLSProvider, MagicMock]:
        provider = CachingHLSProvider("/cache", "rpi_lab", "Lab")
        return provider, cast(MagicMock, provider._client)

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
    def _make(self, mock_cls: MagicMock) -> tuple[LocalHLSCacheProvider, MagicMock]:
        provider = LocalHLSCacheProvider("/cache", "rpi_lab", "Lab")
        return provider, cast(MagicMock, provider._client)

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


class TestNoaaProvider:
    def test_parse_noaa_filename(self):
        parsed = parse_noaa_filename("07_25_2015_00_00_09.aif")
        assert parsed == datetime(2015, 7, 25, 0, 0, 9, tzinfo=timezone.utc)
        assert parse_noaa_filename("2022_09_19_13_12_31.wav") == datetime(
            2022, 9, 19, 13, 12, 31, tzinfo=timezone.utc
        )
        assert parse_noaa_filename("2023-07-27T16-55-59Z.aiff") == datetime(
            2023, 7, 27, 16, 55, 59, tzinfo=timezone.utc
        )
        assert parse_noaa_filename(
            "SanctSound_CI01_01_671379494_20181031T220000Z.flac"
        ) == datetime(2018, 10, 31, 22, 0, 0, tzinfo=timezone.utc)
        assert parse_noaa_filename(
            "SanctSound_CI01_02_671883305_190325210000.flac"
        ) == datetime(2019, 3, 25, 21, 0, 0, tzinfo=timezone.utc)
        assert parse_noaa_filename(
            "SanctSound_CI01_08_671379494_210708180002_o.flac"
        ) == datetime(2021, 7, 8, 18, 0, 2, tzinfo=timezone.utc)
        assert parse_noaa_filename("bad_name.aif") is None

    def test_build_timeline_orders_segments_and_caches_listing(self):
        prefix = "archive/"
        bucket = _FakeBucket(
            [
                _FakeBlob(prefix + "README.txt"),
                _FakeBlob(prefix + "07_25_2015_00_05_09.aif", size=20),
                _FakeBlob(prefix + "07_25_2015_00_00_09.aif", size=10),
                _FakeBlob(prefix + "07_25_2015_00_15_09.aif", size=30),
            ]
        )
        provider = NoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            prefix=prefix,
            bucket_obj=bucket,
        )

        timeline = provider.build_timeline(
            _ts(2015, 7, 25, 0, 0, 0),
            _ts(2015, 7, 25, 0, 20, 0),
        )
        again = provider.build_timeline(
            _ts(2015, 7, 25, 0, 0, 0),
            _ts(2015, 7, 25, 0, 20, 0),
        )

        assert bucket.list_calls == 1
        assert [segment.key for segment in timeline] == [
            prefix + "07_25_2015_00_00_09.aif",
            prefix + "07_25_2015_00_05_09.aif",
            prefix + "07_25_2015_00_15_09.aif",
        ]
        assert timeline == again
        assert timeline[0].duration_sec == pytest.approx(300.0)
        # Gaps are [300, 600]; median=450 → files with next_gap >= 450 use 450
        assert timeline[1].duration_sec == pytest.approx(450.0)
        assert timeline[2].duration_sec == pytest.approx(450.0)

    def test_build_timeline_filters_by_range(self):
        prefix = "archive/"
        bucket = _FakeBucket(
            [
                _FakeBlob(prefix + "07_25_2015_00_00_09.aif"),
                _FakeBlob(prefix + "07_25_2015_00_05_09.aif"),
                _FakeBlob(prefix + "07_25_2015_00_10_09.aif"),
            ]
        )
        provider = NoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            prefix=prefix,
            bucket_obj=bucket,
        )

        timeline = provider.build_timeline(
            _ts(2015, 7, 25, 0, 7, 0),
            _ts(2015, 7, 25, 0, 12, 0),
        )

        assert [segment.key for segment in timeline] == [
            prefix + "07_25_2015_00_05_09.aif",
            prefix + "07_25_2015_00_10_09.aif",
        ]
        assert (
            provider.count_segments(
                _ts(2015, 7, 25, 0, 7, 0),
                _ts(2015, 7, 25, 0, 12, 0),
            )
            == 2
        )

    def test_build_timeline_uses_matching_child_hints(self):
        root = "sanctsound/audio/ci01/"
        bucket = _FakeBucket(
            [
                _FakeBlob(
                    root
                    + "sanctsound_ci01_01/audio/"
                    + "SanctSound_CI01_01_671379494_20181031T220000Z.flac"
                ),
                _FakeBlob(
                    root
                    + "sanctsound_ci01_02/audio/"
                    + "SanctSound_CI01_02_671883305_190325210000.flac"
                ),
            ]
        )
        provider = NoaaGCSProvider(
            "sanctsound_ci01",
            "NOAA SanctSound (Channel Islands)",
            prefix=root,
            audio_subpath="audio/",
            child_folder_hints=[
                {
                    "prefix": "sanctsound_ci01_01/",
                    "start_utc": "2018-10-31T22:00:00Z",
                    "end_utc": "2018-12-15T04:26:35Z",
                },
                {
                    "prefix": "sanctsound_ci01_02/",
                    "start_utc": "2019-03-25T21:00:00Z",
                    "end_utc": "2019-08-04T00:21:55Z",
                },
            ],
            bucket_obj=bucket,
        )

        timeline = provider.build_timeline(
            _ts(2019, 3, 25, 20, 0, 0),
            _ts(2019, 3, 26, 0, 0, 0),
        )

        assert bucket.list_calls == 1
        assert [segment.key for segment in timeline] == [
            root
            + "sanctsound_ci01_02/audio/"
            + "SanctSound_CI01_02_671883305_190325210000.flac"
        ]

    def test_build_timeline_falls_back_to_root_listing_when_hints_missing(self):
        root = "sanctsound/audio/ci01/"
        key = (
            root
            + "sanctsound_ci01_02/audio/"
            + "SanctSound_CI01_02_671883305_190325210000.flac"
        )
        bucket = _FakeBucket([_FakeBlob(key)])
        provider = NoaaGCSProvider(
            "sanctsound_ci01",
            "NOAA SanctSound (Channel Islands)",
            prefix=root,
            audio_subpath="audio/",
            bucket_obj=bucket,
        )

        timeline = provider.build_timeline(
            _ts(2019, 3, 25, 20, 0, 0),
            _ts(2019, 3, 26, 0, 0, 0),
        )

        assert bucket.list_calls == 1
        assert [segment.key for segment in timeline] == [key]

    def test_build_timeline_raises_when_hints_exist_but_none_match(self):
        """When child_folder_hints exist but none overlap with the requested
        time range, the provider must not fall back to scanning the broad base
        prefix.  Instead it should raise FileNotFoundError without issuing any
        GCS list calls — preventing accidental full-archive scans for
        multi-site sources."""
        root = "sanctsound/audio/"
        bucket = _FakeBucket([])
        provider = NoaaGCSProvider(
            "sanctsound_oc01",
            "NOAA SanctSound (Olympic Coast)",
            prefix=root,
            audio_subpath="audio/",
            child_folder_hints=[
                {
                    "prefix": "oc01/sanctsound_oc01_03/",
                    "start_utc": "2019-11-02T19:00:00Z",
                    "end_utc": "2020-04-10T18:48:11Z",
                },
                {
                    "prefix": "oc01/sanctsound_oc01_04/",
                    "start_utc": "2020-07-31T02:00:00Z",
                    "end_utc": "2020-09-22T23:51:47Z",
                },
            ],
            bucket_obj=bucket,
        )
        # July 4 2020 falls in the gap between oc01_03 and oc01_04
        with pytest.raises(FileNotFoundError, match="No NOAA audio data"):
            provider.build_timeline(
                _ts(2020, 7, 4, 0, 0, 0),
                _ts(2020, 7, 4, 12, 0, 0),
            )
        # No GCS listing should have been attempted
        assert bucket.list_calls == 0

    def test_build_timeline_raises_when_no_overlap(self):
        provider = NoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            bucket_obj=_FakeBucket(
                [_FakeBlob("archive/07_25_2015_00_00_09.aif", size=10)]
            ),
            prefix="archive/",
        )
        with pytest.raises(FileNotFoundError, match="No NOAA stream segments"):
            provider.build_timeline(
                _ts(2015, 7, 25, 1, 0, 0),
                _ts(2015, 7, 25, 1, 5, 0),
            )
        assert (
            provider.count_segments(
                _ts(2015, 7, 25, 1, 0, 0),
                _ts(2015, 7, 25, 1, 5, 0),
            )
            == 0
        )

    def test_fetch_segment_downloads_from_bucket(self):
        key = "archive/07_25_2015_00_00_09.aif"
        bucket = _FakeBucket([_FakeBlob(key, data=b"raw-aif")])
        provider = NoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            bucket_obj=bucket,
            prefix="archive/",
        )

        result = provider.fetch_segment(key)

        assert result == b"raw-aif"
        assert bucket.blob_calls == [key]

    @patch("humpback.classifier.providers.noaa_gcs.decode_noaa_audio_bytes")
    def test_decode_segment_delegates(self, mock_decode):
        provider = NoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            bucket_obj=_FakeBucket([]),
        )
        mock_decode.return_value = np.zeros(320, dtype=np.float32)

        audio = provider.decode_segment(b"raw-aif", 32000)

        mock_decode.assert_called_once_with(b"raw-aif", 32000)
        assert audio.shape == (320,)

    def test_invalidate_segment_returns_false(self):
        provider = NoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            bucket_obj=_FakeBucket([]),
        )
        assert provider.invalidate_cached_segment("anything") is False

    def test_noaa_prefetch_defaults_true(self):
        provider = NoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            bucket_obj=_FakeBucket([]),
        )
        assert provider.supports_segment_prefetch is True

    def test_noaa_prefetch_can_be_disabled_per_source(self):
        provider = NoaaGCSProvider(
            "sanctsound_ci01",
            "NOAA SanctSound (Channel Islands)",
            supports_segment_prefetch=False,
            bucket_obj=_FakeBucket([]),
        )
        assert provider.supports_segment_prefetch is False

    @patch("humpback.classifier.providers.noaa_gcs.iter_decode_noaa_audio_bytes")
    def test_iter_decoded_segment_chunks_delegates_to_bytes_helper(self, mock_iter):
        provider = NoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            bucket_obj=_FakeBucket([]),
        )
        mock_chunk = np.zeros(320, dtype=np.float32)
        mock_iter.return_value = iter([(mock_chunk, 5.0)])

        chunks = list(
            provider.iter_decoded_segment_chunks(
                "archive/07_25_2015_00_00_09.aif",
                b"raw-aif",
                32000,
                clip_start_sec=5.0,
                clip_end_sec=65.0,
                chunk_seconds=60.0,
            )
        )

        mock_iter.assert_called_once_with(
            b"raw-aif",
            32000,
            clip_start_sec=5.0,
            clip_end_sec=65.0,
            chunk_seconds=60.0,
        )
        assert chunks == [(mock_chunk, 5.0)]

    @patch("humpback.classifier.providers.noaa_gcs.iter_decode_noaa_audio_file")
    def test_caching_iter_decoded_segment_chunks_prefers_cached_file(
        self, mock_iter_file, tmp_path
    ):
        provider = CachingNoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            str(tmp_path),
            bucket="noaa-passive-bioacoustic",
            bucket_obj=_FakeBucket([]),
        )
        key = "archive/07_25_2015_00_00_09.aif"
        cached_path = tmp_path / "noaa-passive-bioacoustic" / key
        cached_path.parent.mkdir(parents=True)
        cached_path.write_bytes(b"cached")
        mock_chunk = np.zeros(320, dtype=np.float32)
        mock_iter_file.return_value = iter([(mock_chunk, 0.0)])

        chunks = list(
            provider.iter_decoded_segment_chunks(
                key,
                b"raw-aif",
                32000,
                clip_start_sec=0.0,
                clip_end_sec=60.0,
                chunk_seconds=60.0,
            )
        )

        mock_iter_file.assert_called_once_with(
            cached_path,
            32000,
            clip_start_sec=0.0,
            clip_end_sec=60.0,
            chunk_seconds=60.0,
        )
        assert chunks == [(mock_chunk, 0.0)]


class TestProviderBuilders:
    @patch("humpback.classifier.providers.orcasound_hls.LocalHLSCacheProvider")
    def test_orcasound_detection_builder_prefers_local_cache(self, mock_local):
        provider = build_orcasound_detection_provider(
            "rpi_lab",
            "Lab",
            local_cache_path="/local-cache",
            s3_cache_path="/s3-cache",
        )
        mock_local.assert_called_once_with("/local-cache", "rpi_lab", "Lab")
        assert provider is mock_local.return_value

    @patch("humpback.classifier.providers.orcasound_hls.CachingHLSProvider")
    def test_orcasound_detection_builder_uses_s3_cache_when_local_absent(
        self, mock_caching
    ):
        provider = build_orcasound_detection_provider(
            "rpi_lab",
            "Lab",
            local_cache_path=None,
            s3_cache_path="/s3-cache",
        )
        mock_caching.assert_called_once_with("/s3-cache", "rpi_lab", "Lab")
        assert provider is mock_caching.return_value

    @patch("humpback.classifier.providers.orcasound_hls.OrcasoundHLSProvider")
    def test_orcasound_detection_builder_falls_back_to_direct_s3(self, mock_direct):
        provider = build_orcasound_detection_provider(
            "rpi_lab",
            "Lab",
            local_cache_path=None,
            s3_cache_path=None,
        )
        mock_direct.assert_called_once_with("rpi_lab", "Lab")
        assert provider is mock_direct.return_value

    @patch("humpback.classifier.providers.orcasound_hls.LocalHLSCacheProvider")
    def test_orcasound_playback_builder_constructs_local_provider(self, mock_local):
        provider = build_orcasound_local_cache_provider(
            "rpi_lab", "Lab", "/local-cache"
        )
        mock_local.assert_called_once_with("/local-cache", "rpi_lab", "Lab")
        assert provider is mock_local.return_value

    @patch("humpback.classifier.providers.build_orcasound_detection_provider")
    def test_archive_detection_builder_routes_orcasound(self, mock_orcasound):
        mock_orcasound.return_value = object()

        provider = build_archive_detection_provider(
            "rpi_orcasound_lab",
            local_cache_path="/cache",
            s3_cache_path="/s3-cache",
        )

        mock_orcasound.assert_called_once_with(
            "rpi_orcasound_lab",
            "Orcasound Lab",
            local_cache_path="/cache",
            s3_cache_path="/s3-cache",
        )
        assert provider is mock_orcasound.return_value

    def test_archive_detection_builder_routes_noaa_direct(self):
        provider = build_archive_detection_provider(
            "noaa_glacier_bay",
            local_cache_path=None,
            s3_cache_path="/ignored",
        )

        assert isinstance(provider, NoaaGCSProvider)
        assert provider.source_id == "noaa_glacier_bay"
        assert provider.supports_segment_prefetch is True

    def test_archive_detection_builder_applies_sanctsound_prefetch_flag(self):
        provider = build_archive_detection_provider(
            "sanctsound_ci01",
            local_cache_path=None,
            s3_cache_path="/ignored",
        )

        assert isinstance(provider, NoaaGCSProvider)
        assert provider.source_id == "sanctsound_ci01"
        assert provider.supports_segment_prefetch is False

    def test_archive_detection_builder_routes_noaa_caching(self, tmp_path):
        provider = build_archive_detection_provider(
            "noaa_glacier_bay",
            local_cache_path=None,
            s3_cache_path="/ignored",
            noaa_cache_path=str(tmp_path / "noaa-cache"),
        )

        from humpback.classifier.providers.noaa_gcs import CachingNoaaGCSProvider

        assert isinstance(provider, CachingNoaaGCSProvider)
        assert provider.source_id == "noaa_glacier_bay"
        assert provider.supports_segment_prefetch is True

    def test_archive_detection_builder_rejects_local_cache_for_noaa(self):
        with pytest.raises(ValueError, match="local_cache_path is only supported"):
            build_archive_detection_provider(
                "noaa_glacier_bay",
                local_cache_path="/cache",
                s3_cache_path=None,
            )

    @patch("humpback.classifier.providers.build_orcasound_local_cache_provider")
    def test_archive_playback_builder_routes_orcasound(self, mock_orcasound):
        mock_orcasound.return_value = object()

        provider = build_archive_playback_provider(
            "rpi_orcasound_lab",
            cache_path="/cache",
        )

        mock_orcasound.assert_called_once_with(
            "rpi_orcasound_lab",
            "Orcasound Lab",
            "/cache",
        )
        assert provider is mock_orcasound.return_value

    def test_archive_playback_builder_rejects_missing_orcasound_cache(self):
        with pytest.raises(ValueError, match="configured cache path"):
            build_archive_playback_provider("rpi_orcasound_lab", cache_path=None)

    def test_archive_playback_builder_routes_noaa_without_cache(self):
        provider = build_archive_playback_provider(
            "noaa_glacier_bay",
            cache_path=None,
        )

        assert isinstance(provider, NoaaGCSProvider)
        assert provider.source_id == "noaa_glacier_bay"

    def test_archive_playback_builder_routes_noaa_with_cache(self, tmp_path):
        from humpback.classifier.providers.noaa_gcs import CachingNoaaGCSProvider

        provider = build_archive_playback_provider(
            "noaa_glacier_bay",
            cache_path=None,
            noaa_cache_path=str(tmp_path / "noaa-cache"),
        )

        assert isinstance(provider, CachingNoaaGCSProvider)
        assert provider.source_id == "noaa_glacier_bay"


class TestNoaaManifest:
    def _make_files(self) -> list:

        return [
            NoaaAudioFile(
                filename="07_25_2015_00_00_09.aif",
                key="archive/07_25_2015_00_00_09.aif",
                timestamp=datetime(2015, 7, 25, 0, 0, 9, tzinfo=timezone.utc),
                size=100,
            ),
            NoaaAudioFile(
                filename="07_25_2015_00_05_09.aif",
                key="archive/07_25_2015_00_05_09.aif",
                timestamp=datetime(2015, 7, 25, 0, 5, 9, tzinfo=timezone.utc),
                size=200,
            ),
        ]

    def test_write_and_read_roundtrip(self, tmp_path):
        from humpback.classifier.providers.noaa_gcs import (
            _manifest_path,
        )

        files = self._make_files()
        path = _manifest_path(str(tmp_path), "bucket", "prefix/sub/")
        write_noaa_manifest(path, files, 300.0)

        result = read_noaa_manifest(path)
        assert result is not None
        loaded_files, interval = result
        assert len(loaded_files) == 2
        assert loaded_files[0].filename == "07_25_2015_00_00_09.aif"
        assert loaded_files[0].key == "archive/07_25_2015_00_00_09.aif"
        assert loaded_files[0].size == 100
        assert loaded_files[1].filename == "07_25_2015_00_05_09.aif"
        assert interval == 300.0

    def test_read_returns_none_for_missing_file(self, tmp_path):

        result = read_noaa_manifest(tmp_path / "nonexistent.json")
        assert result is None

    def test_read_returns_none_for_corrupt_json(self, tmp_path):

        path = tmp_path / "manifest.json"
        path.write_text("not valid json!!!")
        result = read_noaa_manifest(path)
        assert result is None


class TestCachingNoaaProvider:
    def _make_provider(
        self, cache_root: str, bucket: _FakeBucket, prefix: str = "archive/"
    ):
        from humpback.classifier.providers.noaa_gcs import CachingNoaaGCSProvider

        return CachingNoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            cache_root,
            prefix=prefix,
            bucket_obj=bucket,
        )

    def test_build_timeline_uses_manifest_when_available(self, tmp_path):
        from humpback.classifier.providers.noaa_gcs import (
            _manifest_path,
        )

        prefix = "archive/"
        files = [
            NoaaAudioFile(
                filename="07_25_2015_00_00_09.aif",
                key=prefix + "07_25_2015_00_00_09.aif",
                timestamp=datetime(2015, 7, 25, 0, 0, 9, tzinfo=timezone.utc),
                size=100,
            ),
        ]
        manifest = _manifest_path(str(tmp_path), "noaa-passive-bioacoustic", prefix)
        write_noaa_manifest(manifest, files, 300.0)

        bucket = _FakeBucket([])
        provider = self._make_provider(str(tmp_path), bucket, prefix)
        timeline = provider.build_timeline(
            _ts(2015, 7, 25, 0, 0, 0),
            _ts(2015, 7, 25, 0, 10, 0),
        )

        assert bucket.list_calls == 0
        assert len(timeline) == 1

    def test_build_timeline_lists_gcs_and_writes_manifest(self, tmp_path):
        from humpback.classifier.providers.noaa_gcs import (
            _manifest_path,
        )

        prefix = "archive/"
        bucket = _FakeBucket(
            [
                _FakeBlob(prefix + "07_25_2015_00_00_09.aif", size=100),
                _FakeBlob(prefix + "07_25_2015_00_05_09.aif", size=200),
            ]
        )
        provider = self._make_provider(str(tmp_path), bucket, prefix)
        timeline = provider.build_timeline(
            _ts(2015, 7, 25, 0, 0, 0),
            _ts(2015, 7, 25, 0, 10, 0),
        )

        assert bucket.list_calls == 1
        assert len(timeline) == 2

        manifest = _manifest_path(str(tmp_path), "noaa-passive-bioacoustic", prefix)
        cached = read_noaa_manifest(manifest)
        assert cached is not None
        assert len(cached[0]) == 2

    def test_build_timeline_writes_manifest_per_matching_child_prefix(self, tmp_path):
        from humpback.classifier.providers.noaa_gcs import _manifest_path

        root = "sanctsound/audio/ci01/"
        matching_prefix = root + "sanctsound_ci01_02/audio/"
        bucket = _FakeBucket(
            [
                _FakeBlob(
                    matching_prefix + "SanctSound_CI01_02_671883305_190325210000.flac",
                    size=100,
                ),
            ]
        )
        from humpback.classifier.providers.noaa_gcs import CachingNoaaGCSProvider

        provider = CachingNoaaGCSProvider(
            "sanctsound_ci01",
            "NOAA SanctSound (Channel Islands)",
            str(tmp_path),
            prefix=root,
            audio_subpath="audio/",
            child_folder_hints=[
                {
                    "prefix": "sanctsound_ci01_01/",
                    "start_utc": "2018-10-31T22:00:00Z",
                    "end_utc": "2018-12-15T04:26:35Z",
                },
                {
                    "prefix": "sanctsound_ci01_02/",
                    "start_utc": "2019-03-25T21:00:00Z",
                    "end_utc": "2019-08-04T00:21:55Z",
                },
            ],
            bucket_obj=bucket,
        )

        timeline = provider.build_timeline(
            _ts(2019, 3, 25, 20, 0, 0),
            _ts(2019, 3, 26, 0, 0, 0),
        )

        assert len(timeline) == 1
        manifest = _manifest_path(
            str(tmp_path),
            "noaa-passive-bioacoustic",
            matching_prefix,
        )
        assert manifest.is_file()

    def test_fetch_segment_reads_local_first(self, tmp_path):
        prefix = "archive/"
        key = prefix + "07_25_2015_00_00_09.aif"
        local_path = tmp_path / "noaa-passive-bioacoustic" / key
        local_path.parent.mkdir(parents=True)
        local_path.write_bytes(b"cached-aif-data")

        bucket = _FakeBucket([])
        provider = self._make_provider(str(tmp_path), bucket, prefix)

        result = provider.fetch_segment(key)

        assert result == b"cached-aif-data"
        assert bucket.blob_calls == []

    def test_fetch_segment_falls_back_to_gcs_and_caches(self, tmp_path):
        prefix = "archive/"
        key = prefix + "07_25_2015_00_00_09.aif"
        bucket = _FakeBucket([_FakeBlob(key, data=b"gcs-aif-data")])
        provider = self._make_provider(str(tmp_path), bucket, prefix)

        result = provider.fetch_segment(key)

        assert result == b"gcs-aif-data"
        assert bucket.blob_calls == [key]

        local_path = tmp_path / "noaa-passive-bioacoustic" / key
        assert local_path.is_file()
        assert local_path.read_bytes() == b"gcs-aif-data"

    def test_invalidate_segment(self, tmp_path):
        prefix = "archive/"
        key = prefix + "07_25_2015_00_00_09.aif"
        local_path = tmp_path / "noaa-passive-bioacoustic" / key
        local_path.parent.mkdir(parents=True)
        local_path.write_bytes(b"data")

        provider = self._make_provider(str(tmp_path), _FakeBucket([]), prefix)

        assert provider.invalidate_cached_segment(key) is True
        assert not local_path.exists()
        assert provider.invalidate_cached_segment(key) is False

    def test_name_and_source_id(self):
        from humpback.classifier.providers.noaa_gcs import CachingNoaaGCSProvider

        provider = CachingNoaaGCSProvider(
            "noaa_glacier_bay",
            "NOAA Glacier Bay",
            "/tmp/cache",
            bucket_obj=_FakeBucket([]),
        )
        assert provider.name == "NOAA Glacier Bay"
        assert provider.source_id == "noaa_glacier_bay"


class TestNoaaFactoryHelpers:
    def test_detection_provider_returns_caching_when_cache_path_set(self, tmp_path):
        from humpback.classifier.providers.noaa_gcs import (
            CachingNoaaGCSProvider,
            build_noaa_detection_provider,
        )

        provider = build_noaa_detection_provider(
            "noaa_test",
            "Test",
            noaa_cache_path=str(tmp_path),
            bucket="test-bucket",
            prefix="test/",
        )
        assert isinstance(provider, CachingNoaaGCSProvider)

    def test_detection_provider_returns_direct_when_no_cache_path(self):
        from humpback.classifier.providers.noaa_gcs import build_noaa_detection_provider

        provider = build_noaa_detection_provider(
            "noaa_test",
            "Test",
            noaa_cache_path=None,
            bucket="test-bucket",
            prefix="test/",
        )
        assert isinstance(provider, NoaaGCSProvider)
        assert not isinstance(provider, object.__class__)

    def test_playback_provider_returns_caching_when_cache_path_set(self, tmp_path):
        from humpback.classifier.providers.noaa_gcs import (
            CachingNoaaGCSProvider,
            build_noaa_playback_provider,
        )

        provider = build_noaa_playback_provider(
            "noaa_test",
            "Test",
            noaa_cache_path=str(tmp_path),
            bucket="test-bucket",
            prefix="test/",
        )
        assert isinstance(provider, CachingNoaaGCSProvider)

    def test_playback_provider_returns_direct_when_no_cache_path(self):
        from humpback.classifier.providers.noaa_gcs import build_noaa_playback_provider

        provider = build_noaa_playback_provider(
            "noaa_test",
            "Test",
            noaa_cache_path=None,
            bucket="test-bucket",
            prefix="test/",
        )
        assert isinstance(provider, NoaaGCSProvider)


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


def _make_noaa_file(ts: float) -> NoaaAudioFile:
    """Create a NoaaAudioFile from a unix timestamp."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    filename = dt.strftime("%m_%d_%Y_%H_%M_%S") + ".aif"
    return NoaaAudioFile(
        filename=filename, key=f"archive/{filename}", timestamp=dt, size=100
    )


class TestEstimateNoaaInterval:
    def test_uniform_intervals(self):
        files = [_make_noaa_file(1000 + i * 300) for i in range(5)]
        assert estimate_noaa_interval_sec(files) == pytest.approx(300.0)

    def test_small_outliers_ignored_by_median(self):
        """3 anomalous 25s gaps among 97 normal 300s gaps → returns 300."""
        files: list[NoaaAudioFile] = []
        ts = 1000.0
        for i in range(100):
            files.append(_make_noaa_file(ts))
            if i in (10, 50, 80):
                ts += 25
            else:
                ts += 300
        assert estimate_noaa_interval_sec(files) == pytest.approx(300.0)

    def test_large_outliers_ignored_by_median(self):
        """One large gap (missing files) among 300s gaps → returns 300."""
        files: list[NoaaAudioFile] = []
        ts = 1000.0
        for i in range(50):
            files.append(_make_noaa_file(ts))
            if i == 25:
                ts += 10000
            else:
                ts += 300
        assert estimate_noaa_interval_sec(files) == pytest.approx(300.0)

    def test_single_file_returns_default(self):
        result = estimate_noaa_interval_sec([_make_noaa_file(1000)])
        assert result == DEFAULT_NOAA_SEGMENT_DURATION_SEC

    def test_empty_returns_default(self):
        result = estimate_noaa_interval_sec([])
        assert result == DEFAULT_NOAA_SEGMENT_DURATION_SEC

    def test_two_files_returns_gap(self):
        files = [_make_noaa_file(1000), _make_noaa_file(1300)]
        assert estimate_noaa_interval_sec(files) == pytest.approx(300.0)


class TestManifestReEstimation:
    def test_read_manifest_recomputes_interval(self, tmp_path):
        """Cached manifest with bad stored interval is auto-healed on read."""
        files = [_make_noaa_file(1000 + i * 300) for i in range(10)]
        path = tmp_path / "manifest.json"
        write_noaa_manifest(path, files, 25.0)

        result = read_noaa_manifest(path)
        assert result is not None
        loaded_files, interval = result
        assert len(loaded_files) == 10
        assert interval == pytest.approx(300.0)
