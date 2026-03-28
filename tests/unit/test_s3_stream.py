"""Tests for S3 streaming module (mocked, no real S3 access)."""

import io
import re
import struct
import threading
import time

import numpy as np
import pytest


def _segment_index_from_key(key: str) -> int:
    match = re.search(r"(\d+)(?=\.ts$)", key)
    assert match is not None
    return int(match.group(1))


def _make_wav_bytes(samples: np.ndarray, sr: int = 32000) -> bytes:
    """Create minimal WAV bytes from float32 samples."""
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    data_size = len(pcm) * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


class _FakeProvider:
    """Minimal ArchiveProvider for iter_audio_chunks / resolve_audio_slice tests.

    Accepts a pre-built timeline and optional fetch/decode callables.
    """

    from humpback.classifier.archive import StreamSegment as _SS

    def __init__(
        self,
        timeline: list,
        *,
        fetch_fn=None,
        decode_fn=None,
        chunk_decode_fn=None,
        invalidate_fn=None,
        supports_segment_prefetch: bool = True,
    ):
        self._timeline = timeline
        self._fetch_fn = fetch_fn
        self._decode_fn = decode_fn
        self._chunk_decode_fn = chunk_decode_fn
        self._invalidate_fn = invalidate_fn
        self.supports_segment_prefetch = supports_segment_prefetch
        self.supports_chunked_segment_decode = chunk_decode_fn is not None

    @property
    def name(self) -> str:
        return "FakeProvider"

    @property
    def source_id(self) -> str:
        return "fake_source"

    def build_timeline(self, start_ts: float, end_ts: float) -> list:
        return self._timeline

    def count_segments(self, start_ts: float, end_ts: float) -> int:
        return len(self._timeline)

    def fetch_segment(self, key: str) -> bytes:
        if self._fetch_fn is not None:
            return self._fetch_fn(key)
        return key.encode()

    def decode_segment(self, raw_bytes: bytes, target_sr: int) -> np.ndarray:
        if self._decode_fn is not None:
            return self._decode_fn(raw_bytes, target_sr)
        return np.zeros(target_sr * 10, dtype=np.float32)

    def iter_decoded_segment_chunks(
        self,
        key: str,
        raw_bytes: bytes,
        target_sr: int,
        *,
        clip_start_sec: float,
        clip_end_sec: float,
        chunk_seconds: float,
    ):
        if self._chunk_decode_fn is None:
            raise AttributeError("chunk decode not configured")
        return self._chunk_decode_fn(
            key,
            raw_bytes,
            target_sr,
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_end_sec,
            chunk_seconds=chunk_seconds,
        )

    def invalidate_cached_segment(self, key: str) -> bool:
        if self._invalidate_fn is not None:
            return self._invalidate_fn(key)
        return False


def _make_timeline(n_segments: int, folder_ts: float = 1000.0, seg_dur: float = 10.0):
    """Build a simple sequential timeline of StreamSegments."""
    from humpback.classifier.archive import StreamSegment

    return [
        StreamSegment(
            key=f"hydro/hls/{int(folder_ts)}/live{i}.ts",
            start_ts=folder_ts + i * seg_dur,
            duration_sec=seg_dur,
        )
        for i in range(n_segments)
    ]


class TestDecodeWavBytes:
    """Test the WAV parsing logic in decode_ts_bytes (via mock ffmpeg)."""

    def test_wav_roundtrip(self):
        """WAV encode → parse should recover audio shape."""
        from unittest.mock import patch, MagicMock

        from humpback.classifier.s3_stream import decode_ts_bytes

        # Create a known signal
        sr = 32000
        duration = 0.5
        n_samples = int(sr * duration)
        original = np.sin(np.linspace(0, 2 * np.pi * 440, n_samples)).astype(np.float32)
        wav_bytes = _make_wav_bytes(original, sr)

        # Mock subprocess.run to return our WAV bytes
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = wav_bytes

        with patch(
            "humpback.classifier.s3_stream.subprocess.run", return_value=mock_result
        ):
            audio = decode_ts_bytes(b"fake-ts-data", sr)

        assert audio.dtype == np.float32
        assert len(audio) == n_samples
        # Roundtrip through 16-bit PCM loses precision; check rough match
        np.testing.assert_allclose(audio, original, atol=1e-3)

    def test_ffmpeg_failure_raises(self):
        """Non-zero ffmpeg exit raises RuntimeError."""
        from unittest.mock import patch, MagicMock

        from humpback.classifier.s3_stream import decode_ts_bytes

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"Error decoding"

        with patch(
            "humpback.classifier.s3_stream.subprocess.run", return_value=mock_result
        ):
            with pytest.raises(RuntimeError, match="ffmpeg decode failed"):
                decode_ts_bytes(b"bad-data")


class TestIterAudioChunks:
    """Test the chunk iterator with ArchiveProvider."""

    def test_yields_chunks(self):
        """Should yield audio chunks from a provider with one segment."""
        from humpback.classifier.s3_stream import iter_audio_chunks

        sr = 10
        timeline = _make_timeline(1, folder_ts=1700000000.0, seg_dur=20.0)

        provider = _FakeProvider(
            timeline,
            decode_fn=lambda _raw, sr: np.zeros(sr * 20, dtype=np.float32),
        )
        chunks = list(
            iter_audio_chunks(
                provider,
                1700000000,
                1700003600,
                chunk_seconds=60.0,
                target_sr=sr,
            )
        )

        assert len(chunks) == 1
        audio, utc, segs_done, segs_total = chunks[0]
        assert len(audio) == sr * 20
        assert segs_done == 1
        assert segs_total == 1

    def test_error_callback(self):
        """Segment decode failures should call on_error and continue."""
        from humpback.classifier.s3_stream import iter_audio_chunks

        timeline = _make_timeline(1, folder_ts=1700000000.0)

        provider = _FakeProvider(
            timeline,
            fetch_fn=lambda _key: (_ for _ in ()).throw(Exception("network error")),
        )
        errors: list = []
        chunks = list(
            iter_audio_chunks(
                provider,
                1700000000,
                1700003600,
                on_error=lambda e: errors.append(e),
            )
        )

        assert len(chunks) == 0
        assert len(errors) == 1
        assert "network error" in errors[0]["message"]
        assert errors[0]["type"] == "warning"

    def test_clips_segments_to_requested_end_timestamp(self):
        """iter_audio_chunks should clip audio at end_ts."""
        from humpback.classifier.s3_stream import iter_audio_chunks

        # 3 segments [1000,1010), [1010,1020), [1020,1030) — timeline already
        # clipped by provider to the overlapping range for end_ts=1025
        timeline = _make_timeline(3, folder_ts=1000.0, seg_dur=10.0)

        provider = _FakeProvider(
            timeline,
            decode_fn=lambda _raw, sr: np.ones(sr * 10, dtype=np.float32),
        )
        chunks = list(
            iter_audio_chunks(
                provider,
                start_ts=1000.0,
                end_ts=1025.0,
                chunk_seconds=60.0,
                target_sr=10,
            )
        )

        assert len(chunks) == 1
        audio, chunk_utc, segs_done, segs_total = chunks[0]
        assert chunk_utc.timestamp() == 1000.0
        assert len(audio) == 250  # 25s * 10 Hz
        assert segs_total == 3
        assert segs_done == 3

    def test_raises_when_no_audio_timeline_exists(self):
        """Missing timeline should raise FileNotFoundError."""
        from humpback.classifier.s3_stream import iter_audio_chunks

        def _raise_no_data(_start, _end):
            raise FileNotFoundError("No audio data found for this time range")

        provider = _FakeProvider([])
        provider.build_timeline = _raise_no_data  # type: ignore[assignment]

        with pytest.raises(
            FileNotFoundError, match="No audio data found for this time range"
        ):
            list(
                iter_audio_chunks(
                    provider,
                    1700000000,
                    1700003600,
                )
            )

    def test_long_segment_chunk_decoder_emits_incremental_progress(self):
        """Chunk-capable providers should yield chunk-sized audio without full decode."""
        from humpback.classifier.s3_stream import iter_audio_chunks

        sr = 10
        timeline = _make_timeline(1, folder_ts=1000.0, seg_dur=120.0)
        fetch_calls: list[str] = []
        decode_calls: list[bytes] = []

        def _fetch(key: str) -> bytes:
            fetch_calls.append(key)
            return key.encode()

        def _chunk_decode(
            key: str,
            raw_bytes: bytes,
            target_sr: int,
            *,
            clip_start_sec: float,
            clip_end_sec: float,
            chunk_seconds: float,
        ):
            assert key.endswith("live0.ts")
            assert raw_bytes == key.encode()
            assert clip_start_sec == 0.0
            assert clip_end_sec == 120.0
            assert chunk_seconds == 30.0
            for idx in range(4):
                decode_calls.append(raw_bytes)
                yield (
                    np.full(target_sr * 30, idx, dtype=np.float32),
                    idx * 30.0,
                )

        provider = _FakeProvider(
            timeline,
            fetch_fn=_fetch,
            decode_fn=lambda raw, _sr: (_ for _ in ()).throw(
                AssertionError(f"full decode should not run for {raw!r}")
            ),
            chunk_decode_fn=_chunk_decode,
        )
        chunks = list(
            iter_audio_chunks(
                provider,
                1000.0,
                1120.0,
                chunk_seconds=30.0,
                target_sr=sr,
            )
        )

        assert fetch_calls == ["hydro/hls/1000/live0.ts"]
        assert len(decode_calls) == 4
        assert [int(chunk[0][0]) for chunk in chunks] == [0, 1, 2, 3]
        assert [chunk[2] for chunk in chunks] == [1, 1, 1, 1]
        assert [chunk[1].timestamp() for chunk in chunks] == [
            1000.0,
            1030.0,
            1060.0,
            1090.0,
        ]


class TestIterAudioChunksPrefetch:
    """Test concurrent prefetch behavior in iter_audio_chunks."""

    def test_prefetch_matches_sequential_ordering(self):
        """Prefetch mode should preserve exact timeline ordering."""
        from humpback.classifier.s3_stream import iter_audio_chunks

        timeline = _make_timeline(6, folder_ts=1000.0, seg_dur=10.0)

        def _fetch(key):
            idx = _segment_index_from_key(key)
            if idx % 2 == 0:
                time.sleep(0.01)
            return key.encode()

        def _decode(raw_bytes, sr):
            key = raw_bytes.decode()
            idx = _segment_index_from_key(key)
            return np.full(sr * 10, idx, dtype=np.float32)

        provider = _FakeProvider(timeline, fetch_fn=_fetch, decode_fn=_decode)

        sequential = list(
            iter_audio_chunks(
                provider,
                1000.0,
                1060.0,
                chunk_seconds=10.0,
                target_sr=10,
                prefetch_enabled=False,
            )
        )
        prefetched = list(
            iter_audio_chunks(
                provider,
                1000.0,
                1060.0,
                chunk_seconds=10.0,
                target_sr=10,
                prefetch_enabled=True,
                prefetch_workers=3,
                prefetch_inflight_segments=3,
            )
        )

        seq_values = [int(chunk[0][0]) for chunk in sequential]
        pre_values = [int(chunk[0][0]) for chunk in prefetched]
        assert seq_values == [0, 1, 2, 3, 4, 5]
        assert pre_values == seq_values
        assert [c[1].timestamp() for c in prefetched] == [
            1000,
            1010,
            1020,
            1030,
            1040,
            1050,
        ]

    def test_prefetch_respects_inflight_bound(self):
        """Concurrent fetches should not exceed prefetch_inflight_segments."""
        from humpback.classifier.s3_stream import iter_audio_chunks

        timeline = _make_timeline(8, folder_ts=1000.0, seg_dur=10.0)
        lock = threading.Lock()
        active = [0]
        max_active = [0]

        def _fetch(key):
            with lock:
                active[0] += 1
                max_active[0] = max(max_active[0], active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1
            return key.encode()

        provider = _FakeProvider(
            timeline,
            fetch_fn=_fetch,
            decode_fn=lambda _raw, sr: np.zeros(sr * 10, dtype=np.float32),
        )
        _ = list(
            iter_audio_chunks(
                provider,
                1000.0,
                1080.0,
                chunk_seconds=10.0,
                target_sr=10,
                prefetch_enabled=True,
                prefetch_workers=8,
                prefetch_inflight_segments=2,
            )
        )

        assert max_active[0] <= 2

    def test_prefetch_fetch_error_reports_and_continues(self):
        """Fetch errors in prefetch mode should emit alerts and continue processing."""
        from humpback.classifier.s3_stream import iter_audio_chunks

        timeline = _make_timeline(3, folder_ts=1000.0, seg_dur=10.0)

        def _fetch(key):
            idx = _segment_index_from_key(key)
            if idx == 1:
                raise RuntimeError("boom")
            return key.encode()

        def _decode(raw_bytes, sr):
            key = raw_bytes.decode()
            idx = _segment_index_from_key(key)
            return np.full(sr * 10, idx, dtype=np.float32)

        provider = _FakeProvider(timeline, fetch_fn=_fetch, decode_fn=_decode)
        errors: list = []
        chunks = list(
            iter_audio_chunks(
                provider,
                1000.0,
                1030.0,
                chunk_seconds=10.0,
                target_sr=10,
                prefetch_enabled=True,
                prefetch_workers=3,
                prefetch_inflight_segments=3,
                on_error=errors.append,
            )
        )

        assert len(errors) == 1
        assert "boom" in errors[0]["message"]
        assert [int(chunk[0][0]) for chunk in chunks] == [0, 2]
        assert chunks[-1][2] == 3
        assert chunks[-1][3] == 3

    def test_prefetch_disabled_when_provider_opts_out(self):
        """Providers can opt out of raw-byte prefetch for very large segments."""
        from humpback.classifier.s3_stream import iter_audio_chunks

        timeline = _make_timeline(6, folder_ts=1000.0, seg_dur=10.0)
        lock = threading.Lock()
        active = [0]
        max_active = [0]

        def _fetch(key: str) -> bytes:
            with lock:
                active[0] += 1
                max_active[0] = max(max_active[0], active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1
            return key.encode()

        provider = _FakeProvider(
            timeline,
            fetch_fn=_fetch,
            decode_fn=lambda _raw, sr: np.zeros(sr * 10, dtype=np.float32),
            supports_segment_prefetch=False,
        )
        _ = list(
            iter_audio_chunks(
                provider,
                1000.0,
                1060.0,
                chunk_seconds=10.0,
                target_sr=10,
                prefetch_enabled=True,
                prefetch_workers=6,
                prefetch_inflight_segments=6,
            )
        )

        assert max_active[0] == 1


class TestFolderLookback:
    """Regression tests for incremental lookback expansion."""

    def test_build_timeline_expands_lookback_until_overlap(self):
        from humpback.classifier.s3_stream import _build_stream_timeline

        class FakeClient:
            def __init__(self):
                self.window_starts: list[float] = []

            def list_hls_folders(self, _hydrophone_id, start_ts: float, end_ts: float):
                self.window_starts.append(start_ts)
                return ["1000"] if start_ts <= 1000 < end_ts else []

            def list_segments(self, _hydrophone_id, folder_ts: str):
                if folder_ts != "1000":
                    return []
                return [f"hydro/hls/1000/live{i}.ts" for i in range(120)]

            def fetch_playlist(self, _hydrophone_id, _folder_ts):
                return None

        client = FakeClient()
        timeline = _build_stream_timeline(
            client=client,
            hydrophone_id="rpi_orcasound_lab",
            stream_start_ts=1500.0,
            stream_end_ts=2000.0,
        )

        assert timeline
        assert (
            len(client.window_starts) == 2
        )  # initial window, then first lookback step

    def test_build_timeline_keeps_expanding_until_start_boundary_is_covered(self):
        from humpback.classifier.s3_stream import _build_stream_timeline

        class FakeClient:
            def __init__(self):
                self.window_starts: list[float] = []

            def list_hls_folders(self, _hydrophone_id, start_ts: float, end_ts: float):
                self.window_starts.append(start_ts)
                if start_ts <= 1100 < end_ts:
                    return ["1100", "1600"]
                if start_ts <= 1600 < end_ts:
                    return ["1600"]
                return []

            def list_segments(self, _hydrophone_id, folder_ts: str):
                if folder_ts == "1100":
                    return [f"hydro/hls/1100/live{i}.ts" for i in range(60)]
                if folder_ts == "1600":
                    return [f"hydro/hls/1600/live{i}.ts" for i in range(60)]
                return []

            def fetch_playlist(self, _hydrophone_id, _folder_ts):
                return None

        client = FakeClient()
        stream_start_ts = 1500.0
        timeline = _build_stream_timeline(
            client=client,
            hydrophone_id="rpi_orcasound_lab",
            stream_start_ts=stream_start_ts,
            stream_end_ts=2000.0,
        )

        assert timeline
        assert len(client.window_starts) == 2
        assert any(seg.start_ts <= stream_start_ts < seg.end_ts for seg in timeline)

    def test_build_timeline_stops_expanding_when_initial_window_has_overlap(self):
        from humpback.classifier.s3_stream import _build_stream_timeline

        class FakeClient:
            def __init__(self):
                self.window_starts: list[float] = []

            def list_hls_folders(self, _hydrophone_id, start_ts: float, end_ts: float):
                self.window_starts.append(start_ts)
                return ["1500"] if start_ts <= 1500 < end_ts else []

            def list_segments(self, _hydrophone_id, folder_ts: str):
                if folder_ts != "1500":
                    return []
                return [f"hydro/hls/1500/live{i}.ts" for i in range(60)]

            def fetch_playlist(self, _hydrophone_id, _folder_ts):
                return None

        client = FakeClient()
        timeline = _build_stream_timeline(
            client=client,
            hydrophone_id="rpi_orcasound_lab",
            stream_start_ts=1500.0,
            stream_end_ts=2000.0,
        )

        assert timeline
        assert len(client.window_starts) == 1


class TestMergeDetectionEvents:
    """Validate the hysteresis merge function used by hydrophone detector."""

    def test_basic_merge(self):
        from humpback.classifier.detector import merge_detection_events

        records = [
            {"offset_sec": 0.0, "end_sec": 1.0, "confidence": 0.8},
            {"offset_sec": 1.0, "end_sec": 2.0, "confidence": 0.6},
            {"offset_sec": 2.0, "end_sec": 3.0, "confidence": 0.3},
            {"offset_sec": 3.0, "end_sec": 4.0, "confidence": 0.9},
        ]
        events = merge_detection_events(records, high_threshold=0.7, low_threshold=0.5)
        assert len(events) == 2
        assert events[0]["start_sec"] == 0.0
        assert events[0]["end_sec"] == 2.0
        assert events[0]["n_windows"] == 2
        assert events[1]["start_sec"] == 3.0


class TestCachingS3Client:
    """Test CachingS3Client with filesystem-based caching."""

    def test_fetch_segment_cache_miss_then_hit(self, tmp_path):
        """First fetch goes to S3 and caches; second fetch reads from disk."""
        from unittest.mock import MagicMock

        from humpback.classifier.s3_stream import CachingS3Client, ORCASOUND_S3_BUCKET

        # Create client with mock S3
        client = CachingS3Client(str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.fetch_segment.return_value = b"audio-data-bytes"
        client._s3 = mock_s3

        key = "rpi_orcasound_lab/hls/1700000000/live000.ts"

        # First fetch: cache miss → S3
        data1 = client.fetch_segment(key)
        assert data1 == b"audio-data-bytes"
        assert mock_s3.fetch_segment.call_count == 1

        # Verify file cached on disk
        cached_path = tmp_path / ORCASOUND_S3_BUCKET / key
        assert cached_path.exists()
        assert cached_path.read_bytes() == b"audio-data-bytes"

        # Second fetch: cache hit → no S3 call
        data2 = client.fetch_segment(key)
        assert data2 == b"audio-data-bytes"
        assert mock_s3.fetch_segment.call_count == 1  # still 1

    def test_fetch_segment_404_marker(self, tmp_path):
        """404 from S3 creates marker; subsequent fetch raises SegmentNotFoundError."""
        import json
        from unittest.mock import MagicMock

        from botocore.exceptions import ClientError

        from humpback.classifier.s3_stream import (
            CachingS3Client,
            ORCASOUND_S3_BUCKET,
            SegmentNotFoundError,
        )

        client = CachingS3Client(str(tmp_path))
        mock_s3 = MagicMock()
        error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
        mock_s3.fetch_segment.side_effect = ClientError(error_response, "GetObject")
        client._s3 = mock_s3

        key = "rpi_orcasound_lab/hls/1700000000/live999.ts"

        with pytest.raises(SegmentNotFoundError):
            client.fetch_segment(key)

        # Verify .404.json marker was created
        marker = (
            tmp_path
            / ORCASOUND_S3_BUCKET
            / "rpi_orcasound_lab/hls/1700000000/live999.ts.404.json"
        )
        assert marker.exists()
        content = json.loads(marker.read_text())
        assert "cached_at_utc" in content

        # Subsequent fetch should raise immediately without calling S3
        mock_s3.fetch_segment.reset_mock()
        with pytest.raises(SegmentNotFoundError):
            client.fetch_segment(key)
        mock_s3.fetch_segment.assert_not_called()

    def test_list_segments_merges_local_and_s3(self, tmp_path):
        """list_segments merges local cached files with S3 listing."""
        from unittest.mock import MagicMock

        from humpback.classifier.s3_stream import CachingS3Client, ORCASOUND_S3_BUCKET

        client = CachingS3Client(str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.list_segments.return_value = [
            "rpi_orcasound_lab/hls/1700000000/live001.ts",
            "rpi_orcasound_lab/hls/1700000000/live002.ts",
        ]
        client._s3 = mock_s3

        # Create a local cached file not on S3
        folder = tmp_path / ORCASOUND_S3_BUCKET / "rpi_orcasound_lab/hls/1700000000"
        folder.mkdir(parents=True)
        (folder / "live000.ts").write_bytes(b"cached")

        segs = client.list_segments("rpi_orcasound_lab", "1700000000")
        assert len(segs) == 3
        assert "rpi_orcasound_lab/hls/1700000000/live000.ts" in segs
        assert "rpi_orcasound_lab/hls/1700000000/live001.ts" in segs
        assert "rpi_orcasound_lab/hls/1700000000/live002.ts" in segs

    def test_list_segments_uses_numeric_segment_order(self, tmp_path):
        """Mixed-width segment names should sort numerically, not lexicographically."""
        from unittest.mock import MagicMock

        from humpback.classifier.s3_stream import CachingS3Client

        client = CachingS3Client(str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.list_segments.return_value = [
            "rpi_orcasound_lab/hls/1700000000/live099.ts",
            "rpi_orcasound_lab/hls/1700000000/live100.ts",
            "rpi_orcasound_lab/hls/1700000000/live1000.ts",
            "rpi_orcasound_lab/hls/1700000000/live101.ts",
        ]
        client._s3 = mock_s3

        segs = client.list_segments("rpi_orcasound_lab", "1700000000")
        names = [s.split("/")[-1] for s in segs]
        assert names == ["live099.ts", "live100.ts", "live101.ts", "live1000.ts"]

    def test_list_segments_404_folder_marker(self, tmp_path):
        """Folder with .404.json marker skips S3 call."""
        import json
        from unittest.mock import MagicMock

        from humpback.classifier.s3_stream import CachingS3Client, ORCASOUND_S3_BUCKET

        client = CachingS3Client(str(tmp_path))
        mock_s3 = MagicMock()
        client._s3 = mock_s3

        # Create folder with .404.json marker
        folder = tmp_path / ORCASOUND_S3_BUCKET / "rpi_orcasound_lab/hls/1700000000"
        folder.mkdir(parents=True)
        (folder / ".404.json").write_text(json.dumps({"cached_at_utc": "test"}))

        segs = client.list_segments("rpi_orcasound_lab", "1700000000")
        assert segs == []
        mock_s3.list_segments.assert_not_called()

    def test_list_hls_folders_merges_local(self, tmp_path):
        """list_hls_folders merges S3 results with locally cached folders."""
        from unittest.mock import MagicMock

        from humpback.classifier.s3_stream import CachingS3Client, ORCASOUND_S3_BUCKET

        client = CachingS3Client(str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.list_hls_folders.return_value = ["1700000000"]
        client._s3 = mock_s3

        # Create a local folder with .ts files (not on S3)
        folder = tmp_path / ORCASOUND_S3_BUCKET / "rpi_orcasound_lab/hls/1700003600"
        folder.mkdir(parents=True)
        (folder / "live000.ts").write_bytes(b"cached")

        folders = client.list_hls_folders("rpi_orcasound_lab", 1699999000, 1700004000)
        assert "1700000000" in folders
        assert "1700003600" in folders

    def test_atomic_write(self, tmp_path):
        """Cached file is written atomically via tmp + os.replace."""
        from unittest.mock import MagicMock

        from humpback.classifier.s3_stream import CachingS3Client, ORCASOUND_S3_BUCKET

        client = CachingS3Client(str(tmp_path))
        mock_s3 = MagicMock()
        mock_s3.fetch_segment.return_value = b"segment-data"
        client._s3 = mock_s3

        key = "rpi_orcasound_lab/hls/1700000000/live000.ts"
        client.fetch_segment(key)

        # No .tmp file should remain
        cached_dir = tmp_path / ORCASOUND_S3_BUCKET / "rpi_orcasound_lab/hls/1700000000"
        tmp_files = list(cached_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

        # Final file should exist
        assert (cached_dir / "live000.ts").exists()


class TestIterAudioChunksTimestamp:
    """Verify chunk_start_ts uses segment start_ts, not caller start_ts."""

    def test_chunk_timestamp_from_segment(self):
        """Chunk timestamp should be based on segment's start_ts."""
        from datetime import datetime, timezone

        from humpback.classifier.s3_stream import iter_audio_chunks

        sr = 10
        # Timeline starts at 1700010000, not the caller's start_ts (1700000000)
        timeline = _make_timeline(1, folder_ts=1700010000.0, seg_dur=20.0)

        provider = _FakeProvider(
            timeline,
            decode_fn=lambda _raw, sr: np.zeros(sr * 20, dtype=np.float32),
        )
        chunks = list(
            iter_audio_chunks(
                provider,
                1700000000,  # start_ts much earlier than segment
                1700020000,
                chunk_seconds=60.0,
                target_sr=sr,
            )
        )

        assert len(chunks) == 1
        _, chunk_utc, _, _ = chunks[0]
        expected = datetime.fromtimestamp(1700010000, tz=timezone.utc)
        assert chunk_utc == expected


class TestResolveHydrophoneAudioSliceOrdering:
    """Regression tests for segment ordering in hydrophone playback resolver."""

    def test_resolver_does_not_jump_to_lexicographic_segment(self):
        """Slice crossing live100->live101 boundary should never include live1000."""

        from humpback.classifier.archive import StreamSegment
        from humpback.classifier.s3_stream import resolve_audio_slice

        def fake_decode(ts_bytes, sr):
            key = ts_bytes.decode()
            match = re.search(r"(\d+)(?=\.ts$)", key)
            assert match is not None
            value = float(int(match.group(1)))
            return np.full(sr * 10, value, dtype=np.float32)

        provider = _FakeProvider(
            [
                StreamSegment("hydro/hls/1500/live99.ts", 1500.0, 10.0),
                StreamSegment("hydro/hls/1500/live100.ts", 1510.0, 10.0),
                StreamSegment("hydro/hls/1500/live101.ts", 1520.0, 10.0),
                StreamSegment("hydro/hls/1500/live1000.ts", 1530.0, 10.0),
            ],
            fetch_fn=lambda key: key.encode(),
            decode_fn=fake_decode,
        )

        # filename epoch 1500 + offset 19 = absolute 1519
        audio = resolve_audio_slice(
            provider=provider,
            stream_start_ts=1000.0,
            stream_end_ts=3000.0,
            start_utc=1519.0,  # 1s before boundary between live100 and live101
            duration_sec=5.0,
            target_sr=10,
        )

        # First 1s (10 samples) from live100, remaining 4s from live101.
        assert len(audio) == 50
        assert np.all(audio[:10] == 100.0)
        assert np.all(audio[10:] == 101.0)
        assert 1000.0 not in set(np.unique(audio))


class TestResolveAudioSliceChunkedDecode:
    """Verify resolve_audio_slice uses chunked decode for NOAA-style providers."""

    def test_chunked_provider_skips_full_decode(self):
        """A chunked provider should use iter_decoded_segment_chunks, not
        decode_segment, when resolving a playback slice."""

        from humpback.classifier.archive import StreamSegment
        from humpback.classifier.s3_stream import resolve_audio_slice

        full_decode_called = False
        chunk_decode_called = False

        def fake_full_decode(_raw, sr):
            nonlocal full_decode_called
            full_decode_called = True
            return np.zeros(sr * 300, dtype=np.float32)

        def fake_chunk_decode(
            key,
            raw_bytes,
            target_sr,
            *,
            clip_start_sec,
            clip_end_sec,
            chunk_seconds,
        ):
            nonlocal chunk_decode_called
            chunk_decode_called = True
            n_samples = int(round((clip_end_sec - clip_start_sec) * target_sr))
            yield np.ones(n_samples, dtype=np.float32), clip_start_sec

        provider = _FakeProvider(
            [
                StreamSegment("noaa/audio/file.flac", 1000.0, 3600.0),
            ],
            fetch_fn=lambda key: b"fake-bytes",
            decode_fn=fake_full_decode,
            chunk_decode_fn=fake_chunk_decode,
        )

        # filename epoch 1000 + offset 10 = absolute 1010
        audio = resolve_audio_slice(
            provider=provider,
            stream_start_ts=1000.0,
            stream_end_ts=4600.0,
            start_utc=1010.0,
            duration_sec=5.0,
            target_sr=100,
        )

        assert chunk_decode_called
        assert not full_decode_called
        assert len(audio) == 500  # 5 seconds * 100 Hz

    def test_cached_segment_skips_fetch(self):
        """When is_segment_cached returns True, fetch_segment should not be called."""

        from humpback.classifier.archive import StreamSegment
        from humpback.classifier.s3_stream import resolve_audio_slice

        fetch_called = False

        def fake_fetch(key):
            nonlocal fetch_called
            fetch_called = True
            return b"fake-bytes"

        def fake_chunk_decode(
            key,
            raw_bytes,
            target_sr,
            *,
            clip_start_sec,
            clip_end_sec,
            chunk_seconds,
        ):
            n_samples = int(round((clip_end_sec - clip_start_sec) * target_sr))
            yield np.ones(n_samples, dtype=np.float32), clip_start_sec

        provider = _FakeProvider(
            [
                StreamSegment("noaa/audio/file.flac", 1000.0, 3600.0),
            ],
            fetch_fn=fake_fetch,
            chunk_decode_fn=fake_chunk_decode,
        )
        # Add is_segment_cached to the provider instance
        provider.is_segment_cached = lambda key: True  # type: ignore[attr-defined]

        # filename epoch 1000 + offset 10 = absolute 1010
        audio = resolve_audio_slice(
            provider=provider,
            stream_start_ts=1000.0,
            stream_end_ts=4600.0,
            start_utc=1010.0,
            duration_sec=5.0,
            target_sr=100,
        )

        assert not fetch_called
        assert len(audio) == 500

    def test_guard_fetch_recovers_boundary_samples_from_following_segment(self):
        """A small over-fetch should recover a few missing boundary samples."""

        from humpback.classifier.archive import StreamSegment
        from humpback.classifier.s3_stream import resolve_audio_slice

        sr = 32000
        expected_samples = sr * 5

        def fake_decode(ts_bytes, _sr):
            key = ts_bytes.decode()
            if key.endswith("live0.ts"):
                return np.ones(expected_samples - 3, dtype=np.float32)
            if key.endswith("live1.ts"):
                return np.full(expected_samples, 2.0, dtype=np.float32)
            raise AssertionError(f"Unexpected key: {key}")

        provider = _FakeProvider(
            [
                StreamSegment("hydro/hls/1000/live0.ts", 1000.0, 5.0),
                StreamSegment("hydro/hls/1000/live1.ts", 1005.0, 5.0),
            ],
            fetch_fn=lambda key: key.encode(),
            decode_fn=fake_decode,
        )

        # filename epoch 1000 + offset 0 = absolute 1000
        audio = resolve_audio_slice(
            provider=provider,
            stream_start_ts=1000.0,
            stream_end_ts=1010.0,
            start_utc=1000.0,
            duration_sec=5.0,
            target_sr=sr,
        )

        assert len(audio) == expected_samples
        assert np.all(audio[:-3] == 1.0)
        assert np.all(audio[-3:] == 2.0)


class TestSparseLocalCacheTimeline:
    """Regression tests for sparse local cache timeline reconstruction."""

    def test_sparse_local_segments_use_playlist_offsets_for_resolution(
        self, tmp_path, monkeypatch
    ):
        """Local-only playback should resolve sparse mid-sequence cached segments."""

        from humpback.classifier.providers import LocalHLSCacheProvider
        from humpback.classifier.s3_stream import (
            ORCASOUND_S3_BUCKET,
            build_stream_timeline,
            resolve_audio_slice,
        )

        hydrophone_id = "rpi_orcasound_lab"
        folder_ts = "1000"
        hls_dir = tmp_path / ORCASOUND_S3_BUCKET / hydrophone_id / "hls" / folder_ts
        hls_dir.mkdir(parents=True)

        playlist_lines = ["#EXTM3U", "#EXT-X-VERSION:3"]
        for i in range(100):
            playlist_lines.extend(["#EXTINF:10.0,", f"live{i:03d}.ts"])
        (hls_dir / "live.m3u8").write_text("\n".join(playlist_lines))

        # Simulate partial cache that starts far into the playlist.
        for i in range(80, 90):
            (hls_dir / f"live{i:03d}.ts").write_bytes(b"fake-ts")

        monkeypatch.setattr(
            "humpback.classifier.providers.orcasound_hls.decode_ts_bytes",
            lambda _ts_bytes, sr: np.ones(sr * 10, dtype=np.float32),
        )

        provider = LocalHLSCacheProvider(str(tmp_path), hydrophone_id, hydrophone_id)
        timeline = build_stream_timeline(
            provider=provider,
            stream_start_ts=1800.0,
            stream_end_ts=1900.0,
        )

        assert len(timeline) == 10
        assert timeline[0].key.endswith("live080.ts")
        assert timeline[0].start_ts == pytest.approx(1800.0, abs=1e-3)
        assert timeline[-1].key.endswith("live089.ts")

        # filename epoch 1800 + offset 0 = absolute 1800
        audio = resolve_audio_slice(
            provider=provider,
            stream_start_ts=1800.0,
            stream_end_ts=1900.0,
            start_utc=1800.0,
            duration_sec=5.0,
            target_sr=10,
            timeline=timeline,
        )
        assert len(audio) == 50


class TestDecodedAudioCache:
    """Unit tests for _DecodedAudioCache LRU in the classifier router."""

    def test_cache_hit_returns_stored_value(self):
        from humpback.api.routers.classifier import _DecodedAudioCache

        cache = _DecodedAudioCache(max_entries=4)
        audio = np.ones(100, dtype=np.float32)
        cache.put("job1", "file.flac", 10.0, 5.0, audio, 32000)

        result = cache.get("job1", "file.flac", 10.0, 5.0)
        assert result is not None
        arr, sr = result
        assert sr == 32000
        assert np.array_equal(arr, audio)

    def test_cache_miss_returns_none(self):
        from humpback.api.routers.classifier import _DecodedAudioCache

        cache = _DecodedAudioCache(max_entries=4)
        assert cache.get("job1", "file.flac", 10.0, 5.0) is None

    def test_cache_evicts_oldest(self):
        from humpback.api.routers.classifier import _DecodedAudioCache

        cache = _DecodedAudioCache(max_entries=2)
        a1 = np.ones(10, dtype=np.float32)
        a2 = np.ones(20, dtype=np.float32) * 2
        a3 = np.ones(30, dtype=np.float32) * 3

        cache.put("j1", "f1", 0.0, 5.0, a1, 100)
        cache.put("j2", "f2", 0.0, 5.0, a2, 100)
        cache.put("j3", "f3", 0.0, 5.0, a3, 100)

        # j1 should have been evicted
        assert cache.get("j1", "f1", 0.0, 5.0) is None
        # j2 and j3 should still be present
        assert cache.get("j2", "f2", 0.0, 5.0) is not None
        assert cache.get("j3", "f3", 0.0, 5.0) is not None


class TestOrcasoundS3ClientRetry:
    """Test retry logic in OrcasoundS3Client.fetch_segment."""

    def test_retries_on_incomplete_read(self):
        """IncompleteRead should be retried and succeed on later attempt."""
        from unittest.mock import MagicMock, patch

        from urllib3.exceptions import IncompleteRead

        from humpback.classifier.s3_stream import OrcasoundS3Client

        with patch("humpback.classifier.s3_stream.time.sleep") as mock_sleep:
            client = OrcasoundS3Client.__new__(OrcasoundS3Client)
            mock_boto = MagicMock()
            client._client = mock_boto
            client._bucket = "test-bucket"

            # First call raises IncompleteRead, second succeeds
            body1 = MagicMock()
            body1.read.side_effect = IncompleteRead(partial=0, expected=109416)
            resp1 = {"Body": body1}

            body2 = MagicMock()
            body2.read.return_value = b"segment-data"
            resp2 = {"Body": body2}

            mock_boto.get_object.side_effect = [resp1, resp2]

            result = client.fetch_segment("hydro/hls/1000/live0.ts")

            assert result == b"segment-data"
            assert mock_boto.get_object.call_count == 2
            mock_sleep.assert_called_once_with(1.0)

    def test_retries_on_read_timeout(self):
        """ReadTimeoutError should be retried."""
        from unittest.mock import MagicMock, patch

        from botocore.exceptions import ReadTimeoutError

        from humpback.classifier.s3_stream import OrcasoundS3Client

        with patch("humpback.classifier.s3_stream.time.sleep") as mock_sleep:
            client = OrcasoundS3Client.__new__(OrcasoundS3Client)
            mock_boto = MagicMock()
            client._client = mock_boto
            client._bucket = "test-bucket"

            # First two calls timeout, third succeeds
            mock_boto.get_object.side_effect = [
                ReadTimeoutError(endpoint_url="https://s3"),
                ReadTimeoutError(endpoint_url="https://s3"),
                {"Body": MagicMock(read=MagicMock(return_value=b"ok"))},
            ]

            result = client.fetch_segment("hydro/hls/1000/live0.ts")

            assert result == b"ok"
            assert mock_boto.get_object.call_count == 3
            assert mock_sleep.call_count == 2
            # Exponential backoff: 1s, 2s
            mock_sleep.assert_any_call(1.0)
            mock_sleep.assert_any_call(2.0)

    def test_no_retry_on_not_found(self):
        """NoSuchKey / 404 should raise immediately without retry."""
        from unittest.mock import MagicMock, patch

        from botocore.exceptions import ClientError

        from humpback.classifier.s3_stream import OrcasoundS3Client

        with patch("humpback.classifier.s3_stream.time.sleep") as mock_sleep:
            client = OrcasoundS3Client.__new__(OrcasoundS3Client)
            mock_boto = MagicMock()
            client._client = mock_boto
            client._bucket = "test-bucket"

            error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
            mock_boto.get_object.side_effect = ClientError(error_response, "GetObject")

            with pytest.raises(ClientError):
                client.fetch_segment("hydro/hls/1000/missing.ts")

            assert mock_boto.get_object.call_count == 1
            mock_sleep.assert_not_called()

    def test_no_retry_on_access_denied(self):
        """AccessDenied should raise immediately without retry."""
        from unittest.mock import MagicMock, patch

        from botocore.exceptions import ClientError

        from humpback.classifier.s3_stream import OrcasoundS3Client

        with patch("humpback.classifier.s3_stream.time.sleep") as mock_sleep:
            client = OrcasoundS3Client.__new__(OrcasoundS3Client)
            mock_boto = MagicMock()
            client._client = mock_boto
            client._bucket = "test-bucket"

            error_response = {"Error": {"Code": "AccessDenied", "Message": "Forbidden"}}
            mock_boto.get_object.side_effect = ClientError(error_response, "GetObject")

            with pytest.raises(ClientError):
                client.fetch_segment("hydro/hls/1000/live0.ts")

            assert mock_boto.get_object.call_count == 1
            mock_sleep.assert_not_called()

    def test_raises_after_all_retries_exhausted(self):
        """Should raise the last exception after all retry attempts fail."""
        from unittest.mock import MagicMock, patch

        from urllib3.exceptions import IncompleteRead

        from humpback.classifier.s3_stream import OrcasoundS3Client

        with patch("humpback.classifier.s3_stream.time.sleep"):
            client = OrcasoundS3Client.__new__(OrcasoundS3Client)
            mock_boto = MagicMock()
            client._client = mock_boto
            client._bucket = "test-bucket"

            body = MagicMock()
            body.read.side_effect = IncompleteRead(partial=0, expected=100)
            mock_boto.get_object.return_value = {"Body": body}

            with pytest.raises(IncompleteRead):
                client.fetch_segment("hydro/hls/1000/live0.ts")

            assert mock_boto.get_object.call_count == 3  # _SEGMENT_FETCH_RETRIES

    def test_retries_on_connection_reset(self):
        """ConnectionResetError should be retried."""
        from unittest.mock import MagicMock, patch

        from humpback.classifier.s3_stream import OrcasoundS3Client

        with patch("humpback.classifier.s3_stream.time.sleep"):
            client = OrcasoundS3Client.__new__(OrcasoundS3Client)
            mock_boto = MagicMock()
            client._client = mock_boto
            client._bucket = "test-bucket"

            mock_boto.get_object.side_effect = [
                ConnectionResetError("Connection reset by peer"),
                {"Body": MagicMock(read=MagicMock(return_value=b"recovered"))},
            ]

            result = client.fetch_segment("hydro/hls/1000/live0.ts")
            assert result == b"recovered"
            assert mock_boto.get_object.call_count == 2

    def test_retryable_client_error_is_retried(self):
        """Transient ClientError (e.g., InternalError) should be retried."""
        from unittest.mock import MagicMock, patch

        from botocore.exceptions import ClientError

        from humpback.classifier.s3_stream import OrcasoundS3Client

        with patch("humpback.classifier.s3_stream.time.sleep"):
            client = OrcasoundS3Client.__new__(OrcasoundS3Client)
            mock_boto = MagicMock()
            client._client = mock_boto
            client._bucket = "test-bucket"

            error_response = {"Error": {"Code": "InternalError", "Message": "Retry me"}}
            mock_boto.get_object.side_effect = [
                ClientError(error_response, "GetObject"),
                {"Body": MagicMock(read=MagicMock(return_value=b"ok"))},
            ]

            result = client.fetch_segment("hydro/hls/1000/live0.ts")
            assert result == b"ok"
            assert mock_boto.get_object.call_count == 2
