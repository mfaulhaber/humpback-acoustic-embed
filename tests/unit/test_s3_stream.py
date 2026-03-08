"""Tests for S3 streaming module (mocked, no real S3 access)."""

import io
import struct

import numpy as np
import pytest


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

        with patch("humpback.classifier.s3_stream.subprocess.run", return_value=mock_result):
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

        with patch("humpback.classifier.s3_stream.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="ffmpeg decode failed"):
                decode_ts_bytes(b"bad-data")


class TestIterAudioChunks:
    """Test the chunk iterator with mocked S3 client."""

    def test_yields_chunks(self):
        """Should yield audio chunks from mocked segments."""
        from unittest.mock import MagicMock, patch

        from humpback.classifier.s3_stream import iter_audio_chunks

        sr = 32000
        # 2-second segment → should accumulate below 60s threshold, yielded as remainder
        segment_audio = np.zeros(sr * 2, dtype=np.float32)
        wav_bytes = _make_wav_bytes(segment_audio, sr)

        mock_client = MagicMock()
        mock_client.list_hls_folders.return_value = ["1700000000"]
        mock_client.list_segments.return_value = ["hydro/hls/1700000000/seg0.ts"]
        mock_client.fetch_segment.return_value = b"fake-ts"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = wav_bytes

        with patch("humpback.classifier.s3_stream.subprocess.run", return_value=mock_result):
            chunks = list(iter_audio_chunks(
                mock_client,
                "rpi_orcasound_lab",
                1700000000,
                1700003600,
                chunk_seconds=60.0,
                target_sr=sr,
            ))

        # Should get one remainder chunk with 2s of audio
        assert len(chunks) == 1
        audio, utc, segs_done, segs_total = chunks[0]
        assert len(audio) == sr * 2
        assert segs_done == 1
        assert segs_total == 1

    def test_error_callback(self):
        """Segment decode failures should call on_error and continue."""
        from unittest.mock import MagicMock, patch

        from humpback.classifier.s3_stream import iter_audio_chunks

        mock_client = MagicMock()
        mock_client.list_hls_folders.return_value = ["1700000000"]
        mock_client.list_segments.return_value = ["seg0.ts"]
        mock_client.fetch_segment.side_effect = Exception("network error")

        errors = []

        chunks = list(iter_audio_chunks(
            mock_client,
            "rpi_orcasound_lab",
            1700000000,
            1700003600,
            on_error=lambda e: errors.append(e),
        ))

        assert len(chunks) == 0  # no audio decoded
        assert len(errors) == 1
        assert "network error" in errors[0]["message"]
        assert errors[0]["type"] == "warning"


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
        marker = tmp_path / ORCASOUND_S3_BUCKET / "rpi_orcasound_lab/hls/1700000000/live999.ts.404.json"
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
    """Verify chunk_start_ts uses first folder timestamp, not start_ts."""

    def test_chunk_timestamp_from_folder(self):
        """Chunk timestamp should be based on first folder's timestamp."""
        from datetime import datetime, timezone
        from unittest.mock import MagicMock, patch

        from humpback.classifier.s3_stream import iter_audio_chunks

        sr = 32000
        segment_audio = np.zeros(sr * 2, dtype=np.float32)
        wav_bytes = _make_wav_bytes(segment_audio, sr)

        mock_client = MagicMock()
        # Folder timestamp is 1700010000, but start_ts is 1700000000
        mock_client.list_hls_folders.return_value = ["1700010000"]
        mock_client.list_segments.return_value = ["hydro/hls/1700010000/seg0.ts"]
        mock_client.fetch_segment.return_value = b"fake-ts"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = wav_bytes

        with patch("humpback.classifier.s3_stream.subprocess.run", return_value=mock_result):
            chunks = list(iter_audio_chunks(
                mock_client,
                "rpi_orcasound_lab",
                1700000000,  # start_ts much earlier than folder
                1700020000,
                chunk_seconds=60.0,
                target_sr=sr,
            ))

        assert len(chunks) == 1
        _, chunk_utc, _, _ = chunks[0]
        # Should be based on folder ts (1700010000), not start_ts (1700000000)
        expected = datetime.fromtimestamp(1700010000, tz=timezone.utc)
        assert chunk_utc == expected
