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
