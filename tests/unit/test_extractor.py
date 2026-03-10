"""Unit tests for labeled sample extraction."""

import csv
import math
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from humpback.classifier.extractor import (
    extract_hydrophone_labeled_samples,
    extract_labeled_samples,
    parse_recording_timestamp,
    write_wav_file,
)


class TestParseRecordingTimestamp:
    def test_basic_timestamp(self):
        ts = parse_recording_timestamp("20250115T143022Z_recording.wav")
        assert ts == datetime(2025, 1, 15, 14, 30, 22, tzinfo=timezone.utc)

    def test_timestamp_with_microseconds(self):
        ts = parse_recording_timestamp("20250115T143022.123456Z_data.wav")
        assert ts == datetime(2025, 1, 15, 14, 30, 22, 123456, tzinfo=timezone.utc)

    def test_timestamp_with_short_fraction(self):
        ts = parse_recording_timestamp("20250115T143022.12Z_data.wav")
        assert ts == datetime(2025, 1, 15, 14, 30, 22, 120000, tzinfo=timezone.utc)

    def test_no_timestamp(self):
        assert parse_recording_timestamp("recording_001.wav") is None

    def test_partial_match(self):
        assert parse_recording_timestamp("file_2025_01_15.wav") is None

    def test_embedded_timestamp(self):
        ts = parse_recording_timestamp("station01_20250615T080000Z.flac")
        assert ts == datetime(2025, 6, 15, 8, 0, 0, tzinfo=timezone.utc)


def _make_wav(path: Path, duration: float = 1.0, sr: int = 16000) -> None:
    """Create a simple sine wave WAV file."""
    n = int(sr * duration)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *samples))


def _make_tsv(path: Path, rows: list[dict]) -> None:
    """Write a detection TSV file."""
    fieldnames = [
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "humpback",
        "ship",
        "background",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class TestWriteWavFile:
    def test_creates_wav(self, tmp_path):
        segment = np.sin(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32)
        out = tmp_path / "sub" / "test.wav"
        write_wav_file(segment, 16000, out)
        assert out.exists()
        with wave.open(str(out), "r") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2


class TestExtractLabeledSamples:
    def test_basic_extraction(self, tmp_path):
        audio_folder = tmp_path / "audio"
        audio_folder.mkdir()
        _make_wav(audio_folder / "test.wav", duration=10.0)

        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "test.wav",
                    "start_sec": "0.0",
                    "end_sec": "5.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                },
                {
                    "filename": "test.wav",
                    "start_sec": "5.0",
                    "end_sec": "10.0",
                    "avg_confidence": "0.8",
                    "peak_confidence": "0.85",
                    "humpback": "",
                    "ship": "1",
                    "background": "",
                },
            ],
        )

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"

        summary = extract_labeled_samples(tsv_path, audio_folder, pos_out, neg_out)
        assert summary["n_humpback"] == 1
        assert summary["n_ship"] == 1
        assert summary["n_background"] == 0

        # Check humpback file exists under positive path
        humpback_files = list(pos_out.rglob("*.wav"))
        assert len(humpback_files) == 1

        # Check ship file exists under negative path
        ship_files = list((neg_out / "ship").rglob("*.wav"))
        assert len(ship_files) == 1

    def test_no_labeled_rows(self, tmp_path):
        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "test.wav",
                    "start_sec": "0",
                    "end_sec": "5",
                    "avg_confidence": "0.5",
                    "peak_confidence": "0.6",
                    "humpback": "",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        summary = extract_labeled_samples(
            tsv_path, tmp_path, tmp_path / "pos", tmp_path / "neg"
        )
        assert summary["n_humpback"] == 0
        assert summary["n_ship"] == 0
        assert summary["n_background"] == 0

    def test_idempotent_skip(self, tmp_path):
        audio_folder = tmp_path / "audio"
        audio_folder.mkdir()
        _make_wav(audio_folder / "test.wav", duration=5.0)

        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "test.wav",
                    "start_sec": "0.0",
                    "end_sec": "5.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"

        # First run
        summary1 = extract_labeled_samples(tsv_path, audio_folder, pos_out, neg_out)
        assert summary1["n_humpback"] == 1

        # Second run should skip existing
        summary2 = extract_labeled_samples(tsv_path, audio_folder, pos_out, neg_out)
        assert summary2["n_humpback"] == 0
        assert summary2["n_skipped"] == 1

    def test_timestamp_based_filename(self, tmp_path):
        audio_folder = tmp_path / "audio"
        audio_folder.mkdir()
        _make_wav(audio_folder / "20250615T080000Z_hydrophone.wav", duration=10.0)

        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "20250615T080000Z_hydrophone.wav",
                    "start_sec": "2.5",
                    "end_sec": "7.5",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        extract_labeled_samples(tsv_path, audio_folder, pos_out, neg_out)

        # Should use date-based folder: 2025/06/15
        humpback_files = list(pos_out.rglob("*.wav"))
        assert len(humpback_files) == 1
        assert "2025/06/15" in str(humpback_files[0])

    def test_fallback_filename_no_timestamp(self, tmp_path):
        audio_folder = tmp_path / "audio"
        audio_folder.mkdir()
        _make_wav(audio_folder / "recording_001.wav", duration=5.0)

        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "recording_001.wav",
                    "start_sec": "0.0",
                    "end_sec": "5.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "",
                    "ship": "",
                    "background": "1",
                },
            ],
        )

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        extract_labeled_samples(tsv_path, audio_folder, pos_out, neg_out)

        bg_files = list((neg_out / "background").rglob("*.wav"))
        assert len(bg_files) == 1
        assert "recording_001" in bg_files[0].name
        assert "unknown_date" in str(bg_files[0])

    def test_multiple_labels_same_row(self, tmp_path):
        """A row labeled as both humpback and ship produces files in both dirs."""
        audio_folder = tmp_path / "audio"
        audio_folder.mkdir()
        _make_wav(audio_folder / "test.wav", duration=5.0)

        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "test.wav",
                    "start_sec": "0.0",
                    "end_sec": "5.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "1",
                    "background": "",
                },
            ],
        )

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        summary = extract_labeled_samples(tsv_path, audio_folder, pos_out, neg_out)
        assert summary["n_humpback"] == 1
        assert summary["n_ship"] == 1


class TestExtractionSnapping:
    def test_extraction_snaps_to_window_multiples(self, tmp_path):
        """Event [2.5, 10.0] with window_size=5.0 snaps to [0.0, 10.0]."""
        audio_folder = tmp_path / "audio"
        audio_folder.mkdir()
        _make_wav(audio_folder / "test.wav", duration=15.0)

        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "test.wav",
                    "start_sec": "2.5",
                    "end_sec": "10.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        extract_labeled_samples(
            tsv_path, audio_folder, pos_out, neg_out, window_size_seconds=5.0
        )

        humpback_files = list(pos_out.rglob("*.wav"))
        assert len(humpback_files) == 1
        # Snapped to [0.0, 10.0] = 10.0s
        import wave

        with wave.open(str(humpback_files[0]), "r") as wf:
            duration = wf.getnframes() / wf.getframerate()
        assert abs(duration - 10.0) < 0.1

    def test_extraction_exact_multiple_unchanged(self, tmp_path):
        """Event [5.0, 15.0] stays [5.0, 15.0] = 10.0s."""
        audio_folder = tmp_path / "audio"
        audio_folder.mkdir()
        _make_wav(audio_folder / "test.wav", duration=20.0)

        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "test.wav",
                    "start_sec": "5.0",
                    "end_sec": "15.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        extract_labeled_samples(
            tsv_path, audio_folder, pos_out, neg_out, window_size_seconds=5.0
        )

        humpback_files = list(pos_out.rglob("*.wav"))
        assert len(humpback_files) == 1
        import wave

        with wave.open(str(humpback_files[0]), "r") as wf:
            duration = wf.getnframes() / wf.getframerate()
        assert abs(duration - 10.0) < 0.1

    def test_extraction_backward_compat_default(self, tmp_path):
        """No window_size param defaults to 5.0."""
        audio_folder = tmp_path / "audio"
        audio_folder.mkdir()
        _make_wav(audio_folder / "test.wav", duration=15.0)

        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "test.wav",
                    "start_sec": "2.5",
                    "end_sec": "7.5",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        # No window_size_seconds → defaults to 5.0 → [2.5, 7.5] snaps to [0.0, 10.0]
        extract_labeled_samples(tsv_path, audio_folder, pos_out, neg_out)

        humpback_files = list(pos_out.rglob("*.wav"))
        assert len(humpback_files) == 1
        import wave

        with wave.open(str(humpback_files[0]), "r") as wf:
            duration = wf.getnframes() / wf.getframerate()
        assert abs(duration - 10.0) < 0.1


class TestExtractHydrophoneLabeledSamples:
    """Test hydrophone-specific extraction with mocked HLS client."""

    def test_basic_hydrophone_extraction(self, tmp_path):
        """Extract labeled hydrophone samples using a mock client."""
        from unittest.mock import MagicMock, patch

        # Create TSV with a labeled detection
        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "20250615T080000Z.wav",
                    "start_sec": "0.0",
                    "end_sec": "5.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        sr = 32000
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, sr * 60)).astype(np.float32)

        mock_client = MagicMock()
        mock_client.list_hls_folders.return_value = ["1718438400"]
        mock_client.list_segments.return_value = ["rpi/hls/1718438400/seg0.ts"]
        mock_client.fetch_segment.return_value = b"fake-ts"

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"

        with patch("humpback.classifier.s3_stream.decode_ts_bytes", return_value=audio):
            summary = extract_hydrophone_labeled_samples(
                tsv_path,
                "rpi_orcasound_lab",
                pos_out,
                neg_out,
                mock_client,
                target_sample_rate=sr,
                window_size_seconds=5.0,
            )

        assert summary["n_humpback"] == 1
        assert summary["n_ship"] == 0

        humpback_files = list(pos_out.rglob("*.wav"))
        assert len(humpback_files) == 1
        rel = humpback_files[0].relative_to(pos_out)
        assert rel.parts[:2] == ("rpi_orcasound_lab", "humpback")
        assert rel.parts[2:5] == ("2025", "06", "15")

    def test_hydrophone_extraction_uses_detection_filename_exact_bounds(self, tmp_path):
        """Hydrophone extraction should use exact detection_filename bounds (no snapping)."""
        from unittest.mock import MagicMock, patch

        tsv_path = tmp_path / "detections.tsv"
        fieldnames = [
            "filename",
            "start_sec",
            "end_sec",
            "avg_confidence",
            "peak_confidence",
            "detection_filename",
            "humpback",
            "ship",
            "background",
        ]
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "20250615T080000Z.wav",
                    "start_sec": "2.0",
                    "end_sec": "7.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "detection_filename": "20250615T080003Z_20250615T080006Z.wav",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                }
            )

        sr = 32000
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, sr * 60)).astype(np.float32)

        mock_client = MagicMock()
        mock_client.list_hls_folders.return_value = ["1718438400"]
        mock_client.list_segments.return_value = ["rpi/hls/1718438400/seg0.ts"]
        mock_client.fetch_segment.return_value = b"fake-ts"

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"

        with patch("humpback.classifier.s3_stream.decode_ts_bytes", return_value=audio):
            summary = extract_hydrophone_labeled_samples(
                tsv_path,
                "rpi_orcasound_lab",
                pos_out,
                neg_out,
                mock_client,
                target_sample_rate=sr,
                window_size_seconds=5.0,
            )

        assert summary["n_humpback"] == 1
        out = (
            pos_out
            / "rpi_orcasound_lab"
            / "humpback"
            / "2025"
            / "06"
            / "15"
            / "20250615T080003Z_20250615T080006Z.wav"
        )
        assert out.exists()
        with wave.open(str(out), "r") as wf:
            duration = wf.getnframes() / wf.getframerate()
        assert abs(duration - 3.0) < 0.1

    def test_hydrophone_negative_paths_include_hydrophone_id(self, tmp_path):
        """Hydrophone negatives write under {negative_root}/{hydrophone_id}/{label}/..."""
        from unittest.mock import MagicMock, patch

        sr = 32000
        audio = np.sin(np.linspace(0, 2 * np.pi * 220, sr * 60)).astype(np.float32)

        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "20250615T080000Z.wav",
                    "start_sec": "0.0",
                    "end_sec": "5.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "",
                    "ship": "1",
                    "background": "",
                },
                {
                    "filename": "20250615T080000Z.wav",
                    "start_sec": "5.0",
                    "end_sec": "10.0",
                    "avg_confidence": "0.8",
                    "peak_confidence": "0.85",
                    "humpback": "",
                    "ship": "",
                    "background": "1",
                },
            ],
        )

        mock_client = MagicMock()
        mock_client.list_hls_folders.return_value = ["1718438400"]
        mock_client.list_segments.return_value = ["rpi/hls/1718438400/seg0.ts"]
        mock_client.fetch_segment.return_value = b"fake-ts"

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"

        with patch("humpback.classifier.s3_stream.decode_ts_bytes", return_value=audio):
            summary = extract_hydrophone_labeled_samples(
                tsv_path,
                "rpi_orcasound_lab",
                pos_out,
                neg_out,
                mock_client,
                target_sample_rate=sr,
                window_size_seconds=5.0,
            )

        assert summary["n_ship"] == 1
        assert summary["n_background"] == 1

        ship_files = list((neg_out / "rpi_orcasound_lab" / "ship").rglob("*.wav"))
        background_files = list(
            (neg_out / "rpi_orcasound_lab" / "background").rglob("*.wav")
        )
        assert len(ship_files) == 1
        assert len(background_files) == 1

        for path in ship_files + background_files:
            rel = path.relative_to(neg_out)
            assert rel.parts[0] == "rpi_orcasound_lab"
            assert rel.parts[2:5] == ("2025", "06", "15")

        # Old layout should remain unused for hydrophone extraction.
        assert not list((neg_out / "ship").rglob("*.wav"))
        assert not list((neg_out / "background").rglob("*.wav"))

    def test_no_labeled_rows(self, tmp_path):
        from unittest.mock import MagicMock

        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "20250615T080000Z.wav",
                    "start_sec": "0",
                    "end_sec": "5",
                    "avg_confidence": "0.5",
                    "peak_confidence": "0.6",
                    "humpback": "",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        mock_client = MagicMock()

        summary = extract_hydrophone_labeled_samples(
            tsv_path,
            "rpi_orcasound_lab",
            tmp_path / "pos",
            tmp_path / "neg",
            mock_client,
        )
        assert summary["n_humpback"] == 0
        mock_client.list_hls_folders.assert_not_called()

    def test_idempotent_skip(self, tmp_path):
        """Running extraction twice skips existing files."""
        from unittest.mock import MagicMock, patch

        sr = 32000
        audio = np.zeros(sr * 60, dtype=np.float32)

        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "20250615T080000Z.wav",
                    "start_sec": "0.0",
                    "end_sec": "5.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        mock_client = MagicMock()
        mock_client.list_hls_folders.return_value = ["1718438400"]
        mock_client.list_segments.return_value = ["seg0.ts"]
        mock_client.fetch_segment.return_value = b"fake"

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"

        with patch("humpback.classifier.s3_stream.decode_ts_bytes", return_value=audio):
            s1 = extract_hydrophone_labeled_samples(
                tsv_path,
                "rpi_orcasound_lab",
                pos_out,
                neg_out,
                mock_client,
            )
            s2 = extract_hydrophone_labeled_samples(
                tsv_path,
                "rpi_orcasound_lab",
                pos_out,
                neg_out,
                mock_client,
            )

        assert s1["n_humpback"] == 1
        assert s2["n_humpback"] == 0
        assert s2["n_skipped"] == 1

    def test_late_timestamp_row_extracts_with_stream_anchor(self, tmp_path):
        """Late rows resolve via first-folder anchor when stream bounds are provided."""
        from unittest.mock import MagicMock, patch

        sr = 32000
        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "19700101T003910Z.wav",
                    "start_sec": "0.0",
                    "end_sec": "5.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        mock_client = MagicMock()

        def _list_hls_folders(_hydrophone_id: str, start_ts: float, end_ts: float):
            return ["1500"] if start_ts <= 1500 <= end_ts else []

        def _list_segments(_hydrophone_id: str, folder_ts: str):
            if folder_ts != "1500":
                return []
            return [f"rpi/hls/1500/seg{i:04d}.ts" for i in range(100)]

        mock_client.list_hls_folders.side_effect = _list_hls_folders
        mock_client.list_segments.side_effect = _list_segments
        mock_client.fetch_segment.return_value = b"fake-ts"

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"

        with patch(
            "humpback.classifier.s3_stream.decode_ts_bytes",
            return_value=np.ones(sr * 10, dtype=np.float32),
        ):
            summary = extract_hydrophone_labeled_samples(
                tsv_path,
                "rpi_orcasound_lab",
                pos_out,
                neg_out,
                mock_client,
                target_sample_rate=sr,
                window_size_seconds=5.0,
                stream_start_timestamp=1000.0,
                stream_end_timestamp=3000.0,
            )

        assert summary["n_humpback"] == 1
        assert summary["n_skipped"] == 0
        assert len(list(pos_out.rglob("*.wav"))) == 1

    def test_stream_timeline_built_once_for_multiple_rows(self, tmp_path):
        """Stream timeline should be built once and reused across extraction rows."""
        from unittest.mock import MagicMock, patch

        sr = 32000
        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "19700101T002500Z.wav",
                    "start_sec": "0.0",
                    "end_sec": "5.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                },
                {
                    "filename": "19700101T002500Z.wav",
                    "start_sec": "10.0",
                    "end_sec": "15.0",
                    "avg_confidence": "0.88",
                    "peak_confidence": "0.92",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        mock_client = MagicMock()
        mock_client.list_hls_folders.return_value = ["1500"]
        mock_client.list_segments.return_value = [
            f"rpi/hls/1500/seg{i:04d}.ts" for i in range(6)
        ]
        mock_client.fetch_segment.return_value = b"fake-ts"

        with patch(
            "humpback.classifier.s3_stream.decode_ts_bytes",
            return_value=np.ones(sr * 10, dtype=np.float32),
        ):
            summary = extract_hydrophone_labeled_samples(
                tsv_path,
                "rpi_orcasound_lab",
                tmp_path / "pos",
                tmp_path / "neg",
                mock_client,
                target_sample_rate=sr,
                window_size_seconds=5.0,
                stream_start_timestamp=1000.0,
                stream_end_timestamp=2000.0,
            )

        assert summary["n_humpback"] == 2
        assert summary["n_skipped"] == 0
        # Initial range lookup plus a single max-lookback boundary-coverage check.
        assert mock_client.list_hls_folders.call_count == 2
        assert mock_client.list_segments.call_count == 1

    def test_missing_local_timeline_skips_rows_without_failure(self, tmp_path):
        """Missing local cache data should skip rows instead of failing extraction."""
        from unittest.mock import MagicMock
        from humpback.classifier.s3_stream import (
            FOLDER_LOOKBACK_STEP_SEC,
            MAX_HYDROPHONE_RANGE_SEC,
        )

        tsv_path = tmp_path / "detections.tsv"
        _make_tsv(
            tsv_path,
            [
                {
                    "filename": "20250615T080000Z.wav",
                    "start_sec": "0.0",
                    "end_sec": "5.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        mock_client = MagicMock()
        mock_client.list_hls_folders.return_value = []

        summary = extract_hydrophone_labeled_samples(
            tsv_path,
            "rpi_orcasound_lab",
            tmp_path / "pos",
            tmp_path / "neg",
            mock_client,
            target_sample_rate=32000,
            window_size_seconds=5.0,
            stream_start_timestamp=1000.0,
            stream_end_timestamp=2000.0,
        )

        assert summary["n_humpback"] == 0
        assert summary["n_skipped"] == 1
        expected_calls = int(MAX_HYDROPHONE_RANGE_SEC // FOLDER_LOOKBACK_STEP_SEC) + 1
        assert mock_client.list_hls_folders.call_count == expected_calls
        assert mock_client.list_hls_folders.call_args_list[0].args[1] == 1000.0
        assert (
            mock_client.list_hls_folders.call_args_list[-1].args[1]
            == 1000.0 - MAX_HYDROPHONE_RANGE_SEC
        )
        mock_client.list_segments.assert_not_called()
