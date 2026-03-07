"""Unit tests for labeled sample extraction."""

import csv
import math
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from humpback.classifier.extractor import (
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
        "filename", "start_sec", "end_sec",
        "avg_confidence", "peak_confidence",
        "humpback", "ship", "background",
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
        _make_tsv(tsv_path, [
            {"filename": "test.wav", "start_sec": "0.0", "end_sec": "5.0",
             "avg_confidence": "0.9", "peak_confidence": "0.95",
             "humpback": "1", "ship": "", "background": ""},
            {"filename": "test.wav", "start_sec": "5.0", "end_sec": "10.0",
             "avg_confidence": "0.8", "peak_confidence": "0.85",
             "humpback": "", "ship": "1", "background": ""},
        ])

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
        _make_tsv(tsv_path, [
            {"filename": "test.wav", "start_sec": "0", "end_sec": "5",
             "avg_confidence": "0.5", "peak_confidence": "0.6",
             "humpback": "", "ship": "", "background": ""},
        ])

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
        _make_tsv(tsv_path, [
            {"filename": "test.wav", "start_sec": "0.0", "end_sec": "5.0",
             "avg_confidence": "0.9", "peak_confidence": "0.95",
             "humpback": "1", "ship": "", "background": ""},
        ])

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
        _make_tsv(tsv_path, [
            {"filename": "20250615T080000Z_hydrophone.wav",
             "start_sec": "2.5", "end_sec": "7.5",
             "avg_confidence": "0.9", "peak_confidence": "0.95",
             "humpback": "1", "ship": "", "background": ""},
        ])

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
        _make_tsv(tsv_path, [
            {"filename": "recording_001.wav",
             "start_sec": "0.0", "end_sec": "5.0",
             "avg_confidence": "0.9", "peak_confidence": "0.95",
             "humpback": "", "ship": "", "background": "1"},
        ])

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
        _make_tsv(tsv_path, [
            {"filename": "test.wav", "start_sec": "0.0", "end_sec": "5.0",
             "avg_confidence": "0.9", "peak_confidence": "0.95",
             "humpback": "1", "ship": "1", "background": ""},
        ])

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
        _make_tsv(tsv_path, [
            {"filename": "test.wav", "start_sec": "2.5", "end_sec": "10.0",
             "avg_confidence": "0.9", "peak_confidence": "0.95",
             "humpback": "1", "ship": "", "background": ""},
        ])

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        extract_labeled_samples(tsv_path, audio_folder, pos_out, neg_out, window_size_seconds=5.0)

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
        _make_tsv(tsv_path, [
            {"filename": "test.wav", "start_sec": "5.0", "end_sec": "15.0",
             "avg_confidence": "0.9", "peak_confidence": "0.95",
             "humpback": "1", "ship": "", "background": ""},
        ])

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        extract_labeled_samples(tsv_path, audio_folder, pos_out, neg_out, window_size_seconds=5.0)

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
        _make_tsv(tsv_path, [
            {"filename": "test.wav", "start_sec": "2.5", "end_sec": "7.5",
             "avg_confidence": "0.9", "peak_confidence": "0.95",
             "humpback": "1", "ship": "", "background": ""},
        ])

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
