"""Unit tests for labeled sample extraction."""

import csv
import math
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf

from humpback.classifier.archive import StreamSegment
from humpback.classifier.extractor import (
    extract_hydrophone_labeled_samples,
    extract_labeled_samples,
    parse_recording_timestamp,
    write_flac_file,
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


def _make_tsv(
    path: Path, rows: list[dict], extra_fields: list[str] | None = None
) -> None:
    """Write a detection TSV file."""
    fieldnames = [
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "humpback",
        "orca",
        "ship",
        "background",
    ]
    if extra_fields:
        for f in extra_fields:
            if f not in fieldnames:
                fieldnames.append(f)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def _make_hydrophone_timeline(
    start_ts: float,
    *,
    n_segments: int = 1,
    seg_dur: float = 60.0,
    source_id: str = "rpi_orcasound_lab",
    folder_ts: int | None = None,
) -> list[StreamSegment]:
    """Build a sequential hydrophone-style timeline for provider tests."""
    folder = folder_ts if folder_ts is not None else int(start_ts)
    return [
        StreamSegment(
            key=f"{source_id}/hls/{folder}/seg{i:04d}.ts",
            start_ts=start_ts + (i * seg_dur),
            duration_sec=seg_dur,
        )
        for i in range(n_segments)
    ]


class _FakeHydrophoneProvider:
    """Minimal ArchiveProvider test double for extraction tests."""

    def __init__(
        self,
        *,
        source_id: str = "rpi_orcasound_lab",
        name: str = "Orcasound Lab",
        timeline: list[StreamSegment] | None = None,
        audio: np.ndarray | None = None,
    ) -> None:
        self._source_id = source_id
        self._name = name
        self._timeline = list(timeline or [])
        self.build_timeline_mock = MagicMock(return_value=self._timeline)
        self.fetch_segment_mock = MagicMock(return_value=b"fake-ts")
        payload = (
            np.array(audio, dtype=np.float32, copy=True)
            if audio is not None
            else np.ones(32000 * 60, dtype=np.float32)
        )
        self.decode_segment_mock = MagicMock(
            side_effect=lambda _raw, _sr: np.array(payload, dtype=np.float32, copy=True)
        )
        self.invalidate_cached_segment_mock = MagicMock(return_value=False)

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def name(self) -> str:
        return self._name

    def build_timeline(self, start_ts: float, end_ts: float) -> list[StreamSegment]:
        return self.build_timeline_mock(start_ts, end_ts)

    def count_segments(self, start_ts: float, end_ts: float) -> int:
        return len(self.build_timeline(start_ts, end_ts))

    def fetch_segment(self, key: str) -> bytes:
        return self.fetch_segment_mock(key)

    def decode_segment(self, raw_bytes: bytes, target_sr: int) -> np.ndarray:
        return self.decode_segment_mock(raw_bytes, target_sr)

    def invalidate_cached_segment(self, key: str) -> bool:
        return self.invalidate_cached_segment_mock(key)


def _provider_for_recording(
    filename: str,
    *,
    audio: np.ndarray,
    source_id: str = "rpi_orcasound_lab",
    n_segments: int = 1,
    seg_dur: float = 60.0,
    start_ts: float | None = None,
) -> _FakeHydrophoneProvider:
    """Create a provider whose timeline covers the recording timestamp."""
    recording_ts = parse_recording_timestamp(filename)
    assert recording_ts is not None
    base_ts = recording_ts.timestamp() if start_ts is None else start_ts
    return _FakeHydrophoneProvider(
        source_id=source_id,
        timeline=_make_hydrophone_timeline(
            base_ts,
            n_segments=n_segments,
            seg_dur=seg_dur,
            source_id=source_id,
        ),
        audio=audio,
    )


class TestWriteFlacFile:
    def test_creates_flac(self, tmp_path):
        segment = np.sin(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32)
        out = tmp_path / "sub" / "test.flac"
        write_flac_file(segment, 16000, out)
        assert out.exists()
        info = sf.info(str(out))
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.subtype == "PCM_16"


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
        humpback_files = list(pos_out.rglob("*.flac"))
        assert len(humpback_files) == 1

        # Check ship file exists under negative path
        ship_files = list((neg_out / "ship").rglob("*.flac"))
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
        assert summary["n_orca"] == 0
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
        humpback_files = list(pos_out.rglob("*.flac"))
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

        bg_files = list((neg_out / "background").rglob("*.flac"))
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


class TestExtractionBounds:
    def test_extraction_uses_exact_tsv_bounds(self, tmp_path):
        """Local extraction should use labeled bounds directly (no extra snapping)."""
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

        humpback_files = list(pos_out.rglob("*.flac"))
        assert len(humpback_files) == 1
        # Exact [2.5, 10.0] = 7.5s
        duration = _audio_duration(humpback_files[0])
        assert abs(duration - 7.5) < 0.1

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

        humpback_files = list(pos_out.rglob("*.flac"))
        assert len(humpback_files) == 1
        duration = _audio_duration(humpback_files[0])
        assert abs(duration - 10.0) < 0.1

    def test_extraction_default_window_param_does_not_widen_bounds(self, tmp_path):
        """Default window_size parameter should not widen local extraction clips."""
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
        # No window_size_seconds argument: extract exact [2.5, 7.5] clip.
        extract_labeled_samples(tsv_path, audio_folder, pos_out, neg_out)

        humpback_files = list(pos_out.rglob("*.flac"))
        assert len(humpback_files) == 1
        duration = _audio_duration(humpback_files[0])
        assert abs(duration - 5.0) < 0.1


class TestExtractHydrophoneLabeledSamples:
    """Test hydrophone-specific extraction with ArchiveProvider inputs."""

    def test_basic_hydrophone_extraction(self, tmp_path):
        """Extract labeled hydrophone samples using a provider."""

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
        provider = _provider_for_recording("20250615T080000Z.wav", audio=audio)

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        summary = extract_hydrophone_labeled_samples(
            tsv_path,
            provider,
            pos_out,
            neg_out,
            target_sample_rate=sr,
            window_size_seconds=5.0,
        )

        assert summary["n_humpback"] == 1
        assert summary["n_ship"] == 0

        humpback_files = list(pos_out.rglob("*.flac"))
        assert len(humpback_files) == 1
        rel = humpback_files[0].relative_to(pos_out)
        assert rel.parts[:2] == ("humpback", "rpi_orcasound_lab")
        assert rel.parts[2:5] == ("2025", "06", "15")

    def test_hydrophone_extraction_uses_detection_filename_exact_bounds(self, tmp_path):
        """Hydrophone extraction should use exact detection_filename bounds (no snapping)."""

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
                    "detection_filename": "20250615T080003Z_20250615T080006Z.flac",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                }
            )

        sr = 32000
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, sr * 60)).astype(np.float32)
        provider = _provider_for_recording("20250615T080000Z.wav", audio=audio)

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        summary = extract_hydrophone_labeled_samples(
            tsv_path,
            provider,
            pos_out,
            neg_out,
            target_sample_rate=sr,
            window_size_seconds=5.0,
        )

        assert summary["n_humpback"] == 1
        out = (
            pos_out
            / "humpback"
            / "rpi_orcasound_lab"
            / "2025"
            / "06"
            / "15"
            / "20250615T080003Z_20250615T080006Z.flac"
        )
        assert out.exists()
        duration = _audio_duration(out)
        assert abs(duration - 3.0) < 0.1

    def test_hydrophone_extraction_uses_extract_filename_when_detection_missing(
        self, tmp_path
    ):
        """Legacy rows should use extract_filename bounds when detection_filename is missing."""

        tsv_path = tmp_path / "detections.tsv"
        fieldnames = [
            "filename",
            "start_sec",
            "end_sec",
            "avg_confidence",
            "peak_confidence",
            "extract_filename",
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
                    "extract_filename": "20250615T080003Z_20250615T080006Z.wav",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                }
            )

        sr = 32000
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, sr * 60)).astype(np.float32)
        provider = _provider_for_recording("20250615T080000Z.wav", audio=audio)

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        summary = extract_hydrophone_labeled_samples(
            tsv_path,
            provider,
            pos_out,
            neg_out,
            target_sample_rate=sr,
            window_size_seconds=5.0,
        )

        assert summary["n_humpback"] == 1
        out = (
            pos_out
            / "humpback"
            / "rpi_orcasound_lab"
            / "2025"
            / "06"
            / "15"
            / "20250615T080003Z_20250615T080006Z.flac"
        )
        assert out.exists()
        duration = _audio_duration(out)
        assert abs(duration - 3.0) < 0.1

    def test_hydrophone_negative_paths_include_hydrophone_id(self, tmp_path):
        """Hydrophone negatives write under {negative_root}/{hydrophone_id}/{label}/..."""

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

        provider = _provider_for_recording("20250615T080000Z.wav", audio=audio)

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        summary = extract_hydrophone_labeled_samples(
            tsv_path,
            provider,
            pos_out,
            neg_out,
            target_sample_rate=sr,
            window_size_seconds=5.0,
        )

        assert summary["n_ship"] == 1
        assert summary["n_background"] == 1

        ship_files = list((neg_out / "ship" / "rpi_orcasound_lab").rglob("*.flac"))
        background_files = list(
            (neg_out / "background" / "rpi_orcasound_lab").rglob("*.flac")
        )
        assert len(ship_files) == 1
        assert len(background_files) == 1

        for path in ship_files + background_files:
            rel = path.relative_to(neg_out)
            # species/category first, then hydrophone_id
            assert rel.parts[0] in ("ship", "background")
            assert rel.parts[1] == "rpi_orcasound_lab"
            assert rel.parts[2:5] == ("2025", "06", "15")

        # Old layout (hydrophone_id first) should not exist.
        assert not list((neg_out / "rpi_orcasound_lab" / "ship").rglob("*.flac"))
        assert not list((neg_out / "rpi_orcasound_lab" / "background").rglob("*.flac"))

    def test_no_labeled_rows(self, tmp_path):
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

        provider = _FakeHydrophoneProvider()

        summary = extract_hydrophone_labeled_samples(
            tsv_path,
            provider,
            tmp_path / "pos",
            tmp_path / "neg",
        )
        assert summary["n_humpback"] == 0
        provider.build_timeline_mock.assert_not_called()

    def test_idempotent_skip(self, tmp_path):
        """Running extraction twice skips existing files."""

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

        provider = _provider_for_recording("20250615T080000Z.wav", audio=audio)

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        s1 = extract_hydrophone_labeled_samples(
            tsv_path,
            provider,
            pos_out,
            neg_out,
        )
        s2 = extract_hydrophone_labeled_samples(
            tsv_path,
            provider,
            pos_out,
            neg_out,
        )

        assert s1["n_humpback"] == 1
        assert s2["n_humpback"] == 0
        assert s2["n_skipped"] == 1

    def test_late_timestamp_row_extracts_with_stream_anchor(self, tmp_path):
        """Late rows resolve via first-folder anchor when stream bounds are provided."""

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

        provider = _FakeHydrophoneProvider(
            timeline=_make_hydrophone_timeline(
                1500.0,
                n_segments=100,
                seg_dur=10.0,
            ),
            audio=np.ones(sr * 10, dtype=np.float32),
        )

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        summary = extract_hydrophone_labeled_samples(
            tsv_path,
            provider,
            pos_out,
            neg_out,
            target_sample_rate=sr,
            window_size_seconds=5.0,
            stream_start_timestamp=1000.0,
            stream_end_timestamp=3000.0,
        )

        assert summary["n_humpback"] == 1
        assert summary["n_skipped"] == 0
        assert len(list(pos_out.rglob("*.flac"))) == 1

    def test_stream_timeline_built_once_for_multiple_rows(self, tmp_path):
        """Stream timeline should be built once and reused across extraction rows."""

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

        provider = _FakeHydrophoneProvider(
            timeline=_make_hydrophone_timeline(
                1500.0,
                n_segments=6,
                seg_dur=10.0,
            ),
            audio=np.ones(sr * 10, dtype=np.float32),
        )
        summary = extract_hydrophone_labeled_samples(
            tsv_path,
            provider,
            tmp_path / "pos",
            tmp_path / "neg",
            target_sample_rate=sr,
            window_size_seconds=5.0,
            stream_start_timestamp=1000.0,
            stream_end_timestamp=2000.0,
        )

        assert summary["n_humpback"] == 2
        assert summary["n_skipped"] == 0
        provider.build_timeline_mock.assert_called_once_with(1000.0, 2000.0)

    def test_missing_local_timeline_skips_rows_without_failure(self, tmp_path):
        """Missing local cache data should skip rows instead of failing extraction."""

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

        provider = _FakeHydrophoneProvider(timeline=[])

        summary = extract_hydrophone_labeled_samples(
            tsv_path,
            provider,
            tmp_path / "pos",
            tmp_path / "neg",
            target_sample_rate=32000,
            window_size_seconds=5.0,
            stream_start_timestamp=1000.0,
            stream_end_timestamp=2000.0,
        )

        assert summary["n_humpback"] == 0
        assert summary["n_skipped"] == 1
        provider.build_timeline_mock.assert_called_once_with(1000.0, 2000.0)


class TestOrcaExtraction:
    """Tests for orca label support in extraction."""

    def test_orca_local_extraction(self, tmp_path):
        """Orca label routes to positive_output_path/orca/."""
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
                    "humpback": "",
                    "orca": "1",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        summary = extract_labeled_samples(tsv_path, audio_folder, pos_out, neg_out)
        assert summary["n_orca"] == 1
        assert summary["n_humpback"] == 0

        orca_files = list((pos_out / "orca").rglob("*.flac"))
        assert len(orca_files) == 1

    def test_orca_hydrophone_extraction(self, tmp_path):
        """Orca routes to positive_output_path/orca/{hydrophone}/."""
        sr = 32000
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, sr * 60)).astype(np.float32)

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
                    "orca": "1",
                    "ship": "",
                    "background": "",
                },
            ],
        )

        provider = _provider_for_recording("20250615T080000Z.wav", audio=audio)

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        summary = extract_hydrophone_labeled_samples(
            tsv_path,
            provider,
            pos_out,
            neg_out,
            target_sample_rate=sr,
            window_size_seconds=5.0,
        )

        assert summary["n_orca"] == 1
        assert summary["n_humpback"] == 0

        orca_files = list(pos_out.rglob("*.flac"))
        assert len(orca_files) == 1
        rel = orca_files[0].relative_to(pos_out)
        assert rel.parts[:2] == ("orca", "rpi_orcasound_lab")

    def test_hydrophone_path_order_species_before_hydrophone(self, tmp_path):
        """All hydrophone labels follow species/category-before-hydrophone path order."""
        sr = 32000
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, sr * 60)).astype(np.float32)

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
                    "orca": "1",
                    "ship": "1",
                    "background": "1",
                },
            ],
        )

        provider = _provider_for_recording(
            "20250615T080000Z.wav",
            audio=audio,
            source_id="rpi_north_sjc",
        )

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        summary = extract_hydrophone_labeled_samples(
            tsv_path,
            provider,
            pos_out,
            neg_out,
            target_sample_rate=sr,
            window_size_seconds=5.0,
        )

        assert summary["n_humpback"] == 1
        assert summary["n_orca"] == 1
        assert summary["n_ship"] == 1
        assert summary["n_background"] == 1

        # Verify species/category is first path component, then hydrophone_id
        for label in ("humpback", "orca"):
            files = list((pos_out / label / "rpi_north_sjc").rglob("*.flac"))
            assert len(files) == 1, f"Expected 1 file for {label}"
        for label in ("ship", "background"):
            files = list((neg_out / label / "rpi_north_sjc").rglob("*.flac"))
            assert len(files) == 1, f"Expected 1 file for {label}"

    def test_backward_compat_tsv_without_orca_column(self, tmp_path):
        """TSV files without an orca column should extract with n_orca == 0."""
        audio_folder = tmp_path / "audio"
        audio_folder.mkdir()
        _make_wav(audio_folder / "test.wav", duration=10.0)

        # Write TSV without orca column (legacy format)
        tsv_path = tmp_path / "detections.tsv"
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
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerow(
                {
                    "filename": "test.wav",
                    "start_sec": "0.0",
                    "end_sec": "5.0",
                    "avg_confidence": "0.9",
                    "peak_confidence": "0.95",
                    "humpback": "1",
                    "ship": "",
                    "background": "",
                }
            )

        pos_out = tmp_path / "positive"
        neg_out = tmp_path / "negative"
        summary = extract_labeled_samples(tsv_path, audio_folder, pos_out, neg_out)
        assert summary["n_humpback"] == 1
        assert summary["n_orca"] == 0
