"""Tests for detection diagnostics: write_window_diagnostics and run_detection with emit_diagnostics."""

import math
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pyarrow.parquet as pq
import pytest

from humpback.classifier.detector import run_detection, write_window_diagnostics
from humpback.processing.inference import FakeTFLiteModel


def _write_wav(path: Path, duration: float = 2.0, sample_rate: int = 16000):
    """Write a simple sine wave WAV file."""
    n_samples = int(sample_rate * duration)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n_samples)]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


def test_write_window_diagnostics(tmp_path):
    """write_window_diagnostics writes valid Parquet with expected schema."""
    records = [
        {"filename": "a.wav", "window_index": 0, "offset_sec": 0.0, "end_sec": 5.0,
         "confidence": 0.8, "is_overlapped": False, "overlap_sec": 0.0},
        {"filename": "a.wav", "window_index": 1, "offset_sec": 5.0, "end_sec": 10.0,
         "confidence": 0.3, "is_overlapped": True, "overlap_sec": 2.0},
    ]
    path = tmp_path / "diag.parquet"
    write_window_diagnostics(records, path)

    assert path.is_file()
    table = pq.read_table(path)
    assert table.num_rows == 2
    assert set(table.column_names) == {
        "filename", "window_index", "offset_sec", "end_sec", "confidence", "is_overlapped", "overlap_sec",
    }
    assert table.column("filename")[0].as_py() == "a.wav"
    assert table.column("is_overlapped")[1].as_py() is True
    assert abs(table.column("overlap_sec")[1].as_py() - 2.0) < 1e-5


def _make_fake_pipeline(n_classes=2, positive_prob=0.9):
    """Create a mock sklearn pipeline that returns constant probabilities."""
    pipeline = MagicMock()
    def predict_proba(X):
        proba = np.column_stack([
            np.full(len(X), 1 - positive_prob),
            np.full(len(X), positive_prob),
        ])
        return proba
    pipeline.predict_proba = predict_proba
    return pipeline


def test_run_detection_returns_diagnostics(tmp_path):
    """run_detection with emit_diagnostics=True returns per-window records."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    # 7 seconds → 2 windows at 5s (second one overlapped)
    _write_wav(audio_dir / "test.wav", duration=7.0, sample_rate=16000)

    model = FakeTFLiteModel(vector_dim=64)
    pipeline = _make_fake_pipeline()

    detections, summary, diagnostics = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        input_format="spectrogram",
        emit_diagnostics=True,
        hop_seconds=5.0,
    )

    assert diagnostics is not None
    assert len(diagnostics) == 2

    # First window: not overlapped
    assert diagnostics[0]["window_index"] == 0
    assert diagnostics[0]["offset_sec"] == 0.0
    assert diagnostics[0]["is_overlapped"] is False
    assert diagnostics[0]["overlap_sec"] == 0.0
    assert diagnostics[0]["end_sec"] == 5.0

    # Second window: overlapped (shifted back to cover remaining 2s)
    assert diagnostics[1]["window_index"] == 1
    assert diagnostics[1]["is_overlapped"] is True
    assert diagnostics[1]["overlap_sec"] > 0.0

    assert summary["n_windows"] == 2


def test_run_detection_no_diagnostics_by_default(tmp_path):
    """run_detection without emit_diagnostics returns None for diagnostics."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    _write_wav(audio_dir / "test.wav", duration=5.0, sample_rate=16000)

    model = FakeTFLiteModel(vector_dim=64)
    pipeline = _make_fake_pipeline()

    detections, summary, diagnostics = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        input_format="spectrogram",
        emit_diagnostics=False,
    )

    assert diagnostics is None


def test_run_detection_skips_short_files(tmp_path):
    """Files shorter than one window are skipped with n_skipped_short counter."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    # 2s file with 5s window → skipped
    _write_wav(audio_dir / "short.wav", duration=2.0, sample_rate=16000)
    # 6s file → 2 windows (second overlapped)
    _write_wav(audio_dir / "normal.wav", duration=6.0, sample_rate=16000)

    model = FakeTFLiteModel(vector_dim=64)
    pipeline = _make_fake_pipeline()

    detections, summary, diagnostics = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        input_format="spectrogram",
        emit_diagnostics=True,
        hop_seconds=5.0,
    )

    assert summary["n_skipped_short"] == 1
    assert summary["n_windows"] == 2  # only from normal.wav
    assert len(diagnostics) == 2
