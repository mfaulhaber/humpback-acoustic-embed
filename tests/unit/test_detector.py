"""Tests for embed_audio_folder and detection pipeline with FakeTFLiteModel."""

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from humpback.classifier.detector import run_detection
from humpback.classifier.trainer import embed_audio_folder, train_binary_classifier
from humpback.processing.inference import FakeTFLiteModel


def _write_wav(path: Path, duration: float = 2.0, sample_rate: int = 16000):
    """Write a simple sine wave WAV file."""
    import math

    n_samples = int(sample_rate * duration)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n_samples)]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


def test_embed_audio_folder(tmp_path):
    """embed_audio_folder produces correct shape with FakeTFLiteModel."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Create 2 WAV files, each ~6 seconds (≥ 5s window)
    _write_wav(audio_dir / "a.wav", duration=6.0)
    _write_wav(audio_dir / "b.wav", duration=6.0)

    model = FakeTFLiteModel(vector_dim=128)
    result = embed_audio_folder(
        folder=audio_dir,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        input_format="spectrogram",
    )

    # Each 6s file → 2 windows (second is overlapped)
    assert result.ndim == 2
    assert result.shape[0] == 4  # 2 files, 2 windows each
    assert result.shape[1] == 128


def test_embed_audio_folder_no_files(tmp_path):
    """Empty folder raises ValueError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    model = FakeTFLiteModel(vector_dim=128)
    with pytest.raises(ValueError, match="No audio files"):
        embed_audio_folder(
            folder=empty_dir,
            model=model,
            window_size_seconds=5.0,
            target_sample_rate=16000,
        )


def test_embed_audio_folder_recursive(tmp_path):
    """Finds audio files in subdirectories."""
    audio_dir = tmp_path / "audio"
    sub_dir = audio_dir / "subdir"
    sub_dir.mkdir(parents=True)

    _write_wav(audio_dir / "a.wav", duration=6.0)
    _write_wav(sub_dir / "b.wav", duration=6.0)

    model = FakeTFLiteModel(vector_dim=64)
    result = embed_audio_folder(
        folder=audio_dir,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
    )

    assert result.shape[0] == 4  # 2 files × 2 windows each
    assert result.shape[1] == 64


def test_embed_audio_folder_longer_file(tmp_path):
    """Longer file produces multiple windows."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # 12 seconds → 3 windows at 5s each (last one overlapped)
    _write_wav(audio_dir / "long.wav", duration=12.0)

    model = FakeTFLiteModel(vector_dim=32)
    result = embed_audio_folder(
        folder=audio_dir,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
    )

    assert result.shape[0] == 3
    assert result.shape[1] == 32


def test_embed_audio_folder_short_files_skipped(tmp_path):
    """Files shorter than one window are skipped; raises if all are short."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # 2s files with 5s window → all skipped
    _write_wav(audio_dir / "a.wav", duration=2.0)
    _write_wav(audio_dir / "b.wav", duration=1.0)

    model = FakeTFLiteModel(vector_dim=64)
    with pytest.raises(ValueError, match="No embeddings produced"):
        embed_audio_folder(
            folder=audio_dir,
            model=model,
            window_size_seconds=5.0,
            target_sample_rate=16000,
        )


def test_embed_audio_folder_mixed_short_and_long(tmp_path):
    """Short files are skipped but long files are still processed."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    _write_wav(audio_dir / "short.wav", duration=2.0)  # skipped
    _write_wav(audio_dir / "long.wav", duration=10.0)   # 2 windows

    model = FakeTFLiteModel(vector_dim=64)
    result = embed_audio_folder(
        folder=audio_dir,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
    )

    assert result.shape[0] == 2
    assert result.shape[1] == 64


def test_confidence_stats_in_summary(tmp_path):
    """run_detection summary includes confidence_stats with expected keys."""
    # Create detection audio
    audio_dir = tmp_path / "detect"
    audio_dir.mkdir()
    _write_wav(audio_dir / "a.wav", duration=12.0)

    # Train a simple classifier on fake embeddings
    model = FakeTFLiteModel(vector_dim=64)
    rng = np.random.RandomState(42)
    pos = rng.randn(20, 64) + 2.0
    neg = rng.randn(20, 64) - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)

    detections, summary, _ = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
    )

    assert "confidence_stats" in summary
    stats = summary["confidence_stats"]
    expected_keys = {"mean", "median", "std", "min", "max", "p10", "p25", "p75", "p90", "pct_above_threshold"}
    assert expected_keys == set(stats.keys())
    assert 0.0 <= stats["pct_above_threshold"] <= 1.0
    assert stats["min"] <= stats["mean"] <= stats["max"]

    # New summary keys
    assert "hop_seconds" in summary
    assert "high_threshold" in summary
    assert "low_threshold" in summary


def test_run_detection_with_hop(tmp_path):
    """With hop < window, more windows are produced per file."""
    audio_dir = tmp_path / "detect"
    audio_dir.mkdir()
    _write_wav(audio_dir / "a.wav", duration=12.0)

    model = FakeTFLiteModel(vector_dim=64)
    rng = np.random.RandomState(42)
    pos = rng.randn(20, 64) + 2.0
    neg = rng.randn(20, 64) - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)

    _, summary_no_hop, _ = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        hop_seconds=5.0,
    )

    _, summary_hop, _ = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        hop_seconds=1.0,
    )

    assert summary_hop["n_windows"] > summary_no_hop["n_windows"]


def test_run_detection_on_file_complete_callback(tmp_path):
    """on_file_complete callback is invoked once per audio file with correct progress."""
    audio_dir = tmp_path / "detect"
    audio_dir.mkdir()
    _write_wav(audio_dir / "a.wav", duration=6.0)
    _write_wav(audio_dir / "b.wav", duration=6.0)
    _write_wav(audio_dir / "c.wav", duration=1.0)  # too short, will be skipped

    model = FakeTFLiteModel(vector_dim=64)
    rng = np.random.RandomState(42)
    pos = rng.randn(20, 64) + 2.0
    neg = rng.randn(20, 64) - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)

    calls = []

    def on_file_complete(file_detections, files_done, files_total):
        calls.append({
            "detections": list(file_detections),
            "files_done": files_done,
            "files_total": files_total,
        })

    run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        on_file_complete=on_file_complete,
    )

    # Should be called 3 times (a.wav, b.wav, c.wav including skipped)
    assert len(calls) == 3
    # files_total is always 3
    assert all(c["files_total"] == 3 for c in calls)
    # files_done increments
    assert [c["files_done"] for c in calls] == [1, 2, 3]
    # The skipped file (c.wav) should have empty detections
    assert calls[2]["detections"] == []


def test_run_detection_hysteresis(tmp_path):
    """Hysteresis thresholds produce different event boundaries than single threshold."""
    audio_dir = tmp_path / "detect"
    audio_dir.mkdir()
    _write_wav(audio_dir / "a.wav", duration=30.0)

    model = FakeTFLiteModel(vector_dim=64)
    rng = np.random.RandomState(42)
    pos = rng.randn(20, 64) + 2.0
    neg = rng.randn(20, 64) - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)

    dets_single, _, _ = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        hop_seconds=5.0,
        high_threshold=0.5,
        low_threshold=0.5,
    )

    dets_hysteresis, _, _ = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        hop_seconds=5.0,
        high_threshold=0.7,
        low_threshold=0.3,
    )

    # With higher start threshold, we should get fewer or equal events
    assert len(dets_hysteresis) <= len(dets_single)

    # All detections should have n_windows
    for det in dets_single + dets_hysteresis:
        assert "n_windows" in det
        assert det["n_windows"] >= 1
