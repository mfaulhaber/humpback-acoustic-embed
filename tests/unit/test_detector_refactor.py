"""Regression tests for the ``compute_hysteresis_events`` extraction.

These tests guard the Phase 0 detector refactor: a snapshot captured
from ``run_detection`` before the refactor is compared against the
current output to ensure bit-identical detection rows, and the newly
extracted helper is exercised against short audio and a synthetic
fixture.
"""

import json
import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from humpback.classifier.detector import compute_hysteresis_events, run_detection
from humpback.classifier.trainer import train_binary_classifier
from humpback.processing.inference import FakeTFLiteModel

SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "detector_refactor_snapshot.json"
)


def _write_sine(path: Path, duration: float, freq: float = 440.0) -> None:
    sample_rate = 16000
    n = int(sample_rate * duration)
    samples = [
        int(32767 * 0.7 * math.sin(2 * math.pi * freq * i / sample_rate))
        for i in range(n)
    ]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))


def _synthetic_classifier() -> Pipeline:
    """Rebuild the same deterministic classifier used to capture the snapshot."""
    rng = np.random.RandomState(42)
    seed_embedding = np.sin(np.arange(64) * (1 + 1) / 64).astype(np.float32)
    pos = np.tile(seed_embedding, (20, 1)) + rng.randn(20, 64) * 0.01
    neg = rng.randn(20, 64) * 0.5 - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)
    return pipeline


def _build_fixture_audio_dir(tmp_path: Path) -> Path:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    # Filename carries a UTC timestamp so ``_file_base_epoch`` is deterministic.
    _write_sine(audio_dir / "sample_20260411T000000Z.wav", duration=12.0)
    return audio_dir


def test_run_detection_matches_pre_refactor_snapshot(tmp_path: Path) -> None:
    """Post-refactor ``run_detection`` output matches the committed snapshot."""
    audio_dir = _build_fixture_audio_dir(tmp_path)
    pipeline = _synthetic_classifier()
    model = FakeTFLiteModel(vector_dim=64)

    detections, summary, diagnostics, _ = run_detection(
        audio_folder=audio_dir,
        pipeline=pipeline,
        model=model,
        window_size_seconds=5.0,
        target_sample_rate=16000,
        confidence_threshold=0.5,
        hop_seconds=1.0,
        high_threshold=0.70,
        low_threshold=0.45,
        emit_diagnostics=True,
    )

    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    assert detections == snapshot["detections"]
    assert diagnostics == snapshot["diagnostics"]
    assert summary["n_files"] == snapshot["summary_subset"]["n_files"]
    assert summary["n_windows"] == snapshot["summary_subset"]["n_windows"]
    assert summary["n_detections"] == snapshot["summary_subset"]["n_detections"]
    assert summary["n_spans"] == snapshot["summary_subset"]["n_spans"]


def test_compute_hysteresis_events_shapes(tmp_path: Path) -> None:
    """Helper returns non-empty records + events and all spans valid."""
    _build_fixture_audio_dir(tmp_path)
    pipeline = _synthetic_classifier()
    model = FakeTFLiteModel(vector_dim=64)

    sample_rate = 16000
    n = int(sample_rate * 12.0)
    audio = np.array(
        [0.7 * math.sin(2 * math.pi * 440 * i / sample_rate) for i in range(n)],
        dtype=np.float32,
    )

    window_records, events = compute_hysteresis_events(
        audio=audio,
        sample_rate=sample_rate,
        perch_model=model,
        classifier=pipeline,
        config={
            "window_size_seconds": 5.0,
            "hop_seconds": 1.0,
            "high_threshold": 0.70,
            "low_threshold": 0.45,
        },
    )

    assert len(window_records) > 0
    assert all(r["end_sec"] > r["offset_sec"] for r in window_records)
    assert len(events) >= 1
    for ev in events:
        assert ev["start_sec"] <= ev["end_sec"]


def test_compute_hysteresis_events_short_audio_returns_empty() -> None:
    """Audio shorter than one window returns two empty lists."""
    pipeline = _synthetic_classifier()
    model = FakeTFLiteModel(vector_dim=64)
    short_audio = np.zeros(int(16000 * 2.0), dtype=np.float32)  # 2s < 5s window

    window_records, events = compute_hysteresis_events(
        audio=short_audio,
        sample_rate=16000,
        perch_model=model,
        classifier=pipeline,
        config={
            "window_size_seconds": 5.0,
            "hop_seconds": 1.0,
            "high_threshold": 0.70,
            "low_threshold": 0.45,
        },
    )

    assert window_records == []
    assert events == []


def test_compute_hysteresis_events_is_importable() -> None:
    """Helper is importable from the module's public surface."""
    # Static import at top of file already validates this; the assertion
    # is a sentinel so a future rename can't silently break the contract.
    assert compute_hysteresis_events.__name__ == "compute_hysteresis_events"


@pytest.mark.skipif(not SNAPSHOT_PATH.exists(), reason="snapshot not committed")
def test_snapshot_file_is_well_formed() -> None:
    """Guardrail: the committed snapshot JSON parses and has expected keys."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    assert "detections" in snapshot
    assert "diagnostics" in snapshot
    assert "summary_subset" in snapshot
