"""Chunk-concatenation equivalence test for ``score_audio_windows``.

Splits a fixture audio buffer in half at a whole-window boundary, calls
``score_audio_windows`` once per half with the appropriate
``time_offset_sec``, concatenates the results, and asserts the combined
list equals a single ``score_audio_windows`` call on the whole buffer to
float64 precision.

This is the correctness proof behind the Pass 1 hydrophone streaming
path: so long as chunk boundaries land on whole-window multiples and no
window ever straddles two chunks, per-chunk scoring and buffer-and-call
scoring produce identical traces.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.pipeline import Pipeline

from humpback.classifier.detector import score_audio_windows
from humpback.classifier.trainer import train_binary_classifier
from humpback.processing.inference import FakeTFLiteModel

SAMPLE_RATE = 16000
WINDOW_SIZE_SEC = 5.0
HOP_SEC = 5.0  # non-overlapping windows so every chunk boundary is clean
TOTAL_DURATION_SEC = 30.0
SPLIT_SEC = 15.0  # multiple of WINDOW_SIZE_SEC


def _synthetic_classifier() -> Pipeline:
    rng = np.random.RandomState(42)
    seed_embedding = np.sin(np.arange(64) * (1 + 1) / 64).astype(np.float32)
    pos = np.tile(seed_embedding, (20, 1)) + rng.randn(20, 64) * 0.01
    neg = rng.randn(20, 64) * 0.5 - 2.0
    pipeline, _ = train_binary_classifier(pos, neg)
    return pipeline


def _sine_audio(duration_sec: float) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_sec)
    return np.array(
        [0.7 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE) for i in range(n)],
        dtype=np.float32,
    )


def _config() -> dict[str, object]:
    return {
        "window_size_seconds": WINDOW_SIZE_SEC,
        "hop_seconds": HOP_SEC,
    }


def test_chunked_score_audio_windows_equals_single_call() -> None:
    pipeline = _synthetic_classifier()
    model = FakeTFLiteModel(vector_dim=64)
    audio = _sine_audio(TOTAL_DURATION_SEC)

    split_samples = int(SAMPLE_RATE * SPLIT_SEC)
    chunk_a = audio[:split_samples]
    chunk_b = audio[split_samples:]

    full = score_audio_windows(
        audio=audio,
        sample_rate=SAMPLE_RATE,
        perch_model=model,
        classifier=pipeline,
        config=_config(),
    )

    part_a = score_audio_windows(
        audio=chunk_a,
        sample_rate=SAMPLE_RATE,
        perch_model=model,
        classifier=pipeline,
        config=_config(),
        time_offset_sec=0.0,
    )
    part_b = score_audio_windows(
        audio=chunk_b,
        sample_rate=SAMPLE_RATE,
        perch_model=model,
        classifier=pipeline,
        config=_config(),
        time_offset_sec=SPLIT_SEC,
    )
    combined = part_a + part_b

    assert len(full) == len(combined) > 0
    for lhs, rhs in zip(full, combined):
        assert lhs["offset_sec"] == rhs["offset_sec"]
        assert lhs["end_sec"] == rhs["end_sec"]
        # Float64-precision equality on the classifier probability output.
        assert lhs["confidence"] == rhs["confidence"]


def test_time_offset_shifts_window_records() -> None:
    pipeline = _synthetic_classifier()
    model = FakeTFLiteModel(vector_dim=64)
    audio = _sine_audio(WINDOW_SIZE_SEC * 3)

    base = score_audio_windows(
        audio=audio,
        sample_rate=SAMPLE_RATE,
        perch_model=model,
        classifier=pipeline,
        config=_config(),
    )
    shifted = score_audio_windows(
        audio=audio,
        sample_rate=SAMPLE_RATE,
        perch_model=model,
        classifier=pipeline,
        config=_config(),
        time_offset_sec=100.0,
    )

    assert len(base) == len(shifted) > 0
    for b, s in zip(base, shifted):
        assert s["offset_sec"] == b["offset_sec"] + 100.0
        assert s["end_sec"] == b["end_sec"] + 100.0
        assert s["confidence"] == b["confidence"]
