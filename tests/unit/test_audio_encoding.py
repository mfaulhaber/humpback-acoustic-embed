"""Tests for the audio encoding helpers used by exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from humpback.processing.audio_encoding import write_flac_clip


def test_write_flac_clip_round_trips_samples(tmp_path: Path) -> None:
    samples = np.linspace(-0.6, 0.6, num=1024, dtype=np.float32)
    out = tmp_path / "clip.flac"
    write_flac_clip(samples, sr=32_000, path=out)

    assert out.exists()
    assert out.stat().st_size > 0
    decoded, sr = sf.read(str(out), dtype="float32")
    assert sr == 32_000
    assert decoded.shape == samples.shape
    # PCM_16 quantization tolerance ~ 1 / 32768 ~ 3e-5.
    assert np.max(np.abs(decoded - samples)) < 1e-3


def test_write_flac_clip_does_not_normalize_loudness(tmp_path: Path) -> None:
    quiet = np.full(2048, 0.05, dtype=np.float32)
    out = tmp_path / "quiet.flac"
    write_flac_clip(quiet, sr=32_000, path=out)

    decoded, _ = sf.read(str(out), dtype="float32")
    assert np.max(np.abs(decoded)) < 0.06


def test_write_flac_clip_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "exports" / "event_encoders" / "job" / "clip.flac"
    write_flac_clip(np.zeros(256, dtype=np.float32), sr=32_000, path=nested)
    assert nested.exists()


def test_write_flac_clip_rejects_non_1d(tmp_path: Path) -> None:
    samples = np.zeros((128, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        write_flac_clip(samples, sr=32_000, path=tmp_path / "x.flac")


def test_write_flac_clip_rejects_non_positive_sample_rate(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_flac_clip(np.zeros(128, dtype=np.float32), sr=0, path=tmp_path / "x.flac")


def test_write_flac_clip_rejects_non_finite(tmp_path: Path) -> None:
    samples = np.array([0.0, np.nan, 0.5], dtype=np.float32)
    with pytest.raises(ValueError):
        write_flac_clip(samples, sr=32_000, path=tmp_path / "x.flac")


def test_write_flac_clip_atomic_no_temp_left_behind(tmp_path: Path) -> None:
    samples = np.zeros(256, dtype=np.float32)
    out = tmp_path / "clip.flac"
    write_flac_clip(samples, sr=32_000, path=out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    assert not tmp.exists()
