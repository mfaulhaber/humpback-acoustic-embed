"""Unit tests for scripts/convert_audio_to_flac.py."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import soundfile as sf

import scripts.convert_audio_to_flac as convert_flac


def _make_wav(path: Path, duration: float = 1.0, sample_rate: int = 16000) -> None:
    n_samples = int(sample_rate * duration)
    samples = [
        int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
        for i in range(n_samples)
    ]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


def test_parser_defaults() -> None:
    parser = convert_flac.build_parser()
    args = parser.parse_args(["/tmp/audio"])
    assert args.verify_samples is False


def test_discover_audio_files_recurses_and_dedupes(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    wav_path = nested / "a.wav"
    flac_path = nested / "b.flac"
    txt_path = nested / "notes.txt"

    _make_wav(wav_path)
    sf.write(str(flac_path), [0.0, 0.1, -0.1], 16000, format="FLAC")
    txt_path.write_text("ignore")

    discovered = convert_flac.discover_audio_files([tmp_path, wav_path])

    assert discovered == [wav_path, flac_path]


def test_convert_audio_file_creates_sibling_flac(tmp_path: Path) -> None:
    wav_path = tmp_path / "clip.wav"
    _make_wav(wav_path, duration=2.0)

    result = convert_flac.convert_audio_file(wav_path)

    assert result.status == "converted"
    assert result.target_path == wav_path.with_suffix(".flac")
    assert result.target_path.exists()
    info = sf.info(str(result.target_path))
    assert info.samplerate == 16000
    assert info.subtype == "PCM_16"


def test_convert_audio_file_skips_existing_target(tmp_path: Path) -> None:
    wav_path = tmp_path / "clip.wav"
    flac_path = tmp_path / "clip.flac"
    _make_wav(wav_path)
    sf.write(str(flac_path), [0.0, 0.1, -0.1], 16000, format="FLAC")

    result = convert_flac.convert_audio_file(wav_path)

    assert result.status == "skipped"
    assert "target exists" in result.detail


def test_convert_audio_file_skips_already_flac(tmp_path: Path) -> None:
    flac_path = tmp_path / "clip.flac"
    sf.write(str(flac_path), [0.0, 0.1, -0.1], 16000, format="FLAC")

    result = convert_flac.convert_audio_file(flac_path)

    assert result.status == "skipped"
    assert result.detail == "already FLAC"


def test_convert_audio_file_verifies_samples(tmp_path: Path) -> None:
    wav_path = tmp_path / "clip.wav"
    _make_wav(wav_path, duration=2.0)

    result = convert_flac.convert_audio_file(wav_path, verify_samples=True)

    assert result.status == "converted"
    assert result.max_abs_error is not None
    assert result.max_abs_error <= convert_flac.VERIFY_TOLERANCE


def test_convert_audio_file_verification_failure_cleans_up(
    tmp_path: Path, monkeypatch
) -> None:
    wav_path = tmp_path / "clip.wav"
    _make_wav(wav_path)

    def _fail_verify(*_args, **_kwargs):
        raise ValueError("verification failed")

    monkeypatch.setattr(convert_flac, "verify_converted_audio", _fail_verify)

    result = convert_flac.convert_audio_file(wav_path, verify_samples=True)

    assert result.status == "failed"
    assert "verification failed" in result.detail
    assert not wav_path.with_suffix(".flac").exists()
