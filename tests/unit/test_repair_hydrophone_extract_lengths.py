"""Unit tests for scripts/repair_hydrophone_extract_lengths.py."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import scripts.repair_hydrophone_extract_lengths as repair_script
from humpback.classifier.extractor import write_flac_file
from humpback.models.audio import AudioFile


def _hydrophone_clip_dir(root: Path, label: str, hydrophone_id: str) -> Path:
    clip_dir = root / label / hydrophone_id / "2025" / "06" / "15"
    clip_dir.mkdir(parents=True, exist_ok=True)
    return clip_dir


def _write_short_clip(path: Path, *, frames: int, sample_rate: int) -> None:
    audio = np.linspace(-0.25, 0.25, frames, dtype=np.float32)
    sf.write(str(path), audio, sample_rate, format="FLAC", subtype="PCM_16")


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _folder_path(root: Path, file_path: Path) -> str:
    return f"{root.name}/{file_path.parent.relative_to(root).as_posix()}"


def _add_audio_file_row(session, *, root: Path, file_path: Path) -> AudioFile:
    audio_file = AudioFile(
        filename=file_path.name,
        folder_path=_folder_path(root, file_path),
        source_folder=str(file_path.parent),
        checksum_sha256=_checksum(file_path),
        duration_seconds=sf.info(str(file_path)).frames
        / sf.info(str(file_path)).samplerate,
        sample_rate_original=sf.info(str(file_path)).samplerate,
    )
    session.add(audio_file)
    return audio_file


def test_parser_defaults() -> None:
    parser = repair_script.build_parser()
    args = parser.parse_args([])
    assert args.apply is False
    assert args.window_seconds == 5.0
    assert args.max_missing_samples == 64


async def test_discover_candidates_and_dry_run(session, settings, monkeypatch) -> None:
    positive_root = Path(settings.positive_sample_path)
    clip_dir = _hydrophone_clip_dir(positive_root, "humpback", "rpi_orcasound_lab")
    clip_path = clip_dir / "20250615T080000Z_20250615T080005Z.flac"
    _write_short_clip(clip_path, frames=159997, sample_rate=32000)
    audio_file = _add_audio_file_row(session, root=positive_root, file_path=clip_path)
    await session.commit()

    candidates = await repair_script.discover_repair_candidates(
        session,
        settings,
        window_seconds=5.0,
        max_missing_samples=64,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.audio_file_id == audio_file.id
    assert candidate.missing_samples == 3

    monkeypatch.setattr(
        repair_script,
        "build_archive_playback_provider",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        repair_script,
        "fetch_hydrophone_audio_range",
        lambda *_args, **_kwargs: np.ones(candidate.expected_samples, dtype=np.float32),
    )

    result = await repair_script.repair_candidate(
        session, settings, candidate, apply=False
    )

    assert result.status == "dry-run"
    assert "Would rewrite clip" in result.detail
    assert sf.info(str(clip_path)).frames == 159997


async def test_apply_repair_rewrites_clip_and_refreshes_db(
    session, settings, monkeypatch
) -> None:
    negative_root = Path(settings.negative_sample_path)
    clip_dir = _hydrophone_clip_dir(negative_root, "ship", "rpi_orcasound_lab")
    clip_path = clip_dir / "20250615T080000Z_20250615T080005Z.flac"
    _write_short_clip(clip_path, frames=159997, sample_rate=32000)
    audio_file = _add_audio_file_row(session, root=negative_root, file_path=clip_path)
    await session.commit()

    candidates = await repair_script.discover_repair_candidates(
        session,
        settings,
        window_seconds=5.0,
        max_missing_samples=64,
    )
    candidate = candidates[0]
    repaired_audio = np.ones(candidate.expected_samples, dtype=np.float32)

    monkeypatch.setattr(
        repair_script,
        "build_archive_playback_provider",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        repair_script,
        "fetch_hydrophone_audio_range",
        lambda *_args, **_kwargs: repaired_audio,
    )

    result = await repair_script.repair_candidate(
        session, settings, candidate, apply=True
    )
    await session.commit()
    await session.refresh(audio_file)

    assert result.status == "repaired"
    assert sf.info(str(clip_path)).frames == candidate.expected_samples
    assert clip_path.with_suffix(".png").exists()
    assert audio_file.sample_rate_original == 32000
    assert audio_file.duration_seconds == pytest.approx(5.0, abs=1e-6)
    assert audio_file.checksum_sha256 == _checksum(clip_path)


async def test_apply_repair_renames_misnamed_flac_clip(
    session, settings, monkeypatch
) -> None:
    negative_root = Path(settings.negative_sample_path)
    clip_dir = _hydrophone_clip_dir(negative_root, "background", "rpi_orcasound_lab")
    clip_path = clip_dir / "20250615T080000Z_20250615T080005Z.wav"
    exact_audio = np.linspace(-0.1, 0.1, 160000, dtype=np.float32)
    sf.write(str(clip_path), exact_audio, 32000, format="FLAC", subtype="PCM_16")
    audio_file = _add_audio_file_row(session, root=negative_root, file_path=clip_path)
    await session.commit()

    candidate = (
        await repair_script.discover_repair_candidates(
            session,
            settings,
            window_seconds=5.0,
            max_missing_samples=64,
        )
    )[0]
    repaired_audio = np.ones(candidate.expected_samples, dtype=np.float32)

    assert candidate.requires_suffix_fix is True
    assert candidate.missing_samples == 0

    monkeypatch.setattr(
        repair_script,
        "build_archive_playback_provider",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        repair_script,
        "fetch_hydrophone_audio_range",
        lambda *_args, **_kwargs: repaired_audio,
    )

    result = await repair_script.repair_candidate(
        session, settings, candidate, apply=True
    )
    await session.commit()
    await session.refresh(audio_file)

    repaired_path = clip_path.with_suffix(".flac")
    assert result.status == "repaired"
    assert "renamed output to .flac" in result.detail
    assert not clip_path.exists()
    assert repaired_path.exists()
    assert repaired_path.with_suffix(".png").exists()
    assert audio_file.filename == repaired_path.name
    assert audio_file.checksum_sha256 == _checksum(repaired_path)


async def test_repair_candidate_skips_irreparable_audio(
    session, settings, monkeypatch
) -> None:
    positive_root = Path(settings.positive_sample_path)
    clip_dir = _hydrophone_clip_dir(positive_root, "orca", "rpi_orcasound_lab")
    clip_path = clip_dir / "20250615T080000Z_20250615T080005Z.flac"
    _write_short_clip(clip_path, frames=159997, sample_rate=32000)
    _add_audio_file_row(session, root=positive_root, file_path=clip_path)
    await session.commit()

    candidate = (
        await repair_script.discover_repair_candidates(
            session,
            settings,
            window_seconds=5.0,
            max_missing_samples=64,
        )
    )[0]

    monkeypatch.setattr(
        repair_script,
        "build_archive_playback_provider",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        repair_script,
        "fetch_hydrophone_audio_range",
        lambda *_args, **_kwargs: np.ones(
            candidate.expected_samples - 1, dtype=np.float32
        ),
    )

    result = await repair_script.repair_candidate(
        session, settings, candidate, apply=True
    )

    assert result.status == "skipped"
    assert "still resolved short" in result.detail
    assert sf.info(str(clip_path)).frames == 159997


async def test_repair_candidate_skips_checksum_conflict(
    session, settings, tmp_path, monkeypatch
) -> None:
    positive_root = Path(settings.positive_sample_path)
    clip_dir = _hydrophone_clip_dir(positive_root, "humpback", "rpi_orcasound_lab")
    clip_path = clip_dir / "20250615T080000Z_20250615T080005Z.flac"
    _write_short_clip(clip_path, frames=159997, sample_rate=32000)
    _add_audio_file_row(session, root=positive_root, file_path=clip_path)
    await session.commit()

    candidate = (
        await repair_script.discover_repair_candidates(
            session,
            settings,
            window_seconds=5.0,
            max_missing_samples=64,
        )
    )[0]
    repaired_audio = np.ones(candidate.expected_samples, dtype=np.float32)
    expected_path = tmp_path / "expected.flac"
    write_flac_file(repaired_audio, candidate.sample_rate, expected_path)
    conflict_checksum = _checksum(expected_path)

    session.add(
        AudioFile(
            filename="conflict.flac",
            folder_path=candidate.folder_path,
            source_folder=str(clip_dir),
            checksum_sha256=conflict_checksum,
            duration_seconds=5.0,
            sample_rate_original=candidate.sample_rate,
        )
    )
    await session.commit()

    monkeypatch.setattr(
        repair_script,
        "build_archive_playback_provider",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        repair_script,
        "fetch_hydrophone_audio_range",
        lambda *_args, **_kwargs: repaired_audio,
    )

    result = await repair_script.repair_candidate(
        session, settings, candidate, apply=True
    )

    assert result.status == "skipped"
    assert "checksum would collide" in result.detail
    assert sf.info(str(clip_path)).frames == 159997
