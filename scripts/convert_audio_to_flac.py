#!/usr/bin/env python3
"""Convert audio files to sibling FLAC files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from humpback.processing.audio_io import decode_audio

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
CONVERTIBLE_EXTENSIONS = {".wav", ".mp3"}
TARGET_EXTENSION = ".flac"
VERIFY_TOLERANCE = 5e-5


@dataclass(frozen=True)
class ConversionResult:
    source_path: Path
    target_path: Path
    status: Literal["converted", "skipped", "failed"]
    detail: str
    max_abs_error: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert audio files to sibling FLAC files"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Audio files or directories to scan recursively",
    )
    parser.add_argument(
        "--verify-samples",
        action="store_true",
        help=(
            "Decode source and output audio and require matching sample rate, "
            "sample count, and max absolute error <= "
            f"{VERIFY_TOLERANCE:g}"
        ),
    )
    return parser


def discover_audio_files(paths: list[Path]) -> list[Path]:
    """Resolve files/dirs into a deduplicated sorted list of supported audio paths."""
    discovered: dict[Path, Path] = {}

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_dir():
            matches = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS
            )
            for match in matches:
                discovered.setdefault(match.resolve(), match)
            continue

        if not path.is_file():
            raise ValueError(f"Not a file or directory: {path}")

        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {path}")

        discovered.setdefault(path.resolve(), path)

    return sorted(discovered.values(), key=lambda p: str(p))


def verify_converted_audio(
    source_audio: np.ndarray,
    source_sr: int,
    target_path: Path,
    tolerance: float = VERIFY_TOLERANCE,
) -> float:
    """Verify decoded output audio matches the decoded source within tolerance."""
    target_audio, target_sr = decode_audio(target_path)

    if source_sr != target_sr:
        raise ValueError(
            f"sample rate mismatch: source={source_sr}, target={target_sr}"
        )
    if len(source_audio) != len(target_audio):
        raise ValueError(
            f"sample count mismatch: source={len(source_audio)}, target={len(target_audio)}"
        )

    max_abs_error = (
        float(np.max(np.abs(source_audio - target_audio))) if len(source_audio) else 0.0
    )
    if max_abs_error > tolerance:
        raise ValueError(
            f"max abs error {max_abs_error:.6g} exceeds tolerance {tolerance:.6g}"
        )
    return max_abs_error


def convert_audio_file(
    source_path: Path,
    *,
    verify_samples: bool = False,
    tolerance: float = VERIFY_TOLERANCE,
) -> ConversionResult:
    """Convert a single audio file to a sibling FLAC file."""
    source_path = source_path.resolve()
    target_path = source_path.with_suffix(TARGET_EXTENSION)
    suffix = source_path.suffix.lower()

    if suffix not in AUDIO_EXTENSIONS:
        return ConversionResult(
            source_path, target_path, "failed", "unsupported format"
        )

    if suffix == TARGET_EXTENSION:
        return ConversionResult(source_path, target_path, "skipped", "already FLAC")

    if target_path.exists():
        return ConversionResult(
            source_path,
            target_path,
            "skipped",
            f"target exists: {target_path}",
        )

    try:
        audio, sample_rate = decode_audio(source_path)
        sf.write(
            str(target_path),
            audio.astype(np.float32),
            sample_rate,
            format="FLAC",
            subtype="PCM_16",
        )

        max_abs_error = None
        if verify_samples:
            max_abs_error = verify_converted_audio(
                audio.astype(np.float32),
                sample_rate,
                target_path,
                tolerance=tolerance,
            )

        detail = f"{source_path} -> {target_path}"
        if max_abs_error is not None:
            detail += f" (max_abs_error={max_abs_error:.6g})"
        return ConversionResult(
            source_path,
            target_path,
            "converted",
            detail,
            max_abs_error=max_abs_error,
        )
    except Exception as exc:
        if target_path.exists():
            target_path.unlink()
        return ConversionResult(source_path, target_path, "failed", str(exc))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        audio_files = discover_audio_files(args.paths)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    converted = 0
    skipped = 0
    failed = 0

    for source_path in audio_files:
        result = convert_audio_file(
            source_path,
            verify_samples=args.verify_samples,
            tolerance=VERIFY_TOLERANCE,
        )
        if result.status == "converted":
            converted += 1
            print(f"CONVERTED: {result.detail}")
        elif result.status == "skipped":
            skipped += 1
            print(f"SKIPPED: {result.source_path} ({result.detail})")
        else:
            failed += 1
            print(f"FAILED: {result.source_path} ({result.detail})")

    print(
        f"Summary: converted={converted}, skipped={skipped}, failed={failed}, scanned={len(audio_files)}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
