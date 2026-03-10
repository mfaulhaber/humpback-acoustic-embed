#!/usr/bin/env python3
"""Remove broken short audio clips from a negatives folder.

Usage:
    uv run python scripts/cleanup_short_negatives.py /path/to/negatives          # dry run
    uv run python scripts/cleanup_short_negatives.py /path/to/negatives --delete  # actually delete
    uv run python scripts/cleanup_short_negatives.py /path/to/negatives --min-duration 3.0
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


def get_duration(path: Path) -> float | None:
    """Decode audio and return duration in seconds, or None on failure."""
    try:
        from humpback.processing.audio_io import decode_audio

        audio, sr = decode_audio(path)
        return len(audio) / sr
    except Exception as e:
        print(f"  WARNING: could not decode {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Remove short audio clips from negatives folder"
    )
    parser.add_argument("folder", type=Path, help="Path to negatives folder")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.5,
        help="Minimum duration in seconds (default: 4.5)",
    )
    parser.add_argument(
        "--delete", action="store_true", help="Actually delete files (default: dry run)"
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"ERROR: {args.folder} is not a directory")
        sys.exit(1)

    audio_files = sorted(
        p for p in args.folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    print(f"Scanning {len(audio_files)} audio files in {args.folder}")
    print(f"Minimum duration: {args.min_duration}s")
    print(f"Mode: {'DELETE' if args.delete else 'DRY RUN'}")
    print()

    to_delete: list[tuple[Path, float]] = []
    to_keep: list[tuple[Path, float]] = []
    failed: list[Path] = []

    for p in audio_files:
        dur = get_duration(p)
        if dur is None:
            failed.append(p)
            to_delete.append((p, 0.0))
        elif dur < args.min_duration:
            to_delete.append((p, dur))
        else:
            to_keep.append((p, dur))

    # Group by subfolder
    delete_by_folder: dict[str, list] = defaultdict(list)
    keep_by_folder: dict[str, list] = defaultdict(list)
    for p, dur in to_delete:
        rel = p.parent.relative_to(args.folder)
        delete_by_folder[str(rel)].append((p.name, dur))
    for p, dur in to_keep:
        rel = p.parent.relative_to(args.folder)
        keep_by_folder[str(rel)].append((p.name, dur))

    print(f"Files to delete: {len(to_delete)}")
    print(f"Files to keep:   {len(to_keep)}")
    if failed:
        print(f"Files that failed to decode: {len(failed)}")
    print()

    for folder in sorted(
        set(list(delete_by_folder.keys()) + list(keep_by_folder.keys()))
    ):
        n_del = len(delete_by_folder.get(folder, []))
        n_keep = len(keep_by_folder.get(folder, []))
        print(f"  {folder}/: delete {n_del}, keep {n_keep}")

    if args.delete and to_delete:
        print()
        print("Deleting files...")
        for p, dur in to_delete:
            p.unlink()
            print(f"  Deleted: {p} ({dur:.3f}s)")
        print(f"\nDone. Deleted {len(to_delete)} files.")
    elif to_delete:
        print(
            f"\nDry run complete. Re-run with --delete to remove {len(to_delete)} files."
        )


if __name__ == "__main__":
    main()
