#!/usr/bin/env python3
"""Reorder extracted vocalization files into the label/species-first layout.

Usage:
    uv run python scripts/reorder_vocalization_layout.py /Users/michael/development/data-vocalizations --dry-run
    uv run python scripts/reorder_vocalization_layout.py /Users/michael/development/data-vocalizations --execute
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

POSITIVE_LABELS = {"humpback", "orca"}
NEGATIVE_LABELS = {"ship", "background"}
SAMPLE_LIMIT = 10


@dataclass(frozen=True)
class MoveOperation:
    source: Path
    destination: Path
    bucket: str
    label: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Move vocalization WAV files from hydrophone-first layout to "
            "label/species-first layout."
        )
    )
    parser.add_argument(
        "root", type=Path, help="Dataset root containing positives/ and negatives/"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned moves without changing files (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files after preflight checks pass",
    )
    return parser


def _is_date_triplet(year: str, month: str, day: str) -> bool:
    return (
        len(year) == 4
        and len(month) == 2
        and len(day) == 2
        and year.isdigit()
        and month.isdigit()
        and day.isdigit()
    )


def _classify_move(rel_path: Path) -> MoveOperation | None:
    if rel_path.suffix.lower() != ".wav":
        return None

    parts = rel_path.parts
    if len(parts) != 7:
        return None

    bucket, first, second, year, month, day, filename = parts
    if not _is_date_triplet(year, month, day):
        return None

    if bucket == "positives":
        if second in POSITIVE_LABELS and first not in POSITIVE_LABELS:
            destination = Path(bucket) / second / first / year / month / day / filename
            return MoveOperation(
                source=rel_path,
                destination=destination,
                bucket=bucket,
                label=second,
            )
        return None

    if bucket == "negatives":
        if second in NEGATIVE_LABELS and first not in NEGATIVE_LABELS:
            destination = Path(bucket) / second / first / year / month / day / filename
            return MoveOperation(
                source=rel_path,
                destination=destination,
                bucket=bucket,
                label=second,
            )
        return None

    return None


def discover_moves(root: Path) -> list[MoveOperation]:
    moves: list[MoveOperation] = []
    for bucket in ("positives", "negatives"):
        bucket_root = root / bucket
        if not bucket_root.exists():
            continue
        for source in sorted(bucket_root.rglob("*")):
            if not source.is_file():
                continue
            rel_source = source.relative_to(root)
            move = _classify_move(rel_source)
            if move is not None:
                moves.append(move)
    return moves


def validate_moves(root: Path, moves: Sequence[MoveOperation]) -> None:
    destinations: dict[Path, Path] = {}
    for move in moves:
        source = root / move.source
        destination = root / move.destination

        if not source.exists():
            raise RuntimeError(f"Source file is missing: {source}")
        if destination.exists():
            raise RuntimeError(f"Destination already exists: {destination}")

        previous_source = destinations.get(move.destination)
        if previous_source is not None:
            raise RuntimeError(
                "Multiple source files map to the same destination: "
                f"{root / previous_source} and {source} -> {destination}"
            )
        destinations[move.destination] = move.source


def print_summary(root: Path, moves: Sequence[MoveOperation], execute: bool) -> None:
    mode = "EXECUTE" if execute else "DRY RUN"
    print(f"Root: {root}")
    print(f"Mode: {mode}")
    print(f"Planned moves: {len(moves)}")

    if not moves:
        return

    by_label = Counter((move.bucket, move.label) for move in moves)
    print()
    print("Counts by bucket/label:")
    for bucket, label in sorted(by_label):
        print(f"  {bucket}/{label}: {by_label[(bucket, label)]}")

    print()
    print(f"Sample moves (up to {SAMPLE_LIMIT}):")
    for move in moves[:SAMPLE_LIMIT]:
        print(f"  {move.source} -> {move.destination}")


def execute_moves(root: Path, moves: Sequence[MoveOperation]) -> None:
    for move in moves:
        source = root / move.source
        destination = root / move.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    execute = bool(args.execute)

    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 1

    moves = discover_moves(root)
    try:
        validate_moves(root, moves)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_summary(root, moves, execute=execute)

    if not moves:
        print()
        print("No old-layout WAV files found.")
        return 0

    if not execute:
        print()
        print("Dry run complete. Re-run with --execute to move files.")
        return 0

    execute_moves(root, moves)
    print()
    print(f"Moved {len(moves)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
