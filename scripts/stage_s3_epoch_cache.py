#!/usr/bin/env python3
"""
Stage S3 data from epoch-style timestamp directories into a local cache.

Expected S3 layout:
    s3://BUCKET/SITE_PREFIX/hls/<epoch_ts>/...

Example:
    s3://audio-orcasound-net/rpi_north_sjc/hls/1752303617/

This script:
1. Converts the requested time range to epoch seconds.
2. Lists top-level timestamp prefixes under the given S3 prefix.
3. Filters prefixes whose directory name is an integer epoch in range.
4. Builds an s5cmd command file.
5. Optionally executes the command file.

This version is tightened for public buckets:
- always uses AWS CLI with --no-sign-request
- no credential/profile handling
- defaults to us-west-2
"""

from __future__ import annotations

import argparse
import bisect
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import which


@dataclass(frozen=True)
class TimestampPrefix:
    epoch: int
    name: str


def require_binary(name: str) -> None:
    if which(name) is None:
        print(f"ERROR: Required executable not found in PATH: {name}", file=sys.stderr)
        sys.exit(2)


def parse_dt(value: str) -> datetime:
    """
    Accepts:
      2025-07-12T00:00:00Z
      2025-07-12T00:00:00+00:00
      2025-07-12 00:00:00
      2025-07-12
    Naive datetimes are treated as UTC.
    """
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        try:
            dt = datetime.strptime(value.strip(), "%Y-%m-%d")
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid datetime '{value}'. Use ISO-8601, e.g. 2025-07-12T00:00:00Z"
            ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def to_epoch(dt: datetime) -> int:
    return int(dt.timestamp())


def iso_utc(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


def aws_ls_prefixes(bucket: str, prefix: str, region: str) -> list[TimestampPrefix]:
    """
    Run:
      aws --region <region> --no-sign-request s3 ls s3://bucket/prefix/

    Parse lines like:
      PRE 1752303617/
    """
    s3_uri = f"s3://{bucket}/{normalize_prefix(prefix)}/"
    cmd = [
        "aws",
        "--region",
        region,
        "--no-sign-request",
        "s3",
        "ls",
        s3_uri,
    ]

    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        print("ERROR: Failed to list S3 prefixes.", file=sys.stderr)
        if exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        else:
            print(" ".join(shlex.quote(x) for x in cmd), file=sys.stderr)
        sys.exit(exc.returncode or 1)

    prefixes: list[TimestampPrefix] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("PRE "):
            continue
        name = line[4:].rstrip("/")
        if name.isdigit():
            prefixes.append(TimestampPrefix(epoch=int(name), name=name))

    prefixes.sort(key=lambda p: p.epoch)
    return prefixes


def select_in_range(
    prefixes: list[TimestampPrefix], start_epoch: int, end_epoch: int
) -> list[TimestampPrefix]:
    epochs = [p.epoch for p in prefixes]
    left = bisect.bisect_left(epochs, start_epoch)
    right = bisect.bisect_left(epochs, end_epoch)
    return prefixes[left:right]


def local_dir_for_prefix(local_root: Path, prefix: str, ts_name: str) -> Path:
    return local_root / normalize_prefix(prefix) / ts_name


def build_s5cmd_commands(
    selected: list[TimestampPrefix],
    bucket: str,
    prefix: str,
    local_root: Path,
    skip_existing: bool,
) -> list[str]:
    commands: list[str] = []

    for item in selected:
        local_dir = local_dir_for_prefix(local_root, prefix, item.name)
        local_dir.mkdir(parents=True, exist_ok=True)

        if skip_existing and local_dir.exists():
            try:
                next(local_dir.iterdir())
                continue
            except StopIteration:
                pass

        src = f"s3://{bucket}/{normalize_prefix(prefix)}/{item.name}/*"
        dst = f"{str(local_dir)}/"
        commands.append(f"cp {shlex.quote(src)} {shlex.quote(dst)}")

    return commands


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def positive_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer, got '{value}'") from exc
    if n <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage public S3 epoch-prefix data into a local cache using s5cmd."
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Root prefix containing epoch directories, e.g. rpi_north_sjc/hls",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start datetime, inclusive. Example: 2025-07-12T00:00:00Z",
    )

    end_group = parser.add_mutually_exclusive_group(required=True)
    end_group.add_argument(
        "--end",
        help="End datetime, exclusive. Example: 2025-07-13T00:00:00Z",
    )
    end_group.add_argument(
        "--hours",
        type=positive_int,
        help="Alternative to --end: number of hours after --start",
    )

    parser.add_argument(
        "--local-root",
        default="/workspace/data_cache",
        help="Local cache root. Default: /workspace/data_cache",
    )
    parser.add_argument(
        "--commands-file",
        default=None,
        help="Optional output path for s5cmd commands file",
    )
    parser.add_argument(
        "--matched-prefixes-file",
        default=None,
        help="Optional output path for matched prefix list",
    )
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region for aws cli calls. Default: us-west-2",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute s5cmd after generating the command file",
    )
    parser.add_argument(
        "--numworkers",
        type=positive_int,
        default=64,
        help="s5cmd worker count. Default: 64",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip prefixes whose local cache directory already exists and is non-empty",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file for s5cmd output when --run is used",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    require_binary("aws")
    if args.run:
        require_binary("s5cmd")

    start_dt = parse_dt(args.start)
    end_dt = parse_dt(args.end) if args.end else start_dt + timedelta(hours=args.hours)

    start_epoch = to_epoch(start_dt)
    end_epoch = to_epoch(end_dt)

    if end_epoch <= start_epoch:
        print("ERROR: End time must be greater than start time.", file=sys.stderr)
        return 2

    local_root = Path(args.local_root).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    safe_prefix = normalize_prefix(args.prefix).replace("/", "_")
    commands_file = (
        Path(args.commands_file).expanduser().resolve()
        if args.commands_file
        else local_root / f"commands_{safe_prefix}.txt"
    )
    matched_file = (
        Path(args.matched_prefixes_file).expanduser().resolve()
        if args.matched_prefixes_file
        else local_root / f"matched_prefixes_{safe_prefix}.txt"
    )

    print(f"Listing prefixes under s3://{args.bucket}/{normalize_prefix(args.prefix)}/")
    all_prefixes = aws_ls_prefixes(args.bucket, args.prefix, args.region)

    if not all_prefixes:
        print("No numeric epoch-style prefixes found.", file=sys.stderr)
        return 1

    selected = select_in_range(all_prefixes, start_epoch, end_epoch)

    print(f"Requested range: [{iso_utc(start_epoch)} .. {iso_utc(end_epoch)})")
    print(f"Discovered numeric prefixes: {len(all_prefixes)}")
    print(f"Matched prefixes in range: {len(selected)}")

    write_lines(matched_file, [p.name for p in selected])

    commands = build_s5cmd_commands(
        selected=selected,
        bucket=args.bucket,
        prefix=args.prefix,
        local_root=local_root,
        skip_existing=args.skip_existing,
    )
    write_lines(commands_file, commands)

    print(f"Matched prefix file: {matched_file}")
    print(f"Commands file: {commands_file}")
    print(f"Copy commands to run: {len(commands)}")
    print(f"Local cache root: {local_root}")

    if not selected:
        print("Nothing matched the requested time range.")
        return 0

    if not args.run:
        print("Dry run only. Add --run to execute s5cmd.")
        return 0

    s5cmd_cmd = [
        "s5cmd",
        "--numworkers",
        str(args.numworkers),
        "run",
        str(commands_file),
    ]

    if args.log_file:
        log_path = Path(args.log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                s5cmd_cmd, text=True, stdout=f, stderr=subprocess.STDOUT
            )
    else:
        proc = subprocess.run(s5cmd_cmd)

    if proc.returncode != 0:
        print(f"s5cmd failed with exit code {proc.returncode}", file=sys.stderr)
        return proc.returncode

    print("s5cmd completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
