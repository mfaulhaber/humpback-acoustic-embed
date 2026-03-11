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
3. Uses coarse epoch filtering, then refines matches by object LastModified overlap.
4. Builds an s5cmd command file.
5. Executes per-prefix downloads (or skips execution in dry-run mode).

This version is tightened for public buckets:
- always uses AWS CLI with --no-sign-request
- always uses s5cmd with --no-sign-request
- no credential/profile handling
- defaults to us-west-2
"""

from __future__ import annotations

import argparse
import bisect
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import which

from tqdm import tqdm


@dataclass(frozen=True)
class TimestampPrefix:
    epoch: int
    name: str


@dataclass(frozen=True)
class PrefixSelection:
    coarse_matches: list[TimestampPrefix]
    overlap_candidates: list[TimestampPrefix]
    final_matches: list[TimestampPrefix]
    overlap_inspected: int
    overlap_filter_used: bool
    overlap_error: str | None = None


class S3ObjectListingError(RuntimeError):
    """Raised when object-level overlap checks cannot be completed."""


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
    return aws_ls_prefixes_optimized(
        bucket=bucket,
        prefix=prefix,
        region=region,
        start_epoch=None,
        end_epoch=None,
    )


def aws_ls_prefixes_optimized(
    bucket: str,
    prefix: str,
    region: str,
    start_epoch: int | None,
    end_epoch: int | None,
    start_after_lookback_seconds: int = 86_400,
) -> list[TimestampPrefix]:
    """
    List top-level numeric timestamp prefixes via s3api list-objects-v2.

    Uses:
    - delimiter='/' to return CommonPrefixes (top-level "directories")
    - start-after near requested start to reduce scan time
    - early stop once numeric prefixes reach/exceed requested end
    """
    root_prefix = f"{normalize_prefix(prefix)}/"
    continuation_token: str | None = None
    start_after: str | None = None

    if start_epoch is not None:
        start_after_epoch = max(0, start_epoch - max(0, start_after_lookback_seconds))
        start_after = f"{root_prefix}{start_after_epoch}/"

    prefixes: list[TimestampPrefix] = []
    stop_early = False

    while True:
        try:
            payload = _aws_list_objects_page(
                bucket=bucket,
                key_prefix=root_prefix,
                region=region,
                continuation_token=continuation_token,
                delimiter="/",
                start_after=start_after,
            )
        except S3ObjectListingError as exc:
            print("ERROR: Failed to list S3 prefixes.", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            sys.exit(1)

        # StartAfter applies to the initial listing position only.
        start_after = None

        common_prefixes = payload.get("CommonPrefixes")
        if isinstance(common_prefixes, list):
            for entry in common_prefixes:
                if not isinstance(entry, dict):
                    continue
                raw_prefix = entry.get("Prefix")
                if not isinstance(raw_prefix, str):
                    continue
                if not raw_prefix.startswith(root_prefix):
                    continue
                ts_name = raw_prefix[len(root_prefix) :].strip("/")
                if not ts_name.isdigit():
                    continue

                epoch = int(ts_name)
                prefixes.append(TimestampPrefix(epoch=epoch, name=ts_name))
                if end_epoch is not None and epoch >= end_epoch:
                    stop_early = True

        if stop_early:
            break
        if not payload.get("IsTruncated"):
            break
        next_token = payload.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token:
            break
        continuation_token = next_token

    return _dedupe_sorted_prefixes(prefixes)


def select_in_range(
    prefixes: list[TimestampPrefix], start_epoch: int, end_epoch: int
) -> list[TimestampPrefix]:
    epochs = [p.epoch for p in prefixes]
    left = bisect.bisect_left(epochs, start_epoch)
    right = bisect.bisect_left(epochs, end_epoch)
    return prefixes[left:right]


def parse_s3_last_modified(value: str) -> int | None:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return to_epoch(dt.astimezone(timezone.utc))


def infer_prefix_cadence_seconds(prefixes: list[TimestampPrefix]) -> int | None:
    if len(prefixes) < 2:
        return None

    deltas = [
        right.epoch - left.epoch
        for left, right in zip(prefixes, prefixes[1:])
        if right.epoch > left.epoch
    ]
    if not deltas:
        return None

    deltas.sort()
    return deltas[len(deltas) // 2]


def _dedupe_sorted_prefixes(prefixes: list[TimestampPrefix]) -> list[TimestampPrefix]:
    by_epoch = {item.epoch: item for item in prefixes}
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def build_overlap_candidates(
    prefixes: list[TimestampPrefix],
    coarse_matches: list[TimestampPrefix],
    start_epoch: int,
    end_epoch: int,
    cadence_seconds: int | None,
) -> list[TimestampPrefix]:
    if cadence_seconds is None or cadence_seconds <= 0:
        return _dedupe_sorted_prefixes(coarse_matches)

    window_start = start_epoch - cadence_seconds
    window_end = end_epoch + cadence_seconds
    expanded_matches = select_in_range(prefixes, window_start, window_end)
    return _dedupe_sorted_prefixes([*coarse_matches, *expanded_matches])


def _aws_list_objects_page(
    bucket: str,
    key_prefix: str,
    region: str,
    continuation_token: str | None = None,
    delimiter: str | None = None,
    start_after: str | None = None,
) -> dict:
    cmd = [
        "aws",
        "--region",
        region,
        "--no-sign-request",
        "s3api",
        "list-objects-v2",
        "--bucket",
        bucket,
        "--prefix",
        key_prefix,
    ]
    if delimiter:
        cmd.extend(["--delimiter", delimiter])
    if start_after and continuation_token is None:
        cmd.extend(["--start-after", start_after])
    if continuation_token:
        cmd.extend(["--continuation-token", continuation_token])

    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() if exc.stderr else "command failed"
        raise S3ObjectListingError(
            f"aws s3api list-objects-v2 failed for {key_prefix}: {details}"
        ) from exc

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise S3ObjectListingError(
            f"Failed to parse aws s3api JSON for {key_prefix}"
        ) from exc

    if not isinstance(payload, dict):
        raise S3ObjectListingError(
            f"Unexpected aws s3api payload for {key_prefix}: not a JSON object"
        )

    return payload


def prefix_overlaps_range(
    bucket: str,
    prefix: str,
    ts_name: str,
    region: str,
    start_epoch: int,
    end_epoch: int,
) -> bool:
    key_prefix = f"{normalize_prefix(prefix)}/{ts_name}/"
    continuation_token: str | None = None

    while True:
        payload = _aws_list_objects_page(
            bucket=bucket,
            key_prefix=key_prefix,
            region=region,
            continuation_token=continuation_token,
        )

        contents = payload.get("Contents")
        if isinstance(contents, list):
            for entry in contents:
                if not isinstance(entry, dict):
                    continue
                last_modified = entry.get("LastModified")
                if not isinstance(last_modified, str):
                    continue
                last_modified_epoch = parse_s3_last_modified(last_modified)
                if last_modified_epoch is None:
                    continue
                if start_epoch <= last_modified_epoch < end_epoch:
                    return True

        if not payload.get("IsTruncated"):
            break
        next_token = payload.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token:
            break
        continuation_token = next_token

    return False


def resolve_matching_prefixes(
    prefixes: list[TimestampPrefix],
    bucket: str,
    prefix: str,
    region: str,
    start_epoch: int,
    end_epoch: int,
) -> PrefixSelection:
    coarse_matches = select_in_range(prefixes, start_epoch, end_epoch)
    cadence_seconds = infer_prefix_cadence_seconds(prefixes)
    overlap_candidates = build_overlap_candidates(
        prefixes=prefixes,
        coarse_matches=coarse_matches,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        cadence_seconds=cadence_seconds,
    )

    if not overlap_candidates:
        return PrefixSelection(
            coarse_matches=coarse_matches,
            overlap_candidates=[],
            final_matches=coarse_matches,
            overlap_inspected=0,
            overlap_filter_used=False,
        )

    overlap_matches: list[TimestampPrefix] = []
    overlap_inspected = 0
    try:
        for item in overlap_candidates:
            overlap_inspected += 1
            if prefix_overlaps_range(
                bucket=bucket,
                prefix=prefix,
                ts_name=item.name,
                region=region,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
            ):
                overlap_matches.append(item)
    except S3ObjectListingError as exc:
        return PrefixSelection(
            coarse_matches=coarse_matches,
            overlap_candidates=overlap_candidates,
            final_matches=coarse_matches,
            overlap_inspected=overlap_inspected,
            overlap_filter_used=False,
            overlap_error=str(exc),
        )

    return PrefixSelection(
        coarse_matches=coarse_matches,
        overlap_candidates=overlap_candidates,
        final_matches=overlap_matches,
        overlap_inspected=overlap_inspected,
        overlap_filter_used=True,
    )


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


def _parse_s5cmd_cp_json_line(line: str) -> tuple[int, int] | None:
    """
    Parse a single JSON log line emitted by `s5cmd --json --log info`.

    Returns:
    - (files_delta, bytes_delta) when this line represents a successful `cp` object copy
    - None for non-JSON lines, non-cp lines, or unsuccessful copy events
    """
    stripped = line.strip()
    if not stripped:
        return None

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("operation") != "cp":
        return None
    if payload.get("success") is not True:
        return None

    size_bytes = 0
    object_meta = payload.get("object")
    if isinstance(object_meta, dict):
        size = object_meta.get("size")
        if isinstance(size, int) and size > 0:
            size_bytes = size
    return (1, size_bytes)


def _extract_ts_name_from_copy_command(command_line: str) -> str | None:
    tokens = shlex.split(command_line)
    if len(tokens) < 2 or tokens[0] != "cp":
        return None
    src = tokens[1]
    if not src.startswith("s3://"):
        return None
    src_no_glob = src[:-2] if src.endswith("/*") else src
    ts_name = src_no_glob.rstrip("/").split("/")[-1]
    return ts_name if ts_name.isdigit() else None


def _safe_update_bar(bar: tqdm, delta: int) -> None:
    if delta <= 0:
        return
    total = bar.total
    if total is None:
        bar.update(delta)
        return
    remaining = int(total - bar.n)
    if remaining <= 0:
        return
    bar.update(min(delta, remaining))


def estimate_prefix_object_totals(
    bucket: str,
    prefix: str,
    ts_name: str,
    region: str,
) -> tuple[int, int]:
    key_prefix = f"{normalize_prefix(prefix)}/{ts_name}/"
    continuation_token: str | None = None
    object_count = 0
    total_bytes = 0

    while True:
        payload = _aws_list_objects_page(
            bucket=bucket,
            key_prefix=key_prefix,
            region=region,
            continuation_token=continuation_token,
        )
        contents = payload.get("Contents")
        if isinstance(contents, list):
            for entry in contents:
                if not isinstance(entry, dict):
                    continue
                object_count += 1
                size = entry.get("Size")
                if isinstance(size, int) and size > 0:
                    total_bytes += size

        if not payload.get("IsTruncated"):
            break
        next_token = payload.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token:
            break
        continuation_token = next_token

    return object_count, total_bytes


def estimate_transfer_totals_with_breakdown(
    selected: list[TimestampPrefix],
    bucket: str,
    prefix: str,
    region: str,
) -> tuple[int, int, dict[str, tuple[int, int]]] | None:
    if not selected:
        return (0, 0, {})

    total_files = 0
    total_bytes = 0
    by_prefix: dict[str, tuple[int, int]] = {}
    with tqdm(total=len(selected), desc="Pre-counting objects", unit="prefix") as bar:
        for item in selected:
            try:
                n_files, n_bytes = estimate_prefix_object_totals(
                    bucket=bucket,
                    prefix=prefix,
                    ts_name=item.name,
                    region=region,
                )
            except S3ObjectListingError as exc:
                print(
                    f"WARNING: Pre-count failed for prefix {item.name}: {exc}",
                    file=sys.stderr,
                )
                return None
            total_files += n_files
            total_bytes += n_bytes
            by_prefix[item.name] = (n_files, n_bytes)
            bar.update(1)

    return total_files, total_bytes, by_prefix


def estimate_transfer_totals(
    selected: list[TimestampPrefix],
    bucket: str,
    prefix: str,
    region: str,
) -> tuple[int, int] | None:
    result = estimate_transfer_totals_with_breakdown(
        selected=selected,
        bucket=bucket,
        prefix=prefix,
        region=region,
    )
    if result is None:
        return None
    total_files, total_bytes, _ = result
    return total_files, total_bytes


def run_s5cmd_copy_commands(
    commands: list[str],
    numworkers: int,
    log_path: Path | None = None,
    expected_files_total: int | None = None,
    expected_bytes_total: int | None = None,
    expected_by_prefix: dict[str, tuple[int, int]] | None = None,
) -> int:
    if not commands:
        return 0

    base_cmd = [
        "s5cmd",
        "--no-sign-request",
        "--numworkers",
        str(numworkers),
        "--json",
        "--log",
        "info",
    ]

    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")

    progress_by_bytes = bool(expected_bytes_total and expected_bytes_total > 0)
    progress_by_files = bool(
        not progress_by_bytes and expected_files_total and expected_files_total > 0
    )
    if progress_by_bytes:
        total_value = int(expected_bytes_total)
        unit = "B"
        unit_scale = True
    elif progress_by_files:
        total_value = int(expected_files_total)
        unit = "file"
        unit_scale = False
    else:
        total_value = len(commands)
        unit = "prefix"
        unit_scale = False

    try:
        with tqdm(
            total=total_value,
            desc="Downloading",
            unit=unit,
            unit_scale=unit_scale,
        ) as bar:
            copied_files_total = 0
            copied_bytes_total = 0
            completed_prefixes = 0

            def _set_download_postfix(current_prefix: str | None) -> None:
                current = current_prefix if current_prefix else "-"
                bar.set_postfix_str(
                    (
                        f"current={current} prefixes={completed_prefixes}/{len(commands)} "
                        f"files={copied_files_total} bytes={copied_bytes_total}"
                    )
                )

            for index, command_line in enumerate(commands, start=1):
                command_tokens = shlex.split(command_line)
                cmd = [*base_cmd, *command_tokens]
                ts_name = _extract_ts_name_from_copy_command(command_line)
                current_prefix_label = ts_name or f"command-{index}"
                expected_cmd_files = (
                    expected_by_prefix.get(ts_name, (0, 0))[0]
                    if expected_by_prefix and ts_name
                    else 0
                )
                expected_cmd_bytes = (
                    expected_by_prefix.get(ts_name, (0, 0))[1]
                    if expected_by_prefix and ts_name
                    else 0
                )

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                cmd_bytes = 0
                cmd_files = 0
                last_output_line: str | None = None

                if log_handle is not None:
                    cmd_line = f"$ {' '.join(shlex.quote(part) for part in cmd)}\n"
                    log_handle.write(cmd_line)
                    log_handle.flush()

                _set_download_postfix(current_prefix_label)

                if proc.stdout is not None:
                    for raw_line in proc.stdout:
                        if log_handle is not None:
                            log_handle.write(raw_line)
                            log_handle.flush()
                        stripped_line = raw_line.strip()
                        if stripped_line:
                            last_output_line = stripped_line
                        parsed = _parse_s5cmd_cp_json_line(raw_line)
                        if parsed is None:
                            continue
                        delta_files, delta_bytes = parsed
                        cmd_files += delta_files
                        cmd_bytes += delta_bytes
                        copied_files_total += delta_files
                        copied_bytes_total += delta_bytes
                        if progress_by_bytes and delta_bytes > 0:
                            _safe_update_bar(bar, delta_bytes)
                        if progress_by_files and delta_files > 0:
                            _safe_update_bar(bar, delta_files)
                        _set_download_postfix(current_prefix_label)

                return_code = proc.wait()
                if log_handle is not None:
                    log_handle.write("\n")
                    log_handle.flush()

                if return_code != 0:
                    print(
                        (
                            "s5cmd failed for prefix command "
                            f"{index}/{len(commands)}: {command_line}"
                        ),
                        file=sys.stderr,
                    )
                    if last_output_line:
                        print(last_output_line, file=sys.stderr)
                    return return_code

                if expected_cmd_bytes > cmd_bytes:
                    missing_bytes = expected_cmd_bytes - cmd_bytes
                    copied_bytes_total += missing_bytes
                    if progress_by_bytes:
                        _safe_update_bar(bar, missing_bytes)
                if expected_cmd_files > cmd_files:
                    missing_files = expected_cmd_files - cmd_files
                    copied_files_total += missing_files
                    if progress_by_files:
                        _safe_update_bar(bar, missing_files)

                completed_prefixes += 1
                if not progress_by_bytes and not progress_by_files:
                    _safe_update_bar(bar, 1)
                _set_download_postfix(None)
    finally:
        if log_handle is not None:
            log_handle.close()

    return 0


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
        "--dry-run",
        action="store_true",
        help="Generate manifest files only; do not execute s5cmd downloads",
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
        help="Optional log file for s5cmd output when execution is enabled",
    )
    parser.add_argument(
        "--pre-count",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pre-count selected prefixes to estimate files/bytes for download progress "
            "(default: enabled)"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    require_binary("aws")
    if not args.dry_run:
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
    all_prefixes = aws_ls_prefixes_optimized(
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
    )

    if not all_prefixes:
        print("No numeric epoch-style prefixes found.", file=sys.stderr)
        return 1

    selection = resolve_matching_prefixes(
        prefixes=all_prefixes,
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
    )
    selected = selection.final_matches

    print(f"Requested range: [{iso_utc(start_epoch)} .. {iso_utc(end_epoch)})")
    print(f"Discovered numeric prefixes: {len(all_prefixes)}")
    print(f"Coarse epoch matches: {len(selection.coarse_matches)}")
    print(f"Overlap candidates considered: {len(selection.overlap_candidates)}")
    print(f"Overlap prefixes inspected: {selection.overlap_inspected}")
    if selection.overlap_filter_used:
        print("Overlap refinement: applied")
    else:
        print("Overlap refinement: unavailable (using coarse epoch matches)")
    if selection.overlap_error:
        print(
            f"WARNING: Overlap refinement failed: {selection.overlap_error}",
            file=sys.stderr,
        )
    print(f"Final matched prefixes: {len(selected)}")

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
    planned_prefixes = [
        ts_name
        for command in commands
        if (ts_name := _extract_ts_name_from_copy_command(command)) is not None
    ]
    print(f"Planned prefixes to download ({len(planned_prefixes)}):")
    for ts_name in planned_prefixes:
        local_dir = local_dir_for_prefix(local_root, args.prefix, ts_name)
        print(f"  - {ts_name} -> {local_dir}")

    if not selected:
        print("Nothing matched the requested time range.")
        return 0

    if args.dry_run:
        if commands:
            print("Dry-run copy commands:")
            for command in commands:
                print(command)
        else:
            print("Dry-run copy commands: (none)")
        print("Dry run requested. Skipping s5cmd execution.")
        return 0

    resolved_log_path = (
        Path(args.log_file).expanduser().resolve() if args.log_file else None
    )
    expected_files_total: int | None = None
    expected_bytes_total: int | None = None
    expected_by_prefix: dict[str, tuple[int, int]] | None = None
    if args.pre_count and commands:
        totals = estimate_transfer_totals_with_breakdown(
            selected=selected,
            bucket=args.bucket,
            prefix=args.prefix,
            region=args.region,
        )
        if totals is not None:
            expected_files_total, expected_bytes_total, expected_by_prefix = totals
            print(
                "Estimated download totals: "
                f"files={expected_files_total} bytes={expected_bytes_total}"
            )
        else:
            print(
                "WARNING: Pre-count failed; falling back to non-estimated progress.",
                file=sys.stderr,
            )
    exit_code = run_s5cmd_copy_commands(
        commands=commands,
        numworkers=args.numworkers,
        log_path=resolved_log_path,
        expected_files_total=expected_files_total,
        expected_bytes_total=expected_bytes_total,
        expected_by_prefix=expected_by_prefix,
    )
    if exit_code != 0:
        return exit_code

    print("s5cmd completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
