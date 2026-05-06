"""Remove retired Sequence Models HMM and MT artifact roots.

By default this is a dry run and only writes a manifest under
``<storage_root>/cleanup-manifests``. Pass ``--apply`` to delete the retired
artifact directories after the safety checks pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from humpback.config import Settings
from humpback.storage import cleanup_manifests_dir, ensure_dir, path_within_root

TARGET_ROOTS = (
    "hmm_sequences",
    "masked_transformer_jobs",
    "motif_extractions",
)


class CleanupSafetyError(RuntimeError):
    """Raised when a cleanup candidate is not safe to inspect or delete."""


@dataclass(frozen=True)
class CleanupRunResult:
    manifest_path: Path
    manifest: dict[str, Any]
    exit_code: int = 0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_slug(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_safe_target(candidate: Path, storage_root: Path) -> None:
    if candidate.is_symlink():
        raise CleanupSafetyError(f"Refusing symlink target: {candidate}")
    if not candidate.exists():
        return
    if not candidate.is_dir():
        raise CleanupSafetyError(f"Expected directory target: {candidate}")
    if not path_within_root(candidate, storage_root):
        raise CleanupSafetyError(
            f"Target path escapes storage root {storage_root}: {candidate}"
        )

    for child in candidate.rglob("*"):
        if child.is_symlink():
            raise CleanupSafetyError(f"Refusing symlink within target: {child}")
        if not path_within_root(child, storage_root):
            raise CleanupSafetyError(
                f"Descendant path escapes storage root {storage_root}: {child}"
            )
        if not child.is_file() and not child.is_dir():
            raise CleanupSafetyError(f"Refusing special filesystem entry: {child}")


def _measure_target(candidate: Path) -> dict[str, Any]:
    exists = candidate.exists()
    info: dict[str, Any] = {
        "path": str(candidate),
        "exists": exists,
        "file_count": 0,
        "directory_count": 0,
        "total_bytes": 0,
        "deleted": False,
        "post_exists": exists,
    }
    if not exists:
        return info

    directory_count = 1
    file_count = 0
    total_bytes = 0
    for child in candidate.rglob("*"):
        if child.is_dir():
            directory_count += 1
        elif child.is_file():
            file_count += 1
            total_bytes += child.stat().st_size

    info.update(
        {
            "file_count": file_count,
            "directory_count": directory_count,
            "total_bytes": total_bytes,
        }
    )
    return info


def _write_manifest(
    storage_root: Path, now: datetime, manifest: dict[str, Any]
) -> Path:
    manifest_dir = ensure_dir(cleanup_manifests_dir(storage_root))
    manifest_path = manifest_dir / f"{_timestamp_slug(now)}-sequence-models-hmm-mt.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path


def cleanup_sequence_model_artifacts(
    *,
    storage_root: Path,
    apply: bool = False,
    now: datetime | None = None,
) -> CleanupRunResult:
    now = now or _utc_now()
    storage_root = storage_root.resolve()

    targets: list[dict[str, Any]] = []
    for target_name in TARGET_ROOTS:
        candidate = storage_root / target_name
        _ensure_safe_target(candidate, storage_root)
        info = _measure_target(candidate)
        info["name"] = target_name
        targets.append(info)

    if apply:
        for info in targets:
            candidate = Path(info["path"])
            if not info["exists"]:
                continue
            shutil.rmtree(candidate)
            info["deleted"] = not candidate.exists()
            info["post_exists"] = candidate.exists()

    summary = {
        "target_count": len(targets),
        "existing_target_count": sum(1 for info in targets if info["exists"]),
        "deleted_target_count": sum(1 for info in targets if info["deleted"]),
        "file_count": sum(info["file_count"] for info in targets),
        "directory_count": sum(info["directory_count"] for info in targets),
        "total_bytes": sum(info["total_bytes"] for info in targets),
    }
    manifest = {
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "mode": "apply" if apply else "dry-run",
        "storage_root": str(storage_root),
        "target_roots": list(TARGET_ROOTS),
        "targets": targets,
        "summary": summary,
    }
    manifest_path = _write_manifest(storage_root, now, manifest)
    return CleanupRunResult(manifest_path=manifest_path, manifest=manifest)


def _print_summary(result: CleanupRunResult) -> None:
    manifest = result.manifest
    summary = manifest["summary"]
    print(f"Manifest: {result.manifest_path}")
    print(f"Mode: {manifest['mode']}")
    print(f"Storage root: {manifest['storage_root']}")
    print(
        "Targets: "
        f"{summary['existing_target_count']} existing, "
        f"{summary['file_count']} file(s), "
        f"{summary['directory_count']} directory/directories, "
        f"{summary['total_bytes']} byte(s)"
    )
    for target in manifest["targets"]:
        status = "deleted" if target["deleted"] else "kept"
        if not target["exists"]:
            status = "missing"
        print(
            f"  - {target['name']}: {status}, "
            f"{target['file_count']} file(s), "
            f"{target['directory_count']} directory/directories, "
            f"{target['total_bytes']} byte(s)"
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=None,
        help="Storage root. Defaults to Settings.from_repo_env().storage_root.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete retired Sequence Models HMM and MT artifact roots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly run without deleting files. This is the default.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.apply and args.dry_run:
        raise ValueError("--apply and --dry-run are mutually exclusive")
    settings = Settings.from_repo_env()
    storage_root = args.storage_root or settings.storage_root
    result = cleanup_sequence_model_artifacts(
        storage_root=storage_root,
        apply=args.apply,
    )
    _print_summary(result)
    if args.apply:
        print("Apply completed.")
    else:
        print("Dry run completed. Pass --apply to delete these artifact roots.")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
