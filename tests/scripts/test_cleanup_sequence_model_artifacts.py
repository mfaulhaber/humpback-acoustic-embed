"""Tests for retired Sequence Models artifact cleanup."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "cleanup_sequence_model_artifacts.py"
)
spec = importlib.util.spec_from_file_location(
    "cleanup_sequence_model_artifacts", SCRIPT_PATH
)
assert spec is not None and spec.loader is not None
cleanup = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cleanup
spec.loader.exec_module(cleanup)

NOW = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)


def _write_artifacts(storage_root: Path) -> None:
    (storage_root / "hmm_sequences/job-1").mkdir(parents=True)
    (storage_root / "hmm_sequences/job-1/model.json").write_text("{}", encoding="utf-8")
    (storage_root / "masked_transformer_jobs/job-2").mkdir(parents=True)
    (storage_root / "masked_transformer_jobs/job-2/checkpoint.pt").write_bytes(
        b"checkpoint"
    )
    (storage_root / "motif_extractions/job-3/examples").mkdir(parents=True)
    (storage_root / "motif_extractions/job-3/examples/example.json").write_text(
        "{}", encoding="utf-8"
    )
    (storage_root / "continuous_embeddings/job-keep").mkdir(parents=True)
    (storage_root / "continuous_embeddings/job-keep/manifest.json").write_text(
        "{}", encoding="utf-8"
    )


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_dry_run_writes_manifest_without_deleting_targets(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    _write_artifacts(storage_root)

    result = cleanup.cleanup_sequence_model_artifacts(
        storage_root=storage_root,
        apply=False,
        now=NOW,
    )

    assert result.manifest_path.exists()
    assert (storage_root / "hmm_sequences/job-1/model.json").exists()
    assert (storage_root / "masked_transformer_jobs/job-2/checkpoint.pt").exists()
    assert (storage_root / "motif_extractions/job-3/examples/example.json").exists()
    assert (storage_root / "continuous_embeddings/job-keep/manifest.json").exists()

    manifest = _read_manifest(result.manifest_path)
    assert manifest["mode"] == "dry-run"
    assert manifest["summary"]["existing_target_count"] == 3
    assert manifest["summary"]["deleted_target_count"] == 0
    assert manifest["summary"]["file_count"] == 3


def test_apply_deletes_only_retired_sequence_model_roots(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    _write_artifacts(storage_root)

    result = cleanup.cleanup_sequence_model_artifacts(
        storage_root=storage_root,
        apply=True,
        now=NOW,
    )

    assert not (storage_root / "hmm_sequences").exists()
    assert not (storage_root / "masked_transformer_jobs").exists()
    assert not (storage_root / "motif_extractions").exists()
    assert (storage_root / "continuous_embeddings/job-keep/manifest.json").exists()
    assert result.manifest_path.exists()

    manifest = _read_manifest(result.manifest_path)
    assert manifest["mode"] == "apply"
    assert manifest["summary"]["existing_target_count"] == 3
    assert manifest["summary"]["deleted_target_count"] == 3
    assert {target["name"] for target in manifest["targets"]} == set(
        cleanup.TARGET_ROOTS
    )


def test_missing_targets_are_reported_without_failure(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"

    result = cleanup.cleanup_sequence_model_artifacts(
        storage_root=storage_root,
        apply=False,
        now=NOW,
    )

    manifest = _read_manifest(result.manifest_path)
    assert manifest["summary"]["existing_target_count"] == 0
    assert manifest["summary"]["file_count"] == 0


def test_refuses_target_symlink(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    storage_root.mkdir(parents=True)
    (storage_root / "hmm_sequences").symlink_to(outside, target_is_directory=True)

    with pytest.raises(cleanup.CleanupSafetyError, match="Refusing symlink target"):
        cleanup.cleanup_sequence_model_artifacts(
            storage_root=storage_root,
            apply=False,
            now=NOW,
        )


def test_refuses_nested_symlink(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    (outside / "payload.txt").write_text("payload", encoding="utf-8")
    (storage_root / "motif_extractions/job-3").mkdir(parents=True)
    (storage_root / "motif_extractions/job-3/payload.txt").symlink_to(
        outside / "payload.txt"
    )

    with pytest.raises(cleanup.CleanupSafetyError, match="Refusing symlink within"):
        cleanup.cleanup_sequence_model_artifacts(
            storage_root=storage_root,
            apply=True,
            now=NOW,
        )

    assert (outside / "payload.txt").exists()
