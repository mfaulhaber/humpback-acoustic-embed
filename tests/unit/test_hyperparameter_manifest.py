"""Unit tests for hyperparameter manifest generation service."""

from __future__ import annotations

from pathlib import Path
from typing import Any


from humpback.services.hyperparameter_service.manifest import (
    _assign_splits,
    _classify_detection_row,
    _new_detection_job_summary,
    _record_included_example,
)
from humpback.storage import (
    hyperparameter_manifest_dir,
    hyperparameter_manifest_path,
    hyperparameter_search_results_dir,
)


# ---------------------------------------------------------------------------
# Storage path helpers
# ---------------------------------------------------------------------------


class TestStoragePaths:
    def test_manifest_dir(self, tmp_path: Path) -> None:
        result = hyperparameter_manifest_dir(tmp_path, "abc123")
        assert result == tmp_path / "hyperparameter" / "manifests" / "abc123"

    def test_manifest_path(self, tmp_path: Path) -> None:
        result = hyperparameter_manifest_path(tmp_path, "abc123")
        assert (
            result
            == tmp_path / "hyperparameter" / "manifests" / "abc123" / "manifest.json"
        )

    def test_search_results_dir(self, tmp_path: Path) -> None:
        result = hyperparameter_search_results_dir(tmp_path, "def456")
        assert result == tmp_path / "hyperparameter" / "searches" / "def456"


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


class TestAssignSplits:
    def test_basic_split(self) -> None:
        # 10 unique files, 70/15/15 → 7/2/1 (rounded)
        audio_ids = [f"file_{i}" for i in range(10)]
        assignments = _assign_splits(audio_ids, (70, 15, 15), seed=42)
        assert len(assignments) == 10
        splits = set(assignments.values())
        assert splits == {"train", "val", "test"}

    def test_deterministic(self) -> None:
        audio_ids = [f"file_{i}" for i in range(20)]
        a1 = _assign_splits(audio_ids, (70, 15, 15), seed=42)
        a2 = _assign_splits(audio_ids, (70, 15, 15), seed=42)
        assert a1 == a2

    def test_different_seed_different_assignment(self) -> None:
        audio_ids = [f"file_{i}" for i in range(20)]
        a1 = _assign_splits(audio_ids, (70, 15, 15), seed=42)
        a2 = _assign_splits(audio_ids, (70, 15, 15), seed=99)
        # With enough files, different seeds should produce different assignments
        assert a1 != a2

    def test_deduplicates_audio_ids(self) -> None:
        # Same audio_file_id repeated → only counted once
        audio_ids = ["file_a"] * 5 + ["file_b"] * 3
        assignments = _assign_splits(audio_ids, (70, 15, 15), seed=42)
        assert len(assignments) == 2

    def test_single_file_goes_to_train(self) -> None:
        assignments = _assign_splits(["only_file"], (70, 15, 15), seed=42)
        assert assignments["only_file"] == "train"


# ---------------------------------------------------------------------------
# Row classification
# ---------------------------------------------------------------------------


class TestClassifyDetectionRow:
    def _make_row(
        self,
        humpback: bool = False,
        orca: bool = False,
        ship: bool = False,
        background: bool = False,
    ) -> dict[str, Any]:
        return {
            "row_id": "r1",
            "start_utc": 1000.0,
            "humpback": humpback,
            "orca": orca,
            "ship": ship,
            "background": background,
        }

    def test_vocalization_positive(self) -> None:
        row = self._make_row()
        result, skip = _classify_detection_row(row, {"Song"})
        assert skip is None
        assert result is not None
        assert result["label"] == 1
        assert result["label_source"] == "vocalization_positive"

    def test_vocalization_negative(self) -> None:
        row = self._make_row()
        result, skip = _classify_detection_row(row, {"(Negative)"})
        assert skip is None
        assert result is not None
        assert result["label"] == 0
        assert result["label_source"] == "vocalization_negative"

    def test_binary_positive_humpback(self) -> None:
        row = self._make_row(humpback=True)
        result, skip = _classify_detection_row(row, set())
        assert skip is None
        assert result is not None
        assert result["label"] == 1
        assert result["label_source"] == "binary_positive"

    def test_binary_positive_orca(self) -> None:
        row = self._make_row(orca=True)
        result, skip = _classify_detection_row(row, set())
        assert skip is None
        assert result is not None
        assert result["label"] == 1

    def test_binary_negative_ship(self) -> None:
        row = self._make_row(ship=True)
        result, skip = _classify_detection_row(row, set())
        assert skip is None
        assert result is not None
        assert result["label"] == 0
        assert result["negative_group"] == "ship"

    def test_binary_negative_background(self) -> None:
        row = self._make_row(background=True)
        result, skip = _classify_detection_row(row, set())
        assert skip is None
        assert result is not None
        assert result["label"] == 0
        assert result["negative_group"] == "background"

    def test_unlabeled_skipped(self) -> None:
        row = self._make_row()
        result, skip = _classify_detection_row(row, set())
        assert result is None
        assert skip == "unlabeled"

    def test_conflict_vocalization_positive_and_negative(self) -> None:
        row = self._make_row()
        result, skip = _classify_detection_row(row, {"Song", "(Negative)"})
        assert result is None
        assert skip == "conflict"

    def test_conflict_binary_positive_and_negative(self) -> None:
        row = self._make_row(humpback=True, ship=True)
        result, skip = _classify_detection_row(row, set())
        assert result is None
        assert skip == "conflict"

    def test_vocalization_positive_takes_precedence_over_binary(self) -> None:
        # Vocalization positive + binary positive → vocalization wins
        row = self._make_row(humpback=True)
        result, skip = _classify_detection_row(row, {"Song"})
        assert skip is None
        assert result is not None
        assert result["label_source"] == "vocalization_positive"


# ---------------------------------------------------------------------------
# Detection job summary tracking
# ---------------------------------------------------------------------------


class TestDetectionJobSummary:
    def test_new_summary_has_zero_counts(self) -> None:
        summary = _new_detection_job_summary()
        assert summary["included_positive"] == 0
        assert summary["included_negative"] == 0
        assert summary["skipped_conflicts"] == 0
        assert summary["skipped_unlabeled"] == 0

    def test_record_positive(self) -> None:
        summary = _new_detection_job_summary()
        _record_included_example(summary, 1, "vocalization_positive")
        assert summary["included_positive"] == 1
        assert summary["included_positives_by_source"]["vocalization_positive"] == 1

    def test_record_negative(self) -> None:
        summary = _new_detection_job_summary()
        _record_included_example(summary, 0, "ship")
        assert summary["included_negative"] == 1
        assert summary["included_negatives_by_source"]["ship"] == 1

    def test_no_score_band_category(self) -> None:
        summary = _new_detection_job_summary()
        # Verify no score_band key exists (removed from service)
        assert "score_band" not in summary["included_negatives_by_source"]
