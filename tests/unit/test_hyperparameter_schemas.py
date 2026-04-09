"""Unit tests for hyperparameter Pydantic schemas."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from humpback.schemas.hyperparameter import (
    ManifestCreate,
    ManifestDetail,
    ManifestSummary,
    SearchCreate,
    SearchDetail,
    SearchSpaceDefaults,
    SearchSummary,
)


# ---------------------------------------------------------------------------
# ManifestCreate validation
# ---------------------------------------------------------------------------


class TestManifestCreate:
    def test_valid_with_training_jobs(self) -> None:
        m = ManifestCreate(name="test", training_job_ids=["t1"])
        assert m.training_job_ids == ["t1"]
        assert m.detection_job_ids == []
        assert m.split_ratio == [70, 15, 15]
        assert m.seed == 42

    def test_valid_with_detection_jobs(self) -> None:
        m = ManifestCreate(name="test", detection_job_ids=["d1", "d2"])
        assert m.detection_job_ids == ["d1", "d2"]

    def test_valid_with_both(self) -> None:
        m = ManifestCreate(
            name="test", training_job_ids=["t1"], detection_job_ids=["d1"]
        )
        assert m.training_job_ids == ["t1"]
        assert m.detection_job_ids == ["d1"]

    def test_rejects_no_sources(self) -> None:
        with pytest.raises(ValidationError, match="At least one"):
            ManifestCreate(name="test")

    def test_rejects_empty_sources(self) -> None:
        with pytest.raises(ValidationError, match="At least one"):
            ManifestCreate(name="test", training_job_ids=[], detection_job_ids=[])

    def test_rejects_wrong_split_ratio_length(self) -> None:
        with pytest.raises(ValidationError, match="exactly 3"):
            ManifestCreate(name="test", training_job_ids=["t1"], split_ratio=[80, 20])

    def test_rejects_negative_split_ratio(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            ManifestCreate(
                name="test", training_job_ids=["t1"], split_ratio=[70, -5, 15]
            )

    def test_rejects_zero_sum_split_ratio(self) -> None:
        with pytest.raises(ValidationError, match="positive number"):
            ManifestCreate(name="test", training_job_ids=["t1"], split_ratio=[0, 0, 0])

    def test_custom_split_ratio(self) -> None:
        m = ManifestCreate(
            name="test", training_job_ids=["t1"], split_ratio=[80, 10, 10]
        )
        assert m.split_ratio == [80, 10, 10]

    def test_custom_seed(self) -> None:
        m = ManifestCreate(name="test", training_job_ids=["t1"], seed=99)
        assert m.seed == 99


# ---------------------------------------------------------------------------
# SearchCreate validation
# ---------------------------------------------------------------------------


class TestSearchCreate:
    def test_valid_minimal(self) -> None:
        s = SearchCreate(name="search1", manifest_id="m1")
        assert s.n_trials == 100
        assert s.seed == 42
        assert s.search_space is None
        assert s.comparison_model_id is None

    def test_valid_with_search_space(self) -> None:
        s = SearchCreate(
            name="s", manifest_id="m1", search_space={"a": [1, 2], "b": ["x"]}
        )
        assert s.search_space == {"a": [1, 2], "b": ["x"]}

    def test_valid_with_comparison(self) -> None:
        s = SearchCreate(
            name="s",
            manifest_id="m1",
            comparison_model_id="model1",
            comparison_threshold=0.85,
        )
        assert s.comparison_model_id == "model1"
        assert s.comparison_threshold == 0.85

    def test_rejects_zero_trials(self) -> None:
        with pytest.raises(ValidationError, match="n_trials must be >= 1"):
            SearchCreate(name="s", manifest_id="m1", n_trials=0)

    def test_rejects_negative_trials(self) -> None:
        with pytest.raises(ValidationError, match="n_trials must be >= 1"):
            SearchCreate(name="s", manifest_id="m1", n_trials=-5)

    def test_rejects_empty_search_space_values(self) -> None:
        with pytest.raises(ValidationError, match="non-empty list"):
            SearchCreate(name="s", manifest_id="m1", search_space={"a": []})

    def test_accepts_one_trial(self) -> None:
        s = SearchCreate(name="s", manifest_id="m1", n_trials=1)
        assert s.n_trials == 1


# ---------------------------------------------------------------------------
# Response schemas (construction from dicts)
# ---------------------------------------------------------------------------


class TestManifestSummary:
    def test_construction(self) -> None:
        now = datetime.now(tz=timezone.utc)
        m = ManifestSummary(
            id="abc",
            name="test",
            status="complete",
            training_job_ids=["t1"],
            detection_job_ids=[],
            split_ratio=[70, 15, 15],
            seed=42,
            example_count=100,
            created_at=now,
        )
        assert m.id == "abc"
        assert m.example_count == 100
        assert m.completed_at is None


class TestManifestDetail:
    def test_includes_summary_fields(self) -> None:
        now = datetime.now(tz=timezone.utc)
        m = ManifestDetail(
            id="abc",
            name="test",
            status="complete",
            training_job_ids=[],
            detection_job_ids=["d1"],
            split_ratio=[70, 15, 15],
            seed=42,
            created_at=now,
            split_summary={"train": {"positive": 50, "negative": 30}},
            detection_job_summaries={"d1": {"included_positive": 20}},
        )
        assert m.split_summary is not None
        assert m.detection_job_summaries is not None
        assert m.manifest_path is None


class TestSearchSummary:
    def test_construction(self) -> None:
        now = datetime.now(tz=timezone.utc)
        s = SearchSummary(
            id="s1",
            name="search",
            status="running",
            manifest_id="m1",
            n_trials=200,
            seed=42,
            objective_name="default",
            trials_completed=45,
            best_objective=0.85,
            created_at=now,
        )
        assert s.trials_completed == 45
        assert s.manifest_name is None


class TestSearchDetail:
    def test_includes_summary_and_detail_fields(self) -> None:
        now = datetime.now(tz=timezone.utc)
        s = SearchDetail(
            id="s1",
            name="search",
            status="complete",
            manifest_id="m1",
            n_trials=100,
            seed=42,
            objective_name="default",
            trials_completed=100,
            best_objective=0.92,
            created_at=now,
            search_space={"classifier": ["logreg", "mlp"]},
            best_config={"classifier": "logreg", "threshold": 0.85},
            best_metrics={"recall": 0.95, "fp_rate": 0.02},
            comparison_result={"metric_deltas": {"recall": 0.05}},
        )
        assert s.search_space == {"classifier": ["logreg", "mlp"]}
        assert s.best_config is not None
        assert s.comparison_result is not None


class TestSearchSpaceDefaults:
    def test_construction(self) -> None:
        d = SearchSpaceDefaults(search_space={"a": [1, 2], "b": ["x", "y"]})
        assert "a" in d.search_space
