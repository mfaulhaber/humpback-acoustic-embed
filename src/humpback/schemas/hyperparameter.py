from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---- Manifest ----


class ManifestCreate(BaseModel):
    name: str
    training_job_ids: list[str] = Field(default_factory=list)
    detection_job_ids: list[str] = Field(default_factory=list)
    split_ratio: list[int] = Field(default=[70, 15, 15])
    seed: int = 42

    @model_validator(mode="after")
    def _validate(self):
        if not self.training_job_ids and not self.detection_job_ids:
            raise ValueError(
                "At least one of training_job_ids or detection_job_ids is required"
            )
        if len(self.split_ratio) != 3:
            raise ValueError("split_ratio must be exactly 3 integers")
        if any(v < 0 for v in self.split_ratio):
            raise ValueError("split_ratio values must be non-negative")
        if sum(self.split_ratio) == 0:
            raise ValueError("split_ratio must sum to a positive number")
        return self


class ManifestSummary(BaseModel):
    id: str
    name: str
    status: str
    training_job_ids: list[str]
    detection_job_ids: list[str]
    split_ratio: list[int]
    seed: int
    example_count: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class ManifestDetail(ManifestSummary):
    manifest_path: Optional[str] = None
    split_summary: Optional[dict[str, Any]] = None
    detection_job_summaries: Optional[dict[str, Any]] = None


# ---- Search ----


class SearchCreate(BaseModel):
    name: str
    manifest_id: str
    search_space: Optional[dict[str, list[Any]]] = None
    n_trials: int = 100
    seed: int = 42
    comparison_model_id: Optional[str] = None
    comparison_threshold: Optional[float] = None

    @model_validator(mode="after")
    def _validate(self):
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if self.search_space is not None:
            for key, values in self.search_space.items():
                if not isinstance(values, list) or len(values) == 0:
                    raise ValueError(f"search_space['{key}'] must be a non-empty list")
        return self


class SearchSummary(BaseModel):
    id: str
    name: str
    status: str
    manifest_id: str
    manifest_name: Optional[str] = None
    n_trials: int
    seed: int
    objective_name: str
    trials_completed: int = 0
    best_objective: Optional[float] = None
    comparison_model_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class SearchDetail(SearchSummary):
    search_space: dict[str, list[Any]]
    results_dir: Optional[str] = None
    best_config: Optional[dict[str, Any]] = None
    best_metrics: Optional[dict[str, Any]] = None
    comparison_threshold: Optional[float] = None
    comparison_result: Optional[dict[str, Any]] = None


# ---- Search Space Defaults ----


class SearchSpaceDefaults(BaseModel):
    search_space: dict[str, list[Any]]
