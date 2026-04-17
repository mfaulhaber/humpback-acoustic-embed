from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class HyperparameterManifest(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "hyperparameter_manifests"

    name: Mapped[str]
    status: Mapped[str] = mapped_column(default="queued")
    training_job_ids: Mapped[str] = mapped_column(Text)  # JSON array
    detection_job_ids: Mapped[str] = mapped_column(Text)  # JSON array
    embedding_model_version: Mapped[str]
    split_ratio: Mapped[str] = mapped_column(Text)  # JSON array e.g. [70, 15, 15]
    seed: Mapped[int] = mapped_column(Integer, default=42)
    manifest_path: Mapped[Optional[str]] = mapped_column(default=None)
    example_count: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    split_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    detection_job_summaries: Mapped[Optional[str]] = mapped_column(
        Text, default=None
    )  # JSON
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)


class HyperparameterSearchJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "hyperparameter_search_jobs"

    name: Mapped[str]
    status: Mapped[str] = mapped_column(default="queued")
    manifest_id: Mapped[str]
    search_space: Mapped[str] = mapped_column(Text)  # JSON
    n_trials: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int] = mapped_column(Integer, default=42)
    objective_name: Mapped[str] = mapped_column(default="default")
    results_dir: Mapped[Optional[str]] = mapped_column(default=None)
    trials_completed: Mapped[int] = mapped_column(Integer, default=0)
    best_objective: Mapped[Optional[float]] = mapped_column(Float, default=None)
    best_config: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    best_metrics: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    comparison_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    comparison_threshold: Mapped[Optional[float]] = mapped_column(Float, default=None)
    comparison_result: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
