from typing import Optional

from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class RetrainWorkflow(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "retrain_workflows"

    status: Mapped[str] = mapped_column(default="queued")
    source_model_id: Mapped[str]
    new_model_name: Mapped[str]
    model_version: Mapped[str]
    window_size_seconds: Mapped[float]
    target_sample_rate: Mapped[int]
    feature_config: Mapped[Optional[str]] = mapped_column(Text, default=None)
    parameters: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    positive_folder_roots: Mapped[str] = mapped_column(Text)  # JSON array
    negative_folder_roots: Mapped[str] = mapped_column(Text)  # JSON array
    import_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    processing_job_ids: Mapped[Optional[str]] = mapped_column(
        Text, default=None
    )  # JSON
    processing_total: Mapped[Optional[int]] = mapped_column(default=None)
    processing_complete: Mapped[Optional[int]] = mapped_column(default=None)
    training_job_id: Mapped[Optional[str]] = mapped_column(default=None)
    new_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
