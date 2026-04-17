from typing import Optional

from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class DetectionEmbeddingJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "detection_embedding_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    detection_job_id: Mapped[str]
    model_version: Mapped[str]
    mode: Mapped[Optional[str]] = mapped_column(default=None)
    progress_current: Mapped[Optional[int]] = mapped_column(default=None)
    progress_total: Mapped[Optional[int]] = mapped_column(default=None)
    rows_processed: Mapped[int] = mapped_column(default=0)
    rows_total: Mapped[Optional[int]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
