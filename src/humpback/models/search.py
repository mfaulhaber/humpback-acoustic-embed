from typing import Optional

from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class SearchJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "search_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    detection_job_id: Mapped[str]
    start_utc: Mapped[float]
    end_utc: Mapped[float]
    top_k: Mapped[int] = mapped_column(default=20)
    metric: Mapped[str] = mapped_column(default="cosine")
    embedding_set_ids: Mapped[Optional[str]] = mapped_column(Text, default=None)
    model_version: Mapped[Optional[str]] = mapped_column(default=None)
    embedding_vector: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
