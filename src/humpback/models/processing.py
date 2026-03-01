import enum
from typing import Optional

from sqlalchemy import ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from humpback.database import Base, TimestampMixin, UUIDMixin


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    complete = "complete"
    failed = "failed"
    canceled = "canceled"


class ProcessingJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "processing_jobs"

    audio_file_id: Mapped[str] = mapped_column(ForeignKey("audio_files.id"))
    status: Mapped[str] = mapped_column(default=JobStatus.queued.value)
    encoding_signature: Mapped[str]
    model_version: Mapped[str]
    window_size_seconds: Mapped[float]
    target_sample_rate: Mapped[int]
    feature_config: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    warning_message: Mapped[Optional[str]] = mapped_column(Text, default=None)

    audio_file: Mapped["AudioFile"] = relationship(back_populates="processing_jobs")  # noqa: F821


class EmbeddingSet(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "embedding_sets"
    __table_args__ = (UniqueConstraint("audio_file_id", "encoding_signature"),)

    audio_file_id: Mapped[str] = mapped_column(ForeignKey("audio_files.id"))
    encoding_signature: Mapped[str]
    model_version: Mapped[str]
    window_size_seconds: Mapped[float]
    target_sample_rate: Mapped[int]
    vector_dim: Mapped[int]
    parquet_path: Mapped[str]

    audio_file: Mapped["AudioFile"] = relationship(back_populates="embedding_sets")  # noqa: F821
