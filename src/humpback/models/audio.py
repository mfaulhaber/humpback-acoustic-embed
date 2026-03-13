from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from humpback.database import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from humpback.models.processing import EmbeddingSet, ProcessingJob


class AudioFile(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "audio_files"
    __table_args__ = (
        UniqueConstraint("checksum_sha256", "folder_path", name="uq_checksum_folder"),
    )

    filename: Mapped[str]
    folder_path: Mapped[str] = mapped_column(default="")
    source_folder: Mapped[Optional[str]] = mapped_column(default=None)
    checksum_sha256: Mapped[str] = mapped_column()
    duration_seconds: Mapped[Optional[float]] = mapped_column(default=None)
    sample_rate_original: Mapped[Optional[int]] = mapped_column(default=None)

    metadata_: Mapped[Optional["AudioMetadata"]] = relationship(
        back_populates="audio_file", uselist=False, cascade="all, delete-orphan"
    )
    processing_jobs: Mapped[list["ProcessingJob"]] = relationship(  # noqa: F821
        back_populates="audio_file"
    )
    embedding_sets: Mapped[list["EmbeddingSet"]] = relationship(  # noqa: F821
        back_populates="audio_file"
    )


class AudioMetadata(UUIDMixin, Base):
    __tablename__ = "audio_metadata"

    audio_file_id: Mapped[str] = mapped_column(
        ForeignKey("audio_files.id"), unique=True
    )
    tag_data: Mapped[Optional[str]] = mapped_column(Text, default=None)
    visual_observations: Mapped[Optional[str]] = mapped_column(Text, default=None)
    group_composition: Mapped[Optional[str]] = mapped_column(Text, default=None)
    prey_density_proxy: Mapped[Optional[str]] = mapped_column(Text, default=None)

    audio_file: Mapped["AudioFile"] = relationship(back_populates="metadata_")
