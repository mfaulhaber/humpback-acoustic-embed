from __future__ import annotations

from typing import Optional

from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


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
