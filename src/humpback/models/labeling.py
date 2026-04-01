from typing import Optional

from sqlalchemy import Index, Text
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class VocalizationLabel(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "vocalization_labels"
    __table_args__ = (
        Index(
            "ix_vocalization_labels_job_utc",
            "detection_job_id",
            "start_utc",
            "end_utc",
        ),
    )

    detection_job_id: Mapped[str]
    start_utc: Mapped[float]
    end_utc: Mapped[float]
    label: Mapped[str]
    confidence: Mapped[Optional[float]] = mapped_column(default=None)
    source: Mapped[str] = mapped_column(default="manual")
    notes: Mapped[Optional[str]] = mapped_column(Text, default=None)
    row_store_version_at_import: Mapped[Optional[int]] = mapped_column(default=None)
