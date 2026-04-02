from typing import Optional

from sqlalchemy import Index, Text
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class VocalizationLabel(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "vocalization_labels"
    __table_args__ = (
        Index(
            "ix_vocalization_labels_job_row_id",
            "detection_job_id",
            "row_id",
        ),
    )

    detection_job_id: Mapped[str]
    row_id: Mapped[str]
    label: Mapped[str]
    confidence: Mapped[Optional[float]] = mapped_column(default=None)
    source: Mapped[str] = mapped_column(default="manual")
    notes: Mapped[Optional[str]] = mapped_column(Text, default=None)
