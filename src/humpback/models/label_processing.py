from typing import Optional

from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class LabelProcessingJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "label_processing_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    workflow: Mapped[str] = mapped_column(default="score_based")
    classifier_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    annotation_folder: Mapped[str]
    audio_folder: Mapped[str]
    output_root: Mapped[str]
    parameters: Mapped[Optional[str]] = mapped_column(Text, default=None)
    files_processed: Mapped[Optional[int]] = mapped_column(default=None)
    files_total: Mapped[Optional[int]] = mapped_column(default=None)
    annotations_total: Mapped[Optional[int]] = mapped_column(default=None)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
