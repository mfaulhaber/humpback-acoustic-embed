from sqlalchemy import Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class TrainingDataset(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "training_datasets"

    name: Mapped[str]
    source_config: Mapped[str] = mapped_column(Text)  # JSON
    parquet_path: Mapped[str]
    total_rows: Mapped[int] = mapped_column(Integer, default=0)
    vocabulary: Mapped[str] = mapped_column(Text, default="[]")  # JSON array


class TrainingDatasetLabel(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "training_dataset_labels"
    __table_args__ = (Index("ix_tdl_dataset_row", "training_dataset_id", "row_index"),)

    training_dataset_id: Mapped[str]
    row_index: Mapped[int] = mapped_column(Integer)
    label: Mapped[str]
    source: Mapped[str] = mapped_column(default="snapshot")  # "snapshot" or "manual"
