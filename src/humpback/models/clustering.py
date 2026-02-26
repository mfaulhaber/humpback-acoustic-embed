from typing import Optional

from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from humpback.database import Base, TimestampMixin, UUIDMixin


class ClusteringJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "clustering_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    embedding_set_ids: Mapped[str] = mapped_column(Text)  # JSON array
    parameters: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)

    clusters: Mapped[list["Cluster"]] = relationship(
        back_populates="clustering_job", cascade="all, delete-orphan"
    )


class Cluster(UUIDMixin, Base):
    __tablename__ = "clusters"

    clustering_job_id: Mapped[str] = mapped_column(ForeignKey("clustering_jobs.id"))
    cluster_label: Mapped[int]
    size: Mapped[int]
    metadata_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)

    clustering_job: Mapped["ClusteringJob"] = relationship(back_populates="clusters")
    assignments: Mapped[list["ClusterAssignment"]] = relationship(
        back_populates="cluster", cascade="all, delete-orphan"
    )


class ClusterAssignment(UUIDMixin, Base):
    __tablename__ = "cluster_assignments"

    cluster_id: Mapped[str] = mapped_column(ForeignKey("clusters.id"))
    embedding_set_id: Mapped[str]
    embedding_row_index: Mapped[int]

    cluster: Mapped["Cluster"] = relationship(back_populates="assignments")
