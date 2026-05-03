"""Add effective event identity columns.

Revision ID: 065
Revises: 064
Create Date: 2026-05-03
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "065"
down_revision = "064"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("event_boundary_corrections") as batch:
        batch.add_column(
            sa.Column("event_segmentation_job_id", sa.String(), nullable=True)
        )
        batch.add_column(sa.Column("source_event_id", sa.String(), nullable=True))

    op.create_index(
        "ix_event_boundary_corrections_region_detection_job",
        "event_boundary_corrections",
        ["region_detection_job_id"],
    )
    op.create_index(
        "ix_event_boundary_corrections_segmentation_job",
        "event_boundary_corrections",
        ["event_segmentation_job_id"],
    )
    op.create_index(
        "ix_event_boundary_corrections_source_event",
        "event_boundary_corrections",
        ["source_event_id"],
    )

    with op.batch_alter_table("continuous_embedding_jobs") as batch:
        batch.add_column(
            sa.Column(
                "event_source_mode",
                sa.String(),
                nullable=False,
                server_default="raw",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("continuous_embedding_jobs") as batch:
        batch.drop_column("event_source_mode")

    op.drop_index(
        "ix_event_boundary_corrections_source_event",
        table_name="event_boundary_corrections",
    )
    op.drop_index(
        "ix_event_boundary_corrections_segmentation_job",
        table_name="event_boundary_corrections",
    )
    op.drop_index(
        "ix_event_boundary_corrections_region_detection_job",
        table_name="event_boundary_corrections",
    )
    with op.batch_alter_table("event_boundary_corrections") as batch:
        batch.drop_column("source_event_id")
        batch.drop_column("event_segmentation_job_id")
