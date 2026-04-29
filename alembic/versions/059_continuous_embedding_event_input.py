"""Refactor continuous_embedding_jobs from region to event input.

- Replace ``region_detection_job_id`` with ``event_segmentation_job_id``
- Rename ``total_regions`` to ``total_events``
- Change ``pad_seconds`` server default to 2.0

Revision ID: 059
Revises: 058
Create Date: 2026-04-28
"""

import sqlalchemy as sa
from alembic import op

revision = "059"
down_revision = "058"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("continuous_embedding_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("event_segmentation_job_id", sa.String(), nullable=True),
        )
        batch_op.create_foreign_key(
            "fk_continuous_embedding_jobs_seg_job",
            "event_segmentation_jobs",
            ["event_segmentation_job_id"],
            ["id"],
        )
        batch_op.add_column(
            sa.Column("total_events", sa.Integer(), nullable=True),
        )
        batch_op.drop_column("region_detection_job_id")
        batch_op.drop_column("total_regions")
        batch_op.alter_column(
            "pad_seconds",
            server_default=sa.text("2.0"),
        )


def downgrade() -> None:
    with op.batch_alter_table("continuous_embedding_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("region_detection_job_id", sa.String(), nullable=True),
        )
        batch_op.add_column(
            sa.Column("total_regions", sa.Integer(), nullable=True),
        )
        batch_op.drop_constraint(
            "fk_continuous_embedding_jobs_seg_job", type_="foreignkey"
        )
        batch_op.drop_column("event_segmentation_job_id")
        batch_op.drop_column("total_events")
        batch_op.alter_column(
            "pad_seconds",
            server_default=None,
        )
