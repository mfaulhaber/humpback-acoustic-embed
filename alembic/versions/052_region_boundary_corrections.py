"""Add region_boundary_corrections table for Pass 1 corrections.

Stores human corrections to region boundaries produced by Pass 1
region detection. Supports adjust (move boundaries), add (new region),
and delete (remove region) correction types.

Revision ID: 052
Revises: 051
Create Date: 2026-04-20
"""

import sqlalchemy as sa
from alembic import op

revision = "052"
down_revision = "051"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "region_boundary_corrections",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("region_detection_job_id", sa.String(), nullable=False),
        sa.Column("region_id", sa.String(), nullable=False),
        sa.Column("correction_type", sa.String(), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=True),
        sa.Column("end_sec", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "region_detection_job_id",
            "region_id",
            name="uq_region_boundary_corrections_job_region",
        ),
    )
    op.create_index(
        "ix_region_boundary_corrections_job_id",
        "region_boundary_corrections",
        ["region_detection_job_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_region_boundary_corrections_job_id",
        table_name="region_boundary_corrections",
    )
    op.drop_table("region_boundary_corrections")
