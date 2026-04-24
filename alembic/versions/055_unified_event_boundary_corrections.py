"""Drop and recreate event_boundary_corrections with region_detection_job_id FK.

Revision ID: 055
Revises: 054
"""

from alembic import op

import sqlalchemy as sa

revision = "055"
down_revision = "054"


def upgrade() -> None:
    op.drop_table("event_boundary_corrections")

    op.create_table(
        "event_boundary_corrections",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("region_detection_job_id", sa.String(), nullable=False),
        sa.Column("region_id", sa.String(), nullable=False),
        sa.Column("correction_type", sa.String(), nullable=False),
        sa.Column("original_start_sec", sa.Float(), nullable=True),
        sa.Column("original_end_sec", sa.Float(), nullable=True),
        sa.Column("corrected_start_sec", sa.Float(), nullable=True),
        sa.Column("corrected_end_sec", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_event_boundary_corrections_detection_job",
        "event_boundary_corrections",
        ["region_detection_job_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_event_boundary_corrections_detection_job",
        table_name="event_boundary_corrections",
    )
    op.drop_table("event_boundary_corrections")

    op.create_table(
        "event_boundary_corrections",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("event_segmentation_job_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("region_id", sa.String(), nullable=False),
        sa.Column("correction_type", sa.String(), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=True),
        sa.Column("end_sec", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_event_boundary_corrections_job_id",
        "event_boundary_corrections",
        ["event_segmentation_job_id"],
    )
