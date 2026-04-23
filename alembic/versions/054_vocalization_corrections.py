"""Create vocalization_corrections, drop window_score_corrections and event_type_corrections.

Revision ID: 054
Revises: 053
"""

from alembic import op

import sqlalchemy as sa

revision = "054"
down_revision = "053"


def upgrade() -> None:
    op.create_table(
        "vocalization_corrections",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("region_detection_job_id", sa.String(), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("type_name", sa.String(), nullable=False),
        sa.Column("correction_type", sa.String(), nullable=False),
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
        sa.UniqueConstraint(
            "region_detection_job_id",
            "start_sec",
            "end_sec",
            "type_name",
            name="uq_vocalization_corrections_job_time_type",
        ),
    )
    op.create_index(
        "ix_vocalization_corrections_detection_job",
        "vocalization_corrections",
        ["region_detection_job_id"],
    )

    op.drop_table("window_score_corrections")
    op.drop_table("event_type_corrections")


def downgrade() -> None:
    op.create_table(
        "event_type_corrections",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("event_classification_job_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("type_name", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "event_classification_job_id",
            "event_id",
            name="uq_event_type_corrections_job_event",
        ),
    )

    op.create_table(
        "window_score_corrections",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("window_classification_job_id", sa.String(), nullable=False),
        sa.Column("time_sec", sa.Float(), nullable=False),
        sa.Column("region_id", sa.String(), nullable=False),
        sa.Column("correction_type", sa.String(), nullable=False),
        sa.Column("type_name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.drop_index(
        "ix_vocalization_corrections_detection_job",
        table_name="vocalization_corrections",
    )
    op.drop_table("vocalization_corrections")
