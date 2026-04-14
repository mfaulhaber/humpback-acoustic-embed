"""Drop event_segmentation_training_jobs table.

Segmentation training is consolidated onto the standard
``SegmentationTrainingDataset`` → ``SegmentationTrainingJob`` path.
The feedback-specific ``EventSegmentationTrainingJob`` table is no
longer used.

Revision ID: 047
Revises: 046
Create Date: 2026-04-14
"""

import sqlalchemy as sa
from alembic import op

revision = "047"
down_revision = "046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("event_segmentation_training_jobs")


def downgrade() -> None:
    op.create_table(
        "event_segmentation_training_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("source_job_ids", sa.Text(), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("segmentation_model_id", sa.String(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
