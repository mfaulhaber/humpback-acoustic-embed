"""Add feedback training tables for human-in-the-loop retraining.

Creates four tables that support human correction storage and
feedback-driven retraining for both Pass 2 (event segmentation)
and Pass 3 (event classification):

- ``event_boundary_corrections`` — human corrections to Pass 2
  event boundaries (adjust, add, delete).
- ``event_type_corrections`` — human corrections to Pass 3 event
  type assignments (single-label or negative).
- ``event_segmentation_training_jobs`` — queued Pass 2 feedback
  training jobs sourcing from corrected segmentation job output.
- ``event_classifier_training_jobs`` — queued Pass 3 feedback
  training jobs sourcing from corrected classification job output.

Revision ID: 046
Revises: 045
Create Date: 2026-04-12
"""

import sqlalchemy as sa
from alembic import op

revision = "046"
down_revision = "045"
branch_labels = None
depends_on = None


def upgrade() -> None:
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
    op.create_index(
        "ix_event_type_corrections_job_id",
        "event_type_corrections",
        ["event_classification_job_id"],
    )

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

    op.create_table(
        "event_classifier_training_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("source_job_ids", sa.Text(), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("vocalization_model_id", sa.String(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("event_classifier_training_jobs")
    op.drop_table("event_segmentation_training_jobs")

    op.drop_index(
        "ix_event_type_corrections_job_id",
        table_name="event_type_corrections",
    )
    op.drop_table("event_type_corrections")

    op.drop_index(
        "ix_event_boundary_corrections_job_id",
        table_name="event_boundary_corrections",
    )
    op.drop_table("event_boundary_corrections")
