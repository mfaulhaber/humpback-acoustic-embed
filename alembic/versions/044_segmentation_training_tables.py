"""Add Pass 2 segmentation training tables.

Creates three new tables that hold the training contract for the
Pass 2 framewise segmentation model:

- ``segmentation_training_datasets`` — top-level named training set.
- ``segmentation_training_samples`` — one audio crop + event bounds per
  row. Writable by both the one-shot bootstrap script and a future
  timeline-viewer event-bound editor (same shape, same columns).
- ``segmentation_training_jobs`` — queued trainer runs that read a
  dataset and produce a ``segmentation_models`` row on success.

The dataset / sample split keeps many-to-one containment cheap, and the
composite index on ``(training_dataset_id, source_ref)`` backs the
bootstrap script's idempotency check.

Revision ID: 044
Revises: 043
Create Date: 2026-04-11
"""

import sqlalchemy as sa
from alembic import op

revision = "044"
down_revision = "043"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "segmentation_training_datasets",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "segmentation_training_samples",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("training_dataset_id", sa.String(), nullable=False),
        sa.Column("audio_file_id", sa.String(), nullable=True),
        sa.Column("hydrophone_id", sa.String(), nullable=True),
        sa.Column("start_timestamp", sa.Float(), nullable=True),
        sa.Column("end_timestamp", sa.Float(), nullable=True),
        sa.Column("crop_start_sec", sa.Float(), nullable=False),
        sa.Column("crop_end_sec", sa.Float(), nullable=False),
        sa.Column("events_json", sa.Text(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("source_ref", sa.String(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_segmentation_training_samples_training_dataset_id",
        "segmentation_training_samples",
        ["training_dataset_id"],
    )
    op.create_index(
        "ix_segmentation_training_samples_dataset_source_ref",
        "segmentation_training_samples",
        ["training_dataset_id", "source_ref"],
    )

    op.create_table(
        "segmentation_training_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("training_dataset_id", sa.String(), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=False),
        sa.Column("segmentation_model_id", sa.String(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_segmentation_training_jobs_training_dataset_id",
        "segmentation_training_jobs",
        ["training_dataset_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_segmentation_training_jobs_training_dataset_id",
        table_name="segmentation_training_jobs",
    )
    op.drop_table("segmentation_training_jobs")

    op.drop_index(
        "ix_segmentation_training_samples_dataset_source_ref",
        table_name="segmentation_training_samples",
    )
    op.drop_index(
        "ix_segmentation_training_samples_training_dataset_id",
        table_name="segmentation_training_samples",
    )
    op.drop_table("segmentation_training_samples")

    op.drop_table("segmentation_training_datasets")
