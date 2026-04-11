"""Add call parsing pipeline tables (Phase 0 scaffold).

Creates the parent `call_parsing_runs` table and three child pass job
tables (`region_detection_jobs`, `event_segmentation_jobs`,
`event_classification_jobs`), plus `segmentation_models` for Pass 2
PyTorch checkpoints. Extends `vocalization_models` and
`vocalization_training_jobs` with `model_family` and `input_mode` columns
so Pass 3 can coexist with the existing sklearn family.

Revision ID: 042
Revises: 041
Create Date: 2026-04-11
"""

import sqlalchemy as sa
from alembic import op

revision = "042"
down_revision = "041"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Parent run table — threads one end-to-end pipeline run across the four
    # child pass job tables. Nullable child FKs are populated by the parent
    # orchestration service as each pass is queued.
    op.create_table(
        "call_parsing_runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("audio_source_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("config_snapshot", sa.Text(), nullable=True),
        sa.Column("region_detection_job_id", sa.String(), nullable=True),
        sa.Column("event_segmentation_job_id", sa.String(), nullable=True),
        sa.Column("event_classification_job_id", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_call_parsing_runs_audio_source_id",
        "call_parsing_runs",
        ["audio_source_id"],
    )

    # Pass 2 segmentation model registry — independent from
    # `vocalization_models` because segmentation is a separate task type
    # (framewise onset/offset) rather than per-event multi-label.
    op.create_table(
        "segmentation_models",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("model_family", sa.String(), nullable=False),
        sa.Column("model_path", sa.String(), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("training_job_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # Pass 1 — region detection jobs
    op.create_table(
        "region_detection_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("parent_run_id", sa.String(), nullable=True),
        sa.Column("audio_source_id", sa.String(), nullable=False),
        sa.Column("model_config_id", sa.String(), nullable=True),
        sa.Column("classifier_model_id", sa.String(), nullable=True),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("trace_row_count", sa.Integer(), nullable=True),
        sa.Column("region_count", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_region_detection_jobs_parent_run_id",
        "region_detection_jobs",
        ["parent_run_id"],
    )

    # Pass 2 — event segmentation jobs
    op.create_table(
        "event_segmentation_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("parent_run_id", sa.String(), nullable=True),
        sa.Column("region_detection_job_id", sa.String(), nullable=False),
        sa.Column("segmentation_model_id", sa.String(), nullable=True),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("event_count", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_event_segmentation_jobs_parent_run_id",
        "event_segmentation_jobs",
        ["parent_run_id"],
    )
    op.create_index(
        "ix_event_segmentation_jobs_region_detection_job_id",
        "event_segmentation_jobs",
        ["region_detection_job_id"],
    )

    # Pass 3 — event classification jobs
    op.create_table(
        "event_classification_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("parent_run_id", sa.String(), nullable=True),
        sa.Column("event_segmentation_job_id", sa.String(), nullable=False),
        sa.Column("vocalization_model_id", sa.String(), nullable=True),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("typed_event_count", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_event_classification_jobs_parent_run_id",
        "event_classification_jobs",
        ["parent_run_id"],
    )
    op.create_index(
        "ix_event_classification_jobs_event_segmentation_job_id",
        "event_classification_jobs",
        ["event_segmentation_job_id"],
    )

    # Extend vocalization_models with model_family + input_mode so Pass 3 can
    # register a new "pytorch_event_cnn" family alongside the existing
    # sklearn_perch_embedding rows. Existing rows get backfilled to the
    # defaults before the columns are made NOT NULL.
    with op.batch_alter_table("vocalization_models") as batch_op:
        batch_op.add_column(
            sa.Column("model_family", sa.String(), nullable=True),
        )
        batch_op.add_column(
            sa.Column("input_mode", sa.String(), nullable=True),
        )

    op.execute(
        "UPDATE vocalization_models "
        "SET model_family = 'sklearn_perch_embedding' "
        "WHERE model_family IS NULL"
    )
    op.execute(
        "UPDATE vocalization_models "
        "SET input_mode = 'detection_row' "
        "WHERE input_mode IS NULL"
    )

    with op.batch_alter_table("vocalization_models") as batch_op:
        batch_op.alter_column("model_family", nullable=False)
        batch_op.alter_column("input_mode", nullable=False)

    # Same column additions on vocalization_training_jobs so Pass 3 training
    # reuses the existing training job table rather than introducing a
    # parallel one.
    with op.batch_alter_table("vocalization_training_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("model_family", sa.String(), nullable=True),
        )
        batch_op.add_column(
            sa.Column("input_mode", sa.String(), nullable=True),
        )

    op.execute(
        "UPDATE vocalization_training_jobs "
        "SET model_family = 'sklearn_perch_embedding' "
        "WHERE model_family IS NULL"
    )
    op.execute(
        "UPDATE vocalization_training_jobs "
        "SET input_mode = 'detection_row' "
        "WHERE input_mode IS NULL"
    )

    with op.batch_alter_table("vocalization_training_jobs") as batch_op:
        batch_op.alter_column("model_family", nullable=False)
        batch_op.alter_column("input_mode", nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("vocalization_training_jobs") as batch_op:
        batch_op.drop_column("input_mode")
        batch_op.drop_column("model_family")

    with op.batch_alter_table("vocalization_models") as batch_op:
        batch_op.drop_column("input_mode")
        batch_op.drop_column("model_family")

    op.drop_index(
        "ix_event_classification_jobs_event_segmentation_job_id",
        table_name="event_classification_jobs",
    )
    op.drop_index(
        "ix_event_classification_jobs_parent_run_id",
        table_name="event_classification_jobs",
    )
    op.drop_table("event_classification_jobs")

    op.drop_index(
        "ix_event_segmentation_jobs_region_detection_job_id",
        table_name="event_segmentation_jobs",
    )
    op.drop_index(
        "ix_event_segmentation_jobs_parent_run_id",
        table_name="event_segmentation_jobs",
    )
    op.drop_table("event_segmentation_jobs")

    op.drop_index(
        "ix_region_detection_jobs_parent_run_id",
        table_name="region_detection_jobs",
    )
    op.drop_table("region_detection_jobs")

    op.drop_table("segmentation_models")

    op.drop_index(
        "ix_call_parsing_runs_audio_source_id",
        table_name="call_parsing_runs",
    )
    op.drop_table("call_parsing_runs")
