"""Add window_classification_jobs and window_score_corrections tables.

Standalone sidecar enrichment that scores cached Perch embeddings from
Pass 1 regions through multi-label vocalization classifiers. Corrections
table stores human overrides for per-window type presence/absence.

Revision ID: 053
Revises: 052
Create Date: 2026-04-23
"""

import sqlalchemy as sa
from alembic import op

revision = "053"
down_revision = "052"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "window_classification_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("region_detection_job_id", sa.String(), nullable=False),
        sa.Column("vocalization_model_id", sa.String(), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=True),
        sa.Column("window_count", sa.Integer(), nullable=True),
        sa.Column("vocabulary_snapshot", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_window_classification_jobs_region_job",
        "window_classification_jobs",
        ["region_detection_job_id"],
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
    op.create_index(
        "ix_window_score_corrections_job_id",
        "window_score_corrections",
        ["window_classification_job_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_window_score_corrections_job_id",
        table_name="window_score_corrections",
    )
    op.drop_table("window_score_corrections")
    op.drop_index(
        "ix_window_classification_jobs_region_job",
        table_name="window_classification_jobs",
    )
    op.drop_table("window_classification_jobs")
