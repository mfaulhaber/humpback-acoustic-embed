"""Add retrain_workflows table for automated retrain pipelines

Revision ID: 014
Revises: 013
Create Date: 2026-03-09
"""

import sqlalchemy as sa
from alembic import op

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "retrain_workflows",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("source_model_id", sa.String(), nullable=False),
        sa.Column("new_model_name", sa.String(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("window_size_seconds", sa.Float(), nullable=False),
        sa.Column("target_sample_rate", sa.Integer(), nullable=False),
        sa.Column("feature_config", sa.Text(), nullable=True),
        sa.Column("parameters", sa.Text(), nullable=True),
        sa.Column("positive_folder_roots", sa.Text(), nullable=False),
        sa.Column("negative_folder_roots", sa.Text(), nullable=False),
        sa.Column("import_summary", sa.Text(), nullable=True),
        sa.Column("processing_job_ids", sa.Text(), nullable=True),
        sa.Column("processing_total", sa.Integer(), nullable=True),
        sa.Column("processing_complete", sa.Integer(), nullable=True),
        sa.Column("training_job_id", sa.String(), nullable=True),
        sa.Column("new_model_id", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade():
    op.drop_table("retrain_workflows")
