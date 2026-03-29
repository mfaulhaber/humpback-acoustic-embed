"""Add vocalization_types, vocalization_models, vocalization_training_jobs,
vocalization_inference_jobs tables.

Revision ID: 030
Revises: 029
Create Date: 2026-03-29
"""

from alembic import op
import sqlalchemy as sa

revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "vocalization_types",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("name", name="uq_vocalization_types_name"),
    )

    op.create_table(
        "vocalization_models",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("model_dir_path", sa.String(), nullable=False),
        sa.Column("vocabulary_snapshot", sa.Text(), nullable=False),
        sa.Column("per_class_thresholds", sa.Text(), nullable=False),
        sa.Column("per_class_metrics", sa.Text(), nullable=True),
        sa.Column("training_summary", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "vocalization_training_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("source_config", sa.Text(), nullable=False),
        sa.Column("parameters", sa.Text(), nullable=True),
        sa.Column("vocalization_model_id", sa.String(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "vocalization_inference_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("vocalization_model_id", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("output_path", sa.String(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("vocalization_inference_jobs")
    op.drop_table("vocalization_training_jobs")
    op.drop_table("vocalization_models")
    op.drop_table("vocalization_types")
