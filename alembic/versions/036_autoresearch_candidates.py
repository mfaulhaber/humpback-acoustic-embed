"""Add autoresearch_candidates table.

Revision ID: 036
Revises: 035
Create Date: 2026-04-03
"""

import sqlalchemy as sa
from alembic import op

revision = "036"
down_revision = "035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "autoresearch_candidates",
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("manifest_path", sa.String(), nullable=False),
        sa.Column("best_run_path", sa.String(), nullable=False),
        sa.Column("comparison_path", sa.String(), nullable=True),
        sa.Column("top_false_positives_path", sa.String(), nullable=True),
        sa.Column("phase", sa.String(), nullable=True),
        sa.Column("objective_name", sa.String(), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("promoted_config", sa.Text(), nullable=False),
        sa.Column("best_run_metrics", sa.Text(), nullable=True),
        sa.Column("split_metrics", sa.Text(), nullable=True),
        sa.Column("metric_deltas", sa.Text(), nullable=True),
        sa.Column("replay_summary", sa.Text(), nullable=True),
        sa.Column("source_counts", sa.Text(), nullable=True),
        sa.Column("warnings", sa.Text(), nullable=True),
        sa.Column("source_model_id", sa.String(), nullable=True),
        sa.Column("source_model_name", sa.String(), nullable=True),
        sa.Column("source_model_metadata", sa.Text(), nullable=True),
        sa.Column("comparison_target", sa.String(), nullable=True),
        sa.Column("top_false_positives_preview", sa.Text(), nullable=True),
        sa.Column("prediction_disagreements_preview", sa.Text(), nullable=True),
        sa.Column("is_reproducible_exact", sa.Boolean(), nullable=False),
        sa.Column("training_job_id", sa.String(), nullable=True),
        sa.Column("new_model_id", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("autoresearch_candidates")
