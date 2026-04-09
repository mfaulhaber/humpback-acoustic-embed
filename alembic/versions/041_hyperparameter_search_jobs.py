"""Add hyperparameter_search_jobs table.

Revision ID: 041
Revises: 040
Create Date: 2026-04-09
"""

import sqlalchemy as sa
from alembic import op

revision = "041"
down_revision = "040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hyperparameter_search_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("manifest_id", sa.String(), nullable=False),
        sa.Column("search_space", sa.Text(), nullable=False),
        sa.Column("n_trials", sa.Integer(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("objective_name", sa.String(), nullable=False),
        sa.Column("results_dir", sa.String(), nullable=True),
        sa.Column("trials_completed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("best_objective", sa.Float(), nullable=True),
        sa.Column("best_config", sa.Text(), nullable=True),
        sa.Column("best_metrics", sa.Text(), nullable=True),
        sa.Column("comparison_model_id", sa.String(), nullable=True),
        sa.Column("comparison_threshold", sa.Float(), nullable=True),
        sa.Column("comparison_result", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["manifest_id"],
            ["hyperparameter_manifests.id"],
            name="fk_search_manifest_id",
        ),
    )


def downgrade() -> None:
    op.drop_table("hyperparameter_search_jobs")
