"""Add hyperparameter_manifests table.

Revision ID: 040
Revises: 039
Create Date: 2026-04-09
"""

import sqlalchemy as sa
from alembic import op

revision = "040"
down_revision = "039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hyperparameter_manifests",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("training_job_ids", sa.Text(), nullable=False),
        sa.Column("detection_job_ids", sa.Text(), nullable=False),
        sa.Column("split_ratio", sa.Text(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("manifest_path", sa.String(), nullable=True),
        sa.Column("example_count", sa.Integer(), nullable=True),
        sa.Column("split_summary", sa.Text(), nullable=True),
        sa.Column("detection_job_summaries", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("hyperparameter_manifests")
