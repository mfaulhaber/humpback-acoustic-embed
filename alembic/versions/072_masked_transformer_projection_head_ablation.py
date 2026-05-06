"""Add projection-head ablation config to masked_transformer_jobs.

Revision ID: 072
Revises: 071
Create Date: 2026-05-06
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "072"
down_revision = "071"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.add_column(
            sa.Column(
                "training_freeze_mode",
                sa.Text(),
                nullable=False,
                server_default="none",
            )
        )
        batch.add_column(
            sa.Column("source_masked_transformer_job_id", sa.String(), nullable=True)
        )
        batch.add_column(
            sa.Column("negative_label_family_policy_json", sa.Text(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.drop_column("negative_label_family_policy_json")
        batch.drop_column("source_masked_transformer_job_id")
        batch.drop_column("training_freeze_mode")
