"""Add retrieval-head config to masked_transformer_jobs.

Revision ID: 067
Revises: 066
Create Date: 2026-05-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "067"
down_revision = "066"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.add_column(
            sa.Column(
                "retrieval_head_enabled",
                sa.Boolean(),
                nullable=False,
                server_default="0",
            )
        )
        batch.add_column(sa.Column("retrieval_dim", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("retrieval_hidden_dim", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column(
                "retrieval_l2_normalize",
                sa.Boolean(),
                nullable=False,
                server_default="1",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.drop_column("retrieval_l2_normalize")
        batch.drop_column("retrieval_hidden_dim")
        batch.drop_column("retrieval_dim")
        batch.drop_column("retrieval_head_enabled")
