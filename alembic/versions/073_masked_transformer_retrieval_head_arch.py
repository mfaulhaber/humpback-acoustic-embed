"""Add retrieval-head architecture to masked_transformer_jobs.

Revision ID: 073
Revises: 072
Create Date: 2026-05-06
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "073"
down_revision = "072"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.add_column(
            sa.Column(
                "retrieval_head_arch",
                sa.Text(),
                nullable=False,
                server_default="mlp",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.drop_column("retrieval_head_arch")
