"""Add batch_size to masked_transformer_jobs.

Revision ID: 070
Revises: 069
Create Date: 2026-05-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "070"
down_revision = "069"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.add_column(
            sa.Column(
                "batch_size",
                sa.Integer(),
                nullable=False,
                server_default="8",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.drop_column("batch_size")
