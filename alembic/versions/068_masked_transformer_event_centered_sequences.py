"""Add event-centered sequence config to masked_transformer_jobs.

Revision ID: 068
Revises: 067
Create Date: 2026-05-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "068"
down_revision = "067"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.add_column(
            sa.Column(
                "sequence_construction_mode",
                sa.Text(),
                nullable=False,
                server_default="region",
            )
        )
        batch.add_column(
            sa.Column(
                "event_centered_fraction",
                sa.Float(),
                nullable=False,
                server_default="0.0",
            )
        )
        batch.add_column(sa.Column("pre_event_context_sec", sa.Float(), nullable=True))
        batch.add_column(sa.Column("post_event_context_sec", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.drop_column("post_event_context_sec")
        batch.drop_column("pre_event_context_sec")
        batch.drop_column("event_centered_fraction")
        batch.drop_column("sequence_construction_mode")
