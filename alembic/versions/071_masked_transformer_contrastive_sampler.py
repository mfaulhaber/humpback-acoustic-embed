"""Add contrastive sampler config to masked_transformer_jobs.

Revision ID: 071
Revises: 070
Create Date: 2026-05-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "071"
down_revision = "070"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.add_column(
            sa.Column(
                "contrastive_sampler_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )
        batch.add_column(
            sa.Column(
                "contrastive_labels_per_batch",
                sa.Integer(),
                nullable=False,
                server_default="4",
            )
        )
        batch.add_column(
            sa.Column(
                "contrastive_events_per_label",
                sa.Integer(),
                nullable=False,
                server_default="4",
            )
        )
        batch.add_column(
            sa.Column(
                "contrastive_max_unlabeled_fraction",
                sa.Float(),
                nullable=False,
                server_default="0.25",
            )
        )
        batch.add_column(
            sa.Column(
                "contrastive_region_balance",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.drop_column("contrastive_region_balance")
        batch.drop_column("contrastive_max_unlabeled_fraction")
        batch.drop_column("contrastive_events_per_label")
        batch.drop_column("contrastive_labels_per_batch")
        batch.drop_column("contrastive_sampler_enabled")
