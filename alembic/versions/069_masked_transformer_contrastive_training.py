"""Add contrastive training config to masked_transformer_jobs.

Revision ID: 069
Revises: 068
Create Date: 2026-05-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "069"
down_revision = "068"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.add_column(
            sa.Column(
                "contrastive_loss_weight",
                sa.Float(),
                nullable=False,
                server_default="0.0",
            )
        )
        batch.add_column(
            sa.Column(
                "contrastive_temperature",
                sa.Float(),
                nullable=False,
                server_default="0.07",
            )
        )
        batch.add_column(
            sa.Column(
                "contrastive_label_source",
                sa.Text(),
                nullable=False,
                server_default="none",
            )
        )
        batch.add_column(
            sa.Column(
                "contrastive_min_events_per_label",
                sa.Integer(),
                nullable=False,
                server_default="4",
            )
        )
        batch.add_column(
            sa.Column(
                "contrastive_min_regions_per_label",
                sa.Integer(),
                nullable=False,
                server_default="2",
            )
        )
        batch.add_column(
            sa.Column(
                "require_cross_region_positive",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )
        batch.add_column(
            sa.Column("related_label_policy_json", sa.Text(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.drop_column("related_label_policy_json")
        batch.drop_column("require_cross_region_positive")
        batch.drop_column("contrastive_min_regions_per_label")
        batch.drop_column("contrastive_min_events_per_label")
        batch.drop_column("contrastive_label_source")
        batch.drop_column("contrastive_temperature")
        batch.drop_column("contrastive_loss_weight")
