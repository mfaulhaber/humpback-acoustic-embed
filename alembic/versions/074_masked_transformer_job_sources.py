"""Add source rows for multi-source masked-transformer training.

Revision ID: 074
Revises: 073
Create Date: 2026-05-06
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "074"
down_revision = "073"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "masked_transformer_job_sources",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("masked_transformer_job_id", sa.String(), nullable=False),
        sa.Column("source_order", sa.Integer(), nullable=False),
        sa.Column("continuous_embedding_job_id", sa.String(), nullable=False),
        sa.Column("event_classification_job_id", sa.String(), nullable=False),
        sa.Column("source_alias", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["masked_transformer_job_id"],
            ["masked_transformer_jobs.id"],
            name="fk_masked_transformer_job_sources_job",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["continuous_embedding_job_id"],
            ["continuous_embedding_jobs.id"],
            name="fk_masked_transformer_job_sources_continuous_embedding_job",
        ),
        sa.UniqueConstraint(
            "masked_transformer_job_id",
            "source_order",
            name="uq_masked_transformer_job_sources_order",
        ),
        sa.UniqueConstraint(
            "masked_transformer_job_id",
            "continuous_embedding_job_id",
            "event_classification_job_id",
            name="uq_masked_transformer_job_sources_pair",
        ),
    )
    op.create_index(
        "ix_masked_transformer_job_sources_job_id",
        "masked_transformer_job_sources",
        ["masked_transformer_job_id"],
    )
    op.create_index(
        "ix_masked_transformer_job_sources_continuous_embedding_job_id",
        "masked_transformer_job_sources",
        ["continuous_embedding_job_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_masked_transformer_job_sources_continuous_embedding_job_id",
        table_name="masked_transformer_job_sources",
    )
    op.drop_index(
        "ix_masked_transformer_job_sources_job_id",
        table_name="masked_transformer_job_sources",
    )
    op.drop_table("masked_transformer_job_sources")
