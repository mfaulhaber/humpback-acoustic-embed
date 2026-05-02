"""Add masked_transformer_jobs table (ADR-061).

Sequence Models: masked-transformer track parallel to HMM. One training
job per ``(continuous_embedding_job_id, training_signature)``; per-k
artifacts fan out under the job directory.

Revision ID: 063
Revises: 062
Create Date: 2026-05-01
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "063"
down_revision = "062"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "masked_transformer_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("status_reason", sa.Text(), nullable=True),
        sa.Column("continuous_embedding_job_id", sa.String(), nullable=False),
        sa.Column("training_signature", sa.String(), nullable=False),
        # Training config
        sa.Column("preset", sa.Text(), nullable=False, server_default="default"),
        sa.Column("mask_fraction", sa.Float(), nullable=False, server_default="0.20"),
        sa.Column("span_length_min", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("span_length_max", sa.Integer(), nullable=False, server_default="6"),
        sa.Column("dropout", sa.Float(), nullable=False, server_default="0.1"),
        sa.Column(
            "mask_weight_bias",
            sa.Boolean(),
            nullable=False,
            server_default="1",
        ),
        sa.Column(
            "cosine_loss_weight",
            sa.Float(),
            nullable=False,
            server_default="0.0",
        ),
        sa.Column("max_epochs", sa.Integer(), nullable=False, server_default="30"),
        sa.Column(
            "early_stop_patience", sa.Integer(), nullable=False, server_default="3"
        ),
        sa.Column("val_split", sa.Float(), nullable=False, server_default="0.1"),
        sa.Column("seed", sa.Integer(), nullable=False, server_default="42"),
        # Tokenization config (JSON-encoded list of int)
        sa.Column("k_values", sa.Text(), nullable=False),
        # Device + outcomes
        sa.Column("chosen_device", sa.Text(), nullable=True),
        sa.Column("fallback_reason", sa.Text(), nullable=True),
        sa.Column("final_train_loss", sa.Float(), nullable=True),
        sa.Column("final_val_loss", sa.Float(), nullable=True),
        sa.Column("total_epochs", sa.Integer(), nullable=True),
        # Storage
        sa.Column("job_dir", sa.Text(), nullable=True),
        sa.Column("total_sequences", sa.Integer(), nullable=True),
        sa.Column("total_chunks", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["continuous_embedding_job_id"],
            ["continuous_embedding_jobs.id"],
            name="fk_masked_transformer_jobs_continuous_embedding_job",
        ),
    )
    op.create_index(
        "ix_masked_transformer_jobs_status",
        "masked_transformer_jobs",
        ["status"],
    )
    op.create_index(
        "ix_masked_transformer_jobs_continuous_embedding_job_id",
        "masked_transformer_jobs",
        ["continuous_embedding_job_id"],
    )
    op.create_index(
        "ix_masked_transformer_jobs_training_signature",
        "masked_transformer_jobs",
        ["training_signature"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_masked_transformer_jobs_training_signature",
        table_name="masked_transformer_jobs",
    )
    op.drop_index(
        "ix_masked_transformer_jobs_continuous_embedding_job_id",
        table_name="masked_transformer_jobs",
    )
    op.drop_index(
        "ix_masked_transformer_jobs_status",
        table_name="masked_transformer_jobs",
    )
    op.drop_table("masked_transformer_jobs")
