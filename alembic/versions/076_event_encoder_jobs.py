"""Add event_encoder_jobs table.

Revision ID: 076
Revises: 075
Create Date: 2026-05-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "076"
down_revision = "075"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "event_encoder_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("event_segmentation_job_id", sa.String(), nullable=False),
        sa.Column("event_source_mode", sa.String(), nullable=False),
        sa.Column("continuous_embedding_job_id", sa.String(), nullable=False),
        sa.Column("continuous_embedding_signature", sa.Text(), nullable=False),
        sa.Column(
            "tokenizer_version",
            sa.Text(),
            nullable=False,
            server_default="crnn-event-encoder-v1",
        ),
        sa.Column("pooling_config_json", sa.Text(), nullable=False),
        sa.Column("descriptor_config_json", sa.Text(), nullable=False),
        sa.Column("preprocessing_config_json", sa.Text(), nullable=False),
        sa.Column("k_values_json", sa.Text(), nullable=False),
        sa.Column("random_seed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokenization_signature", sa.String(), nullable=False),
        sa.Column("event_vector_dim", sa.Integer(), nullable=True),
        sa.Column("total_events", sa.Integer(), nullable=True),
        sa.Column("encoded_events", sa.Integer(), nullable=True),
        sa.Column("skipped_events", sa.Integer(), nullable=True),
        sa.Column("event_vectors_path", sa.Text(), nullable=True),
        sa.Column("event_tokens_path", sa.Text(), nullable=True),
        sa.Column("token_sequences_path", sa.Text(), nullable=True),
        sa.Column("manifest_path", sa.Text(), nullable=True),
        sa.Column("report_path", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tokenization_signature",
            name="uq_event_encoder_jobs_tokenization_signature",
        ),
        sa.ForeignKeyConstraint(
            ["event_segmentation_job_id"],
            ["event_segmentation_jobs.id"],
            name="fk_event_encoder_jobs_segmentation_job",
        ),
        sa.ForeignKeyConstraint(
            ["continuous_embedding_job_id"],
            ["continuous_embedding_jobs.id"],
            name="fk_event_encoder_jobs_continuous_embedding_job",
        ),
    )
    op.create_index(
        "ix_event_encoder_jobs_status",
        "event_encoder_jobs",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index("ix_event_encoder_jobs_status", table_name="event_encoder_jobs")
    op.drop_table("event_encoder_jobs")
