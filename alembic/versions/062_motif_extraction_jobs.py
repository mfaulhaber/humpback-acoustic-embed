"""Add motif extraction jobs.

Revision ID: 062
Revises: 061
Create Date: 2026-04-30
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "062"
down_revision = "061"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "motif_extraction_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("hmm_sequence_job_id", sa.String(), nullable=False),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column("min_ngram", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("max_ngram", sa.Integer(), nullable=False, server_default="8"),
        sa.Column(
            "minimum_occurrences", sa.Integer(), nullable=False, server_default="5"
        ),
        sa.Column(
            "minimum_event_sources", sa.Integer(), nullable=False, server_default="2"
        ),
        sa.Column("frequency_weight", sa.Float(), nullable=False, server_default="0.4"),
        sa.Column(
            "event_source_weight", sa.Float(), nullable=False, server_default="0.3"
        ),
        sa.Column(
            "event_core_weight", sa.Float(), nullable=False, server_default="0.2"
        ),
        sa.Column(
            "low_background_weight", sa.Float(), nullable=False, server_default="0.1"
        ),
        sa.Column("call_probability_weight", sa.Float(), nullable=True),
        sa.Column("config_signature", sa.String(), nullable=False),
        sa.Column("total_groups", sa.Integer(), nullable=True),
        sa.Column("total_collapsed_tokens", sa.Integer(), nullable=True),
        sa.Column("total_candidate_occurrences", sa.Integer(), nullable=True),
        sa.Column("total_motifs", sa.Integer(), nullable=True),
        sa.Column("artifact_dir", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["hmm_sequence_job_id"],
            ["hmm_sequence_jobs.id"],
            name="fk_motif_extraction_jobs_hmm_sequence_job",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_motif_extraction_jobs_status",
        "motif_extraction_jobs",
        ["status"],
    )
    op.create_index(
        "ix_motif_extraction_jobs_hmm_sequence_job_id",
        "motif_extraction_jobs",
        ["hmm_sequence_job_id"],
    )
    op.create_index(
        "ix_motif_extraction_jobs_config_signature",
        "motif_extraction_jobs",
        ["config_signature"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_motif_extraction_jobs_config_signature",
        table_name="motif_extraction_jobs",
    )
    op.drop_index(
        "ix_motif_extraction_jobs_hmm_sequence_job_id",
        table_name="motif_extraction_jobs",
    )
    op.drop_index("ix_motif_extraction_jobs_status", table_name="motif_extraction_jobs")
    op.drop_table("motif_extraction_jobs")
