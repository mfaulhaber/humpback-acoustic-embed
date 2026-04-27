"""Add continuous_embedding_jobs table.

Sequence Models PR 1 producer: region-bounded, hydrophone-only 1-second-hop
SurfPerch embeddings padded around Pass-1 region detections. Idempotent on
``encoding_signature``.

Revision ID: 057
Revises: 056
Create Date: 2026-04-27
"""

import sqlalchemy as sa
from alembic import op

revision = "057"
down_revision = "056"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "continuous_embedding_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("region_detection_job_id", sa.String(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("window_size_seconds", sa.Float(), nullable=False),
        sa.Column("hop_seconds", sa.Float(), nullable=False),
        sa.Column("pad_seconds", sa.Float(), nullable=False),
        sa.Column("target_sample_rate", sa.Integer(), nullable=False),
        sa.Column("feature_config_json", sa.Text(), nullable=True),
        sa.Column("encoding_signature", sa.String(), nullable=False),
        sa.Column("vector_dim", sa.Integer(), nullable=True),
        sa.Column("total_regions", sa.Integer(), nullable=True),
        sa.Column("merged_spans", sa.Integer(), nullable=True),
        sa.Column("total_windows", sa.Integer(), nullable=True),
        sa.Column("parquet_path", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["region_detection_job_id"],
            ["region_detection_jobs.id"],
            name="fk_continuous_embedding_jobs_region_job",
        ),
    )
    op.create_index(
        "ix_continuous_embedding_jobs_encoding_signature",
        "continuous_embedding_jobs",
        ["encoding_signature"],
    )
    op.create_index(
        "ix_continuous_embedding_jobs_status",
        "continuous_embedding_jobs",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_continuous_embedding_jobs_status",
        table_name="continuous_embedding_jobs",
    )
    op.drop_index(
        "ix_continuous_embedding_jobs_encoding_signature",
        table_name="continuous_embedding_jobs",
    )
    op.drop_table("continuous_embedding_jobs")
