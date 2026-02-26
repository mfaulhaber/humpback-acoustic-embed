"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-02-26
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "audio_files",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("checksum_sha256", sa.String(), nullable=False, unique=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("sample_rate_original", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "audio_metadata",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "audio_file_id",
            sa.String(),
            sa.ForeignKey("audio_files.id"),
            unique=True,
            nullable=False,
        ),
        sa.Column("tag_data", sa.Text(), nullable=True),
        sa.Column("visual_observations", sa.Text(), nullable=True),
        sa.Column("group_composition", sa.Text(), nullable=True),
        sa.Column("prey_density_proxy", sa.Text(), nullable=True),
    )
    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "audio_file_id",
            sa.String(),
            sa.ForeignKey("audio_files.id"),
            nullable=False,
        ),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("encoding_signature", sa.String(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("window_size_seconds", sa.Float(), nullable=False),
        sa.Column("target_sample_rate", sa.Integer(), nullable=False),
        sa.Column("feature_config", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "embedding_sets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "audio_file_id",
            sa.String(),
            sa.ForeignKey("audio_files.id"),
            nullable=False,
        ),
        sa.Column("encoding_signature", sa.String(), nullable=False, unique=True),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("window_size_seconds", sa.Float(), nullable=False),
        sa.Column("target_sample_rate", sa.Integer(), nullable=False),
        sa.Column("vector_dim", sa.Integer(), nullable=False),
        sa.Column("parquet_path", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "clustering_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("embedding_set_ids", sa.Text(), nullable=False),
        sa.Column("parameters", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "clusters",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "clustering_job_id",
            sa.String(),
            sa.ForeignKey("clustering_jobs.id"),
            nullable=False,
        ),
        sa.Column("cluster_label", sa.Integer(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("metadata_summary", sa.Text(), nullable=True),
    )
    op.create_table(
        "cluster_assignments",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "cluster_id",
            sa.String(),
            sa.ForeignKey("clusters.id"),
            nullable=False,
        ),
        sa.Column("embedding_set_id", sa.String(), nullable=False),
        sa.Column("embedding_row_index", sa.Integer(), nullable=False),
    )


def downgrade():
    op.drop_table("cluster_assignments")
    op.drop_table("clusters")
    op.drop_table("clustering_jobs")
    op.drop_table("embedding_sets")
    op.drop_table("processing_jobs")
    op.drop_table("audio_metadata")
    op.drop_table("audio_files")
