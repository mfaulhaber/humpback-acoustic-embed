"""Add search_jobs table

Revision ID: 019
Revises: 018
Create Date: 2026-03-18
"""

import sqlalchemy as sa
from alembic import op

revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "search_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("detection_job_id", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("start_sec", sa.Float(), nullable=False),
        sa.Column("end_sec", sa.Float(), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False, server_default="20"),
        sa.Column("metric", sa.String(), nullable=False, server_default="cosine"),
        sa.Column("embedding_set_ids", sa.Text(), nullable=True),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("embedding_vector", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )


def downgrade():
    op.drop_table("search_jobs")
