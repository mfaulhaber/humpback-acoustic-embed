"""Add detection_embedding_jobs table for post-hoc embedding generation.

Revision ID: 031
Revises: 030
Create Date: 2026-03-29
"""

from alembic import op
import sqlalchemy as sa

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "detection_embedding_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("detection_job_id", sa.String(), nullable=False),
        sa.Column("progress_current", sa.Integer(), nullable=True),
        sa.Column("progress_total", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("detection_embedding_jobs")
