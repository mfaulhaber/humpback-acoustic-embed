"""Add warning_message to processing_jobs

Revision ID: 004
Revises: 003
Create Date: 2026-02-28
"""

from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "processing_jobs",
        sa.Column("warning_message", sa.Text(), nullable=True),
    )


def downgrade():
    op.drop_column("processing_jobs", "warning_message")
