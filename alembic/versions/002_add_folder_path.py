"""Add folder_path to audio_files

Revision ID: 002
Revises: 001
Create Date: 2026-02-27
"""

from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "audio_files",
        sa.Column("folder_path", sa.String(), nullable=False, server_default=""),
    )


def downgrade():
    op.drop_column("audio_files", "folder_path")
