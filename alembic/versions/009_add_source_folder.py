"""Add source_folder column to audio_files

Revision ID: 009
Revises: 008
Create Date: 2026-03-06
"""

import sqlalchemy as sa
from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("audio_files") as batch_op:
        batch_op.add_column(sa.Column("source_folder", sa.String(), nullable=True))


def downgrade():
    with op.batch_alter_table("audio_files") as batch_op:
        batch_op.drop_column("source_folder")
