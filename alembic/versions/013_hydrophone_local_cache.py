"""Add local_cache_path to detection_jobs for local HLS cache support

Revision ID: 013
Revises: 012
Create Date: 2026-03-08
"""

import sqlalchemy as sa
from alembic import op

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(sa.Column("local_cache_path", sa.String(), nullable=True))


def downgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("local_cache_path")
