"""Add extraction columns to detection_jobs

Revision ID: 008
Revises: 007
Create Date: 2026-03-05
"""

import sqlalchemy as sa
from alembic import op

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(sa.Column("extract_status", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("extract_error", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("extract_summary", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("extract_config", sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("extract_config")
        batch_op.drop_column("extract_summary")
        batch_op.drop_column("extract_error")
        batch_op.drop_column("extract_status")
