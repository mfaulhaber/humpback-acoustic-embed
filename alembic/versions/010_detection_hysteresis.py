"""Add hop_seconds, high_threshold, low_threshold to detection_jobs

Revision ID: 010
Revises: 009
Create Date: 2026-03-07
"""

import sqlalchemy as sa
from alembic import op

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("hop_seconds", sa.Float(), nullable=True, server_default="1.0")
        )
        batch_op.add_column(
            sa.Column(
                "high_threshold", sa.Float(), nullable=True, server_default="0.70"
            )
        )
        batch_op.add_column(
            sa.Column("low_threshold", sa.Float(), nullable=True, server_default="0.45")
        )


def downgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("low_threshold")
        batch_op.drop_column("high_threshold")
        batch_op.drop_column("hop_seconds")
