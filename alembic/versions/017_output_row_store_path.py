"""Add output_row_store_path column to detection_jobs

Revision ID: 017
Revises: 016
Create Date: 2026-03-15
"""

import sqlalchemy as sa
from alembic import op

revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("output_row_store_path", sa.String(), nullable=True)
        )


def downgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("output_row_store_path")
