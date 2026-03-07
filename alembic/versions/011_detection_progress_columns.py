"""Add files_processed and files_total to detection_jobs

Revision ID: 011
Revises: 010
Create Date: 2026-03-07
"""

import sqlalchemy as sa
from alembic import op

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(sa.Column("files_processed", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("files_total", sa.Integer(), nullable=True))


def downgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("files_total")
        batch_op.drop_column("files_processed")
