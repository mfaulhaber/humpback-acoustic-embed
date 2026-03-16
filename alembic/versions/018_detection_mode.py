"""Add detection_mode column to detection_jobs

Revision ID: 018
Revises: 017
Create Date: 2026-03-16
"""

import sqlalchemy as sa
from alembic import op

revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(sa.Column("detection_mode", sa.String(), nullable=True))


def downgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("detection_mode")
