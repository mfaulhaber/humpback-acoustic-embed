"""Add has_humpback_labels column to detection_jobs

Revision ID: 015
Revises: 014
Create Date: 2026-03-09
"""

import sqlalchemy as sa
from alembic import op

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("has_humpback_labels", sa.Boolean(), nullable=True)
        )


def downgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("has_humpback_labels")
