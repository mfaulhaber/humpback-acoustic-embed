"""Add window_selection and min_prominence columns to detection_jobs.

Revision ID: 038
Revises: 037
Create Date: 2026-04-04
"""

import sqlalchemy as sa
from alembic import op

revision = "038"
down_revision = "037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(sa.Column("window_selection", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("min_prominence", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("min_prominence")
        batch_op.drop_column("window_selection")
