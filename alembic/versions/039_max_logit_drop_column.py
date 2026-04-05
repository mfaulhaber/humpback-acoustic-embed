"""Add max_logit_drop column to detection_jobs.

Revision ID: 039
Revises: 038
Create Date: 2026-04-04
"""

import sqlalchemy as sa
from alembic import op

revision = "039"
down_revision = "038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(sa.Column("max_logit_drop", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("max_logit_drop")
