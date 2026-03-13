"""Rename has_humpback_labels to has_positive_labels on detection_jobs

Revision ID: 016
Revises: 015
Create Date: 2026-03-12
"""

import sqlalchemy as sa
from alembic import op

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.alter_column(
            "has_humpback_labels",
            new_column_name="has_positive_labels",
            existing_type=sa.Boolean(),
            existing_nullable=True,
        )


def downgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.alter_column(
            "has_positive_labels",
            new_column_name="has_humpback_labels",
            existing_type=sa.Boolean(),
            existing_nullable=True,
        )
