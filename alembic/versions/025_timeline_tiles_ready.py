"""Add timeline_tiles_ready column to detection_jobs.

Revision ID: 025
Revises: 024
Create Date: 2026-03-24
"""

from alembic import op
import sqlalchemy as sa

revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("timeline_tiles_ready", sa.Boolean(), server_default="0")
        )


def downgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("timeline_tiles_ready")
