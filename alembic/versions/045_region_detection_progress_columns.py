"""Add progress tracking columns to region_detection_jobs.

Adds ``chunks_total`` and ``chunks_completed`` nullable integer columns
for cheap API polling of hydrophone streaming progress. File-based jobs
leave both columns NULL.

Revision ID: 045
Revises: 044
Create Date: 2026-04-12
"""

import sqlalchemy as sa
from alembic import op

revision = "045"
down_revision = "044"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("region_detection_jobs") as batch_op:
        batch_op.add_column(sa.Column("chunks_total", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("chunks_completed", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("region_detection_jobs") as batch_op:
        batch_op.drop_column("chunks_completed")
        batch_op.drop_column("chunks_total")
