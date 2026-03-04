"""Add refined_from_job_id to clustering_jobs

Revision ID: 006
Revises: 005
Create Date: 2026-03-04
"""

import sqlalchemy as sa
from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("clustering_jobs") as batch_op:
        batch_op.add_column(sa.Column("refined_from_job_id", sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table("clustering_jobs") as batch_op:
        batch_op.drop_column("refined_from_job_id")
