"""Add detection_job_ids column to clustering_jobs.

Revision ID: 056
Revises: 055
"""

from alembic import op

import sqlalchemy as sa

revision = "056"
down_revision = "055"


def upgrade() -> None:
    with op.batch_alter_table("clustering_jobs") as batch_op:
        batch_op.add_column(sa.Column("detection_job_ids", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("clustering_jobs") as batch_op:
        batch_op.drop_column("detection_job_ids")
