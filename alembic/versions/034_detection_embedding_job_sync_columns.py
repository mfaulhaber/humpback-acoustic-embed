"""Add mode and result_summary columns to detection_embedding_jobs.

Revision ID: 034
Revises: 033
Create Date: 2026-04-02
"""

from alembic import op
import sqlalchemy as sa

revision = "034"
down_revision = "033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("detection_embedding_jobs") as batch_op:
        batch_op.add_column(sa.Column("mode", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("result_summary", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("detection_embedding_jobs") as batch_op:
        batch_op.drop_column("result_summary")
        batch_op.drop_column("mode")
