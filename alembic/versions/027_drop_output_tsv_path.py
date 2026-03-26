"""Drop output_tsv_path from detection_jobs.

Revision ID: 027
Revises: 026
Create Date: 2026-03-25
"""

from alembic import op
import sqlalchemy as sa

revision = "027"
down_revision = "026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("output_tsv_path")


def downgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(sa.Column("output_tsv_path", sa.String(), nullable=True))
