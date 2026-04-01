"""Add row_store_version to detection_jobs and row_store_version_at_import to vocalization_labels.

Revision ID: 033
Revises: 032
Create Date: 2026-04-01
"""

from alembic import op
import sqlalchemy as sa

revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "row_store_version",
                sa.Integer(),
                nullable=False,
                server_default="1",
            )
        )

    with op.batch_alter_table("vocalization_labels") as batch_op:
        batch_op.add_column(
            sa.Column(
                "row_store_version_at_import",
                sa.Integer(),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("vocalization_labels") as batch_op:
        batch_op.drop_column("row_store_version_at_import")

    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("row_store_version")
