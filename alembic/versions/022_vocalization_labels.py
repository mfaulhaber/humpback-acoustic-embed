"""Add vocalization_labels table for extensible vocalization type labeling.

Revision ID: 022
Revises: 021
Create Date: 2026-03-23
"""

from alembic import op
import sqlalchemy as sa

revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "vocalization_labels",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("detection_job_id", sa.String(), nullable=False),
        sa.Column("row_id", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("source", sa.String(), nullable=False, server_default="manual"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_vocalization_labels_job_row",
        "vocalization_labels",
        ["detection_job_id", "row_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_vocalization_labels_job_row", table_name="vocalization_labels")
    op.drop_table("vocalization_labels")
