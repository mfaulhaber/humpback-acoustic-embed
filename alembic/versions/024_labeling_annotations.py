"""Add labeling_annotations table for sub-window annotations.

Revision ID: 024
Revises: 023
Create Date: 2026-03-23
"""

from alembic import op
import sqlalchemy as sa

revision = "024"
down_revision = "023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "labeling_annotations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("detection_job_id", sa.String(), nullable=False),
        sa.Column("row_id", sa.String(), nullable=False),
        sa.Column("start_offset_sec", sa.Float(), nullable=False),
        sa.Column("end_offset_sec", sa.Float(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_labeling_annotations_job_row",
        "labeling_annotations",
        ["detection_job_id", "row_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_labeling_annotations_job_row", table_name="labeling_annotations")
    op.drop_table("labeling_annotations")
