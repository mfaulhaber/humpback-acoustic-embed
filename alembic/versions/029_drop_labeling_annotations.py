"""Drop labeling_annotations table (sub-window annotations replaced by window-level multi-hot labels).

Revision ID: 029
Revises: 028
Create Date: 2026-03-29
"""

from alembic import op
import sqlalchemy as sa

revision = "029"
down_revision = "028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("labeling_annotations")


def downgrade() -> None:
    op.create_table(
        "labeling_annotations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("detection_job_id", sa.String(), nullable=False),
        sa.Column("start_utc", sa.Float(), nullable=False),
        sa.Column("end_utc", sa.Float(), nullable=False),
        sa.Column("start_offset_sec", sa.Float(), nullable=False),
        sa.Column("end_offset_sec", sa.Float(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_labeling_annotations_job_utc",
        "labeling_annotations",
        ["detection_job_id", "start_utc", "end_utc"],
    )
