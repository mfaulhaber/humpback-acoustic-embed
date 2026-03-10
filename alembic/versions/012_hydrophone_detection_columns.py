"""Add hydrophone detection columns to detection_jobs

Revision ID: 012
Revises: 011
Create Date: 2026-03-08
"""

import sqlalchemy as sa
from alembic import op

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        # Make audio_folder nullable for hydrophone jobs
        batch_op.alter_column(
            "audio_folder",
            existing_type=sa.String(),
            nullable=True,
        )
        # Hydrophone identification
        batch_op.add_column(sa.Column("hydrophone_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("hydrophone_name", sa.String(), nullable=True))
        # Time range
        batch_op.add_column(sa.Column("start_timestamp", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("end_timestamp", sa.Float(), nullable=True))
        # Progress tracking for streaming
        batch_op.add_column(
            sa.Column("segments_processed", sa.Integer(), nullable=True)
        )
        batch_op.add_column(sa.Column("segments_total", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("time_covered_sec", sa.Float(), nullable=True))
        # Alerts (JSON array)
        batch_op.add_column(sa.Column("alerts", sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("alerts")
        batch_op.drop_column("time_covered_sec")
        batch_op.drop_column("segments_total")
        batch_op.drop_column("segments_processed")
        batch_op.drop_column("end_timestamp")
        batch_op.drop_column("start_timestamp")
        batch_op.drop_column("hydrophone_name")
        batch_op.drop_column("hydrophone_id")
        batch_op.alter_column(
            "audio_folder",
            existing_type=sa.String(),
            nullable=False,
        )
