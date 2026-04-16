"""Record compute device on Pass 2 and Pass 3 job rows.

Adds ``compute_device`` and ``gpu_fallback_reason`` columns to both
``event_segmentation_jobs`` and ``event_classification_jobs`` so the
worker can persist the device chosen at job start (after load-time
validation) and surface any fallback reason to the UI.

Revision ID: 048
Revises: 047
Create Date: 2026-04-16
"""

import sqlalchemy as sa
from alembic import op

revision = "048"
down_revision = "047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("event_segmentation_jobs") as batch_op:
        batch_op.add_column(sa.Column("compute_device", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("gpu_fallback_reason", sa.Text(), nullable=True))

    with op.batch_alter_table("event_classification_jobs") as batch_op:
        batch_op.add_column(sa.Column("compute_device", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("gpu_fallback_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("event_classification_jobs") as batch_op:
        batch_op.drop_column("gpu_fallback_reason")
        batch_op.drop_column("compute_device")

    with op.batch_alter_table("event_segmentation_jobs") as batch_op:
        batch_op.drop_column("gpu_fallback_reason")
        batch_op.drop_column("compute_device")
