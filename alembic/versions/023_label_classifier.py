"""Add classifier_purpose, job_purpose, and source_detection_job_ids columns.

Revision ID: 023
Revises: 022
Create Date: 2026-03-23
"""

from alembic import op
import sqlalchemy as sa

revision = "023"
down_revision = "022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("classifier_models") as batch_op:
        batch_op.add_column(
            sa.Column(
                "classifier_purpose",
                sa.String(),
                server_default="detection",
                nullable=False,
            )
        )

    with op.batch_alter_table("classifier_training_jobs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "job_purpose",
                sa.String(),
                server_default="detection",
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column("source_detection_job_ids", sa.Text(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("classifier_training_jobs") as batch_op:
        batch_op.drop_column("source_detection_job_ids")
        batch_op.drop_column("job_purpose")

    with op.batch_alter_table("classifier_models") as batch_op:
        batch_op.drop_column("classifier_purpose")
