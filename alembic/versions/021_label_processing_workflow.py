"""Add workflow column and make classifier_model_id nullable.

Revision ID: 021
Revises: 020
Create Date: 2026-03-21
"""

from alembic import op
import sqlalchemy as sa

revision = "021"
down_revision = "020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("label_processing_jobs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "workflow", sa.String(), server_default="score_based", nullable=False
            )
        )
        # Make classifier_model_id nullable for sample_builder workflow
        batch_op.alter_column(
            "classifier_model_id",
            existing_type=sa.String(),
            nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("label_processing_jobs") as batch_op:
        batch_op.alter_column(
            "classifier_model_id",
            existing_type=sa.String(),
            nullable=False,
        )
        batch_op.drop_column("workflow")
