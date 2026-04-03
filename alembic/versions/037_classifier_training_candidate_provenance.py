"""Add candidate-backed training provenance columns.

Revision ID: 037
Revises: 036
Create Date: 2026-04-03
"""

import sqlalchemy as sa
from alembic import op

revision = "037"
down_revision = "036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("classifier_training_jobs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "source_mode",
                sa.String(),
                nullable=False,
                server_default="embedding_sets",
            )
        )
        batch_op.add_column(
            sa.Column("source_candidate_id", sa.String(), nullable=True)
        )
        batch_op.add_column(sa.Column("source_model_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("manifest_path", sa.String(), nullable=True))
        batch_op.add_column(
            sa.Column("training_split_name", sa.String(), nullable=True)
        )
        batch_op.add_column(sa.Column("promoted_config", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("source_comparison_context", sa.Text(), nullable=True)
        )

    with op.batch_alter_table("classifier_models") as batch_op:
        batch_op.add_column(
            sa.Column(
                "training_source_mode",
                sa.String(),
                nullable=False,
                server_default="embedding_sets",
            )
        )
        batch_op.add_column(
            sa.Column("source_candidate_id", sa.String(), nullable=True)
        )
        batch_op.add_column(sa.Column("source_model_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("promotion_provenance", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("classifier_models") as batch_op:
        batch_op.drop_column("promotion_provenance")
        batch_op.drop_column("source_model_id")
        batch_op.drop_column("source_candidate_id")
        batch_op.drop_column("training_source_mode")

    with op.batch_alter_table("classifier_training_jobs") as batch_op:
        batch_op.drop_column("source_comparison_context")
        batch_op.drop_column("promoted_config")
        batch_op.drop_column("training_split_name")
        batch_op.drop_column("manifest_path")
        batch_op.drop_column("source_model_id")
        batch_op.drop_column("source_candidate_id")
        batch_op.drop_column("source_mode")
