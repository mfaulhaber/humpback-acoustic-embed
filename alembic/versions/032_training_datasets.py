"""Add training_datasets and training_dataset_labels tables.

Revision ID: 032
Revises: 031
Create Date: 2026-03-31
"""

from alembic import op
import sqlalchemy as sa

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_datasets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("source_config", sa.Text(), nullable=False),
        sa.Column("parquet_path", sa.String(), nullable=False),
        sa.Column("total_rows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("vocabulary", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )

    op.create_table(
        "training_dataset_labels",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("training_dataset_id", sa.String(), nullable=False),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False, server_default="snapshot"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_tdl_dataset_row",
        "training_dataset_labels",
        ["training_dataset_id", "row_index"],
    )

    with op.batch_alter_table("vocalization_models") as batch_op:
        batch_op.add_column(
            sa.Column("training_dataset_id", sa.String(), nullable=True)
        )

    with op.batch_alter_table("vocalization_training_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("training_dataset_id", sa.String(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("vocalization_training_jobs") as batch_op:
        batch_op.drop_column("training_dataset_id")

    with op.batch_alter_table("vocalization_models") as batch_op:
        batch_op.drop_column("training_dataset_id")

    op.drop_index("ix_tdl_dataset_row", table_name="training_dataset_labels")
    op.drop_table("training_dataset_labels")
    op.drop_table("training_datasets")
