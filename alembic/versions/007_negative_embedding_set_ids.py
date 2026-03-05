"""Replace negative_audio_folder with negative_embedding_set_ids

Revision ID: 007
Revises: 006
Create Date: 2026-03-04
"""

import sqlalchemy as sa
from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("classifier_training_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("negative_embedding_set_ids", sa.Text(), nullable=True)
        )

    # Migrate existing data: set empty JSON array for any existing rows
    op.execute(
        "UPDATE classifier_training_jobs SET negative_embedding_set_ids = '[]' "
        "WHERE negative_embedding_set_ids IS NULL"
    )

    with op.batch_alter_table("classifier_training_jobs") as batch_op:
        batch_op.drop_column("negative_audio_folder")


def downgrade():
    with op.batch_alter_table("classifier_training_jobs") as batch_op:
        batch_op.add_column(sa.Column("negative_audio_folder", sa.String(), nullable=True))
        batch_op.drop_column("negative_embedding_set_ids")
