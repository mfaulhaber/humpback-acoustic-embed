"""Add model_type and input_format to model_configs

Revision ID: 003
Revises: 002
Create Date: 2026-02-27
"""

from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "model_configs",
        sa.Column("model_type", sa.String(), nullable=False, server_default="tflite"),
    )
    op.add_column(
        "model_configs",
        sa.Column(
            "input_format", sa.String(), nullable=False, server_default="spectrogram"
        ),
    )


def downgrade():
    op.drop_column("model_configs", "input_format")
    op.drop_column("model_configs", "model_type")
