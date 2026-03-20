"""Add label_processing_jobs table

Revision ID: 020
Revises: 019
Create Date: 2026-03-20
"""

import sqlalchemy as sa
from alembic import op

revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "label_processing_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("classifier_model_id", sa.String(), nullable=False),
        sa.Column("annotation_folder", sa.String(), nullable=False),
        sa.Column("audio_folder", sa.String(), nullable=False),
        sa.Column("output_root", sa.String(), nullable=False),
        sa.Column("parameters", sa.Text(), nullable=True),
        sa.Column("files_processed", sa.Integer(), nullable=True),
        sa.Column("files_total", sa.Integer(), nullable=True),
        sa.Column("annotations_total", sa.Integer(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
    )


def downgrade():
    op.drop_table("label_processing_jobs")
