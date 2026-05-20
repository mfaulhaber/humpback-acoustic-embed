"""Add piano_roll_notes_jobs table.

Tracks per-Event-Encoder-job MIDI note extraction runs. Idempotent on
``(event_encoder_job_id, extractor_version)``. The worker writes an
``event_notes_v{N}.parquet`` sidecar under the Event Encoder's job directory
on success.

Revision ID: 077
Revises: 076
Create Date: 2026-05-20
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "077"
down_revision = "076"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "piano_roll_notes_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("event_encoder_job_id", sa.String(), nullable=False),
        sa.Column(
            "extractor_version",
            sa.String(),
            nullable=False,
            server_default="v1",
        ),
        sa.Column(
            "status",
            sa.String(),
            nullable=False,
            server_default="queued",
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("notes_path", sa.Text(), nullable=True),
        sa.Column("n_events", sa.Integer(), nullable=True),
        sa.Column("n_notes", sa.Integer(), nullable=True),
        sa.Column("compute_seconds", sa.Float(), nullable=True),
        sa.Column("params_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "event_encoder_job_id",
            "extractor_version",
            name="uq_piano_roll_notes_jobs_encoder_version",
        ),
        sa.ForeignKeyConstraint(
            ["event_encoder_job_id"],
            ["event_encoder_jobs.id"],
            name="fk_piano_roll_notes_jobs_event_encoder_job",
        ),
    )
    op.create_index(
        "ix_piano_roll_notes_jobs_event_encoder_job_id",
        "piano_roll_notes_jobs",
        ["event_encoder_job_id"],
    )
    op.create_index(
        "ix_piano_roll_notes_jobs_status",
        "piano_roll_notes_jobs",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_piano_roll_notes_jobs_status",
        table_name="piano_roll_notes_jobs",
    )
    op.drop_index(
        "ix_piano_roll_notes_jobs_event_encoder_job_id",
        table_name="piano_roll_notes_jobs",
    )
    op.drop_table("piano_roll_notes_jobs")
