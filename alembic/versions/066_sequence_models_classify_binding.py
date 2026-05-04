"""Bind Sequence Models jobs to an EventClassificationJob for label source.

Adds nullable ``event_classification_job_id`` FK columns to
``hmm_sequence_jobs`` and ``masked_transformer_jobs`` so each Sequence
Models job records which Pass 3 Classify run feeds its
``label_distribution.json`` and exemplar annotations.

The column is nullable for the in-transaction window between row insert
and FK resolution; in committed state it is non-NULL after the
submit-validation step. ``ondelete="RESTRICT"`` prevents deleting a
Classify job that is still bound by an HMM/MT job.

Revision ID: 066
Revises: 065
Create Date: 2026-05-04
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "066"
down_revision = "065"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("hmm_sequence_jobs") as batch:
        batch.add_column(
            sa.Column(
                "event_classification_job_id",
                sa.String(),
                sa.ForeignKey(
                    "event_classification_jobs.id",
                    name="fk_event_classification_job_id",
                    ondelete="RESTRICT",
                ),
                nullable=True,
            )
        )

    op.create_index(
        "ix_hmm_sequence_jobs_event_classification_job_id",
        "hmm_sequence_jobs",
        ["event_classification_job_id"],
    )

    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.add_column(
            sa.Column(
                "event_classification_job_id",
                sa.String(),
                sa.ForeignKey(
                    "event_classification_jobs.id",
                    name="fk_event_classification_job_id",
                    ondelete="RESTRICT",
                ),
                nullable=True,
            )
        )

    op.create_index(
        "ix_masked_transformer_jobs_event_classification_job_id",
        "masked_transformer_jobs",
        ["event_classification_job_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_masked_transformer_jobs_event_classification_job_id",
        table_name="masked_transformer_jobs",
    )
    with op.batch_alter_table("masked_transformer_jobs") as batch:
        batch.drop_column("event_classification_job_id")

    op.drop_index(
        "ix_hmm_sequence_jobs_event_classification_job_id",
        table_name="hmm_sequence_jobs",
    )
    with op.batch_alter_table("hmm_sequence_jobs") as batch:
        batch.drop_column("event_classification_job_id")
