"""Remove retired HMM, masked-transformer, and motif tables.

Revision ID: 075
Revises: 074
Create Date: 2026-05-06

This is a destructive retirement migration. Alembic cannot restore dropped rows
or filesystem artifacts; rollback requires restoring the database backup taken
before upgrade.
"""

from __future__ import annotations

from alembic import op

revision = "075"
down_revision = "074"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("motif_extraction_jobs")
    op.drop_table("masked_transformer_job_sources")
    op.drop_table("masked_transformer_jobs")
    op.drop_table("hmm_sequence_jobs")


def downgrade() -> None:
    raise NotImplementedError(
        "Migration 075 destructively removes retired Sequence Models tables; "
        "restore the pre-upgrade database backup to roll back."
    )
