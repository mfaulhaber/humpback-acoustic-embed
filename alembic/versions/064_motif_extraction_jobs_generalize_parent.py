"""Generalize ``motif_extraction_jobs`` parent (ADR-061).

Adds ``parent_kind``, ``masked_transformer_job_id`` (nullable FK), and
``k`` (nullable). Makes ``hmm_sequence_job_id`` nullable. Adds a CHECK
constraint that enforces XOR between the two parent FKs and consistency
with ``parent_kind`` (``k`` is required iff
``parent_kind = 'masked_transformer'``). Existing rows backfill with
``parent_kind = 'hmm'``.

Revision ID: 064
Revises: 063
Create Date: 2026-05-01
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "064"
down_revision = "063"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns + relax existing FK nullability via batch_alter_table.
    with op.batch_alter_table("motif_extraction_jobs") as batch:
        batch.add_column(
            sa.Column(
                "parent_kind",
                sa.Text(),
                nullable=False,
                server_default="hmm",
            )
        )
        batch.add_column(
            sa.Column("masked_transformer_job_id", sa.String(), nullable=True)
        )
        batch.add_column(sa.Column("k", sa.Integer(), nullable=True))
        batch.alter_column(
            "hmm_sequence_job_id",
            existing_type=sa.String(),
            nullable=True,
        )

    # Backfill not strictly necessary because of the server_default, but
    # explicit for any pre-existing NULLs (sqlite would not have any).
    op.execute(
        "UPDATE motif_extraction_jobs SET parent_kind='hmm' "
        "WHERE parent_kind IS NULL OR parent_kind=''"
    )

    # Add CHECK + FK in a second pass so the backfill is committed first.
    with op.batch_alter_table("motif_extraction_jobs") as batch:
        batch.create_check_constraint(
            "ck_motif_extraction_jobs_parent_xor",
            "(parent_kind = 'hmm' AND hmm_sequence_job_id IS NOT NULL "
            "  AND masked_transformer_job_id IS NULL AND k IS NULL) "
            "OR (parent_kind = 'masked_transformer' AND hmm_sequence_job_id IS NULL "
            "  AND masked_transformer_job_id IS NOT NULL AND k IS NOT NULL)",
        )
        batch.create_foreign_key(
            "fk_motif_extraction_jobs_masked_transformer_job",
            "masked_transformer_jobs",
            ["masked_transformer_job_id"],
            ["id"],
        )

    op.create_index(
        "ix_motif_extraction_jobs_masked_transformer_job_id",
        "motif_extraction_jobs",
        ["masked_transformer_job_id"],
    )
    op.create_index(
        "ix_motif_extraction_jobs_parent_kind",
        "motif_extraction_jobs",
        ["parent_kind"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_motif_extraction_jobs_parent_kind",
        table_name="motif_extraction_jobs",
    )
    op.drop_index(
        "ix_motif_extraction_jobs_masked_transformer_job_id",
        table_name="motif_extraction_jobs",
    )
    with op.batch_alter_table("motif_extraction_jobs") as batch:
        batch.drop_constraint(
            "fk_motif_extraction_jobs_masked_transformer_job",
            type_="foreignkey",
        )
        batch.drop_constraint(
            "ck_motif_extraction_jobs_parent_xor",
            type_="check",
        )
        batch.alter_column(
            "hmm_sequence_job_id",
            existing_type=sa.String(),
            nullable=False,
        )
        batch.drop_column("k")
        batch.drop_column("masked_transformer_job_id")
        batch.drop_column("parent_kind")
