"""hyperparameter_manifests: add embedding_model_version.

Adds ``embedding_model_version`` (NOT NULL after backfill). Backfill strategy:

- For manifests whose first training_job_id source resolves, pull the
  training job's ``model_version``.
- Otherwise for manifests whose first detection_job_id source resolves,
  pull the source classifier's ``model_version``.
- Rows where neither source resolves are tagged ``unknown`` so the NOT NULL
  constraint can hold; operators should rebuild those manifests.

Revision ID: 050
Revises: 049
Create Date: 2026-04-17
"""

from __future__ import annotations

import json

import sqlalchemy as sa
from alembic import op

revision = "050"
down_revision = "049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()

    with op.batch_alter_table("hyperparameter_manifests") as batch_op:
        batch_op.add_column(
            sa.Column("embedding_model_version", sa.String(), nullable=True)
        )

    rows = bind.execute(
        sa.text(
            "SELECT id, training_job_ids, detection_job_ids "
            "FROM hyperparameter_manifests"
        )
    ).fetchall()

    for row_id, tjids_json, djids_json in rows:
        tjids = json.loads(tjids_json) if tjids_json else []
        djids = json.loads(djids_json) if djids_json else []
        resolved: str | None = None

        for tjid in tjids:
            tj = bind.execute(
                sa.text(
                    "SELECT model_version FROM classifier_training_jobs WHERE id = :id"
                ),
                {"id": str(tjid)},
            ).fetchone()
            if tj and tj[0]:
                resolved = str(tj[0])
                break

        if resolved is None:
            for djid in djids:
                dj = bind.execute(
                    sa.text(
                        "SELECT cm.model_version "
                        "FROM detection_jobs dj "
                        "JOIN classifier_models cm ON cm.id = dj.classifier_model_id "
                        "WHERE dj.id = :id"
                    ),
                    {"id": str(djid)},
                ).fetchone()
                if dj and dj[0]:
                    resolved = str(dj[0])
                    break

        if resolved is None:
            resolved = "unknown"

        bind.execute(
            sa.text(
                "UPDATE hyperparameter_manifests "
                "SET embedding_model_version = :mv WHERE id = :id"
            ),
            {"mv": resolved, "id": row_id},
        )

    with op.batch_alter_table("hyperparameter_manifests") as batch_op:
        batch_op.alter_column(
            "embedding_model_version",
            existing_type=sa.String(),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("hyperparameter_manifests") as batch_op:
        batch_op.drop_column("embedding_model_version")
