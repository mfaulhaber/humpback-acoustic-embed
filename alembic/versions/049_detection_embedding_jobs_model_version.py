"""detection_embedding_jobs: add model_version + progress, relocate parquet.

Adds ``model_version`` (backfilled from the detection job's source classifier),
replaces single-column uniqueness on ``detection_job_id`` with a composite
unique ``(detection_job_id, model_version)``, and adds ``rows_processed`` /
``rows_total`` progress columns. Physically moves any existing detection
embeddings parquet from the legacy path to the new model-versioned path.

Revision ID: 049
Revises: 048
Create Date: 2026-04-17
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import sqlalchemy as sa
from alembic import op

from humpback.config import Settings

revision = "049"
down_revision = "048"
branch_labels = None
depends_on = None


def _legacy_path(storage_root: Path, detection_job_id: str) -> Path:
    return (
        storage_root / "detections" / detection_job_id / "detection_embeddings.parquet"
    )


def _new_path(storage_root: Path, detection_job_id: str, model_version: str) -> Path:
    return (
        storage_root
        / "detections"
        / detection_job_id
        / "embeddings"
        / model_version
        / "detection_embeddings.parquet"
    )


def upgrade() -> None:
    bind = op.get_bind()

    # 1. Add model_version nullable, rows_processed (default 0), rows_total nullable.
    with op.batch_alter_table("detection_embedding_jobs") as batch_op:
        batch_op.add_column(sa.Column("model_version", sa.String(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "rows_processed",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.add_column(sa.Column("rows_total", sa.Integer(), nullable=True))

    # 2. Backfill model_version from source classifier.
    bind.execute(
        sa.text(
            "UPDATE detection_embedding_jobs "
            "SET model_version = ("
            "    SELECT cm.model_version "
            "    FROM detection_jobs dj "
            "    JOIN classifier_models cm ON cm.id = dj.classifier_model_id "
            "    WHERE dj.id = detection_embedding_jobs.detection_job_id"
            ") "
            "WHERE model_version IS NULL"
        )
    )

    # Rows where the source classifier/detection job no longer exists cannot be
    # backfilled. Prune failed orphans; anything else is a data-integrity issue
    # the operator must resolve.
    bind.execute(
        sa.text(
            "DELETE FROM detection_embedding_jobs "
            "WHERE model_version IS NULL AND status = 'failed'"
        )
    )
    orphan_count = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM detection_embedding_jobs WHERE model_version IS NULL"
        )
    ).scalar()
    if orphan_count:
        msg = (
            f"Cannot backfill model_version for {orphan_count} detection_embedding_jobs "
            "rows (source classifier missing). Resolve manually before retrying."
        )
        raise RuntimeError(msg)

    # Collapse legacy duplicates: the pre-migration schema allowed multiple rows
    # per (detection_job_id) after repeated re-embedding attempts. Keep the most
    # recent row per (detection_job_id, model_version) so the new composite
    # unique constraint holds.
    bind.execute(
        sa.text(
            "DELETE FROM detection_embedding_jobs "
            "WHERE id NOT IN ("
            "    SELECT id FROM ("
            "        SELECT id, "
            "               ROW_NUMBER() OVER ("
            "                   PARTITION BY detection_job_id, model_version "
            "                   ORDER BY updated_at DESC, created_at DESC"
            "               ) AS rn "
            "        FROM detection_embedding_jobs"
            "    ) t WHERE t.rn = 1"
            ")"
        )
    )

    # 3. Physically move any legacy parquet files to the new path.
    storage_root_env = os.getenv("HUMPBACK_STORAGE_ROOT")
    if storage_root_env:
        storage_root = Path(storage_root_env)
    else:
        try:
            storage_root = Path(Settings().storage_root)
        except Exception:
            storage_root = None

    if storage_root is not None and storage_root.exists():
        rows = bind.execute(
            sa.text(
                "SELECT DISTINCT detection_job_id, model_version "
                "FROM detection_embedding_jobs"
            )
        ).fetchall()
        for det_job_id, mv in rows:
            legacy = _legacy_path(storage_root, det_job_id)
            if not legacy.exists():
                continue
            new = _new_path(storage_root, det_job_id, mv)
            new.parent.mkdir(parents=True, exist_ok=True)
            if not new.exists():
                shutil.move(str(legacy), str(new))

    # 4. Drop any existing unique/index on detection_job_id, alter to NOT NULL,
    #    and create the composite unique index.
    with op.batch_alter_table("detection_embedding_jobs") as batch_op:
        batch_op.alter_column(
            "model_version",
            existing_type=sa.String(),
            nullable=False,
        )
        # Composite unique index; names chosen to be stable across SQLite recreate.
        batch_op.create_unique_constraint(
            "uq_detection_embedding_jobs_det_job_model",
            ["detection_job_id", "model_version"],
        )


def downgrade() -> None:
    bind = op.get_bind()

    # Move files back to the legacy path (best effort).
    storage_root_env = os.getenv("HUMPBACK_STORAGE_ROOT")
    if storage_root_env:
        storage_root = Path(storage_root_env)
    else:
        try:
            storage_root = Path(Settings().storage_root)
        except Exception:
            storage_root = None

    if storage_root is not None and storage_root.exists():
        rows = bind.execute(
            sa.text(
                "SELECT DISTINCT detection_job_id, model_version "
                "FROM detection_embedding_jobs"
            )
        ).fetchall()
        for det_job_id, mv in rows:
            new = _new_path(storage_root, det_job_id, mv)
            if not new.exists():
                continue
            legacy = _legacy_path(storage_root, det_job_id)
            legacy.parent.mkdir(parents=True, exist_ok=True)
            if not legacy.exists():
                shutil.move(str(new), str(legacy))

    with op.batch_alter_table("detection_embedding_jobs") as batch_op:
        batch_op.drop_constraint(
            "uq_detection_embedding_jobs_det_job_model", type_="unique"
        )
        batch_op.drop_column("rows_total")
        batch_op.drop_column("rows_processed")
        batch_op.drop_column("model_version")
