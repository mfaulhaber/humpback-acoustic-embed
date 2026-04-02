"""Add row_id to vocalization_labels, drop timestamp identity columns and row_store_version.

Revision ID: 035
Revises: 034
Create Date: 2026-04-02
"""

import logging
import os
from pathlib import Path

import sqlalchemy as sa
from alembic import op

revision = "035"
down_revision = "034"
branch_labels = None
depends_on = None

logger = logging.getLogger(__name__)

_MATCH_TOLERANCE_SEC = 0.5


def _match_labels_to_row_store(connection: sa.Connection) -> None:
    """Assign row_id to existing vocalization labels by matching against row stores."""
    storage_root = Path(os.environ.get("HUMPBACK_STORAGE_ROOT", "data"))

    # Get all distinct detection_job_ids that have vocalization labels.
    result = connection.execute(
        sa.text("SELECT DISTINCT detection_job_id FROM vocalization_labels")
    )
    job_ids = [row[0] for row in result]

    if not job_ids:
        logger.info("No vocalization labels to migrate.")
        return

    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not available; skipping row_id assignment for labels.")
        return

    total_matched = 0
    total_orphaned = 0

    for job_id in job_ids:
        rs_path = storage_root / "detections" / job_id / "detection_rows.parquet"
        if not rs_path.is_file():
            logger.info("No row store for job %s; deleting its labels.", job_id)
            connection.execute(
                sa.text(
                    "DELETE FROM vocalization_labels WHERE detection_job_id = :jid"
                ),
                {"jid": job_id},
            )
            continue

        # Read row store to get (start_utc, end_utc, row_id) tuples.
        table = pq.read_table(str(rs_path))
        col_names = set(table.column_names)
        if "start_utc" not in col_names:
            logger.info("Legacy row store for job %s; skipping label matching.", job_id)
            connection.execute(
                sa.text(
                    "DELETE FROM vocalization_labels WHERE detection_job_id = :jid"
                ),
                {"jid": job_id},
            )
            continue

        columns = table.to_pydict()
        rs_entries: list[tuple[float, float, str]] = []
        for i in range(table.num_rows):
            s = columns["start_utc"][i]
            e = columns["end_utc"][i]
            rid = columns.get("row_id", [None] * table.num_rows)[i]
            if s is None or e is None:
                continue
            try:
                rs_entries.append((float(s), float(e), rid or ""))
            except (ValueError, TypeError):
                continue

        # Get labels for this job.
        labels = connection.execute(
            sa.text(
                "SELECT id, start_utc, end_utc FROM vocalization_labels "
                "WHERE detection_job_id = :jid"
            ),
            {"jid": job_id},
        ).fetchall()

        matched_ids: list[tuple[str, str]] = []  # (label_id, row_id)
        orphan_ids: list[str] = []

        for label_id, label_start, label_end in labels:
            found = False
            for rs_start, rs_end, rs_row_id in rs_entries:
                if (
                    abs(label_start - rs_start) <= _MATCH_TOLERANCE_SEC
                    and abs(label_end - rs_end) <= _MATCH_TOLERANCE_SEC
                ):
                    matched_ids.append((label_id, rs_row_id))
                    found = True
                    break
            if not found:
                orphan_ids.append(label_id)

        # Apply matches.
        for label_id, row_id in matched_ids:
            connection.execute(
                sa.text("UPDATE vocalization_labels SET row_id = :rid WHERE id = :lid"),
                {"rid": row_id, "lid": label_id},
            )
        total_matched += len(matched_ids)

        # Delete orphans.
        for label_id in orphan_ids:
            connection.execute(
                sa.text("DELETE FROM vocalization_labels WHERE id = :lid"),
                {"lid": label_id},
            )
        total_orphaned += len(orphan_ids)

    logger.info(
        "Label migration: %d matched, %d orphaned/deleted.",
        total_matched,
        total_orphaned,
    )


def upgrade() -> None:
    # Phase 1: Add row_id column (nullable) to vocalization_labels.
    with op.batch_alter_table("vocalization_labels") as batch_op:
        batch_op.add_column(sa.Column("row_id", sa.String(), nullable=True))

    # Phase 2: Populate row_id from row store matching.
    connection = op.get_bind()
    _match_labels_to_row_store(connection)

    # Delete any labels that still have NULL row_id (unmatched).
    connection.execute(sa.text("DELETE FROM vocalization_labels WHERE row_id IS NULL"))

    # Phase 3: Recreate vocalization_labels with final schema.
    with op.batch_alter_table("vocalization_labels") as batch_op:
        batch_op.alter_column("row_id", nullable=False)
        batch_op.drop_column("start_utc")
        batch_op.drop_column("end_utc")
        batch_op.drop_column("row_store_version_at_import")
        batch_op.drop_index("ix_vocalization_labels_job_utc")
        batch_op.create_index(
            "ix_vocalization_labels_job_row_id",
            ["detection_job_id", "row_id"],
        )

    # Phase 4: Drop row_store_version from detection_jobs.
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("row_store_version")


def downgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "row_store_version", sa.Integer(), nullable=False, server_default="1"
            )
        )

    with op.batch_alter_table("vocalization_labels") as batch_op:
        batch_op.add_column(sa.Column("start_utc", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("end_utc", sa.Float(), nullable=True))
        batch_op.add_column(
            sa.Column("row_store_version_at_import", sa.Integer(), nullable=True)
        )
        batch_op.drop_index("ix_vocalization_labels_job_row_id")
        batch_op.create_index(
            "ix_vocalization_labels_job_utc",
            ["detection_job_id", "start_utc", "end_utc"],
        )
        batch_op.drop_column("row_id")
