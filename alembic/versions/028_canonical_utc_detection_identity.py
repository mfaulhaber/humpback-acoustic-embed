"""Canonical UTC detection identity for vocalization_labels, labeling_annotations, search_jobs.

Replaces row_id / filename+start_sec+end_sec with start_utc/end_utc float pairs.

Revision ID: 028
Revises: 027
Create Date: 2026-03-28
"""

from alembic import op
import sqlalchemy as sa

revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- vocalization_labels: row_id -> start_utc + end_utc ---
    with op.batch_alter_table("vocalization_labels") as batch_op:
        batch_op.add_column(sa.Column("start_utc", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("end_utc", sa.Float(), nullable=True))
        batch_op.drop_index("ix_vocalization_labels_job_row")
        batch_op.drop_column("row_id")
        batch_op.create_index(
            "ix_vocalization_labels_job_utc",
            ["detection_job_id", "start_utc", "end_utc"],
        )

    # Default orphaned rows to 0.0 (labels without a row store match)
    op.execute("UPDATE vocalization_labels SET start_utc = 0.0 WHERE start_utc IS NULL")
    op.execute("UPDATE vocalization_labels SET end_utc = 0.0 WHERE end_utc IS NULL")

    with op.batch_alter_table("vocalization_labels") as batch_op:
        batch_op.alter_column("start_utc", nullable=False)
        batch_op.alter_column("end_utc", nullable=False)

    # --- labeling_annotations: row_id -> start_utc + end_utc ---
    with op.batch_alter_table("labeling_annotations") as batch_op:
        batch_op.add_column(sa.Column("start_utc", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("end_utc", sa.Float(), nullable=True))
        batch_op.drop_index("ix_labeling_annotations_job_row")
        batch_op.drop_column("row_id")
        batch_op.create_index(
            "ix_labeling_annotations_job_utc",
            ["detection_job_id", "start_utc", "end_utc"],
        )

    op.execute(
        "UPDATE labeling_annotations SET start_utc = 0.0 WHERE start_utc IS NULL"
    )
    op.execute("UPDATE labeling_annotations SET end_utc = 0.0 WHERE end_utc IS NULL")

    with op.batch_alter_table("labeling_annotations") as batch_op:
        batch_op.alter_column("start_utc", nullable=False)
        batch_op.alter_column("end_utc", nullable=False)

    # --- search_jobs: filename+start_sec+end_sec -> start_utc+end_utc ---
    with op.batch_alter_table("search_jobs") as batch_op:
        batch_op.add_column(sa.Column("start_utc", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("end_utc", sa.Float(), nullable=True))
        batch_op.drop_column("filename")
        batch_op.drop_column("start_sec")
        batch_op.drop_column("end_sec")

    # Search jobs are ephemeral (deleted after results); default any survivors.
    op.execute("UPDATE search_jobs SET start_utc = 0.0 WHERE start_utc IS NULL")
    op.execute("UPDATE search_jobs SET end_utc = 0.0 WHERE end_utc IS NULL")

    with op.batch_alter_table("search_jobs") as batch_op:
        batch_op.alter_column("start_utc", nullable=False)
        batch_op.alter_column("end_utc", nullable=False)


def downgrade() -> None:
    # --- search_jobs ---
    with op.batch_alter_table("search_jobs") as batch_op:
        batch_op.add_column(sa.Column("filename", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("start_sec", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("end_sec", sa.Float(), nullable=True))
        batch_op.drop_column("start_utc")
        batch_op.drop_column("end_utc")

    # --- labeling_annotations ---
    with op.batch_alter_table("labeling_annotations") as batch_op:
        batch_op.drop_index("ix_labeling_annotations_job_utc")
        batch_op.add_column(sa.Column("row_id", sa.String(), nullable=True))
        batch_op.drop_column("start_utc")
        batch_op.drop_column("end_utc")
        batch_op.create_index(
            "ix_labeling_annotations_job_row", ["detection_job_id", "row_id"]
        )

    # --- vocalization_labels ---
    with op.batch_alter_table("vocalization_labels") as batch_op:
        batch_op.drop_index("ix_vocalization_labels_job_utc")
        batch_op.add_column(sa.Column("row_id", sa.String(), nullable=True))
        batch_op.drop_column("start_utc")
        batch_op.drop_column("end_utc")
        batch_op.create_index(
            "ix_vocalization_labels_job_row", ["detection_job_id", "row_id"]
        )
