"""Replace call-parsing ``audio_source_id`` placeholder with proper source columns.

Phase 0 gave ``call_parsing_runs`` and ``region_detection_jobs`` a single
``audio_source_id: String (not null)`` placeholder column. Pass 1 accepts
two distinct source shapes — an uploaded audio file, or a hydrophone time
range — so we drop the placeholder and add the four real columns:
``audio_file_id``, ``hydrophone_id``, ``start_timestamp``, ``end_timestamp``.

Exactly-one-of validation lives in the Pydantic request model and the
service layer, matching the project's existing ``DetectionJob`` pattern.
No DB CHECK constraint.

Revision ID: 043
Revises: 042
Create Date: 2026-04-11
"""

import sqlalchemy as sa
from alembic import op

revision = "043"
down_revision = "042"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the index that Phase 0 created on the placeholder column before
    # batch_alter_table reflects the table — the column itself is about to
    # go, and letting the reflected metadata carry a stale index into the
    # batch copy causes a SQLite recreate to fail.
    op.drop_index(
        "ix_call_parsing_runs_audio_source_id",
        table_name="call_parsing_runs",
    )

    with op.batch_alter_table("call_parsing_runs") as batch_op:
        batch_op.drop_column("audio_source_id")
        batch_op.add_column(sa.Column("audio_file_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("hydrophone_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("start_timestamp", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("end_timestamp", sa.Float(), nullable=True))

    with op.batch_alter_table("region_detection_jobs") as batch_op:
        batch_op.drop_column("audio_source_id")
        batch_op.add_column(sa.Column("audio_file_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("hydrophone_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("start_timestamp", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("end_timestamp", sa.Float(), nullable=True))


def downgrade() -> None:
    # Restore ``audio_source_id`` as nullable (not NOT NULL) so the
    # downgrade does not require backfilling a placeholder value on
    # existing rows. The Phase 0 migration's NOT NULL constraint was only
    # reachable from a freshly-created DB.
    with op.batch_alter_table("region_detection_jobs") as batch_op:
        batch_op.drop_column("end_timestamp")
        batch_op.drop_column("start_timestamp")
        batch_op.drop_column("hydrophone_id")
        batch_op.drop_column("audio_file_id")
        batch_op.add_column(sa.Column("audio_source_id", sa.String(), nullable=True))

    with op.batch_alter_table("call_parsing_runs") as batch_op:
        batch_op.drop_column("end_timestamp")
        batch_op.drop_column("start_timestamp")
        batch_op.drop_column("hydrophone_id")
        batch_op.drop_column("audio_file_id")
        batch_op.add_column(sa.Column("audio_source_id", sa.String(), nullable=True))

    op.create_index(
        "ix_call_parsing_runs_audio_source_id",
        "call_parsing_runs",
        ["audio_source_id"],
    )
