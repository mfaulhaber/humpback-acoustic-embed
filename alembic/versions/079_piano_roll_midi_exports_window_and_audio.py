"""Extend piano_roll_midi_exports for windowed bundled MIDI + FLAC exports.

Adds the window bounds (``window_start_utc`` / ``window_end_utc``) and the
co-exported FLAC artifact's metadata columns. All new columns are NOT NULL.

Because the existing rows from the canonical-per-job era have no window
bounds and no FLAC artifact, the upgrade drops all existing rows first and
removes any matching on-disk ``.mid`` files under
``<storage_root>/exports/event_encoders/*/notes_*.mid``. The storage root is
read from runtime settings; deletions verify path containment before
unlinking.

Revision ID: 079
Revises: 078
Create Date: 2026-05-21
"""

from __future__ import annotations

import logging
from pathlib import Path

import sqlalchemy as sa
from alembic import op

revision = "079"
down_revision = "078"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.revision.079")


def _exports_dir_under_storage_root() -> Path | None:
    """Resolve ``<storage_root>/exports/event_encoders`` if storage is configured.

    Returns ``None`` when settings can't be loaded (test environments that
    skip runtime configuration), so the migration still proceeds with the
    row-only cleanup.
    """
    try:
        from humpback.config import Settings

        settings = Settings.from_repo_env()
        return Path(settings.storage_root) / "exports" / "event_encoders"
    except Exception:
        return None


def _delete_legacy_midi_files() -> None:
    base = _exports_dir_under_storage_root()
    if base is None or not base.exists():
        return

    base_resolved = base.resolve()
    for mid_path in base.glob("*/notes_*.mid"):
        try:
            resolved = mid_path.resolve()
            resolved.relative_to(base_resolved)
        except ValueError:
            logger.warning("skipping legacy MIDI outside storage root: %s", mid_path)
            continue
        try:
            resolved.unlink()
        except OSError as exc:
            logger.warning("could not remove legacy MIDI %s: %s", resolved, exc)


def upgrade() -> None:
    op.execute("DELETE FROM piano_roll_midi_exports")
    _delete_legacy_midi_files()

    with op.batch_alter_table("piano_roll_midi_exports") as batch_op:
        batch_op.add_column(sa.Column("window_start_utc", sa.Float(), nullable=False))
        batch_op.add_column(sa.Column("window_end_utc", sa.Float(), nullable=False))
        batch_op.add_column(sa.Column("audio_path", sa.Text(), nullable=False))
        batch_op.add_column(sa.Column("audio_size_bytes", sa.Integer(), nullable=False))
        batch_op.add_column(
            sa.Column("audio_sample_rate", sa.Integer(), nullable=False)
        )
        batch_op.add_column(sa.Column("audio_duration_s", sa.Float(), nullable=False))
        batch_op.create_check_constraint(
            "ck_piano_roll_midi_exports_window_positive",
            "window_end_utc > window_start_utc",
        )


def downgrade() -> None:
    with op.batch_alter_table("piano_roll_midi_exports") as batch_op:
        batch_op.drop_constraint(
            "ck_piano_roll_midi_exports_window_positive", type_="check"
        )
        batch_op.drop_column("audio_duration_s")
        batch_op.drop_column("audio_sample_rate")
        batch_op.drop_column("audio_size_bytes")
        batch_op.drop_column("audio_path")
        batch_op.drop_column("window_end_utc")
        batch_op.drop_column("window_start_utc")
