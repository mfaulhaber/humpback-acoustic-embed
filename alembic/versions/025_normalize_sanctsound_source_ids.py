"""Normalize SanctSound umbrella source IDs in historical detection jobs.

Revision ID: 025
Revises: 024
Create Date: 2026-03-24
"""

from alembic import op

revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE detection_jobs
        SET hydrophone_id = 'sanctsound_ci',
            hydrophone_name = 'NOAA SanctSound (Channel Islands)'
        WHERE hydrophone_id = 'sanctsound_ci01'
        """
    )
    op.execute(
        """
        UPDATE detection_jobs
        SET hydrophone_id = 'sanctsound_oc',
            hydrophone_name = 'NOAA SanctSound (Olympic Coast)'
        WHERE hydrophone_id = 'sanctsound_oc01'
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE detection_jobs
        SET hydrophone_id = 'sanctsound_ci01',
            hydrophone_name = 'NOAA SanctSound (Channel Islands)'
        WHERE hydrophone_id = 'sanctsound_ci'
        """
    )
    op.execute(
        """
        UPDATE detection_jobs
        SET hydrophone_id = 'sanctsound_oc01',
            hydrophone_name = 'NOAA SanctSound (Olympic Coast)'
        WHERE hydrophone_id = 'sanctsound_oc'
        """
    )
