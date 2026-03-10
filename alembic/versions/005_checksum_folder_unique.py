"""Replace global checksum unique with (checksum, folder_path) composite unique

Revision ID: 005
Revises: 004
Create Date: 2026-03-02
"""

from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None

naming_convention = {
    "uq": "uq_%(table_name)s_%(column_0_name)s",
}


def upgrade():
    with op.batch_alter_table(
        "audio_files", naming_convention=naming_convention
    ) as batch_op:
        batch_op.drop_constraint("uq_audio_files_checksum_sha256", type_="unique")
        batch_op.create_unique_constraint(
            "uq_checksum_folder", ["checksum_sha256", "folder_path"]
        )


def downgrade():
    with op.batch_alter_table(
        "audio_files", naming_convention=naming_convention
    ) as batch_op:
        batch_op.drop_constraint("uq_checksum_folder", type_="unique")
        batch_op.create_unique_constraint(
            "uq_audio_files_checksum_sha256", ["checksum_sha256"]
        )
