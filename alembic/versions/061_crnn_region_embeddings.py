"""Add CRNN region-embedding columns and HMM training-mode columns.

Adds nullable columns to ``continuous_embedding_jobs`` (CRNN
region-based embedding source) and ``hmm_sequence_jobs`` (training-mode
+ tier configuration) per the CRNN region-HMM design spec
(``docs/specs/2026-04-29-crnn-region-hmm-design.md``). Existing rows
populated by the SurfPerch event-padded path keep all new columns NULL.

Revision ID: 061
Revises: 060
Create Date: 2026-04-29
"""

import sqlalchemy as sa
from alembic import op

revision = "061"
down_revision = "060"
branch_labels = None
depends_on = None


_DEFAULT_PROPORTIONS_JSON = (
    '{"event_core": 0.4, "near_event": 0.35, "background": 0.25}'
)


def upgrade() -> None:
    with op.batch_alter_table("continuous_embedding_jobs") as batch_op:
        # SurfPerch-only fields become nullable in interpretation: CRNN
        # region-source jobs do not populate them.
        batch_op.alter_column(
            "window_size_seconds",
            existing_type=sa.Float(),
            nullable=True,
        )
        batch_op.alter_column(
            "hop_seconds",
            existing_type=sa.Float(),
            nullable=True,
        )
        batch_op.alter_column(
            "pad_seconds",
            existing_type=sa.Float(),
            nullable=True,
        )
        batch_op.add_column(
            sa.Column("region_detection_job_id", sa.String(), nullable=True),
        )
        batch_op.create_foreign_key(
            "fk_continuous_embedding_jobs_region_job",
            "region_detection_jobs",
            ["region_detection_job_id"],
            ["id"],
        )
        batch_op.add_column(sa.Column("chunk_size_seconds", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("chunk_hop_seconds", sa.Float(), nullable=True))
        batch_op.add_column(
            sa.Column("crnn_checkpoint_sha256", sa.Text(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("crnn_segmentation_model_id", sa.String(), nullable=True),
        )
        batch_op.create_foreign_key(
            "fk_continuous_embedding_jobs_seg_model",
            "segmentation_models",
            ["crnn_segmentation_model_id"],
            ["id"],
        )
        batch_op.add_column(sa.Column("projection_kind", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("projection_dim", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("total_regions", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("total_chunks", sa.Integer(), nullable=True))

    with op.batch_alter_table("hmm_sequence_jobs") as batch_op:
        batch_op.add_column(sa.Column("training_mode", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "event_core_overlap_threshold",
                sa.Float(),
                nullable=True,
                server_default=sa.text("0.5"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "near_event_window_seconds",
                sa.Float(),
                nullable=True,
                server_default=sa.text("5.0"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "event_balanced_proportions",
                sa.Text(),
                nullable=True,
                server_default=sa.text(f"'{_DEFAULT_PROPORTIONS_JSON}'"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "subsequence_length_chunks",
                sa.Integer(),
                nullable=True,
                server_default=sa.text("32"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "subsequence_stride_chunks",
                sa.Integer(),
                nullable=True,
                server_default=sa.text("16"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "target_train_chunks",
                sa.Integer(),
                nullable=True,
                server_default=sa.text("200000"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "min_region_length_seconds",
                sa.Float(),
                nullable=True,
                server_default=sa.text("2.0"),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("hmm_sequence_jobs") as batch_op:
        batch_op.drop_column("min_region_length_seconds")
        batch_op.drop_column("target_train_chunks")
        batch_op.drop_column("subsequence_stride_chunks")
        batch_op.drop_column("subsequence_length_chunks")
        batch_op.drop_column("event_balanced_proportions")
        batch_op.drop_column("near_event_window_seconds")
        batch_op.drop_column("event_core_overlap_threshold")
        batch_op.drop_column("training_mode")

    with op.batch_alter_table("continuous_embedding_jobs") as batch_op:
        batch_op.drop_constraint(
            "fk_continuous_embedding_jobs_seg_model", type_="foreignkey"
        )
        batch_op.drop_constraint(
            "fk_continuous_embedding_jobs_region_job", type_="foreignkey"
        )
        batch_op.drop_column("total_chunks")
        batch_op.drop_column("total_regions")
        batch_op.drop_column("projection_dim")
        batch_op.drop_column("projection_kind")
        batch_op.drop_column("crnn_segmentation_model_id")
        batch_op.drop_column("crnn_checkpoint_sha256")
        batch_op.drop_column("chunk_hop_seconds")
        batch_op.drop_column("chunk_size_seconds")
        batch_op.drop_column("region_detection_job_id")
        batch_op.alter_column(
            "pad_seconds",
            existing_type=sa.Float(),
            nullable=False,
        )
        batch_op.alter_column(
            "hop_seconds",
            existing_type=sa.Float(),
            nullable=False,
        )
        batch_op.alter_column(
            "window_size_seconds",
            existing_type=sa.Float(),
            nullable=False,
        )
