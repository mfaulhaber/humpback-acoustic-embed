"""Add hmm_sequence_jobs table.

Sequence Models PR 2: HMM training + Viterbi decode on continuous embeddings.
Given a completed ContinuousEmbeddingJob, fits PCA + GaussianHMM, decodes
Viterbi states, and persists artifacts for visualization.

Revision ID: 058
Revises: 057
Create Date: 2026-04-27
"""

import sqlalchemy as sa
from alembic import op

revision = "058"
down_revision = "057"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hmm_sequence_jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("continuous_embedding_job_id", sa.String(), nullable=False),
        sa.Column("n_states", sa.Integer(), nullable=False),
        sa.Column("pca_dims", sa.Integer(), nullable=False),
        sa.Column("pca_whiten", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("l2_normalize", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column(
            "covariance_type", sa.String(), nullable=False, server_default="diag"
        ),
        sa.Column("n_iter", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("random_seed", sa.Integer(), nullable=False, server_default="42"),
        sa.Column(
            "min_sequence_length_frames",
            sa.Integer(),
            nullable=False,
            server_default="10",
        ),
        sa.Column("tol", sa.Float(), nullable=False, server_default="0.0001"),
        sa.Column("library", sa.String(), nullable=False, server_default="hmmlearn"),
        sa.Column("train_log_likelihood", sa.Float(), nullable=True),
        sa.Column("n_train_sequences", sa.Integer(), nullable=True),
        sa.Column("n_train_frames", sa.Integer(), nullable=True),
        sa.Column("n_decoded_sequences", sa.Integer(), nullable=True),
        sa.Column("artifact_dir", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["continuous_embedding_job_id"],
            ["continuous_embedding_jobs.id"],
            name="fk_hmm_sequence_jobs_continuous_embedding_job",
        ),
    )
    op.create_index(
        "ix_hmm_sequence_jobs_status",
        "hmm_sequence_jobs",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_hmm_sequence_jobs_status",
        table_name="hmm_sequence_jobs",
    )
    op.drop_table("hmm_sequence_jobs")
