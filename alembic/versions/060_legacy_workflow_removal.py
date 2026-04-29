"""Retire legacy workflow tables and columns.

Revision ID: 060
Revises: 059
Create Date: 2026-04-29

Downgrade recreates dropped tables and columns, but it cannot restore the user
data that was archived and removed before this migration.
"""

from __future__ import annotations

import json

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision = "060"
down_revision = "059"
branch_labels = None
depends_on = None


def _table_exists(bind, table_name: str) -> bool:
    return inspect(bind).has_table(table_name)


def _column_names(bind, table_name: str) -> set[str]:
    return {column["name"] for column in inspect(bind).get_columns(table_name)}


def _count(bind, sql: str, **params) -> int:
    return int(bind.execute(sa.text(sql), params).scalar() or 0)


def _json_array(raw: str | None) -> list[str]:
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data]


def _preflight_blockers(bind) -> dict[str, int]:
    blockers: dict[str, int] = {}

    if _table_exists(bind, "processing_jobs"):
        count = _count(
            bind,
            "SELECT COUNT(*) FROM processing_jobs WHERE status IN ('queued', 'running')",
        )
        if count:
            blockers["processing_jobs_active"] = count

    if _table_exists(bind, "search_jobs"):
        count = _count(
            bind,
            "SELECT COUNT(*) FROM search_jobs WHERE status IN ('queued', 'running')",
        )
        if count:
            blockers["search_jobs_active"] = count

    if _table_exists(bind, "label_processing_jobs"):
        count = _count(
            bind,
            "SELECT COUNT(*) FROM label_processing_jobs "
            "WHERE status IN ('queued', 'running')",
        )
        if count:
            blockers["label_processing_jobs_active"] = count

    if _table_exists(bind, "retrain_workflows"):
        count = _count(
            bind,
            "SELECT COUNT(*) FROM retrain_workflows "
            "WHERE status IN ('queued', 'importing', 'processing', 'training')",
        )
        if count:
            blockers["retrain_workflows_active"] = count

    if _table_exists(bind, "clustering_jobs") and "embedding_set_ids" in _column_names(
        bind, "clustering_jobs"
    ):
        count = _count(
            bind,
            "SELECT COUNT(*) FROM clustering_jobs "
            "WHERE embedding_set_ids IS NOT NULL "
            "AND TRIM(embedding_set_ids) NOT IN ('', '[]')",
        )
        if count:
            blockers["legacy_clustering_jobs"] = count

    if _table_exists(
        bind, "vocalization_training_jobs"
    ) and "source_config" in _column_names(bind, "vocalization_training_jobs"):
        count = _count(
            bind,
            "SELECT COUNT(*) FROM vocalization_training_jobs "
            "WHERE source_config LIKE '%embedding_set_ids%'",
        )
        if count:
            blockers["vocalization_training_jobs_embedding_sets"] = count

    if _table_exists(bind, "training_datasets") and "source_config" in _column_names(
        bind, "training_datasets"
    ):
        count = _count(
            bind,
            "SELECT COUNT(*) FROM training_datasets "
            "WHERE source_config LIKE '%embedding_set_ids%'",
        )
        if count:
            blockers["training_datasets_embedding_sets"] = count

    return blockers


def _backfill_legacy_source_summary(bind) -> None:
    if not _table_exists(bind, "classifier_training_jobs"):
        return

    rows = bind.execute(
        sa.text(
            "SELECT id, positive_embedding_set_ids, negative_embedding_set_ids "
            "FROM classifier_training_jobs "
            "WHERE source_mode = 'embedding_sets'"
        )
    ).fetchall()
    for row_id, pos_raw, neg_raw in rows:
        positive_ids = _json_array(pos_raw)
        negative_ids = _json_array(neg_raw)
        summary = {
            "legacy_input_kind": "embedding_sets",
            "positive_embedding_set_ids": positive_ids,
            "negative_embedding_set_ids": negative_ids,
            "positive_embedding_set_count": len(positive_ids),
            "negative_embedding_set_count": len(negative_ids),
            "total_sources": len(positive_ids) + len(negative_ids),
        }
        bind.execute(
            sa.text(
                "UPDATE classifier_training_jobs "
                "SET legacy_source_summary = :summary "
                "WHERE id = :row_id"
            ),
            {"row_id": row_id, "summary": json.dumps(summary)},
        )


def _restore_legacy_source_columns(bind) -> None:
    if not _table_exists(bind, "classifier_training_jobs"):
        return

    rows = bind.execute(
        sa.text(
            "SELECT id, legacy_source_summary "
            "FROM classifier_training_jobs "
            "WHERE legacy_source_summary IS NOT NULL"
        )
    ).fetchall()
    for row_id, raw_summary in rows:
        try:
            summary = json.loads(raw_summary)
        except (TypeError, json.JSONDecodeError):
            summary = {}
        positive_ids = summary.get("positive_embedding_set_ids") or []
        negative_ids = summary.get("negative_embedding_set_ids") or []
        bind.execute(
            sa.text(
                "UPDATE classifier_training_jobs "
                "SET positive_embedding_set_ids = :positive_ids, "
                "    negative_embedding_set_ids = :negative_ids "
                "WHERE id = :row_id"
            ),
            {
                "row_id": row_id,
                "positive_ids": json.dumps(positive_ids),
                "negative_ids": json.dumps(negative_ids),
            },
        )


def upgrade() -> None:
    bind = op.get_bind()

    blockers = _preflight_blockers(bind)
    if blockers:
        detail = ", ".join(
            f"{code}={count}" for code, count in sorted(blockers.items())
        )
        raise RuntimeError(
            "Legacy workflow cleanup blockers remain; run the cleanup/remediation "
            f"step before upgrading: {detail}"
        )

    with op.batch_alter_table("classifier_training_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("legacy_source_summary", sa.Text(), nullable=True)
        )

    _backfill_legacy_source_summary(bind)

    with op.batch_alter_table("classifier_training_jobs") as batch_op:
        batch_op.alter_column(
            "source_mode",
            existing_type=sa.String(),
            existing_nullable=False,
            server_default="detection_manifest",
        )
        batch_op.drop_column("positive_embedding_set_ids")
        batch_op.drop_column("negative_embedding_set_ids")

    with op.batch_alter_table("classifier_models") as batch_op:
        batch_op.alter_column(
            "training_source_mode",
            existing_type=sa.String(),
            existing_nullable=False,
            server_default="detection_manifest",
        )

    with op.batch_alter_table("cluster_assignments") as batch_op:
        batch_op.add_column(sa.Column("source_id", sa.String(), nullable=True))

    bind.execute(
        sa.text(
            "UPDATE cluster_assignments SET source_id = embedding_set_id "
            "WHERE source_id IS NULL"
        )
    )

    with op.batch_alter_table("cluster_assignments") as batch_op:
        batch_op.alter_column(
            "source_id",
            existing_type=sa.String(),
            nullable=False,
        )
        batch_op.drop_column("embedding_set_id")

    with op.batch_alter_table("clustering_jobs") as batch_op:
        batch_op.drop_column("embedding_set_ids")

    if _table_exists(bind, "search_jobs"):
        op.drop_table("search_jobs")
    if _table_exists(bind, "label_processing_jobs"):
        op.drop_table("label_processing_jobs")
    if _table_exists(bind, "processing_jobs"):
        op.drop_table("processing_jobs")
    if _table_exists(bind, "embedding_sets"):
        op.drop_table("embedding_sets")
    if _table_exists(bind, "audio_metadata"):
        op.drop_table("audio_metadata")


def downgrade() -> None:
    bind = op.get_bind()

    if not _table_exists(bind, "audio_metadata"):
        op.create_table(
            "audio_metadata",
            sa.Column("audio_file_id", sa.String(), nullable=False),
            sa.Column("tag_data", sa.Text(), nullable=True),
            sa.Column("visual_observations", sa.Text(), nullable=True),
            sa.Column("group_composition", sa.Text(), nullable=True),
            sa.Column("prey_density_proxy", sa.Text(), nullable=True),
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["audio_file_id"], ["audio_files.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("audio_file_id"),
        )

    if not _table_exists(bind, "embedding_sets"):
        op.create_table(
            "embedding_sets",
            sa.Column("audio_file_id", sa.String(), nullable=False),
            sa.Column("encoding_signature", sa.String(), nullable=False),
            sa.Column("model_version", sa.String(), nullable=False),
            sa.Column("window_size_seconds", sa.Float(), nullable=False),
            sa.Column("target_sample_rate", sa.Integer(), nullable=False),
            sa.Column("vector_dim", sa.Integer(), nullable=False),
            sa.Column("parquet_path", sa.String(), nullable=False),
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["audio_file_id"], ["audio_files.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("audio_file_id", "encoding_signature"),
        )

    if not _table_exists(bind, "processing_jobs"):
        op.create_table(
            "processing_jobs",
            sa.Column("audio_file_id", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, server_default="queued"),
            sa.Column("encoding_signature", sa.String(), nullable=False),
            sa.Column("model_version", sa.String(), nullable=False),
            sa.Column("window_size_seconds", sa.Float(), nullable=False),
            sa.Column("target_sample_rate", sa.Integer(), nullable=False),
            sa.Column("feature_config", sa.Text(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("warning_message", sa.Text(), nullable=True),
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["audio_file_id"], ["audio_files.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _table_exists(bind, "label_processing_jobs"):
        op.create_table(
            "label_processing_jobs",
            sa.Column("status", sa.String(), nullable=False, server_default="queued"),
            sa.Column(
                "workflow", sa.String(), nullable=False, server_default="score_based"
            ),
            sa.Column("classifier_model_id", sa.String(), nullable=True),
            sa.Column("annotation_folder", sa.String(), nullable=False),
            sa.Column("audio_folder", sa.String(), nullable=False),
            sa.Column("output_root", sa.String(), nullable=False),
            sa.Column("parameters", sa.Text(), nullable=True),
            sa.Column("files_processed", sa.Integer(), nullable=True),
            sa.Column("files_total", sa.Integer(), nullable=True),
            sa.Column("annotations_total", sa.Integer(), nullable=True),
            sa.Column("result_summary", sa.Text(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _table_exists(bind, "search_jobs"):
        op.create_table(
            "search_jobs",
            sa.Column("status", sa.String(), nullable=False, server_default="queued"),
            sa.Column("detection_job_id", sa.String(), nullable=False),
            sa.Column("start_utc", sa.Float(), nullable=False),
            sa.Column("end_utc", sa.Float(), nullable=False),
            sa.Column("top_k", sa.Integer(), nullable=False, server_default="20"),
            sa.Column("metric", sa.String(), nullable=False, server_default="cosine"),
            sa.Column("embedding_set_ids", sa.Text(), nullable=True),
            sa.Column("model_version", sa.String(), nullable=True),
            sa.Column("embedding_vector", sa.Text(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    with op.batch_alter_table("clustering_jobs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "embedding_set_ids", sa.Text(), nullable=False, server_default="[]"
            )
        )

    with op.batch_alter_table("cluster_assignments") as batch_op:
        batch_op.add_column(sa.Column("embedding_set_id", sa.String(), nullable=True))

    bind.execute(
        sa.text(
            "UPDATE cluster_assignments SET embedding_set_id = source_id "
            "WHERE embedding_set_id IS NULL"
        )
    )

    with op.batch_alter_table("cluster_assignments") as batch_op:
        batch_op.alter_column(
            "embedding_set_id",
            existing_type=sa.String(),
            nullable=False,
        )
        batch_op.drop_column("source_id")

    with op.batch_alter_table("classifier_training_jobs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "positive_embedding_set_ids",
                sa.Text(),
                nullable=False,
                server_default="[]",
            )
        )
        batch_op.add_column(
            sa.Column(
                "negative_embedding_set_ids",
                sa.Text(),
                nullable=False,
                server_default="[]",
            )
        )
        batch_op.alter_column(
            "source_mode",
            existing_type=sa.String(),
            existing_nullable=False,
            server_default="embedding_sets",
        )

    _restore_legacy_source_columns(bind)

    with op.batch_alter_table("classifier_training_jobs") as batch_op:
        batch_op.drop_column("legacy_source_summary")

    with op.batch_alter_table("classifier_models") as batch_op:
        batch_op.alter_column(
            "training_source_mode",
            existing_type=sa.String(),
            existing_nullable=False,
            server_default="embedding_sets",
        )
