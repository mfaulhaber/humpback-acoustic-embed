"""Archive and remove retired workflow artifacts.

Usage:
    uv run python scripts/cleanup_legacy_workflows.py --archive-root /path/to/archive
    uv run python scripts/cleanup_legacy_workflows.py --apply --archive-root /path/to/archive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    func,
    inspect,
    select,
    update,
)

from humpback.config import Settings
from humpback.database import create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import CallParsingRun, RegionDetectionJob
from humpback.models.classifier import ClassifierTrainingJob
from humpback.models.clustering import ClusteringJob
from humpback.models.segmentation_training import SegmentationTrainingSample
from humpback.models.training_dataset import TrainingDataset
from humpback.models.vocalization import VocalizationTrainingJob
from humpback.storage import cleanup_manifests_dir, path_within_root

LEGACY_TABLES_METADATA = MetaData()
LEGACY_AUDIO_METADATA_TABLE = Table(
    "audio_metadata",
    LEGACY_TABLES_METADATA,
    Column("id", String, primary_key=True),
    Column("audio_file_id", String),
    Column("tag_data", Text),
    Column("visual_observations", Text),
    Column("group_composition", Text),
    Column("prey_density_proxy", Text),
)
LEGACY_PROCESSING_JOBS_TABLE = Table(
    "processing_jobs",
    LEGACY_TABLES_METADATA,
    Column("id", String, primary_key=True),
    Column("audio_file_id", String),
    Column("status", String),
    Column("encoding_signature", String),
    Column("model_version", String),
    Column("window_size_seconds", Float),
    Column("target_sample_rate", Integer),
    Column("feature_config", Text),
    Column("error_message", Text),
)
LEGACY_EMBEDDING_SETS_TABLE = Table(
    "embedding_sets",
    LEGACY_TABLES_METADATA,
    Column("id", String, primary_key=True),
    Column("audio_file_id", String),
    Column("encoding_signature", String),
    Column("model_version", String),
    Column("window_size_seconds", Float),
    Column("target_sample_rate", Integer),
    Column("vector_dim", Integer),
    Column("parquet_path", Text),
)
LEGACY_SEARCH_JOBS_TABLE = Table(
    "search_jobs",
    LEGACY_TABLES_METADATA,
    Column("id", String, primary_key=True),
    Column("status", String),
    Column("detection_job_id", String),
    Column("start_utc", Float),
    Column("end_utc", Float),
    Column("error_message", Text),
)
LEGACY_LABEL_PROCESSING_JOBS_TABLE = Table(
    "label_processing_jobs",
    LEGACY_TABLES_METADATA,
    Column("id", String, primary_key=True),
    Column("status", String),
    Column("annotation_folder", Text),
    Column("audio_folder", Text),
    Column("output_root", Text),
    Column("error_message", Text),
)
LEGACY_CLUSTERING_JOBS_TABLE = Table(
    "clustering_jobs",
    MetaData(),
    Column("id", String, primary_key=True),
    Column("status", String),
    Column("embedding_set_ids", Text),
    Column("detection_job_ids", Text),
)

ACTIVE_STATUSES = ("queued", "running")
LEGACY_ROOTS = {
    "audio_raw": Path("audio/raw"),
    "embeddings": Path("embeddings"),
    "label_processing": Path("label_processing"),
    "legacy_clusters": Path("clusters"),
}
PRUNE_EMPTY_ROOTS = {"audio_raw", "embeddings", "label_processing"}
REMEDIABLE_BLOCKER_CODES = {
    "processing_jobs_active",
    "search_jobs_active",
    "label_processing_jobs_active",
    "classifier_training_jobs_embedding_sets_active",
    "legacy_clustering_jobs",
    "vocalization_training_jobs_embedding_sets",
    "training_datasets_embedding_sets",
}


class CleanupSafetyError(RuntimeError):
    """Raised when cleanup would operate on unsafe filesystem paths."""


@dataclass
class CleanupRunResult:
    exit_code: int
    manifest_path: Path
    manifest: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Archive and delete retired workflow artifacts."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Archive and delete files. Dry-run is the default.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        help="Destination root for archived copies of legacy artifacts.",
    )
    return parser


def _json_list(raw: str | None) -> list[Any]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _json_has_nonempty_list(raw: str | None, key: str | None = None) -> bool:
    if not raw:
        return False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if key is not None:
        data = data.get(key) if isinstance(data, dict) else None
    return isinstance(data, list) and len(data) > 0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_slug(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _model_to_dict(model: Any) -> dict[str, Any]:
    return {
        column.name: _json_default(getattr(model, column.name))
        for column in model.__table__.columns
    }


def _row_mapping_to_dict(row_mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_default(value) for key, value in row_mapping.items()}


def _ensure_safe_tree(candidate: Path, expected_root: Path, storage_root: Path) -> None:
    if not candidate.exists():
        return
    if candidate.is_symlink():
        raise CleanupSafetyError(f"Refusing symlink candidate: {candidate}")
    if not path_within_root(candidate, expected_root):
        raise CleanupSafetyError(
            f"Candidate path escapes expected root {expected_root}: {candidate}"
        )
    if not path_within_root(candidate, storage_root):
        raise CleanupSafetyError(
            f"Candidate path escapes storage root {storage_root}: {candidate}"
        )
    if candidate.is_dir():
        for child in candidate.rglob("*"):
            if child.is_symlink():
                raise CleanupSafetyError(f"Refusing symlink within candidate: {child}")
            if not path_within_root(child, storage_root):
                raise CleanupSafetyError(
                    f"Descendant path escapes storage root {storage_root}: {child}"
                )


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _copy_candidate(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def _delete_candidate(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _prune_empty_parents(path: Path, stop_at: Path) -> None:
    current = path.parent
    while current != stop_at and current.exists():
        try:
            next(current.iterdir())
            return
        except StopIteration:
            current.rmdir()
            current = current.parent


async def _count_rows(session, model) -> int:
    result = await session.execute(select(func.count()).select_from(model))
    return int(result.scalar_one())


async def _table_exists(session, table_name: str) -> bool:
    conn = await session.connection()
    return bool(
        await conn.run_sync(lambda sync_conn: inspect(sync_conn).has_table(table_name))
    )


async def _column_exists(session, table_name: str, column_name: str) -> bool:
    if not await _table_exists(session, table_name):
        return False
    conn = await session.connection()
    columns = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).get_columns(table_name)
    )
    return any(column["name"] == column_name for column in columns)


async def _count_table_rows(session, table: Table) -> int:
    if not await _table_exists(session, table.name):
        return 0
    result = await session.execute(select(func.count()).select_from(table))
    return int(result.scalar_one())


async def _collect_db_summary(session) -> dict[str, Any]:
    direct_counts = {
        "audio_files": await _count_rows(session, AudioFile),
        "audio_metadata": await _count_table_rows(session, LEGACY_AUDIO_METADATA_TABLE),
        "processing_jobs": await _count_table_rows(
            session, LEGACY_PROCESSING_JOBS_TABLE
        ),
        "embedding_sets": await _count_table_rows(session, LEGACY_EMBEDDING_SETS_TABLE),
        "search_jobs": await _count_table_rows(session, LEGACY_SEARCH_JOBS_TABLE),
        "label_processing_jobs": await _count_table_rows(
            session, LEGACY_LABEL_PROCESSING_JOBS_TABLE
        ),
    }

    blockers: list[dict[str, Any]] = []

    async def add_status_blocker(code: str, model, label: str, extra_where=()):
        stmt = (
            select(func.count())
            .select_from(model)
            .where(model.status.in_(ACTIVE_STATUSES), *extra_where)
        )
        count = int((await session.execute(stmt)).scalar_one())
        if count:
            blockers.append({"code": code, "count": count, "detail": label})

    async def add_status_blocker_for_table(
        code: str,
        table: Table,
        label: str,
    ) -> None:
        if not await _table_exists(session, table.name):
            return
        stmt = (
            select(func.count())
            .select_from(table)
            .where(table.c.status.in_(ACTIVE_STATUSES))
        )
        count = int((await session.execute(stmt)).scalar_one())
        if count:
            blockers.append({"code": code, "count": count, "detail": label})

    await add_status_blocker_for_table(
        "processing_jobs_active",
        LEGACY_PROCESSING_JOBS_TABLE,
        "queued/running processing jobs",
    )
    await add_status_blocker_for_table(
        "search_jobs_active",
        LEGACY_SEARCH_JOBS_TABLE,
        "queued/running search jobs",
    )
    await add_status_blocker_for_table(
        "label_processing_jobs_active",
        LEGACY_LABEL_PROCESSING_JOBS_TABLE,
        "queued/running label-processing jobs",
    )
    await add_status_blocker(
        "classifier_training_jobs_embedding_sets_active",
        ClassifierTrainingJob,
        "queued/running classifier training jobs using embedding_sets",
        extra_where=(ClassifierTrainingJob.source_mode == "embedding_sets",),
    )

    retained_cluster_rows = (
        await session.execute(select(ClusteringJob.id, ClusteringJob.detection_job_ids))
    ).all()
    clustering_rows: list[Any] = []
    if await _column_exists(session, "clustering_jobs", "embedding_set_ids"):
        clustering_rows = (
            await session.execute(
                select(
                    LEGACY_CLUSTERING_JOBS_TABLE.c.id,
                    LEGACY_CLUSTERING_JOBS_TABLE.c.embedding_set_ids,
                    LEGACY_CLUSTERING_JOBS_TABLE.c.detection_job_ids,
                )
            )
        ).all()
    legacy_cluster_rows = sum(
        1
        for row in clustering_rows
        if _json_has_nonempty_list(row.embedding_set_ids)
        and not _json_has_nonempty_list(row.detection_job_ids)
    )
    if legacy_cluster_rows:
        blockers.append(
            {
                "code": "legacy_clustering_jobs",
                "count": legacy_cluster_rows,
                "detail": "clustering jobs still backed by embedding_set_ids",
            }
        )

    voc_training_rows = (
        await session.execute(
            select(VocalizationTrainingJob.id, VocalizationTrainingJob.source_config)
        )
    ).all()
    voc_training_with_embedding_sets = sum(
        1
        for row in voc_training_rows
        if _json_has_nonempty_list(row.source_config, "embedding_set_ids")
    )
    if voc_training_with_embedding_sets:
        blockers.append(
            {
                "code": "vocalization_training_jobs_embedding_sets",
                "count": voc_training_with_embedding_sets,
                "detail": "vocalization training jobs still reference embedding_set_ids",
            }
        )

    dataset_rows = (
        await session.execute(select(TrainingDataset.id, TrainingDataset.source_config))
    ).all()
    datasets_with_embedding_sets = sum(
        1
        for row in dataset_rows
        if _json_has_nonempty_list(row.source_config, "embedding_set_ids")
    )
    if datasets_with_embedding_sets:
        blockers.append(
            {
                "code": "training_datasets_embedding_sets",
                "count": datasets_with_embedding_sets,
                "detail": "training datasets still reference embedding_set_ids",
            }
        )

    call_parsing_audio_refs = int(
        (
            await session.execute(
                select(func.count())
                .select_from(CallParsingRun)
                .where(CallParsingRun.audio_file_id.is_not(None))
            )
        ).scalar_one()
    )
    region_detection_audio_refs = int(
        (
            await session.execute(
                select(func.count())
                .select_from(RegionDetectionJob)
                .where(RegionDetectionJob.audio_file_id.is_not(None))
            )
        ).scalar_one()
    )
    segmentation_training_audio_refs = int(
        (
            await session.execute(
                select(func.count())
                .select_from(SegmentationTrainingSample)
                .where(SegmentationTrainingSample.audio_file_id.is_not(None))
            )
        ).scalar_one()
    )
    retained_audio_ref_count = (
        call_parsing_audio_refs
        + region_detection_audio_refs
        + segmentation_training_audio_refs
    )
    if retained_audio_ref_count:
        blockers.append(
            {
                "code": "retained_audio_file_references",
                "count": retained_audio_ref_count,
                "detail": {
                    "call_parsing_runs": call_parsing_audio_refs,
                    "region_detection_jobs": region_detection_audio_refs,
                    "segmentation_training_samples": segmentation_training_audio_refs,
                },
            }
        )

    retained_cluster_rows = [
        row.id
        for row in retained_cluster_rows
        if _json_has_nonempty_list(row.detection_job_ids)
    ]

    return {
        "direct_legacy_table_counts": direct_counts,
        "blockers": blockers,
        "retained_cluster_job_ids": retained_cluster_rows,
    }


def _split_blockers(
    blockers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    remediable = []
    unremediable = []
    for blocker in blockers:
        if blocker["code"] in REMEDIABLE_BLOCKER_CODES:
            remediable.append(blocker)
        else:
            unremediable.append(blocker)
    return remediable, unremediable


def _db_remediation_archive_path(archive_root: Path, now: datetime, code: str) -> Path:
    return archive_root / "db-remediations" / f"{_timestamp_slug(now)}-{code}.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_legacy_source_config(
    raw_config: str,
    *,
    now: datetime,
    archive_path: Path,
) -> str:
    try:
        config = json.loads(raw_config) if raw_config else {}
    except json.JSONDecodeError:
        config = {}
    if not isinstance(config, dict):
        config = {}

    embedding_set_ids = config.pop("embedding_set_ids", [])
    archived_sources = config.get("archived_legacy_sources")
    if not isinstance(archived_sources, list):
        archived_sources = []
    archived_sources.append(
        {
            "type": "embedding_sets",
            "count": len(embedding_set_ids)
            if isinstance(embedding_set_ids, list)
            else 0,
            "archived_at": now.isoformat(),
            "archive_path": str(archive_path),
        }
    )
    config["archived_legacy_sources"] = archived_sources
    return json.dumps(config)


async def _remediate_blockers(
    session,
    *,
    archive_root: Path,
    now: datetime,
) -> list[dict[str, Any]]:
    remediations: list[dict[str, Any]] = []

    async def archive_and_fail_active_jobs(code: str, model, *, extra_where=()):
        jobs = list(
            (
                await session.execute(
                    select(model).where(model.status.in_(ACTIVE_STATUSES), *extra_where)
                )
            )
            .scalars()
            .all()
        )
        if not jobs:
            return
        archive_path = _db_remediation_archive_path(archive_root, now, code)
        _write_json(archive_path, [_model_to_dict(job) for job in jobs])
        for job in jobs:
            job.status = "failed"
            if hasattr(job, "error_message"):
                job.error_message = "Retired during legacy workflow cleanup; this workflow is no longer supported."
        remediations.append(
            {
                "code": code,
                "action": "mark_failed",
                "count": len(jobs),
                "archive_path": str(archive_path),
            }
        )

    async def archive_and_fail_active_rows(code: str, table: Table) -> None:
        if not await _table_exists(session, table.name):
            return
        rows = list(
            (
                await session.execute(
                    select(table).where(table.c.status.in_(ACTIVE_STATUSES))
                )
            )
            .mappings()
            .all()
        )
        if not rows:
            return
        archive_path = _db_remediation_archive_path(archive_root, now, code)
        _write_json(
            archive_path,
            [_row_mapping_to_dict(dict(row)) for row in rows],
        )
        row_ids = [row["id"] for row in rows]
        values: dict[str, Any] = {"status": "failed"}
        if "error_message" in table.c:
            values["error_message"] = (
                "Retired during legacy workflow cleanup; this workflow is no longer supported."
            )
        await session.execute(
            update(table).where(table.c.id.in_(row_ids)).values(**values)
        )
        remediations.append(
            {
                "code": code,
                "action": "mark_failed",
                "count": len(rows),
                "archive_path": str(archive_path),
            }
        )

    await archive_and_fail_active_rows(
        "processing_jobs_active", LEGACY_PROCESSING_JOBS_TABLE
    )
    await archive_and_fail_active_rows("search_jobs_active", LEGACY_SEARCH_JOBS_TABLE)
    await archive_and_fail_active_rows(
        "label_processing_jobs_active", LEGACY_LABEL_PROCESSING_JOBS_TABLE
    )
    await archive_and_fail_active_jobs(
        "classifier_training_jobs_embedding_sets_active",
        ClassifierTrainingJob,
        extra_where=(ClassifierTrainingJob.source_mode == "embedding_sets",),
    )

    clustering_jobs: list[dict[str, Any]] = []
    if await _column_exists(session, "clustering_jobs", "embedding_set_ids"):
        clustering_jobs = [
            dict(row)
            for row in (await session.execute(select(LEGACY_CLUSTERING_JOBS_TABLE)))
            .mappings()
            .all()
            if _json_has_nonempty_list(row["embedding_set_ids"])
            and not _json_has_nonempty_list(row["detection_job_ids"])
        ]
    if clustering_jobs:
        archive_path = _db_remediation_archive_path(
            archive_root, now, "legacy_clustering_jobs"
        )
        _write_json(
            archive_path,
            [_row_mapping_to_dict(job) for job in clustering_jobs],
        )
        await session.execute(
            delete(LEGACY_CLUSTERING_JOBS_TABLE).where(
                LEGACY_CLUSTERING_JOBS_TABLE.c.id.in_(
                    [job["id"] for job in clustering_jobs]
                )
            )
        )
        remediations.append(
            {
                "code": "legacy_clustering_jobs",
                "action": "delete_rows",
                "count": len(clustering_jobs),
                "archive_path": str(archive_path),
            }
        )

    vocalization_jobs = [
        job
        for job in (await session.execute(select(VocalizationTrainingJob)))
        .scalars()
        .all()
        if _json_has_nonempty_list(job.source_config, "embedding_set_ids")
    ]
    if vocalization_jobs:
        archive_path = _db_remediation_archive_path(
            archive_root, now, "vocalization_training_jobs_embedding_sets"
        )
        _write_json(archive_path, [_model_to_dict(job) for job in vocalization_jobs])
        for job in vocalization_jobs:
            job.source_config = _normalize_legacy_source_config(
                job.source_config,
                now=now,
                archive_path=archive_path,
            )
        remediations.append(
            {
                "code": "vocalization_training_jobs_embedding_sets",
                "action": "rewrite_source_config",
                "count": len(vocalization_jobs),
                "archive_path": str(archive_path),
            }
        )

    datasets = [
        dataset
        for dataset in (await session.execute(select(TrainingDataset))).scalars().all()
        if _json_has_nonempty_list(dataset.source_config, "embedding_set_ids")
    ]
    if datasets:
        archive_path = _db_remediation_archive_path(
            archive_root, now, "training_datasets_embedding_sets"
        )
        _write_json(archive_path, [_model_to_dict(dataset) for dataset in datasets])
        for dataset in datasets:
            dataset.source_config = _normalize_legacy_source_config(
                dataset.source_config,
                now=now,
                archive_path=archive_path,
            )
        remediations.append(
            {
                "code": "training_datasets_embedding_sets",
                "action": "rewrite_source_config",
                "count": len(datasets),
                "archive_path": str(archive_path),
            }
        )

    if remediations:
        await session.commit()

    return remediations


def _discover_candidates(
    *,
    storage_root: Path,
    archive_root: Path | None,
    retained_cluster_job_ids: set[str],
) -> dict[str, dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}

    for class_name, relative_root in LEGACY_ROOTS.items():
        source_root = storage_root / relative_root
        candidates: list[dict[str, Any]] = []
        total_bytes = 0
        all_within_root = True

        if source_root.exists():
            for child in sorted(source_root.iterdir()):
                if (
                    class_name == "legacy_clusters"
                    and child.name in retained_cluster_job_ids
                ):
                    continue
                _ensure_safe_tree(child, source_root, storage_root)
                relative_source = child.relative_to(storage_root)
                size_bytes = _path_size(child)
                total_bytes += size_bytes
                archive_path = (
                    str((archive_root / relative_source).resolve())
                    if archive_root is not None
                    else None
                )
                candidates.append(
                    {
                        "source_path": str(child.resolve()),
                        "relative_source_path": str(relative_source),
                        "archive_path": archive_path,
                        "kind": "directory" if child.is_dir() else "file",
                        "bytes": size_bytes,
                    }
                )
                all_within_root = all_within_root and path_within_root(
                    child, source_root
                )

        discovered[class_name] = {
            "source_root": str(source_root.resolve()),
            "candidate_count": len(candidates),
            "total_bytes": total_bytes,
            "candidates": candidates,
            "verification": {
                "candidate_count": len(candidates),
                "total_bytes": total_bytes,
                "all_candidates_within_root": all_within_root,
                "archive_copy_count": 0,
                "source_deleted_count": 0,
            },
        }

    return discovered


def _write_manifest(
    storage_root: Path, now: datetime, manifest: dict[str, Any]
) -> Path:
    manifest_dir = cleanup_manifests_dir(storage_root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        manifest_dir / f"{_timestamp_slug(now)}-legacy-workflow-removal.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path


def _print_summary(manifest_path: Path, manifest: dict[str, Any]) -> None:
    print(f"Manifest: {manifest_path}")
    print(f"Mode: {manifest['mode']}")
    print(f"Storage root: {manifest['storage_root']}")
    archive_root = manifest.get("archive_root")
    print(f"Archive root: {archive_root or '(not provided)'}")

    if manifest["db"]["blockers"]:
        print("Blockers:")
        for blocker in manifest["db"]["blockers"]:
            print(f"  - {blocker['code']}: {blocker['count']}")
    else:
        print("Blockers: none")

    remediations = manifest["db"].get("remediations") or []
    if remediations:
        print("Remediations:")
        for remediation in remediations:
            print(f"  - {remediation['code']}: {remediation['count']}")

    print("Artifact layout:")
    for class_name, info in manifest["artifact_classes"].items():
        print(
            f"  - {class_name}: {info['candidate_count']} candidate(s), "
            f"{info['total_bytes']} byte(s)"
        )
        for candidate in info["candidates"]:
            target = candidate["archive_path"] or "(archive-root not provided)"
            print(f"      {candidate['source_path']} -> {target}")


async def execute_cleanup(
    *,
    settings: Settings,
    apply: bool = False,
    archive_root: Path | None = None,
    now: datetime | None = None,
) -> CleanupRunResult:
    if apply and archive_root is None:
        raise ValueError("--apply requires --archive-root")

    now = now or _utc_now()
    storage_root = settings.storage_root.resolve()
    archive_root = archive_root.resolve() if archive_root is not None else None

    if archive_root is not None and path_within_root(archive_root, storage_root):
        raise CleanupSafetyError(
            f"Archive root must live outside storage_root: {archive_root}"
        )

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    remediations: list[dict[str, Any]] = []
    async with session_factory() as session:
        initial_db_summary = await _collect_db_summary(session)
        initial_blockers = list(initial_db_summary["blockers"])
        _remediable, unremediable = _split_blockers(initial_blockers)

        if apply and archive_root is not None and not unremediable:
            remediations = await _remediate_blockers(
                session,
                archive_root=archive_root,
                now=now,
            )
            current_db_summary = await _collect_db_summary(session)
        else:
            current_db_summary = initial_db_summary
    await engine.dispose()

    retained_cluster_job_ids = set(current_db_summary.pop("retained_cluster_job_ids"))
    artifact_classes = _discover_candidates(
        storage_root=storage_root,
        archive_root=archive_root,
        retained_cluster_job_ids=retained_cluster_job_ids,
    )

    blockers = current_db_summary["blockers"]
    exit_code = 1 if blockers else 0

    if apply and not blockers:
        assert archive_root is not None
        archive_root.mkdir(parents=True, exist_ok=True)
        for class_name, info in artifact_classes.items():
            source_root = Path(info["source_root"])
            archive_copy_count = 0
            source_deleted_count = 0
            for candidate in info["candidates"]:
                src = Path(candidate["source_path"])
                dst = archive_root / candidate["relative_source_path"]
                _copy_candidate(src, dst)
                if dst.exists():
                    archive_copy_count += 1
                _delete_candidate(src)
                if not src.exists():
                    source_deleted_count += 1
                if class_name in PRUNE_EMPTY_ROOTS:
                    _prune_empty_parents(src, source_root)

            info["verification"]["archive_copy_count"] = archive_copy_count
            info["verification"]["source_deleted_count"] = source_deleted_count

    manifest = {
        "generated_at": now.isoformat(),
        "mode": "apply" if apply else "dry-run",
        "storage_root": str(storage_root),
        "archive_root": str(archive_root) if archive_root is not None else None,
        "db": {
            **current_db_summary,
            "initial_blockers": initial_blockers,
            "remediations": remediations,
        },
        "artifact_classes": artifact_classes,
        "summary": {
            "total_candidate_count": sum(
                info["candidate_count"] for info in artifact_classes.values()
            ),
            "total_candidate_bytes": sum(
                info["total_bytes"] for info in artifact_classes.values()
            ),
        },
    }
    manifest_path = _write_manifest(storage_root, now, manifest)
    return CleanupRunResult(
        exit_code=exit_code,
        manifest_path=manifest_path,
        manifest=manifest,
    )


async def _async_main(args: argparse.Namespace) -> int:
    settings = Settings.from_repo_env()
    result = await execute_cleanup(
        settings=settings,
        apply=args.apply,
        archive_root=args.archive_root,
    )
    _print_summary(result.manifest_path, result.manifest)
    if result.exit_code:
        print("Cleanup preflight found blockers.", file=sys.stderr)
    elif args.apply:
        print("Archive and deletion completed.")
    else:
        print("Dry run completed. Use --apply to archive and delete.")
    return result.exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.apply and args.archive_root is None:
        parser.error("--apply requires --archive-root")
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
