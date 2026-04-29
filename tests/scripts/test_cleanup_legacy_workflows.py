"""Tests for legacy workflow cleanup script."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from humpback.config import Settings
from humpback.database import Base, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.classifier import ClassifierTrainingJob
from humpback.models.clustering import ClusteringJob
from humpback.models.training_dataset import TrainingDataset
from humpback.models.vocalization import VocalizationTrainingJob


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "cleanup_legacy_workflows.py"
)
spec = importlib.util.spec_from_file_location("cleanup_legacy_workflows", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
cleanup = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cleanup
spec.loader.exec_module(cleanup)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        storage_root=tmp_path / "storage",
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
    )


async def _init_db(settings: Settings) -> None:
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(cleanup.LEGACY_TABLES_METADATA.create_all)
        await conn.execute(
            text("ALTER TABLE clustering_jobs ADD COLUMN embedding_set_ids TEXT")
        )
    await engine.dispose()


async def _seed_nominal_rows(settings: Settings, *, retained_cluster_id: str) -> None:
    engine = create_async_engine(settings.database_url)
    factory = create_session_factory(engine)
    async with factory() as session:
        audio = AudioFile(
            id="audio-1",
            filename="sample.wav",
            folder_path="legacy/folder",
            checksum_sha256="abc123",
        )
        session.add(audio)
        await session.flush()
        await session.execute(
            cleanup.LEGACY_AUDIO_METADATA_TABLE.insert().values(
                id="audio-metadata-1",
                audio_file_id=audio.id,
            )
        )
        await session.execute(
            cleanup.LEGACY_PROCESSING_JOBS_TABLE.insert().values(
                id="processing-job-1",
                audio_file_id=audio.id,
                status="complete",
                encoding_signature="sig-1",
                model_version="perch_v1",
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.execute(
            cleanup.LEGACY_EMBEDDING_SETS_TABLE.insert().values(
                id="embedding-set-1",
                audio_file_id=audio.id,
                encoding_signature="sig-1",
                model_version="perch_v1",
                window_size_seconds=5.0,
                target_sample_rate=32000,
                vector_dim=1280,
                parquet_path=str(
                    settings.storage_root / "embeddings/perch_v1/audio-1/sig-1.parquet"
                ),
            )
        )
        await session.execute(
            cleanup.LEGACY_SEARCH_JOBS_TABLE.insert().values(
                id="search-job-1",
                status="complete",
                detection_job_id="det-1",
                start_utc=1.0,
                end_utc=2.0,
            )
        )
        await session.execute(
            cleanup.LEGACY_LABEL_PROCESSING_JOBS_TABLE.insert().values(
                id="label-processing-job-1",
                status="complete",
                annotation_folder="/ann",
                audio_folder="/audio",
                output_root="/out",
            )
        )
        session.add(
            ClusteringJob(
                id=retained_cluster_id,
                detection_job_ids=json.dumps(["det-1"]),
                status="complete",
            )
        )
        await session.flush()
        await session.execute(
            text(
                "UPDATE clustering_jobs SET embedding_set_ids = :embedding_set_ids "
                "WHERE id = :job_id"
            ),
            {"embedding_set_ids": "[]", "job_id": retained_cluster_id},
        )
        await session.commit()
    await engine.dispose()


def _create_legacy_artifacts(
    settings: Settings,
    *,
    retained_cluster_id: str,
    include_legacy_cluster: bool = True,
) -> None:
    storage_root = settings.storage_root
    (storage_root / "audio/raw/audio-1").mkdir(parents=True, exist_ok=True)
    (storage_root / "audio/raw/audio-1/original.wav").write_bytes(b"audio")

    (storage_root / "embeddings/perch_v1/audio-1").mkdir(parents=True, exist_ok=True)
    (storage_root / "embeddings/perch_v1/audio-1/sig-1.parquet").write_bytes(
        b"embedding"
    )

    (storage_root / "label_processing/job-1").mkdir(parents=True, exist_ok=True)
    (storage_root / "label_processing/job-1/result.json").write_text("{}")

    if include_legacy_cluster:
        (storage_root / "clusters/legacy-orphan").mkdir(parents=True, exist_ok=True)
        (storage_root / "clusters/legacy-orphan/clusters.json").write_text("{}")

    (storage_root / f"clusters/{retained_cluster_id}").mkdir(
        parents=True, exist_ok=True
    )
    (storage_root / f"clusters/{retained_cluster_id}/keep.txt").write_text("keep")


def test_apply_requires_archive_root():
    try:
        cleanup.main(["--apply"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected parser failure when --archive-root is omitted")


def test_dry_run_writes_manifest_and_keeps_files(tmp_path):
    settings = _settings(tmp_path)
    retained_cluster_id = "cluster-retained"
    archive_root = tmp_path / "archive"

    asyncio.run(_init_db(settings))
    asyncio.run(_seed_nominal_rows(settings, retained_cluster_id=retained_cluster_id))
    _create_legacy_artifacts(settings, retained_cluster_id=retained_cluster_id)

    result = asyncio.run(
        cleanup.execute_cleanup(
            settings=settings,
            apply=False,
            archive_root=archive_root,
        )
    )

    assert result.exit_code == 0
    assert result.manifest_path.exists()
    manifest = result.manifest
    assert manifest["mode"] == "dry-run"
    assert manifest["db"]["direct_legacy_table_counts"]["audio_files"] == 1
    assert manifest["artifact_classes"]["audio_raw"]["candidate_count"] == 1
    assert manifest["artifact_classes"]["embeddings"]["candidate_count"] == 1
    assert manifest["artifact_classes"]["label_processing"]["candidate_count"] == 1
    assert manifest["artifact_classes"]["legacy_clusters"]["candidate_count"] == 1
    assert manifest["artifact_classes"]["legacy_clusters"]["candidates"][0][
        "archive_path"
    ] == str((archive_root / "clusters/legacy-orphan").resolve())

    assert (settings.storage_root / "audio/raw/audio-1/original.wav").exists()
    assert (
        settings.storage_root / "embeddings/perch_v1/audio-1/sig-1.parquet"
    ).exists()
    assert (settings.storage_root / "label_processing/job-1/result.json").exists()
    assert (settings.storage_root / "clusters/legacy-orphan/clusters.json").exists()
    assert (settings.storage_root / f"clusters/{retained_cluster_id}/keep.txt").exists()


def test_refuses_candidate_outside_storage_root(tmp_path):
    settings = _settings(tmp_path)
    retained_cluster_id = "cluster-retained"
    archive_root = tmp_path / "archive"

    asyncio.run(_init_db(settings))
    asyncio.run(_seed_nominal_rows(settings, retained_cluster_id=retained_cluster_id))
    (settings.storage_root / "audio/raw").mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope")
    (settings.storage_root / "audio/raw/escape").symlink_to(
        outside, target_is_directory=True
    )

    try:
        asyncio.run(
            cleanup.execute_cleanup(
                settings=settings,
                apply=False,
                archive_root=archive_root,
            )
        )
    except cleanup.CleanupSafetyError as exc:
        assert "symlink" in str(exc).lower()
    else:
        raise AssertionError("Expected CleanupSafetyError for out-of-root candidate")


def test_reports_blockers_and_exits_nonzero(tmp_path):
    settings = _settings(tmp_path)
    archive_root = tmp_path / "archive"

    asyncio.run(_init_db(settings))

    async def seed_blockers() -> None:
        engine = create_async_engine(settings.database_url)
        factory = create_session_factory(engine)
        async with factory() as session:
            await session.execute(
                cleanup.LEGACY_PROCESSING_JOBS_TABLE.insert().values(
                    id="processing-job-blocked",
                    audio_file_id="audio-1",
                    status="queued",
                    encoding_signature="sig-queued",
                    model_version="perch_v1",
                    window_size_seconds=5.0,
                    target_sample_rate=32000,
                )
            )
            session.add(
                ClassifierTrainingJob(
                    status="queued",
                    name="legacy-train",
                    model_version="perch_v1",
                    window_size_seconds=5.0,
                    target_sample_rate=32000,
                    source_mode="embedding_sets",
                )
            )
            session.add(
                ClusteringJob(
                    id="legacy-cluster-blocked",
                    detection_job_ids=None,
                    status="complete",
                )
            )
            await session.flush()
            await session.execute(
                text(
                    "UPDATE clustering_jobs SET embedding_set_ids = :embedding_set_ids "
                    "WHERE id = :job_id"
                ),
                {
                    "embedding_set_ids": json.dumps(["es-1"]),
                    "job_id": "legacy-cluster-blocked",
                },
            )
            session.add(
                VocalizationTrainingJob(
                    source_config=json.dumps({"embedding_set_ids": ["es-1"]}),
                    status="complete",
                )
            )
            session.add(
                TrainingDataset(
                    name="legacy-dataset",
                    source_config=json.dumps({"embedding_set_ids": ["es-1"]}),
                    parquet_path="/tmp/dataset.parquet",
                )
            )
            session.add(
                RegionDetectionJob(
                    audio_file_id="audio-1",
                    status="complete",
                )
            )
            await session.commit()
        await engine.dispose()

    asyncio.run(seed_blockers())

    result = asyncio.run(
        cleanup.execute_cleanup(
            settings=settings,
            apply=False,
            archive_root=archive_root,
        )
    )

    assert result.exit_code == 1
    blocker_codes = {item["code"] for item in result.manifest["db"]["blockers"]}
    assert "processing_jobs_active" in blocker_codes
    assert "classifier_training_jobs_embedding_sets_active" in blocker_codes
    assert "legacy_clustering_jobs" in blocker_codes
    assert "vocalization_training_jobs_embedding_sets" in blocker_codes
    assert "training_datasets_embedding_sets" in blocker_codes
    assert "retained_audio_file_references" in blocker_codes
    assert result.manifest["db"]["remediations"] == []
    assert result.manifest_path.exists()


def test_apply_archives_deletes_and_is_idempotent(tmp_path):
    settings = _settings(tmp_path)
    retained_cluster_id = "cluster-retained"
    archive_root = tmp_path / "archive"

    asyncio.run(_init_db(settings))
    asyncio.run(_seed_nominal_rows(settings, retained_cluster_id=retained_cluster_id))
    _create_legacy_artifacts(settings, retained_cluster_id=retained_cluster_id)

    first = asyncio.run(
        cleanup.execute_cleanup(
            settings=settings,
            apply=True,
            archive_root=archive_root,
        )
    )

    assert first.exit_code == 0
    assert not (settings.storage_root / "audio/raw/audio-1").exists()
    assert not (settings.storage_root / "embeddings/perch_v1").exists()
    assert not (settings.storage_root / "label_processing/job-1").exists()
    assert not (settings.storage_root / "clusters/legacy-orphan").exists()
    assert (settings.storage_root / f"clusters/{retained_cluster_id}/keep.txt").exists()

    assert (archive_root / "audio/raw/audio-1/original.wav").exists()
    assert (archive_root / "embeddings/perch_v1/audio-1/sig-1.parquet").exists()
    assert (archive_root / "label_processing/job-1/result.json").exists()
    assert (archive_root / "clusters/legacy-orphan/clusters.json").exists()

    verification = first.manifest["artifact_classes"]["audio_raw"]["verification"]
    assert verification["archive_copy_count"] == 1
    assert verification["source_deleted_count"] == 1

    second = asyncio.run(
        cleanup.execute_cleanup(
            settings=settings,
            apply=True,
            archive_root=archive_root,
        )
    )

    assert second.exit_code == 0
    assert second.manifest["artifact_classes"]["audio_raw"]["candidate_count"] == 0
    assert second.manifest["artifact_classes"]["embeddings"]["candidate_count"] == 0
    assert (
        second.manifest["artifact_classes"]["label_processing"]["candidate_count"] == 0
    )
    assert (
        second.manifest["artifact_classes"]["legacy_clusters"]["candidate_count"] == 0
    )


def test_apply_remediates_supported_blockers_then_continues(tmp_path):
    settings = _settings(tmp_path)
    archive_root = tmp_path / "archive"

    asyncio.run(_init_db(settings))

    async def seed_remediable_blockers() -> None:
        engine = create_async_engine(settings.database_url)
        factory = create_session_factory(engine)
        async with factory() as session:
            session.add(
                ClassifierTrainingJob(
                    id="legacy-classifier-job",
                    status="queued",
                    name="legacy-train",
                    model_version="perch_v1",
                    window_size_seconds=5.0,
                    target_sample_rate=32000,
                    source_mode="embedding_sets",
                )
            )
            session.add(
                ClusteringJob(
                    id="legacy-cluster-job",
                    detection_job_ids=None,
                    status="complete",
                )
            )
            await session.flush()
            await session.execute(
                text(
                    "UPDATE clustering_jobs SET embedding_set_ids = :embedding_set_ids "
                    "WHERE id = :job_id"
                ),
                {
                    "embedding_set_ids": json.dumps(["es-1"]),
                    "job_id": "legacy-cluster-job",
                },
            )
            session.add(
                VocalizationTrainingJob(
                    id="legacy-voc-job",
                    source_config=json.dumps({"embedding_set_ids": ["es-1"]}),
                    status="complete",
                )
            )
            session.add(
                TrainingDataset(
                    id="legacy-dataset",
                    name="legacy-dataset",
                    source_config=json.dumps({"embedding_set_ids": ["es-1"]}),
                    parquet_path="/tmp/dataset.parquet",
                )
            )
            await session.commit()
        await engine.dispose()

    asyncio.run(seed_remediable_blockers())

    (settings.storage_root / "embeddings/perch_v1/audio-1").mkdir(
        parents=True, exist_ok=True
    )
    (settings.storage_root / "embeddings/perch_v1/audio-1/sig-1.parquet").write_bytes(
        b"embedding"
    )
    (settings.storage_root / "clusters/legacy-cluster-job").mkdir(
        parents=True, exist_ok=True
    )
    (settings.storage_root / "clusters/legacy-cluster-job/clusters.json").write_text(
        "{}"
    )

    result = asyncio.run(
        cleanup.execute_cleanup(
            settings=settings,
            apply=True,
            archive_root=archive_root,
        )
    )

    assert result.exit_code == 0
    remediation_codes = {item["code"] for item in result.manifest["db"]["remediations"]}
    assert "classifier_training_jobs_embedding_sets_active" in remediation_codes
    assert "legacy_clustering_jobs" in remediation_codes
    assert "vocalization_training_jobs_embedding_sets" in remediation_codes
    assert "training_datasets_embedding_sets" in remediation_codes
    assert result.manifest["db"]["blockers"] == []

    async def assert_db_state() -> None:
        engine = create_async_engine(settings.database_url)
        factory = create_session_factory(engine)
        async with factory() as session:
            classifier_job = await session.get(
                ClassifierTrainingJob, "legacy-classifier-job"
            )
            assert classifier_job is not None
            assert classifier_job.status == "failed"
            assert classifier_job.error_message is not None

            clustering_row = (
                (
                    await session.execute(
                        cleanup.LEGACY_CLUSTERING_JOBS_TABLE.select().where(
                            cleanup.LEGACY_CLUSTERING_JOBS_TABLE.c.id
                            == "legacy-cluster-job"
                        )
                    )
                )
                .mappings()
                .first()
            )
            assert clustering_row is None

            voc_job = await session.get(VocalizationTrainingJob, "legacy-voc-job")
            assert voc_job is not None
            assert "embedding_set_ids" not in json.loads(voc_job.source_config)

            dataset = await session.get(TrainingDataset, "legacy-dataset")
            assert dataset is not None
            assert "embedding_set_ids" not in json.loads(dataset.source_config)
        await engine.dispose()

    asyncio.run(assert_db_state())

    assert not (settings.storage_root / "clusters/legacy-cluster-job").exists()
    assert not (settings.storage_root / "embeddings/perch_v1").exists()
    remediation_archives = list((archive_root / "db-remediations").glob("*.json"))
    assert remediation_archives
