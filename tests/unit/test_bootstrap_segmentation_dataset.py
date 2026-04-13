"""Unit tests for the segmentation dataset bootstrap script.

Exercises the core ``run_bootstrap()`` function with an in-memory
SQLite DB and synthetic detection row stores. Each test seeds the
minimum fixtures (DetectionJob with hydrophone source, row store parquet,
VocalizationLabel) and asserts the expected skip/insert behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select

from humpback.classifier.detection_rows import write_detection_row_store
from humpback.database import Base, create_engine, create_session_factory
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.labeling import VocalizationLabel
from humpback.models.segmentation_training import (
    SegmentationTrainingDataset,
    SegmentationTrainingSample,
)
from humpback.storage import detection_row_store_path

from scripts.bootstrap_segmentation_dataset import (
    _discover_row_ids_from_jobs,
    parse_args,
    read_row_ids,
    run_bootstrap,
)

JOB_START_TS = 1000.0
JOB_END_TS = 2000.0
HYDROPHONE_ID = "test_hydrophone"


@pytest.fixture
async def session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


async def _seed_fixture(
    session_factory,
    tmp_path: Path,
    *,
    row_id: str = "row-aaa",
    start_utc: float = 1100.0,
    end_utc: float = 1105.0,
    labels: list[str] | None = None,
    job_start_ts: float = JOB_START_TS,
    job_end_ts: float = JOB_END_TS,
    hydrophone_id: str = HYDROPHONE_ID,
) -> dict[str, str]:
    """Seed DB + disk for one hydrophone detection row with vocalization labels."""
    storage_root = tmp_path / "storage"

    if labels is None:
        labels = ["upcall"]

    async with session_factory() as session:
        cm = ClassifierModel(
            name="fake-cm",
            model_path="/tmp/not-a-real-file.joblib",
            model_version="perch_v1",
            vector_dim=64,
            window_size_seconds=5.0,
            target_sample_rate=16000,
        )
        session.add(cm)
        await session.flush()

        dj = DetectionJob(
            classifier_model_id=cm.id,
            hydrophone_id=hydrophone_id,
            start_timestamp=job_start_ts,
            end_timestamp=job_end_ts,
            status="complete",
        )
        session.add(dj)
        await session.flush()

        for label in labels:
            session.add(
                VocalizationLabel(
                    detection_job_id=dj.id,
                    row_id=row_id,
                    label=label,
                    source="manual",
                )
            )

        await session.commit()
        ids = {
            "detection_job_id": dj.id,
            "classifier_model_id": cm.id,
        }

    rs_path = detection_row_store_path(storage_root, ids["detection_job_id"])
    rs_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "row_id": row_id,
        "start_utc": str(start_utc),
        "end_utc": str(end_utc),
        "avg_confidence": "0.8",
        "peak_confidence": "0.9",
        "n_windows": "2",
    }
    write_detection_row_store(rs_path, [row])

    return ids


async def test_happy_path_inserts_sample(session_factory, tmp_path):
    await _seed_fixture(session_factory, tmp_path)
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        result = await run_bootstrap(
            session,
            row_ids=["row-aaa"],
            dataset_name="test-ds",
            dataset_id=None,
            crop_seconds=10.0,
            allow_multi_label=False,
            dry_run=False,
            storage_root=storage_root,
        )

    assert result.inserted == 1
    assert sum(result.skipped.values()) == 0
    assert result.dataset_id != ""

    async with session_factory() as session:
        samples = (
            (await session.execute(select(SegmentationTrainingSample))).scalars().all()
        )
        assert len(samples) == 1
        sample = samples[0]
        assert sample.training_dataset_id == result.dataset_id
        assert sample.audio_file_id is None
        assert sample.hydrophone_id == HYDROPHONE_ID
        assert sample.start_timestamp is not None
        assert sample.end_timestamp is not None
        assert sample.crop_start_sec == 0.0
        crop_duration = sample.end_timestamp - sample.start_timestamp
        assert sample.crop_end_sec == pytest.approx(crop_duration, abs=0.01)
        assert sample.source == "bootstrap_vocalization_row"
        assert sample.source_ref == "row-aaa"
        events = json.loads(sample.events_json)
        assert len(events) == 1
        assert events[0]["start_sec"] >= 0.0
        assert events[0]["end_sec"] > events[0]["start_sec"]


async def test_no_vocalization_label_skips(session_factory, tmp_path):
    await _seed_fixture(session_factory, tmp_path, labels=[])
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        result = await run_bootstrap(
            session,
            row_ids=["row-aaa"],
            dataset_name="test-ds",
            dataset_id=None,
            crop_seconds=10.0,
            allow_multi_label=False,
            dry_run=False,
            storage_root=storage_root,
        )

    assert result.inserted == 0
    assert result.skipped.get("no vocalization label") == 1


async def test_multi_label_without_flag_skips(session_factory, tmp_path):
    await _seed_fixture(session_factory, tmp_path, labels=["upcall", "downsweep"])
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        result = await run_bootstrap(
            session,
            row_ids=["row-aaa"],
            dataset_name="test-ds",
            dataset_id=None,
            crop_seconds=10.0,
            allow_multi_label=False,
            dry_run=False,
            storage_root=storage_root,
        )

    assert result.inserted == 0
    assert result.skipped.get("multi-label") == 1


async def test_multi_label_with_flag_inserts(session_factory, tmp_path):
    await _seed_fixture(session_factory, tmp_path, labels=["upcall", "downsweep"])
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        result = await run_bootstrap(
            session,
            row_ids=["row-aaa"],
            dataset_name="test-ds",
            dataset_id=None,
            crop_seconds=10.0,
            allow_multi_label=True,
            dry_run=False,
            storage_root=storage_root,
        )

    assert result.inserted == 1
    assert sum(result.skipped.values()) == 0


async def test_idempotency_no_duplicate_insert(session_factory, tmp_path):
    await _seed_fixture(session_factory, tmp_path)
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        first = await run_bootstrap(
            session,
            row_ids=["row-aaa"],
            dataset_name="test-ds",
            dataset_id=None,
            crop_seconds=10.0,
            allow_multi_label=False,
            dry_run=False,
            storage_root=storage_root,
        )
    assert first.inserted == 1

    async with session_factory() as session:
        second = await run_bootstrap(
            session,
            row_ids=["row-aaa"],
            dataset_name=None,
            dataset_id=first.dataset_id,
            crop_seconds=10.0,
            allow_multi_label=False,
            dry_run=False,
            storage_root=storage_root,
        )
    assert second.inserted == 0
    assert second.skipped.get("already present") == 1

    async with session_factory() as session:
        count = len(
            (await session.execute(select(SegmentationTrainingSample))).scalars().all()
        )
        assert count == 1


async def test_dry_run_no_db_changes(session_factory, tmp_path):
    await _seed_fixture(session_factory, tmp_path)
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        result = await run_bootstrap(
            session,
            row_ids=["row-aaa"],
            dataset_name="test-ds",
            dataset_id=None,
            crop_seconds=10.0,
            allow_multi_label=False,
            dry_run=True,
            storage_root=storage_root,
        )

    assert result.inserted == 1

    async with session_factory() as session:
        samples = (
            (await session.execute(select(SegmentationTrainingSample))).scalars().all()
        )
        assert len(samples) == 0
        datasets = (
            (await session.execute(select(SegmentationTrainingDataset))).scalars().all()
        )
        assert len(datasets) == 0


async def test_unknown_row_id_skips_and_continues(session_factory, tmp_path):
    await _seed_fixture(session_factory, tmp_path)
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        result = await run_bootstrap(
            session,
            row_ids=["unknown-row", "row-aaa"],
            dataset_name="test-ds",
            dataset_id=None,
            crop_seconds=10.0,
            allow_multi_label=False,
            dry_run=False,
            storage_root=storage_root,
        )

    assert result.inserted == 1
    assert result.skipped.get("no vocalization label") == 1


async def test_crop_too_short_at_boundary(session_factory, tmp_path):
    await _seed_fixture(
        session_factory,
        tmp_path,
        start_utc=1001.0,
        end_utc=1002.0,
        job_start_ts=1000.0,
        job_end_ts=1003.0,
    )
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        result = await run_bootstrap(
            session,
            row_ids=["row-aaa"],
            dataset_name="test-ds",
            dataset_id=None,
            crop_seconds=20.0,
            allow_multi_label=False,
            dry_run=False,
            storage_root=storage_root,
        )

    assert result.inserted == 0
    assert result.skipped.get("crop too short at boundary") == 1


async def test_crop_clamped_to_job_range(session_factory, tmp_path):
    """Crop window is clamped to job's [start_timestamp, end_timestamp]."""
    await _seed_fixture(
        session_factory,
        tmp_path,
        start_utc=1002.0,
        end_utc=1004.0,
        job_start_ts=1000.0,
        job_end_ts=1010.0,
    )
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        result = await run_bootstrap(
            session,
            row_ids=["row-aaa"],
            dataset_name="test-ds",
            dataset_id=None,
            crop_seconds=10.0,
            allow_multi_label=False,
            dry_run=False,
            storage_root=storage_root,
        )

    assert result.inserted == 1

    async with session_factory() as session:
        samples = (
            (await session.execute(select(SegmentationTrainingSample))).scalars().all()
        )
        sample = samples[0]
        assert sample.start_timestamp >= 1000.0
        assert sample.end_timestamp <= 1010.0


def test_parse_args_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(
            ["--row-ids-file", "f.txt", "--dataset-name", "x", "--dataset-id", "y"]
        )

    with pytest.raises(SystemExit):
        parse_args(["--row-ids-file", "f.txt"])


async def test_nonexistent_dataset_id_errors(session_factory, tmp_path):
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(SystemExit, match="not found"):
        async with session_factory() as session:
            await run_bootstrap(
                session,
                row_ids=["row-aaa"],
                dataset_name=None,
                dataset_id="does-not-exist",
                crop_seconds=10.0,
                allow_multi_label=False,
                dry_run=False,
                storage_root=storage_root,
            )

    async with session_factory() as session:
        samples = (
            (await session.execute(select(SegmentationTrainingSample))).scalars().all()
        )
        assert len(samples) == 0


def test_read_row_ids_skips_blanks_and_comments(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("# header comment\nrow-1\n\n  row-2  \n# another\nrow-3\n")
    ids = read_row_ids(f)
    assert ids == ["row-1", "row-2", "row-3"]


# -- _discover_row_ids_from_jobs tests --


async def _seed_discovery_fixture(
    session_factory,
    *,
    rows: list[tuple[str, list[tuple[str, str]]]],
) -> str:
    """Seed a hydrophone detection job with multiple rows, each having labels.

    ``rows`` is a list of ``(row_id, [(label, source), ...])``.
    Returns the detection job ID.
    """
    async with session_factory() as session:
        cm = ClassifierModel(
            name="fake-cm",
            model_path="/tmp/not-a-real-file.joblib",
            model_version="perch_v1",
            vector_dim=64,
            window_size_seconds=5.0,
            target_sample_rate=16000,
        )
        session.add(cm)
        await session.flush()

        dj = DetectionJob(
            classifier_model_id=cm.id,
            hydrophone_id=HYDROPHONE_ID,
            start_timestamp=JOB_START_TS,
            end_timestamp=JOB_END_TS,
            status="complete",
        )
        session.add(dj)
        await session.flush()

        for row_id, label_pairs in rows:
            for label, source in label_pairs:
                session.add(
                    VocalizationLabel(
                        detection_job_id=dj.id,
                        row_id=row_id,
                        label=label,
                        source=source,
                    )
                )

        await session.commit()
        return dj.id


async def test_discover_single_label_rows(session_factory):
    dj_id = await _seed_discovery_fixture(
        session_factory,
        rows=[
            ("row-single", [("upcall", "manual")]),
            ("row-multi", [("upcall", "manual"), ("downsweep", "manual")]),
            ("row-neg", [("(Negative)", "manual")]),
        ],
    )

    async with session_factory() as session:
        row_ids = await _discover_row_ids_from_jobs(session, [dj_id])

    assert row_ids == ["row-single"]


async def test_discover_skips_inference_labels(session_factory):
    dj_id = await _seed_discovery_fixture(
        session_factory,
        rows=[
            ("row-manual", [("upcall", "manual")]),
            ("row-infer", [("upcall", "inference")]),
            ("row-mixed", [("upcall", "manual"), ("downsweep", "inference")]),
        ],
    )

    async with session_factory() as session:
        row_ids = await _discover_row_ids_from_jobs(session, [dj_id])

    assert set(row_ids) == {"row-manual", "row-mixed"}


async def test_discover_nonexistent_job_returns_empty(session_factory):
    async with session_factory() as session:
        row_ids = await _discover_row_ids_from_jobs(session, ["does-not-exist"])

    assert row_ids == []


async def test_inference_labels_skipped_in_row_ids_path(session_factory, tmp_path):
    """Rows with only inference labels are skipped even via --row-ids-file."""
    storage_root = tmp_path / "storage"

    async with session_factory() as session:
        cm = ClassifierModel(
            name="fake-cm",
            model_path="/tmp/not-a-real-file.joblib",
            model_version="perch_v1",
            vector_dim=64,
            window_size_seconds=5.0,
            target_sample_rate=16000,
        )
        session.add(cm)
        await session.flush()

        dj = DetectionJob(
            classifier_model_id=cm.id,
            hydrophone_id=HYDROPHONE_ID,
            start_timestamp=JOB_START_TS,
            end_timestamp=JOB_END_TS,
            status="complete",
        )
        session.add(dj)
        await session.flush()

        session.add(
            VocalizationLabel(
                detection_job_id=dj.id,
                row_id="row-infer",
                label="upcall",
                source="inference",
            )
        )
        await session.commit()
        dj_id = dj.id

    rs_path = detection_row_store_path(storage_root, dj_id)
    rs_path.parent.mkdir(parents=True, exist_ok=True)
    write_detection_row_store(
        rs_path,
        [
            {
                "row_id": "row-infer",
                "start_utc": "1100.0",
                "end_utc": "1105.0",
                "avg_confidence": "0.8",
                "peak_confidence": "0.9",
                "n_windows": "2",
            }
        ],
    )

    async with session_factory() as session:
        result = await run_bootstrap(
            session,
            row_ids=["row-infer"],
            dataset_name="test-ds",
            dataset_id=None,
            crop_seconds=10.0,
            allow_multi_label=False,
            dry_run=False,
            storage_root=storage_root,
        )

    assert result.inserted == 0
    assert result.skipped.get("no vocalization label") == 1
