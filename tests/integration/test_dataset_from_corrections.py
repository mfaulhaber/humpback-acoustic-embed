"""Integration tests for POST /call-parsing/segmentation-training-datasets/from-corrections."""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient

from humpback.call_parsing.storage import (
    region_job_dir,
    segmentation_job_dir,
    write_events,
    write_regions,
)
from humpback.call_parsing.types import Event, Region
from humpback.database import create_engine, create_session_factory
from humpback.models.call_parsing import (
    EventSegmentationJob,
    RegionDetectionJob,
    SegmentationModel,
)
from humpback.models.call_parsing import EventBoundaryCorrection
from humpback.models.segmentation_training import (
    SegmentationTrainingDataset,
    SegmentationTrainingSample,
)
from humpback.schemas.call_parsing import SegmentationTrainingConfig
from humpback.storage import ensure_dir

BASE = "/call-parsing"


async def _seed_job_with_corrections(
    app_settings,
    *,
    with_corrections: bool = True,
    correction_scope: str = "scoped",
    add_extra_segmentation_job: bool = False,
    include_legacy_corrections: bool = False,
):
    """Create a completed segmentation job with regions, events, and corrections.

    Returns (seg_job_id, region_detection_job_id).
    """
    storage_root = app_settings.storage_root

    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            rd = RegionDetectionJob(
                hydrophone_id="orcasound_lab",
                start_timestamp=1000.0,
                end_timestamp=2000.0,
                status="complete",
                config_json="{}",
            )
            session.add(rd)
            await session.flush()

            sm = SegmentationModel(
                name="test-model",
                model_family="pytorch_crnn",
                model_path="/fake/path",
                config_json="{}",
            )
            session.add(sm)
            await session.flush()

            seg_job = EventSegmentationJob(
                region_detection_job_id=rd.id,
                segmentation_model_id=sm.id,
                status="complete",
                event_count=2,
            )
            session.add(seg_job)
            await session.flush()

            if add_extra_segmentation_job:
                session.add(
                    EventSegmentationJob(
                        region_detection_job_id=rd.id,
                        segmentation_model_id=sm.id,
                        status="complete",
                        event_count=2,
                    )
                )

            # Write regions.parquet
            regions_dir = ensure_dir(region_job_dir(storage_root, rd.id))
            write_regions(
                regions_dir / "regions.parquet",
                [
                    Region(
                        region_id="r1",
                        start_sec=100.0,
                        end_sec=120.0,
                        padded_start_sec=99.0,
                        padded_end_sec=121.0,
                        max_score=0.95,
                        mean_score=0.8,
                        n_windows=5,
                    ),
                    Region(
                        region_id="r2",
                        start_sec=200.0,
                        end_sec=210.0,
                        padded_start_sec=199.0,
                        padded_end_sec=211.0,
                        max_score=0.85,
                        mean_score=0.7,
                        n_windows=3,
                    ),
                ],
            )

            # Write events.parquet
            seg_dir = ensure_dir(segmentation_job_dir(storage_root, seg_job.id))
            write_events(
                seg_dir / "events.parquet",
                [
                    Event(
                        event_id="e1",
                        region_id="r1",
                        start_sec=105.0,
                        end_sec=107.0,
                        center_sec=106.0,
                        segmentation_confidence=0.9,
                    ),
                    Event(
                        event_id="e2",
                        region_id="r2",
                        start_sec=203.0,
                        end_sec=205.0,
                        center_sec=204.0,
                        segmentation_confidence=0.85,
                    ),
                ],
            )

            if with_corrections:
                correction_seg_job_id = (
                    seg_job.id if correction_scope == "scoped" else None
                )
                # Add corrections for region r1 only
                session.add(
                    EventBoundaryCorrection(
                        region_detection_job_id=rd.id,
                        event_segmentation_job_id=correction_seg_job_id,
                        region_id="r1",
                        source_event_id="e1",
                        correction_type="adjust",
                        original_start_sec=105.0,
                        original_end_sec=107.0,
                        corrected_start_sec=105.5,
                        corrected_end_sec=107.5,
                    )
                )
                session.add(
                    EventBoundaryCorrection(
                        region_detection_job_id=rd.id,
                        event_segmentation_job_id=correction_seg_job_id,
                        region_id="r1",
                        correction_type="add",
                        corrected_start_sec=110.0,
                        corrected_end_sec=112.0,
                    )
                )
            if include_legacy_corrections:
                session.add(
                    EventBoundaryCorrection(
                        region_detection_job_id=rd.id,
                        region_id="r1",
                        correction_type="add",
                        corrected_start_sec=115.0,
                        corrected_end_sec=116.0,
                    )
                )

            await session.commit()
            return seg_job.id, rd.id
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_create_dataset_from_corrections(client: AsyncClient, app_settings):
    seg_job_id, _ = await _seed_job_with_corrections(app_settings)

    resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={"segmentation_job_ids": [seg_job_id]},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["name"] == f"corrections-{seg_job_id[:8]}"
    assert data["sample_count"] > 0
    assert data["selected_job_count"] == 1
    assert data["source_job_count"] == 1
    assert data["skipped_job_count"] == 0
    assert data["skipped_jobs"] == []
    assert "id" in data

    # Verify dataset exists in DB
    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            ds = await session.get(SegmentationTrainingDataset, data["id"])
            assert ds is not None
            assert ds.name == data["name"]

            from sqlalchemy import select

            samples_result = await session.execute(
                select(SegmentationTrainingSample).where(
                    SegmentationTrainingSample.training_dataset_id == data["id"]
                )
            )
            samples = list(samples_result.scalars().all())
            assert len(samples) == data["sample_count"]

            # All samples should have correct source metadata
            for s in samples:
                assert s.source == "boundary_correction"
                assert s.source_ref == seg_job_id
                assert s.hydrophone_id == "orcasound_lab"
                events = json.loads(s.events_json)
                assert isinstance(events, list)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_create_dataset_custom_name(client: AsyncClient, app_settings):
    seg_job_id, _ = await _seed_job_with_corrections(app_settings)

    resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={
            "segmentation_job_ids": [seg_job_id],
            "name": "my-dataset",
            "description": "Test dataset",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "my-dataset"


@pytest.mark.asyncio
async def test_create_dataset_no_corrections_returns_400(
    client: AsyncClient, app_settings
):
    seg_job_id, _ = await _seed_job_with_corrections(
        app_settings, with_corrections=False
    )

    resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={"segmentation_job_ids": [seg_job_id]},
    )
    assert resp.status_code == 400
    assert "No corrected regions" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_create_dataset_missing_job_returns_404(
    client: AsyncClient, app_settings
):
    resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={"segmentation_job_ids": ["nonexistent-id"]},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_dataset_incomplete_job_returns_400(
    client: AsyncClient, app_settings
):
    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            rd = RegionDetectionJob(
                hydrophone_id="h1",
                start_timestamp=0.0,
                end_timestamp=100.0,
                status="complete",
                config_json="{}",
            )
            session.add(rd)
            await session.flush()

            sm = SegmentationModel(
                name="test-model",
                model_family="pytorch_crnn",
                model_path="/fake",
                config_json="{}",
            )
            session.add(sm)
            await session.flush()

            seg_job = EventSegmentationJob(
                region_detection_job_id=rd.id,
                segmentation_model_id=sm.id,
                status="running",
            )
            session.add(seg_job)
            await session.commit()
            seg_job_id = seg_job.id
    finally:
        await engine.dispose()

    resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={"segmentation_job_ids": [seg_job_id]},
    )
    assert resp.status_code == 400
    assert "not complete" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_only_corrected_regions_included(client: AsyncClient, app_settings):
    """Only region r1 has corrections; r2 should not appear in samples."""
    seg_job_id, _ = await _seed_job_with_corrections(app_settings)

    resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={"segmentation_job_ids": [seg_job_id]},
    )
    assert resp.status_code == 201
    data = resp.json()

    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            from sqlalchemy import select

            samples_result = await session.execute(
                select(SegmentationTrainingSample).where(
                    SegmentationTrainingSample.training_dataset_id == data["id"]
                )
            )
            samples = list(samples_result.scalars().all())
            # Region r1 spans 99.0-121.0 = 22s, which is < 30s max crop
            # So we should get exactly 1 sample
            assert len(samples) == 1
            events = json.loads(samples[0].events_json)
            # Should have 2 events: adjusted e1 and added new1
            assert len(events) == 2
            starts = sorted(e["start_sec"] for e in events)
            assert starts == [105.5, 110.0]
    finally:
        await engine.dispose()


# ---- Multi-job dataset tests ------------------------------------------------


@pytest.mark.asyncio
async def test_multi_job_dataset_combines_samples(client: AsyncClient, app_settings):
    """Two jobs with corrections produce samples with distinct source_ref values."""
    seg_id_1, _ = await _seed_job_with_corrections(app_settings)
    seg_id_2, _ = await _seed_job_with_corrections(app_settings)

    resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={"segmentation_job_ids": [seg_id_1, seg_id_2]},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["sample_count"] >= 2
    assert data["name"] == f"corrections-2jobs-{seg_id_1[:8]}"
    assert data["selected_job_count"] == 2
    assert data["source_job_count"] == 2
    assert data["skipped_job_count"] == 0

    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            from sqlalchemy import select

            samples_result = await session.execute(
                select(SegmentationTrainingSample).where(
                    SegmentationTrainingSample.training_dataset_id == data["id"]
                )
            )
            samples = list(samples_result.scalars().all())
            source_refs = {s.source_ref for s in samples}
            assert seg_id_1 in source_refs
            assert seg_id_2 in source_refs
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_multi_job_skips_jobs_without_corrections(
    client: AsyncClient, app_settings
):
    """One job with corrections + one without → only the corrected job contributes."""
    seg_id_1, _ = await _seed_job_with_corrections(app_settings)
    seg_id_2, _ = await _seed_job_with_corrections(app_settings, with_corrections=False)

    resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={"segmentation_job_ids": [seg_id_1, seg_id_2]},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["sample_count"] >= 1
    assert data["selected_job_count"] == 2
    assert data["source_job_count"] == 1
    assert data["skipped_job_count"] == 1
    assert data["skipped_jobs"][0]["segmentation_job_id"] == seg_id_2
    assert data["skipped_jobs"][0]["correction_mode"] == "none"

    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            from sqlalchemy import select

            samples_result = await session.execute(
                select(SegmentationTrainingSample).where(
                    SegmentationTrainingSample.training_dataset_id == data["id"]
                )
            )
            samples = list(samples_result.scalars().all())
            source_refs = {s.source_ref for s in samples}
            assert seg_id_1 in source_refs
            assert seg_id_2 not in source_refs
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_multi_job_includes_unambiguous_legacy_corrections(
    client: AsyncClient, app_settings
):
    """Legacy region-scoped corrections contribute when ownership is unambiguous."""
    scoped_seg_id, _ = await _seed_job_with_corrections(app_settings)
    legacy_seg_id, _ = await _seed_job_with_corrections(
        app_settings,
        correction_scope="legacy",
    )

    resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={"segmentation_job_ids": [scoped_seg_id, legacy_seg_id]},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["selected_job_count"] == 2
    assert data["source_job_count"] == 2
    assert data["skipped_job_count"] == 0

    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            from sqlalchemy import select

            samples_result = await session.execute(
                select(SegmentationTrainingSample).where(
                    SegmentationTrainingSample.training_dataset_id == data["id"]
                )
            )
            samples = list(samples_result.scalars().all())
            source_refs = {s.source_ref for s in samples}
            assert scoped_seg_id in source_refs
            assert legacy_seg_id in source_refs
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_multi_job_skips_ambiguous_legacy_corrections(
    client: AsyncClient, app_settings
):
    """Legacy rows are skipped when multiple segmentation jobs share one region job."""
    scoped_seg_id, _ = await _seed_job_with_corrections(app_settings)
    ambiguous_seg_id, _ = await _seed_job_with_corrections(
        app_settings,
        correction_scope="legacy",
        add_extra_segmentation_job=True,
    )

    resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={"segmentation_job_ids": [scoped_seg_id, ambiguous_seg_id]},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["selected_job_count"] == 2
    assert data["source_job_count"] == 1
    assert data["skipped_job_count"] == 1
    assert data["skipped_jobs"] == [
        {
            "segmentation_job_id": ambiguous_seg_id,
            "reason": (
                "legacy region-scoped corrections are ambiguous because the "
                "region detection job has multiple segmentation jobs"
            ),
            "correction_mode": "legacy_ambiguous",
        }
    ]

    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            from sqlalchemy import select

            samples_result = await session.execute(
                select(SegmentationTrainingSample).where(
                    SegmentationTrainingSample.training_dataset_id == data["id"]
                )
            )
            samples = list(samples_result.scalars().all())
            source_refs = {s.source_ref for s in samples}
            assert scoped_seg_id in source_refs
            assert ambiguous_seg_id not in source_refs
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_scoped_corrections_take_precedence_over_legacy(
    client: AsyncClient, app_settings
):
    """Modern scoped corrections are authoritative when legacy rows coexist."""
    seg_id, _ = await _seed_job_with_corrections(
        app_settings,
        include_legacy_corrections=True,
    )

    resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={"segmentation_job_ids": [seg_id]},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["source_job_count"] == 1
    assert data["skipped_job_count"] == 0

    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            from sqlalchemy import select

            samples_result = await session.execute(
                select(SegmentationTrainingSample).where(
                    SegmentationTrainingSample.training_dataset_id == data["id"]
                )
            )
            samples = list(samples_result.scalars().all())
            assert len(samples) == 1
            events = json.loads(samples[0].events_json)
            starts = sorted(e["start_sec"] for e in events)
            assert starts == [105.5, 110.0]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_multi_job_all_no_corrections_returns_400(
    client: AsyncClient, app_settings
):
    """All jobs having no corrections raises 400."""
    seg_id_1, _ = await _seed_job_with_corrections(app_settings, with_corrections=False)
    seg_id_2, _ = await _seed_job_with_corrections(app_settings, with_corrections=False)

    resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={"segmentation_job_ids": [seg_id_1, seg_id_2]},
    )
    assert resp.status_code == 400
    assert "No corrected regions" in resp.json()["detail"]


# ---- Quick retrain tests ----------------------------------------------------


@pytest.mark.asyncio
async def test_quick_retrain_creates_dataset_and_training_job(
    client: AsyncClient, app_settings
):
    seg_job_id, _ = await _seed_job_with_corrections(app_settings)

    resp = await client.post(
        f"{BASE}/segmentation-training/quick-retrain",
        json={"segmentation_job_id": seg_job_id},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert "dataset_id" in data
    assert "training_job_id" in data
    assert data["sample_count"] > 0

    # Verify the training job exists and is queued
    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            from humpback.models.segmentation_training import SegmentationTrainingJob

            job = await session.get(SegmentationTrainingJob, data["training_job_id"])
            assert job is not None
            assert job.status == "queued"
            assert job.training_dataset_id == data["dataset_id"]
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_quick_retrain_no_corrections_returns_400(
    client: AsyncClient, app_settings
):
    seg_job_id, _ = await _seed_job_with_corrections(
        app_settings, with_corrections=False
    )

    resp = await client.post(
        f"{BASE}/segmentation-training/quick-retrain",
        json={"segmentation_job_id": seg_job_id},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_quick_retrain_missing_job_returns_404(client: AsyncClient, app_settings):
    resp = await client.post(
        f"{BASE}/segmentation-training/quick-retrain",
        json={"segmentation_job_id": "nonexistent"},
    )
    assert resp.status_code == 404


# ---- Segmentation training job from existing dataset ----------------------


@pytest.mark.asyncio
async def test_create_training_job_from_dataset(client: AsyncClient, app_settings):
    """POST /segmentation-training-jobs creates a queued job for an existing dataset."""
    seg_job_id, _ = await _seed_job_with_corrections(app_settings)

    # Create a dataset first
    ds_resp = await client.post(
        f"{BASE}/segmentation-training-datasets/from-corrections",
        json={"segmentation_job_ids": [seg_job_id]},
    )
    assert ds_resp.status_code == 201
    dataset_id = ds_resp.json()["id"]

    # Create a training job from the dataset
    resp = await client.post(
        f"{BASE}/segmentation-training-jobs",
        json={"training_dataset_id": dataset_id},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["training_dataset_id"] == dataset_id
    assert data["status"] == "queued"
    assert "id" in data


@pytest.mark.asyncio
async def test_create_training_job_from_segmentation_jobs(
    client: AsyncClient, app_settings
):
    """POST /segmentation-training-jobs can build the dataset internally."""
    seg_job_id, _ = await _seed_job_with_corrections(app_settings)

    resp = await client.post(
        f"{BASE}/segmentation-training-jobs",
        json={"segmentation_job_ids": [seg_job_id]},
    )

    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["status"] == "queued"
    assert data["training_dataset_id"]
    assert data["segmentation_model_id"] is None

    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            dataset = await session.get(
                SegmentationTrainingDataset, data["training_dataset_id"]
            )
            assert dataset is not None

            from sqlalchemy import select

            samples_result = await session.execute(
                select(SegmentationTrainingSample).where(
                    SegmentationTrainingSample.training_dataset_id
                    == data["training_dataset_id"]
                )
            )
            samples = list(samples_result.scalars().all())
            assert samples
            assert {s.source_ref for s in samples} == {seg_job_id}
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_list_segmentation_training_jobs(client: AsyncClient, app_settings):
    seg_job_id, _ = await _seed_job_with_corrections(app_settings)
    create_resp = await client.post(
        f"{BASE}/segmentation-training-jobs",
        json={"segmentation_job_ids": [seg_job_id]},
    )
    assert create_resp.status_code == 201, create_resp.text
    created_id = create_resp.json()["id"]

    resp = await client.get(f"{BASE}/segmentation-training-jobs")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data
    assert data[0]["id"] == created_id
    assert data[0]["status"] == "queued"


@pytest.mark.asyncio
async def test_create_training_job_rejects_missing_source(client: AsyncClient):
    resp = await client.post(f"{BASE}/segmentation-training-jobs", json={})

    assert resp.status_code == 422
    assert "exactly one" in resp.text


@pytest.mark.asyncio
async def test_create_training_job_rejects_multiple_sources(
    client: AsyncClient, app_settings
):
    seg_job_id, _ = await _seed_job_with_corrections(app_settings)

    resp = await client.post(
        f"{BASE}/segmentation-training-jobs",
        json={
            "training_dataset_id": "dataset-1",
            "segmentation_job_ids": [seg_job_id],
        },
    )

    assert resp.status_code == 422
    assert "exactly one" in resp.text


@pytest.mark.asyncio
async def test_create_training_job_accepts_advanced_config(
    client: AsyncClient, app_settings
):
    seg_job_id, _ = await _seed_job_with_corrections(app_settings)

    resp = await client.post(
        f"{BASE}/segmentation-training-jobs",
        json={
            "segmentation_job_ids": [seg_job_id],
            "config": {
                "epochs": 4,
                "batch_size": 2,
                "learning_rate": 0.0005,
                "n_mels": 48,
                "feature_config": {
                    "sample_rate": 24000,
                    "n_fft": 1024,
                    "hop_length": 256,
                    "n_mels": 48,
                    "fmin": 30.0,
                    "fmax": 5000.0,
                    "normalize": "per_region_zscore",
                },
            },
        },
    )

    assert resp.status_code == 201, resp.text
    config = json.loads(resp.json()["config_json"])
    assert config["epochs"] == 4
    assert config["batch_size"] == 2
    assert config["learning_rate"] == 0.0005
    assert config["n_mels"] == 48
    assert config["feature_config"]["sample_rate"] == 24000
    assert config["feature_config"]["hop_length"] == 256


def test_segmentation_training_config_preserves_legacy_flat_n_mels():
    config = SegmentationTrainingConfig.model_validate({"n_mels": 32})

    assert config.n_mels == 32
    assert config.feature_config.n_mels == 32


def test_segmentation_training_config_rejects_mismatched_feature_n_mels():
    with pytest.raises(ValueError, match="feature_config.n_mels"):
        SegmentationTrainingConfig.model_validate(
            {
                "n_mels": 64,
                "feature_config": {"n_mels": 32},
            }
        )


@pytest.mark.asyncio
async def test_create_training_job_missing_dataset_returns_404(
    client: AsyncClient, app_settings
):
    resp = await client.post(
        f"{BASE}/segmentation-training-jobs",
        json={"training_dataset_id": "nonexistent"},
    )
    assert resp.status_code == 404


# ---- load_corrected_events tests -------------------------------------------


@pytest.mark.asyncio
async def test_load_corrected_events_no_corrections(client: AsyncClient, app_settings):
    """No corrections → original events returned unchanged."""
    from humpback.call_parsing.segmentation.extraction import load_corrected_events

    seg_job_id, rd_id = await _seed_job_with_corrections(
        app_settings, with_corrections=False
    )

    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            events = await load_corrected_events(
                session, rd_id, seg_job_id, app_settings.storage_root
            )
            assert len(events) == 2
            ids = {e.event_id for e in events}
            assert ids == {"e1", "e2"}
            # Original boundaries preserved
            e1 = next(e for e in events if e.event_id == "e1")
            assert e1.start_sec == 105.0
            assert e1.end_sec == 107.0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_load_corrected_events_with_corrections(
    client: AsyncClient, app_settings
):
    """Corrections applied: adjust e1, add new1, e2 unchanged."""
    from humpback.call_parsing.segmentation.extraction import load_corrected_events

    seg_job_id, rd_id = await _seed_job_with_corrections(app_settings)

    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            events = await load_corrected_events(
                session, rd_id, seg_job_id, app_settings.storage_root
            )
            # e2 unchanged, e1 adjusted, plus one added event
            assert len(events) == 3

            # e1 adjusted
            e1 = next(e for e in events if e.start_sec == 105.5)
            assert e1.end_sec == 107.5
            assert e1.region_id == "r1"
            assert e1.center_sec == pytest.approx((105.5 + 107.5) / 2.0)

            # e2 unchanged
            e2 = next(e for e in events if e.start_sec == 203.0)
            assert e2.end_sec == 205.0

            # added event
            added = next(e for e in events if e.start_sec == 110.0)
            assert added.end_sec == 112.0
            assert added.region_id == "r1"
            assert added.segmentation_confidence == 0.0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_load_corrected_events_delete(client: AsyncClient, app_settings):
    """Delete correction removes the event from the result."""
    from humpback.call_parsing.segmentation.extraction import load_corrected_events

    storage_root = app_settings.storage_root
    engine = create_engine(app_settings.database_url)
    try:
        sf = create_session_factory(engine)
        async with sf() as session:
            rd = RegionDetectionJob(
                hydrophone_id="h1",
                start_timestamp=0.0,
                end_timestamp=100.0,
                status="complete",
                config_json="{}",
            )
            session.add(rd)
            await session.flush()

            sm = SegmentationModel(
                name="test-model",
                model_family="pytorch_crnn",
                model_path="/fake",
                config_json="{}",
            )
            session.add(sm)
            await session.flush()

            seg_job = EventSegmentationJob(
                region_detection_job_id=rd.id,
                segmentation_model_id=sm.id,
                status="complete",
                event_count=1,
            )
            session.add(seg_job)
            await session.flush()

            seg_dir = ensure_dir(segmentation_job_dir(storage_root, seg_job.id))
            write_events(
                seg_dir / "events.parquet",
                [
                    Event(
                        event_id="e1",
                        region_id="r1",
                        start_sec=10.0,
                        end_sec=12.0,
                        center_sec=11.0,
                        segmentation_confidence=0.9,
                    ),
                ],
            )

            session.add(
                EventBoundaryCorrection(
                    region_detection_job_id=rd.id,
                    region_id="r1",
                    correction_type="delete",
                    original_start_sec=10.0,
                    original_end_sec=12.0,
                )
            )
            await session.commit()

            events = await load_corrected_events(
                session, rd.id, seg_job.id, storage_root
            )
            assert len(events) == 0
    finally:
        await engine.dispose()
