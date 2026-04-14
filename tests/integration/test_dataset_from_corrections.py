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
from humpback.models.feedback_training import EventBoundaryCorrection
from humpback.models.segmentation_training import (
    SegmentationTrainingDataset,
    SegmentationTrainingSample,
)
from humpback.storage import ensure_dir

BASE = "/call-parsing"


async def _seed_job_with_corrections(app_settings, *, with_corrections: bool = True):
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
                # Add corrections for region r1 only
                session.add(
                    EventBoundaryCorrection(
                        event_segmentation_job_id=seg_job.id,
                        event_id="e1",
                        region_id="r1",
                        correction_type="adjust",
                        start_sec=105.5,
                        end_sec=107.5,
                    )
                )
                session.add(
                    EventBoundaryCorrection(
                        event_segmentation_job_id=seg_job.id,
                        event_id="new1",
                        region_id="r1",
                        correction_type="add",
                        start_sec=110.0,
                        end_sec=112.0,
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
async def test_create_training_job_missing_dataset_returns_404(
    client: AsyncClient, app_settings
):
    resp = await client.post(
        f"{BASE}/segmentation-training-jobs",
        json={"training_dataset_id": "nonexistent"},
    )
    assert resp.status_code == 404
