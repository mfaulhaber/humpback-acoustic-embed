"""Unit tests for the Pass 1 region boundary correction overlay."""

from __future__ import annotations

from pathlib import Path

import pytest

from humpback.call_parsing.regions_overlay import load_corrected_regions
from humpback.call_parsing.storage import write_regions
from humpback.call_parsing.types import Region
from humpback.database import Base, create_engine, create_session_factory
from humpback.models.feedback_training import RegionBoundaryCorrection


def _make_region(region_id: str, start: float, end: float) -> Region:
    return Region(
        region_id=region_id,
        start_sec=start,
        end_sec=end,
        padded_start_sec=start - 1.0,
        padded_end_sec=end + 1.0,
        max_score=0.9,
        mean_score=0.7,
        n_windows=10,
    )


@pytest.fixture
async def db_session(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sf = create_session_factory(engine)
    async with sf() as session:
        yield session
    await engine.dispose()


@pytest.fixture
def regions_parquet(tmp_path) -> Path:
    path = tmp_path / "regions.parquet"
    write_regions(
        path,
        [
            _make_region("r1", 10.0, 20.0),
            _make_region("r2", 30.0, 40.0),
            _make_region("r3", 50.0, 60.0),
        ],
    )
    return path


JOB_ID = "test-region-job-id"


@pytest.mark.asyncio
async def test_no_corrections_returns_originals(db_session, regions_parquet):
    result = await load_corrected_regions(db_session, JOB_ID, regions_parquet)
    assert len(result) == 3
    assert [r.region_id for r in result] == ["r1", "r2", "r3"]


@pytest.mark.asyncio
async def test_adjust_modifies_boundaries(db_session, regions_parquet):
    db_session.add(
        RegionBoundaryCorrection(
            region_detection_job_id=JOB_ID,
            region_id="r2",
            correction_type="adjust",
            start_sec=32.0,
            end_sec=38.0,
        )
    )
    await db_session.commit()

    result = await load_corrected_regions(db_session, JOB_ID, regions_parquet)
    assert len(result) == 3
    r2 = [r for r in result if r.region_id == "r2"][0]
    assert r2.start_sec == 32.0
    assert r2.end_sec == 38.0
    assert r2.padded_start_sec == 32.0
    assert r2.padded_end_sec == 38.0


@pytest.mark.asyncio
async def test_delete_removes_region(db_session, regions_parquet):
    db_session.add(
        RegionBoundaryCorrection(
            region_detection_job_id=JOB_ID,
            region_id="r1",
            correction_type="delete",
        )
    )
    await db_session.commit()

    result = await load_corrected_regions(db_session, JOB_ID, regions_parquet)
    assert len(result) == 2
    ids = [r.region_id for r in result]
    assert "r1" not in ids


@pytest.mark.asyncio
async def test_add_inserts_new_region(db_session, regions_parquet):
    db_session.add(
        RegionBoundaryCorrection(
            region_detection_job_id=JOB_ID,
            region_id="new-r",
            correction_type="add",
            start_sec=25.0,
            end_sec=28.0,
        )
    )
    await db_session.commit()

    result = await load_corrected_regions(db_session, JOB_ID, regions_parquet)
    assert len(result) == 4
    new_regions = [r for r in result if r.start_sec == 25.0]
    assert len(new_regions) == 1
    assert new_regions[0].end_sec == 28.0
    assert new_regions[0].region_id == "new-r"


@pytest.mark.asyncio
async def test_multiple_corrections_compose(db_session, regions_parquet):
    db_session.add_all(
        [
            RegionBoundaryCorrection(
                region_detection_job_id=JOB_ID,
                region_id="r1",
                correction_type="delete",
            ),
            RegionBoundaryCorrection(
                region_detection_job_id=JOB_ID,
                region_id="r2",
                correction_type="adjust",
                start_sec=31.0,
                end_sec=39.0,
            ),
            RegionBoundaryCorrection(
                region_detection_job_id=JOB_ID,
                region_id="new-r",
                correction_type="add",
                start_sec=5.0,
                end_sec=8.0,
            ),
        ]
    )
    await db_session.commit()

    result = await load_corrected_regions(db_session, JOB_ID, regions_parquet)
    assert len(result) == 3
    ids = {r.region_id for r in result}
    assert "r1" not in ids
    assert "r3" in ids
    assert result[0].start_sec == 5.0
    r2 = [r for r in result if r.region_id == "r2"][0]
    assert r2.start_sec == 31.0


@pytest.mark.asyncio
async def test_output_sorted_by_start_sec(db_session, regions_parquet):
    db_session.add(
        RegionBoundaryCorrection(
            region_detection_job_id=JOB_ID,
            region_id="new-first",
            correction_type="add",
            start_sec=1.0,
            end_sec=3.0,
        )
    )
    await db_session.commit()

    result = await load_corrected_regions(db_session, JOB_ID, regions_parquet)
    starts = [r.start_sec for r in result]
    assert starts == sorted(starts)
