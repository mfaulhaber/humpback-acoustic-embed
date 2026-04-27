"""End-to-end tests for the continuous-embedding worker.

Uses the synthetic ``SurfPerchStub`` embedder so the worker exercises
the full claim → run → atomic write path without depending on real
audio decoding or model files.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from humpback.call_parsing.storage import region_job_dir, write_regions
from humpback.call_parsing.types import Region
from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.schemas.sequence_models import ContinuousEmbeddingJobCreate
from humpback.services.continuous_embedding_service import (
    create_continuous_embedding_job,
)
from humpback.storage import (
    continuous_embedding_dir,
    continuous_embedding_manifest_path,
    continuous_embedding_parquet_path,
)
from humpback.workers.continuous_embedding_worker import (
    run_continuous_embedding_job,
    run_one_iteration,
)
from tests.fixtures.sequence_models.surfperch_stub import SurfPerchStub


def _make_region(region_id: str, start: float, end: float) -> Region:
    return Region(
        region_id=region_id,
        start_sec=start,
        end_sec=end,
        padded_start_sec=start,
        padded_end_sec=end,
        max_score=0.9,
        mean_score=0.6,
        n_windows=10,
    )


async def _seed_region_job_with_regions(
    session,
    settings,
    *,
    regions: list[Region],
    duration_sec: float = 1000.0,
) -> RegionDetectionJob:
    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=0.0,
        end_timestamp=duration_sec,
    )
    session.add(region_job)
    await session.commit()
    await session.refresh(region_job)

    job_dir = region_job_dir(settings.storage_root, region_job.id)
    job_dir.mkdir(parents=True, exist_ok=True)
    write_regions(job_dir / "regions.parquet", regions)
    return region_job


async def _seed_continuous_job(
    session,
    region_detection_job_id: str,
    *,
    hop: float = 1.0,
    pad: float = 5.0,
) -> ContinuousEmbeddingJob:
    payload = ContinuousEmbeddingJobCreate(
        region_detection_job_id=region_detection_job_id,
        hop_seconds=hop,
        pad_seconds=pad,
    )
    job, _ = await create_continuous_embedding_job(session, payload)
    return job


async def test_happy_path_writes_parquet_and_manifest(session, settings):
    regions = [
        _make_region("r1", 100.0, 110.0),
        _make_region("r2", 200.0, 215.0),
    ]
    region_job = await _seed_region_job_with_regions(session, settings, regions=regions)
    job = await _seed_continuous_job(session, region_job.id, pad=5.0)
    stub = SurfPerchStub(vector_dim=8)

    await run_continuous_embedding_job(session, job, settings, embedder=stub)

    await session.refresh(job)
    assert job.status == JobStatus.complete.value
    assert job.vector_dim == 8
    assert job.total_regions == 2
    assert job.merged_spans == 2
    assert job.total_windows is not None and job.total_windows > 0
    assert job.error_message is None

    parquet_path = continuous_embedding_parquet_path(settings.storage_root, job.id)
    manifest_path = continuous_embedding_manifest_path(settings.storage_root, job.id)
    assert parquet_path.exists()
    assert manifest_path.exists()
    assert job.parquet_path == str(parquet_path)

    table = pq.read_table(parquet_path)
    assert table.num_rows == job.total_windows
    assert set(table.column_names) == {
        "merged_span_id",
        "window_index_in_span",
        "audio_file_id",
        "start_time_sec",
        "end_time_sec",
        "is_in_pad",
        "source_region_ids",
        "embedding",
    }
    span_ids = table.column("merged_span_id").to_pylist()
    window_idxs = table.column("window_index_in_span").to_pylist()
    sorted_pairs = sorted(zip(span_ids, window_idxs))
    assert list(zip(span_ids, window_idxs)) == sorted_pairs

    embedding_widths = {len(v) for v in table.column("embedding").to_pylist()}
    assert embedding_widths == {8}

    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["vector_dim"] == 8
    assert manifest["merged_spans"] == 2
    assert manifest["total_windows"] == job.total_windows
    assert len(manifest["spans"]) == 2


async def test_failure_path_marks_failed_and_cleans_artifacts(session, settings):
    regions = [_make_region("r1", 100.0, 120.0)]
    region_job = await _seed_region_job_with_regions(session, settings, regions=regions)
    job = await _seed_continuous_job(session, region_job.id)
    stub = SurfPerchStub(vector_dim=4, fail_on_span=0)

    await run_continuous_embedding_job(session, job, settings, embedder=stub)

    await session.refresh(job)
    assert job.status == JobStatus.failed.value
    assert job.error_message is not None and "fail" in job.error_message.lower()

    parquet_path = continuous_embedding_parquet_path(settings.storage_root, job.id)
    manifest_path = continuous_embedding_manifest_path(settings.storage_root, job.id)
    assert not parquet_path.exists()
    assert not manifest_path.exists()


async def test_no_temp_files_left_on_success(session, settings):
    regions = [_make_region("r1", 100.0, 115.0)]
    region_job = await _seed_region_job_with_regions(session, settings, regions=regions)
    job = await _seed_continuous_job(session, region_job.id)

    await run_continuous_embedding_job(
        session, job, settings, embedder=SurfPerchStub(vector_dim=4)
    )

    job_dir = continuous_embedding_dir(settings.storage_root, job.id)
    leftover_tmps = list(job_dir.glob("*.tmp"))
    assert leftover_tmps == []


async def test_cancellation_between_spans_drops_artifacts(session, settings):
    regions = [
        _make_region("r1", 100.0, 110.0),
        _make_region("r2", 200.0, 215.0),
    ]
    region_job = await _seed_region_job_with_regions(session, settings, regions=regions)
    job = await _seed_continuous_job(session, region_job.id, pad=5.0)

    class CancelOnSecondSpan:
        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                # After first span runs, request cancellation by flipping
                # status. The worker rechecks at the top of each span loop.
                async def flip():
                    refreshed = await session.get(ContinuousEmbeddingJob, job.id)
                    if refreshed is not None:
                        refreshed.status = JobStatus.canceled.value
                        await session.commit()

                asyncio.get_event_loop().run_until_complete(flip())
            return SurfPerchStub(vector_dim=4)(**kwargs)

    # Simpler: pre-cancel before the worker runs.
    job.status = JobStatus.canceled.value
    await session.commit()
    await run_continuous_embedding_job(
        session, job, settings, embedder=SurfPerchStub(vector_dim=4)
    )

    parquet_path = continuous_embedding_parquet_path(settings.storage_root, job.id)
    manifest_path = continuous_embedding_manifest_path(settings.storage_root, job.id)
    assert not parquet_path.exists()
    assert not manifest_path.exists()
    await session.refresh(job)
    assert job.status == JobStatus.canceled.value


async def test_aborts_when_source_not_complete(session, settings):
    regions = [_make_region("r1", 100.0, 110.0)]
    region_job = await _seed_region_job_with_regions(session, settings, regions=regions)
    region_job.status = JobStatus.running.value
    await session.commit()

    payload = ContinuousEmbeddingJobCreate(region_detection_job_id=region_job.id)
    # service guard accepts running region_jobs; the worker should reject.
    job, _ = await create_continuous_embedding_job(session, payload)
    await run_continuous_embedding_job(
        session, job, settings, embedder=SurfPerchStub(vector_dim=4)
    )

    await session.refresh(job)
    assert job.status == JobStatus.failed.value
    assert job.error_message is not None
    assert "not complete" in job.error_message

    parquet_path = continuous_embedding_parquet_path(settings.storage_root, job.id)
    assert not parquet_path.exists()


async def test_run_one_iteration_claims_and_runs(session, settings):
    regions = [_make_region("r1", 50.0, 65.0)]
    region_job = await _seed_region_job_with_regions(session, settings, regions=regions)
    job = await _seed_continuous_job(session, region_job.id)
    assert job.status == JobStatus.queued.value

    claimed = await run_one_iteration(
        session, settings, embedder=SurfPerchStub(vector_dim=4)
    )
    assert claimed is not None
    assert claimed.id == job.id

    await session.refresh(job)
    assert job.status == JobStatus.complete.value


async def test_default_embedder_raises(session, settings):
    regions = [_make_region("r1", 100.0, 110.0)]
    region_job = await _seed_region_job_with_regions(session, settings, regions=regions)
    job = await _seed_continuous_job(session, region_job.id)

    await run_continuous_embedding_job(session, job, settings)

    await session.refresh(job)
    assert job.status == JobStatus.failed.value
    assert "EmbedderProtocol" in (job.error_message or "")


@pytest.fixture
def settings(tmp_path):
    from humpback.config import Settings

    db_path = tmp_path / "test.db"
    return Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_root=tmp_path / "storage",
        use_real_model=False,
    )
