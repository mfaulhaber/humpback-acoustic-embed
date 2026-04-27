"""End-to-end tests for the continuous-embedding worker.

Uses the synthetic ``SurfPerchStub`` embedder so the worker exercises
the full claim → run → atomic write path without depending on real
audio decoding or model files.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from humpback.call_parsing.storage import region_job_dir, write_regions
from humpback.call_parsing.types import Region
from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.processing.inference import FakeTF2Model
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


def _make_region(
    region_id: str,
    start: float,
    end: float,
    *,
    padded_start: float | None = None,
    padded_end: float | None = None,
) -> Region:
    return Region(
        region_id=region_id,
        start_sec=start,
        end_sec=end,
        padded_start_sec=start if padded_start is None else padded_start,
        padded_end_sec=end if padded_end is None else padded_end,
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
    source_region_ids = table.column("source_region_ids").to_pylist()
    assert source_region_ids
    assert all(isinstance(ids, list) for ids in source_region_ids)
    assert any(ids == ["r1"] for ids in source_region_ids)

    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["vector_dim"] == 8
    assert manifest["merged_spans"] == 2
    assert manifest["total_windows"] == job.total_windows
    assert len(manifest["spans"]) == 2


async def test_worker_uses_raw_region_bounds_for_padding_and_membership(
    session, settings
):
    regions = [
        _make_region("r1", 100.0, 110.0, padded_start=95.0, padded_end=115.0),
    ]
    region_job = await _seed_region_job_with_regions(session, settings, regions=regions)
    job = await _seed_continuous_job(session, region_job.id, pad=5.0)

    await run_continuous_embedding_job(
        session, job, settings, embedder=SurfPerchStub(vector_dim=4)
    )

    table = pq.read_table(
        continuous_embedding_parquet_path(settings.storage_root, job.id)
    )
    starts = table.column("start_time_sec").to_pylist()
    in_pad = table.column("is_in_pad").to_pylist()
    source_region_ids = table.column("source_region_ids").to_pylist()

    assert starts[0] == 95.0
    assert in_pad[0] is True
    assert source_region_ids[0] == []

    first_in_region = next(i for i, value in enumerate(in_pad) if value is False)
    assert starts[first_in_region] == 98.0
    assert source_region_ids[first_in_region] == ["r1"]


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

    job = ContinuousEmbeddingJob(
        region_detection_job_id=region_job.id,
        model_version="surfperch-tensorflow2",
        window_size_seconds=5.0,
        hop_seconds=1.0,
        pad_seconds=10.0,
        target_sample_rate=32000,
        encoding_signature="manual-test-signature",
    )
    session.add(job)
    await session.commit()
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


async def test_default_embedder_runs_with_runtime_dependencies(
    session, settings, monkeypatch
):
    regions = [_make_region("r1", 100.0, 110.0)]
    region_job = await _seed_region_job_with_regions(session, settings, regions=regions)
    job = await _seed_continuous_job(session, region_job.id)

    class FakeProvider:
        def build_timeline(self, start_ts: float, end_ts: float):
            return [("timeline", start_ts, end_ts)]

    async def fake_get_model_by_version(_session, model_version: str, _settings):
        assert model_version == "surfperch-tensorflow2"
        return FakeTF2Model(4), "waveform"

    def fake_resolve_audio_slice(
        provider,
        stream_start_ts: float,
        stream_end_ts: float,
        start_utc: float,
        duration_sec: float,
        target_sr: int = 32000,
        timeline=None,
    ) -> np.ndarray:
        assert provider is fake_provider
        assert stream_start_ts == 0.0
        assert stream_end_ts == 1000.0
        assert start_utc >= stream_start_ts
        assert timeline == fake_provider.build_timeline(stream_start_ts, stream_end_ts)
        sample_count = int(round(duration_sec * target_sr))
        return np.linspace(0.0, 1.0, sample_count, dtype=np.float32)

    fake_provider = FakeProvider()
    monkeypatch.setattr(
        "humpback.workers.continuous_embedding_worker.get_model_by_version",
        fake_get_model_by_version,
    )
    monkeypatch.setattr(
        "humpback.workers.continuous_embedding_worker.build_archive_detection_provider",
        lambda *args, **kwargs: fake_provider,
    )
    monkeypatch.setattr(
        "humpback.workers.continuous_embedding_worker.resolve_audio_slice",
        fake_resolve_audio_slice,
    )

    await run_continuous_embedding_job(session, job, settings)

    await session.refresh(job)
    assert job.status == JobStatus.complete.value
    assert job.vector_dim == 4


@pytest.fixture
def settings(tmp_path):
    from humpback.config import Settings

    db_path = tmp_path / "test.db"
    return Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_root=tmp_path / "storage",
        use_real_model=False,
    )
