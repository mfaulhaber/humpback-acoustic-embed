"""End-to-end tests for the continuous-embedding worker.

Uses the synthetic ``SurfPerchStub`` embedder so the worker exercises
the full claim -> run -> atomic write path without depending on real
audio decoding or model files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from humpback.call_parsing.storage import (
    segmentation_job_dir,
    write_events,
)
from humpback.call_parsing.types import Event
from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
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


def _make_event(
    event_id: str,
    region_id: str,
    start: float,
    end: float,
) -> Event:
    return Event(
        event_id=event_id,
        region_id=region_id,
        start_sec=start,
        end_sec=end,
        center_sec=(start + end) / 2.0,
        segmentation_confidence=0.9,
    )


async def _seed_seg_job_with_events(
    session,
    settings,
    *,
    events: list[Event],
    duration_sec: float = 1000.0,
    start_timestamp: float = 1000.0,
) -> EventSegmentationJob:
    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=start_timestamp,
        end_timestamp=start_timestamp + duration_sec,
    )
    session.add(region_job)
    await session.commit()
    await session.refresh(region_job)

    seg_job = EventSegmentationJob(
        status=JobStatus.complete.value,
        region_detection_job_id=region_job.id,
    )
    session.add(seg_job)
    await session.commit()
    await session.refresh(seg_job)

    seg_dir = segmentation_job_dir(settings.storage_root, seg_job.id)
    seg_dir.mkdir(parents=True, exist_ok=True)
    write_events(seg_dir / "events.parquet", events)
    return seg_job


async def _seed_continuous_job(
    session,
    event_segmentation_job_id: str,
    *,
    hop: float = 1.0,
    pad: float = 2.0,
) -> ContinuousEmbeddingJob:
    payload = ContinuousEmbeddingJobCreate(
        event_segmentation_job_id=event_segmentation_job_id,
        hop_seconds=hop,
        pad_seconds=pad,
    )
    job, _ = await create_continuous_embedding_job(session, payload)
    return job


async def test_happy_path_writes_parquet_and_manifest(session, settings):
    events = [
        _make_event("e1", "r1", 100.0, 110.0),
        _make_event("e2", "r1", 200.0, 215.0),
    ]
    seg_job = await _seed_seg_job_with_events(session, settings, events=events)
    job = await _seed_continuous_job(session, seg_job.id, pad=2.0)
    stub = SurfPerchStub(vector_dim=8)

    await run_continuous_embedding_job(session, job, settings, embedder=stub)

    await session.refresh(job)
    assert job.status == JobStatus.complete.value
    assert job.vector_dim == 8
    assert job.total_events == 2
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
        "event_id",
        "window_index_in_span",
        "audio_file_id",
        "start_timestamp",
        "end_timestamp",
        "is_in_pad",
        "embedding",
    }
    span_ids = table.column("merged_span_id").to_pylist()
    window_idxs = table.column("window_index_in_span").to_pylist()
    sorted_pairs = sorted(zip(span_ids, window_idxs))
    assert list(zip(span_ids, window_idxs)) == sorted_pairs

    embedding_widths = {len(v) for v in table.column("embedding").to_pylist()}
    assert embedding_widths == {8}
    event_ids = table.column("event_id").to_pylist()
    assert "e1" in event_ids
    assert "e2" in event_ids

    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["vector_dim"] == 8
    assert manifest["merged_spans"] == 2
    assert manifest["total_events"] == 2
    assert manifest["total_windows"] == job.total_windows
    assert len(manifest["spans"]) == 2
    assert manifest["spans"][0]["event_id"] == "e1"
    assert manifest["spans"][1]["event_id"] == "e2"


async def test_event_span_padding_and_clamping(session, settings):
    events = [
        _make_event("e1", "r1", 100.0, 110.0),
    ]
    seg_job = await _seed_seg_job_with_events(
        session, settings, events=events, start_timestamp=1000.0, duration_sec=1000.0
    )
    job = await _seed_continuous_job(session, seg_job.id, pad=3.0)

    await run_continuous_embedding_job(
        session, job, settings, embedder=SurfPerchStub(vector_dim=4)
    )

    table = pq.read_table(
        continuous_embedding_parquet_path(settings.storage_root, job.id)
    )
    starts = table.column("start_timestamp").to_pylist()
    in_pad = table.column("is_in_pad").to_pylist()

    assert starts[0] == 1000.0 + 97.0
    assert in_pad[0] is True

    first_in_event = next(i for i, v in enumerate(in_pad) if v is False)
    assert starts[first_in_event] >= 1000.0 + 98.0


async def test_sequential_span_ids_one_per_event(session, settings):
    events = [
        _make_event("e1", "r1", 100.0, 105.0),
        _make_event("e2", "r1", 106.0, 111.0),
        _make_event("e3", "r2", 200.0, 208.0),
    ]
    seg_job = await _seed_seg_job_with_events(session, settings, events=events)
    job = await _seed_continuous_job(session, seg_job.id, pad=2.0)

    await run_continuous_embedding_job(
        session, job, settings, embedder=SurfPerchStub(vector_dim=4)
    )

    await session.refresh(job)
    assert job.merged_spans == 3
    assert job.total_events == 3

    manifest_path = continuous_embedding_manifest_path(settings.storage_root, job.id)
    manifest = json.loads(Path(manifest_path).read_text())
    span_ids = [s["merged_span_id"] for s in manifest["spans"]]
    assert span_ids == [0, 1, 2]


async def test_failure_path_marks_failed_and_cleans_artifacts(session, settings):
    events = [_make_event("e1", "r1", 100.0, 120.0)]
    seg_job = await _seed_seg_job_with_events(session, settings, events=events)
    job = await _seed_continuous_job(session, seg_job.id)
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
    events = [_make_event("e1", "r1", 100.0, 115.0)]
    seg_job = await _seed_seg_job_with_events(session, settings, events=events)
    job = await _seed_continuous_job(session, seg_job.id)

    await run_continuous_embedding_job(
        session, job, settings, embedder=SurfPerchStub(vector_dim=4)
    )

    job_dir = continuous_embedding_dir(settings.storage_root, job.id)
    leftover_tmps = list(job_dir.glob("*.tmp"))
    assert leftover_tmps == []


async def test_cancellation_between_spans_drops_artifacts(session, settings):
    events = [
        _make_event("e1", "r1", 100.0, 110.0),
        _make_event("e2", "r1", 200.0, 215.0),
    ]
    seg_job = await _seed_seg_job_with_events(session, settings, events=events)
    job = await _seed_continuous_job(session, seg_job.id, pad=2.0)

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
    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=1000.0,
        end_timestamp=2000.0,
    )
    session.add(region_job)
    await session.commit()
    await session.refresh(region_job)

    seg_job = EventSegmentationJob(
        status=JobStatus.running.value,
        region_detection_job_id=region_job.id,
    )
    session.add(seg_job)
    await session.commit()
    await session.refresh(seg_job)

    job = ContinuousEmbeddingJob(
        event_segmentation_job_id=seg_job.id,
        model_version="surfperch-tensorflow2",
        window_size_seconds=5.0,
        hop_seconds=1.0,
        pad_seconds=2.0,
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
    events = [_make_event("e1", "r1", 50.0, 65.0)]
    seg_job = await _seed_seg_job_with_events(session, settings, events=events)
    job = await _seed_continuous_job(session, seg_job.id)
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
    events = [_make_event("e1", "r1", 100.0, 110.0)]
    seg_job = await _seed_seg_job_with_events(session, settings, events=events)
    job = await _seed_continuous_job(session, seg_job.id)

    seg_job_obj = await session.get(EventSegmentationJob, seg_job.id)
    region_job = await session.get(
        RegionDetectionJob, seg_job_obj.region_detection_job_id
    )

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
        assert stream_start_ts == region_job.start_timestamp
        assert stream_end_ts == region_job.end_timestamp
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
