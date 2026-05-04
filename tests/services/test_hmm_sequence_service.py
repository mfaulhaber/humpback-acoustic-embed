"""Tests for the HMM sequence service — create, list, get, cancel lifecycle.

Label-distribution generation is exercised via the event-scoped helper
unit tests in ``tests/sequence_models/test_label_distribution.py`` plus
the loader integration test
``tests/sequence_models/test_load_effective_event_labels.py``. The full
``generate_interpretations`` end-to-end (PCA + UMAP + overlay + label
distribution) is covered by the API integration tests and the manual
smoke pass (see spec §8.7).
"""

from typing import Any

import pytest

from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.schemas.sequence_models import HMMSequenceJobCreate
from humpback.services.hmm_sequence_service import (
    CancelTerminalJobError,
    cancel_hmm_sequence_job,
    create_hmm_sequence_job,
    delete_hmm_sequence_job,
    get_hmm_sequence_job,
    list_hmm_sequence_jobs,
)
from humpback.storage import hmm_sequence_dir


async def _seed_continuous_embedding_job(
    session,
    *,
    status: str = JobStatus.complete.value,
) -> ContinuousEmbeddingJob:
    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=0.0,
        end_timestamp=300.0,
    )
    session.add(region_job)
    await session.flush()

    seg_job = EventSegmentationJob(
        status=JobStatus.complete.value,
        region_detection_job_id=region_job.id,
    )
    session.add(seg_job)
    await session.commit()
    await session.refresh(seg_job)

    ce_job = ContinuousEmbeddingJob(
        event_segmentation_job_id=seg_job.id,
        model_version="surfperch-tensorflow2",
        window_size_seconds=5.0,
        hop_seconds=1.0,
        pad_seconds=2.0,
        target_sample_rate=32000,
        encoding_signature=f"test-sig-{seg_job.id}",
        status=status,
    )
    session.add(ce_job)
    await session.commit()
    await session.refresh(ce_job)
    return ce_job


def _make_payload(ce_job_id: str, **overrides: Any) -> HMMSequenceJobCreate:
    defaults: dict[str, Any] = dict(
        continuous_embedding_job_id=ce_job_id,
        n_states=4,
    )
    defaults.update(overrides)
    return HMMSequenceJobCreate(**defaults)


async def test_create_returns_queued_job(session):
    ce_job = await _seed_continuous_embedding_job(session)
    payload = _make_payload(ce_job.id)
    job = await create_hmm_sequence_job(session, payload)
    assert job.status == JobStatus.queued.value
    assert job.continuous_embedding_job_id == ce_job.id
    assert job.n_states == 4
    assert job.pca_dims == 50
    assert job.covariance_type == "diag"
    assert job.l2_normalize is True
    assert job.pca_whiten is False
    assert job.n_iter == 100
    assert job.random_seed == 42
    assert job.min_sequence_length_frames == 3


async def test_rejects_nonexistent_source(session):
    payload = _make_payload("nonexistent")
    with pytest.raises(ValueError, match="continuous_embedding_job not found"):
        await create_hmm_sequence_job(session, payload)


async def test_rejects_non_complete_source(session):
    ce_job = await _seed_continuous_embedding_job(
        session, status=JobStatus.running.value
    )
    payload = _make_payload(ce_job.id)
    with pytest.raises(ValueError, match="completed continuous_embedding_job"):
        await create_hmm_sequence_job(session, payload)


async def test_list_all(session):
    ce_job = await _seed_continuous_embedding_job(session)
    await create_hmm_sequence_job(session, _make_payload(ce_job.id))
    await create_hmm_sequence_job(session, _make_payload(ce_job.id, n_states=6))

    jobs = await list_hmm_sequence_jobs(session)
    assert len(jobs) == 2


async def test_list_filter_by_status(session):
    ce_job = await _seed_continuous_embedding_job(session)
    job = await create_hmm_sequence_job(session, _make_payload(ce_job.id))
    job.status = JobStatus.complete.value
    await session.commit()

    await create_hmm_sequence_job(session, _make_payload(ce_job.id, n_states=6))

    queued = await list_hmm_sequence_jobs(session, status=JobStatus.queued.value)
    assert len(queued) == 1

    complete = await list_hmm_sequence_jobs(session, status=JobStatus.complete.value)
    assert len(complete) == 1


async def test_list_filter_by_source(session):
    ce1 = await _seed_continuous_embedding_job(session)
    ce2 = await _seed_continuous_embedding_job(session)
    await create_hmm_sequence_job(session, _make_payload(ce1.id))
    await create_hmm_sequence_job(session, _make_payload(ce2.id))

    jobs = await list_hmm_sequence_jobs(session, continuous_embedding_job_id=ce1.id)
    assert len(jobs) == 1
    assert jobs[0].continuous_embedding_job_id == ce1.id


async def test_get_returns_job(session):
    ce_job = await _seed_continuous_embedding_job(session)
    job = await create_hmm_sequence_job(session, _make_payload(ce_job.id))

    fetched = await get_hmm_sequence_job(session, job.id)
    assert fetched is not None
    assert fetched.id == job.id


async def test_get_missing_returns_none(session):
    result = await get_hmm_sequence_job(session, "nonexistent")
    assert result is None


async def test_cancel_queued(session):
    ce_job = await _seed_continuous_embedding_job(session)
    job = await create_hmm_sequence_job(session, _make_payload(ce_job.id))
    canceled = await cancel_hmm_sequence_job(session, job.id)
    assert canceled is not None
    assert canceled.status == JobStatus.canceled.value


async def test_cancel_running(session):
    ce_job = await _seed_continuous_embedding_job(session)
    job = await create_hmm_sequence_job(session, _make_payload(ce_job.id))
    job.status = JobStatus.running.value
    await session.commit()

    canceled = await cancel_hmm_sequence_job(session, job.id)
    assert canceled is not None
    assert canceled.status == JobStatus.canceled.value


async def test_cancel_complete_raises(session):
    ce_job = await _seed_continuous_embedding_job(session)
    job = await create_hmm_sequence_job(session, _make_payload(ce_job.id))
    job.status = JobStatus.complete.value
    await session.commit()

    with pytest.raises(CancelTerminalJobError):
        await cancel_hmm_sequence_job(session, job.id)


async def test_cancel_missing_returns_none(session):
    result = await cancel_hmm_sequence_job(session, "nonexistent")
    assert result is None


async def test_delete_removes_db_row_and_disk_artifacts(session, settings):
    ce_job = await _seed_continuous_embedding_job(session)
    job = await create_hmm_sequence_job(session, _make_payload(ce_job.id))

    artifact_dir = hmm_sequence_dir(settings.storage_root, job.id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "states.parquet").write_text("fake")

    result = await delete_hmm_sequence_job(session, job.id, settings)
    assert result is True
    assert not artifact_dir.exists()
    assert await get_hmm_sequence_job(session, job.id) is None


async def test_delete_nonexistent_returns_false(session, settings):
    result = await delete_hmm_sequence_job(session, "nonexistent", settings)
    assert result is False


async def test_delete_succeeds_when_no_disk_artifacts(session, settings):
    ce_job = await _seed_continuous_embedding_job(session)
    job = await create_hmm_sequence_job(session, _make_payload(ce_job.id))

    result = await delete_hmm_sequence_job(session, job.id, settings)
    assert result is True
    assert await get_hmm_sequence_job(session, job.id) is None
