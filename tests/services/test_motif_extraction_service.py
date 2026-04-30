from types import SimpleNamespace

import pytest

from humpback.models.processing import JobStatus
from humpback.models.sequence_models import (
    ContinuousEmbeddingJob,
    HMMSequenceJob,
    MotifExtractionJob,
)
from humpback.services.motif_extraction_service import (
    CancelTerminalJobError,
    cancel_motif_extraction_job,
    create_motif_extraction_job,
    delete_motif_extraction_job,
    list_motif_extraction_jobs,
)
from humpback.storage import motif_extraction_dir


def _payload(hmm_id: str, **overrides):
    defaults = {
        "hmm_sequence_job_id": hmm_id,
        "min_ngram": 2,
        "max_ngram": 8,
        "minimum_occurrences": 5,
        "minimum_event_sources": 2,
        "frequency_weight": 0.4,
        "event_source_weight": 0.3,
        "event_core_weight": 0.2,
        "low_background_weight": 0.1,
        "call_probability_weight": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


async def _source_jobs(session, *, hmm_status: str = JobStatus.complete.value):
    cej = ContinuousEmbeddingJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id="seg-1",
        model_version="surfperch-tensorflow2",
        window_size_seconds=5.0,
        hop_seconds=1.0,
        pad_seconds=2.0,
        target_sample_rate=32000,
        encoding_signature="enc-1",
    )
    session.add(cej)
    await session.commit()
    await session.refresh(cej)
    hmm = HMMSequenceJob(
        status=hmm_status,
        continuous_embedding_job_id=cej.id,
        n_states=4,
        pca_dims=8,
    )
    session.add(hmm)
    await session.commit()
    await session.refresh(hmm)
    return cej, hmm


async def test_create_requires_completed_hmm_job(session):
    _, hmm = await _source_jobs(session, hmm_status=JobStatus.running.value)

    with pytest.raises(ValueError, match="completed hmm_sequence_job"):
        await create_motif_extraction_job(session, _payload(hmm.id))


async def test_create_is_idempotent_for_active_and_complete_jobs(session):
    _, hmm = await _source_jobs(session)

    first, created = await create_motif_extraction_job(session, _payload(hmm.id))
    second, created_again = await create_motif_extraction_job(session, _payload(hmm.id))
    assert created is True
    assert created_again is False
    assert second.id == first.id

    first.status = JobStatus.complete.value
    await session.commit()
    third, created_third = await create_motif_extraction_job(session, _payload(hmm.id))
    assert created_third is False
    assert third.id == first.id


async def test_failed_and_canceled_jobs_do_not_block_retry(session):
    _, hmm = await _source_jobs(session)
    first, _ = await create_motif_extraction_job(session, _payload(hmm.id))
    first.status = JobStatus.failed.value
    await session.commit()

    second, created = await create_motif_extraction_job(session, _payload(hmm.id))
    assert created is True
    assert second.id != first.id

    second.status = JobStatus.canceled.value
    await session.commit()
    third, created_third = await create_motif_extraction_job(session, _payload(hmm.id))
    assert created_third is True
    assert third.id not in {first.id, second.id}


async def test_list_cancel_and_delete(session, settings):
    _, hmm = await _source_jobs(session)
    job, _ = await create_motif_extraction_job(session, _payload(hmm.id))

    jobs = await list_motif_extraction_jobs(
        session, status=JobStatus.queued.value, hmm_sequence_job_id=hmm.id
    )
    assert [j.id for j in jobs] == [job.id]

    canceled = await cancel_motif_extraction_job(session, job.id)
    assert canceled is not None
    assert canceled.status == JobStatus.canceled.value

    with pytest.raises(CancelTerminalJobError):
        await cancel_motif_extraction_job(session, job.id)

    artifact_dir = motif_extraction_dir(settings.storage_root, job.id)
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")

    assert await delete_motif_extraction_job(session, job.id, settings) is True
    assert not artifact_dir.exists()
    assert await session.get(MotifExtractionJob, job.id) is None
