from types import SimpleNamespace

import pytest

from humpback.models.processing import JobStatus
from humpback.models.sequence_models import (
    ContinuousEmbeddingJob,
    HMMSequenceJob,
    MaskedTransformerJob,
    MotifExtractionJob,
)
from humpback.services.masked_transformer_service import serialize_k_values
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


async def _masked_transformer_jobs(
    session,
    *,
    mt_status: str = JobStatus.complete.value,
    k_values: list[int] | None = None,
):
    cej = ContinuousEmbeddingJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id="seg-2",
        model_version="crnn-call-parsing-pytorch",
        target_sample_rate=32000,
        encoding_signature="enc-mt-1",
    )
    session.add(cej)
    await session.commit()
    await session.refresh(cej)
    mt = MaskedTransformerJob(
        status=mt_status,
        continuous_embedding_job_id=cej.id,
        training_signature="sig-1",
        k_values=serialize_k_values(k_values or [50, 100]),
    )
    session.add(mt)
    await session.commit()
    await session.refresh(mt)
    return cej, mt


def _mt_payload(mt_id: str, k: int | None, **overrides):
    defaults = {
        "parent_kind": "masked_transformer",
        "hmm_sequence_job_id": None,
        "masked_transformer_job_id": mt_id,
        "k": k,
        "min_ngram": 2,
        "max_ngram": 4,
        "minimum_occurrences": 2,
        "minimum_event_sources": 1,
        "frequency_weight": 0.4,
        "event_source_weight": 0.3,
        "event_core_weight": 0.2,
        "low_background_weight": 0.1,
        "call_probability_weight": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


async def test_create_with_masked_transformer_parent(session):
    _, mt = await _masked_transformer_jobs(session)

    job, created = await create_motif_extraction_job(session, _mt_payload(mt.id, k=100))
    assert created is True
    assert job.parent_kind == "masked_transformer"
    assert job.masked_transformer_job_id == mt.id
    assert job.k == 100
    assert job.hmm_sequence_job_id is None
    assert job.source_kind == "region_crnn"


async def test_create_masked_transformer_requires_completed_job(session):
    _, mt = await _masked_transformer_jobs(session, mt_status=JobStatus.running.value)

    with pytest.raises(ValueError, match="completed masked_transformer_job"):
        await create_motif_extraction_job(session, _mt_payload(mt.id, k=100))


async def test_create_masked_transformer_rejects_k_not_in_k_values(session):
    _, mt = await _masked_transformer_jobs(session, k_values=[100])

    with pytest.raises(ValueError, match="not in the masked_transformer_job"):
        await create_motif_extraction_job(session, _mt_payload(mt.id, k=200))


async def test_create_masked_transformer_requires_k(session):
    _, mt = await _masked_transformer_jobs(session)

    with pytest.raises(ValueError, match="k is required"):
        await create_motif_extraction_job(session, _mt_payload(mt.id, k=None))


async def test_idempotency_for_masked_transformer_parent(session):
    _, mt = await _masked_transformer_jobs(session)

    first, c1 = await create_motif_extraction_job(session, _mt_payload(mt.id, k=100))
    second, c2 = await create_motif_extraction_job(session, _mt_payload(mt.id, k=100))
    third, c3 = await create_motif_extraction_job(session, _mt_payload(mt.id, k=50))

    assert c1 is True
    assert c2 is False
    assert second.id == first.id
    # Different k → different signature → different job.
    assert c3 is True
    assert third.id != first.id


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
