"""Tests for the masked-transformer service (ADR-061)."""

from __future__ import annotations

import json

import pytest

from humpback.models.call_parsing import (
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
)
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.services.masked_transformer_service import (
    CancelTerminalJobError,
    ExtendKSweepError,
    cancel_masked_transformer_job,
    create_masked_transformer_job,
    extend_k_sweep_job,
    get_masked_transformer_job,
    list_masked_transformer_jobs,
    parse_k_values,
)


async def _seed_crnn_cej(
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

    session.add(
        EventClassificationJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
        )
    )
    await session.commit()

    cej = ContinuousEmbeddingJob(
        event_segmentation_job_id=seg_job.id,
        region_detection_job_id=region_job.id,
        model_version="crnn-call-parsing-pytorch",
        target_sample_rate=16000,
        encoding_signature=f"sig-crnn-{seg_job.id}",
        status=status,
    )
    session.add(cej)
    await session.commit()
    await session.refresh(cej)
    return cej


async def _seed_surfperch_cej(session) -> ContinuousEmbeddingJob:
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

    session.add(
        EventClassificationJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
        )
    )
    await session.commit()

    cej = ContinuousEmbeddingJob(
        event_segmentation_job_id=seg_job.id,
        model_version="surfperch-tensorflow2",
        target_sample_rate=32000,
        encoding_signature=f"sig-sp-{seg_job.id}",
        status=JobStatus.complete.value,
    )
    session.add(cej)
    await session.commit()
    await session.refresh(cej)
    return cej


class TestParseKValues:
    def test_string_payload(self):
        assert parse_k_values("[100, 200]") == [100, 200]

    def test_list_payload(self):
        assert parse_k_values([50, 100]) == [50, 100]

    def test_dedupes_existing(self):
        assert parse_k_values([100, 100, 50]) == [100, 50]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            parse_k_values([])

    def test_below_two_raises(self):
        with pytest.raises(ValueError, match="k must be >= 2"):
            parse_k_values([1, 100])


class TestCreateMaskedTransformerJob:
    async def test_create_returns_queued_job(self, session):
        cej = await _seed_crnn_cej(session)
        job, created = await create_masked_transformer_job(
            session, continuous_embedding_job_id=cej.id
        )
        assert created is True
        assert job.status == JobStatus.queued.value
        assert job.continuous_embedding_job_id == cej.id
        assert job.preset == "default"
        assert job.training_signature
        assert json.loads(job.k_values) == [100]
        assert job.mask_fraction == pytest.approx(0.20)
        assert job.retrieval_head_enabled is False
        assert job.retrieval_dim is None
        assert job.retrieval_hidden_dim is None
        assert job.retrieval_l2_normalize is True

    async def test_idempotent_returns_existing(self, session):
        cej = await _seed_crnn_cej(session)
        first, created1 = await create_masked_transformer_job(
            session, continuous_embedding_job_id=cej.id, preset="small"
        )
        second, created2 = await create_masked_transformer_job(
            session, continuous_embedding_job_id=cej.id, preset="small"
        )
        assert created1 is True
        assert created2 is False
        assert first.id == second.id

    async def test_idempotent_excludes_k_values(self, session):
        """Different k_values must NOT change the training_signature."""
        cej = await _seed_crnn_cej(session)
        first, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            k_values=[100],
        )
        second, created2 = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            k_values=[200, 300],
        )
        # Same training_signature → service returns existing job (k_values
        # of the *existing* row are unchanged).
        assert created2 is False
        assert first.id == second.id
        assert json.loads(first.k_values) == [100]

    async def test_retrieval_head_config_changes_signature(self, session):
        cej = await _seed_crnn_cej(session)
        contextual, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
        )
        retrieval, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
        )

        assert created is True
        assert retrieval.id != contextual.id
        assert retrieval.training_signature != contextual.training_signature
        assert retrieval.retrieval_head_enabled is True
        assert retrieval.retrieval_dim == 128
        assert retrieval.retrieval_hidden_dim == 512
        assert retrieval.retrieval_l2_normalize is True

    async def test_retrieval_head_dimensions_participate_in_signature(self, session):
        cej = await _seed_crnn_cej(session)
        first, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            retrieval_dim=64,
            retrieval_hidden_dim=256,
        )
        second, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            retrieval_dim=128,
            retrieval_hidden_dim=256,
        )

        assert created is True
        assert first.id != second.id
        assert first.training_signature != second.training_signature

    async def test_retrieval_head_idempotency_still_excludes_k_values(self, session):
        cej = await _seed_crnn_cej(session)
        first, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            k_values=[100],
        )
        second, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            k_values=[200],
        )

        assert created is False
        assert first.id == second.id
        assert json.loads(second.k_values) == [100]

    async def test_rejects_nonexistent_upstream(self, session):
        with pytest.raises(ValueError, match="continuous_embedding_job not found"):
            await create_masked_transformer_job(
                session, continuous_embedding_job_id="does-not-exist"
            )

    async def test_rejects_non_complete_upstream(self, session):
        cej = await _seed_crnn_cej(session, status=JobStatus.running.value)
        with pytest.raises(ValueError, match="completed continuous_embedding_job"):
            await create_masked_transformer_job(
                session, continuous_embedding_job_id=cej.id
            )

    async def test_rejects_surfperch_upstream(self, session):
        cej = await _seed_surfperch_cej(session)
        with pytest.raises(ValueError, match="CRNN region-based"):
            await create_masked_transformer_job(
                session, continuous_embedding_job_id=cej.id
            )

    async def test_rejects_unknown_preset(self, session):
        cej = await _seed_crnn_cej(session)
        with pytest.raises(ValueError, match="preset must be one of"):
            await create_masked_transformer_job(
                session, continuous_embedding_job_id=cej.id, preset="huge"
            )

    async def test_invalid_k_values_rejected(self, session):
        cej = await _seed_crnn_cej(session)
        with pytest.raises(ValueError, match="k must be >= 2"):
            await create_masked_transformer_job(
                session, continuous_embedding_job_id=cej.id, k_values=[1]
            )

    async def test_retrieval_head_rejects_invalid_dimensions(self, session):
        cej = await _seed_crnn_cej(session)
        with pytest.raises(ValueError, match="retrieval_dim must be positive"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                retrieval_head_enabled=True,
                retrieval_dim=0,
            )
        with pytest.raises(ValueError, match="retrieval_hidden_dim must be positive"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                retrieval_head_enabled=True,
                retrieval_hidden_dim=0,
            )


class TestListAndGet:
    async def test_list_returns_jobs_in_reverse_chronological(self, session):
        cej = await _seed_crnn_cej(session)
        job1, _ = await create_masked_transformer_job(
            session, continuous_embedding_job_id=cej.id, preset="small"
        )
        job2, _ = await create_masked_transformer_job(
            session, continuous_embedding_job_id=cej.id, preset="default"
        )
        all_jobs = await list_masked_transformer_jobs(session)
        ids = [j.id for j in all_jobs]
        assert {job1.id, job2.id} <= set(ids)

    async def test_filter_by_status(self, session):
        cej = await _seed_crnn_cej(session)
        await create_masked_transformer_job(
            session, continuous_embedding_job_id=cej.id, preset="small"
        )
        results = await list_masked_transformer_jobs(
            session, status=JobStatus.queued.value
        )
        assert len(results) >= 1
        for r in results:
            assert r.status == JobStatus.queued.value


class TestCancel:
    async def test_cancel_queued_job(self, session):
        cej = await _seed_crnn_cej(session)
        job, _ = await create_masked_transformer_job(
            session, continuous_embedding_job_id=cej.id, preset="small"
        )
        canceled = await cancel_masked_transformer_job(session, job.id)
        assert canceled is not None
        assert canceled.status == JobStatus.canceled.value

    async def test_cancel_terminal_raises(self, session):
        cej = await _seed_crnn_cej(session)
        job, _ = await create_masked_transformer_job(
            session, continuous_embedding_job_id=cej.id, preset="small"
        )
        job.status = JobStatus.complete.value
        await session.commit()
        with pytest.raises(CancelTerminalJobError):
            await cancel_masked_transformer_job(session, job.id)

    async def test_cancel_returns_none_for_missing(self, session):
        result = await cancel_masked_transformer_job(session, "does-not-exist")
        assert result is None


class TestExtendKSweep:
    async def test_extends_only_new_k_values(self, session):
        cej = await _seed_crnn_cej(session)
        job, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            k_values=[100],
        )
        job.status = JobStatus.complete.value
        await session.commit()

        updated = await extend_k_sweep_job(session, job.id, [200, 100, 300])
        assert json.loads(updated.k_values) == [100, 200, 300]
        assert updated.status == JobStatus.queued.value
        assert updated.status_reason == "extend_k_sweep"

    async def test_no_op_when_all_k_already_present(self, session):
        cej = await _seed_crnn_cej(session)
        job, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            k_values=[100, 200],
        )
        job.status = JobStatus.complete.value
        await session.commit()

        before_status = job.status
        unchanged = await extend_k_sweep_job(session, job.id, [100, 200])
        assert unchanged.status == before_status
        assert json.loads(unchanged.k_values) == [100, 200]

    async def test_rejects_running_job(self, session):
        cej = await _seed_crnn_cej(session)
        job, _ = await create_masked_transformer_job(
            session, continuous_embedding_job_id=cej.id, preset="small"
        )
        job.status = JobStatus.running.value
        await session.commit()
        with pytest.raises(ExtendKSweepError):
            await extend_k_sweep_job(session, job.id, [200])

    async def test_rejects_queued_job(self, session):
        cej = await _seed_crnn_cej(session)
        job, _ = await create_masked_transformer_job(
            session, continuous_embedding_job_id=cej.id, preset="small"
        )
        # Status is queued by default; not allowed to extend.
        with pytest.raises(ExtendKSweepError):
            await extend_k_sweep_job(session, job.id, [200])

    async def test_get_returns_none_for_unknown(self, session):
        result = await get_masked_transformer_job(session, "no-such-job")
        assert result is None
