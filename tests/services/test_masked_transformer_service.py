"""Tests for the masked-transformer service (ADR-061)."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

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
    vector_dim: int | None = 8,
    chunk_size_seconds: float | None = 0.25,
    chunk_hop_seconds: float | None = 0.25,
    projection_kind: str | None = "identity",
    projection_dim: int | None = 8,
    crnn_checkpoint_sha256: str | None = "test-crnn-checkpoint",
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
        vector_dim=vector_dim,
        chunk_size_seconds=chunk_size_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
        projection_kind=projection_kind,
        projection_dim=projection_dim,
        crnn_checkpoint_sha256=crnn_checkpoint_sha256,
        status=status,
    )
    session.add(cej)
    await session.commit()
    await session.refresh(cej)
    return cej


async def _classify_id_for_cej(session, cej: ContinuousEmbeddingJob) -> str:
    result = await session.execute(
        select(EventClassificationJob.id)
        .where(
            EventClassificationJob.event_segmentation_job_id
            == cej.event_segmentation_job_id
        )
        .order_by(EventClassificationJob.created_at.desc())
        .limit(1)
    )
    classify_id = result.scalar_one()
    return str(classify_id)


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
        assert job.batch_size == 8
        assert job.retrieval_head_enabled is False
        assert job.retrieval_dim is None
        assert job.retrieval_hidden_dim is None
        assert job.retrieval_l2_normalize is True
        assert job.retrieval_head_arch == "mlp"
        assert job.sequence_construction_mode == "region"
        assert job.event_centered_fraction == pytest.approx(0.0)
        assert job.pre_event_context_sec is None
        assert job.post_event_context_sec is None
        assert job.contrastive_loss_weight == pytest.approx(0.0)
        assert job.contrastive_temperature == pytest.approx(0.07)
        assert job.contrastive_label_source == "none"
        assert job.contrastive_min_events_per_label == 4
        assert job.contrastive_min_regions_per_label == 2
        assert job.require_cross_region_positive is True
        assert job.related_label_policy_json is None
        assert job.contrastive_sampler_enabled is True
        assert job.contrastive_labels_per_batch == 4
        assert job.contrastive_events_per_label == 4
        assert job.contrastive_max_unlabeled_fraction == pytest.approx(0.25)
        assert job.contrastive_region_balance is True

    async def test_create_accepts_ordered_source_pairs(self, session):
        cej1 = await _seed_crnn_cej(session)
        cej2 = await _seed_crnn_cej(session)
        classify1 = await _classify_id_for_cej(session, cej1)
        classify2 = await _classify_id_for_cej(session, cej2)

        job, created = await create_masked_transformer_job(
            session,
            sources=[
                {
                    "continuous_embedding_job_id": cej1.id,
                    "event_classification_job_id": classify1,
                    "source_alias": "north",
                },
                {
                    "continuous_embedding_job_id": cej2.id,
                    "event_classification_job_id": classify2,
                    "source_alias": "south",
                },
            ],
            preset="small",
            k_values=[50, 100],
        )

        assert created is True
        assert job.continuous_embedding_job_id == cej1.id
        assert job.event_classification_job_id == classify1
        assert [source.source_order for source in job.sources] == [0, 1]
        assert [source.continuous_embedding_job_id for source in job.sources] == [
            cej1.id,
            cej2.id,
        ]
        assert [source.event_classification_job_id for source in job.sources] == [
            classify1,
            classify2,
        ]
        assert [source.source_alias for source in job.sources] == ["north", "south"]

        fetched = await get_masked_transformer_job(session, job.id)
        assert fetched is not None
        assert [source.source_order for source in fetched.sources] == [0, 1]

    async def test_multi_source_idempotency_excludes_k_values(self, session):
        cej1 = await _seed_crnn_cej(session)
        cej2 = await _seed_crnn_cej(session)
        classify1 = await _classify_id_for_cej(session, cej1)
        classify2 = await _classify_id_for_cej(session, cej2)
        sources = [
            {
                "continuous_embedding_job_id": cej1.id,
                "event_classification_job_id": classify1,
            },
            {
                "continuous_embedding_job_id": cej2.id,
                "event_classification_job_id": classify2,
            },
        ]

        first, created1 = await create_masked_transformer_job(
            session,
            sources=sources,
            preset="small",
            k_values=[50],
        )
        second, created2 = await create_masked_transformer_job(
            session,
            sources=sources,
            preset="small",
            k_values=[200, 300],
        )

        assert created1 is True
        assert created2 is False
        assert first.id == second.id
        assert json.loads(second.k_values) == [50]

    async def test_multi_source_order_participates_in_signature(self, session):
        cej1 = await _seed_crnn_cej(session)
        cej2 = await _seed_crnn_cej(session)
        classify1 = await _classify_id_for_cej(session, cej1)
        classify2 = await _classify_id_for_cej(session, cej2)

        first, _ = await create_masked_transformer_job(
            session,
            sources=[
                {
                    "continuous_embedding_job_id": cej1.id,
                    "event_classification_job_id": classify1,
                },
                {
                    "continuous_embedding_job_id": cej2.id,
                    "event_classification_job_id": classify2,
                },
            ],
            preset="small",
        )
        second, created = await create_masked_transformer_job(
            session,
            sources=[
                {
                    "continuous_embedding_job_id": cej2.id,
                    "event_classification_job_id": classify2,
                },
                {
                    "continuous_embedding_job_id": cej1.id,
                    "event_classification_job_id": classify1,
                },
            ],
            preset="small",
        )

        assert created is True
        assert first.id != second.id
        assert first.training_signature != second.training_signature

    async def test_multi_source_rejects_duplicate_pairs(self, session):
        cej = await _seed_crnn_cej(session)
        classify_id = await _classify_id_for_cej(session, cej)

        with pytest.raises(ValueError, match="duplicate"):
            await create_masked_transformer_job(
                session,
                sources=[
                    {
                        "continuous_embedding_job_id": cej.id,
                        "event_classification_job_id": classify_id,
                    },
                    {
                        "continuous_embedding_job_id": cej.id,
                        "event_classification_job_id": classify_id,
                    },
                ],
            )

    async def test_multi_source_rejects_mismatched_classify_job(self, session):
        cej1 = await _seed_crnn_cej(session)
        cej2 = await _seed_crnn_cej(session)
        classify1 = await _classify_id_for_cej(session, cej1)

        with pytest.raises(ValueError, match="does not match"):
            await create_masked_transformer_job(
                session,
                sources=[
                    {
                        "continuous_embedding_job_id": cej1.id,
                        "event_classification_job_id": classify1,
                    },
                    {
                        "continuous_embedding_job_id": cej2.id,
                        "event_classification_job_id": classify1,
                    },
                ],
            )

    async def test_multi_source_rejects_incompatible_embedding_jobs(self, session):
        cej1 = await _seed_crnn_cej(session, vector_dim=8)
        cej2 = await _seed_crnn_cej(session, vector_dim=16)
        classify1 = await _classify_id_for_cej(session, cej1)
        classify2 = await _classify_id_for_cej(session, cej2)

        with pytest.raises(ValueError, match="compatible vector_dim"):
            await create_masked_transformer_job(
                session,
                sources=[
                    {
                        "continuous_embedding_job_id": cej1.id,
                        "event_classification_job_id": classify1,
                    },
                    {
                        "continuous_embedding_job_id": cej2.id,
                        "event_classification_job_id": classify2,
                    },
                ],
            )

    async def test_multi_source_rejects_checkpoint_mismatch_when_known(self, session):
        cej1 = await _seed_crnn_cej(session, crnn_checkpoint_sha256="ckpt-a")
        cej2 = await _seed_crnn_cej(session, crnn_checkpoint_sha256="ckpt-b")
        classify1 = await _classify_id_for_cej(session, cej1)
        classify2 = await _classify_id_for_cej(session, cej2)

        with pytest.raises(ValueError, match="crnn_checkpoint_sha256"):
            await create_masked_transformer_job(
                session,
                sources=[
                    {
                        "continuous_embedding_job_id": cej1.id,
                        "event_classification_job_id": classify1,
                    },
                    {
                        "continuous_embedding_job_id": cej2.id,
                        "event_classification_job_id": classify2,
                    },
                ],
            )

    async def test_multi_source_allows_unknown_checkpoint_mixed_with_known(
        self, session
    ):
        cej1 = await _seed_crnn_cej(session, crnn_checkpoint_sha256=None)
        cej2 = await _seed_crnn_cej(session, crnn_checkpoint_sha256="ckpt-b")
        classify1 = await _classify_id_for_cej(session, cej1)
        classify2 = await _classify_id_for_cej(session, cej2)

        job, created = await create_masked_transformer_job(
            session,
            sources=[
                {
                    "continuous_embedding_job_id": cej1.id,
                    "event_classification_job_id": classify1,
                },
                {
                    "continuous_embedding_job_id": cej2.id,
                    "event_classification_job_id": classify2,
                },
            ],
        )

        assert created is True
        assert len(job.sources) == 2

    async def test_multi_source_rejects_contrastive_and_ablation(self, session):
        cej1 = await _seed_crnn_cej(session)
        cej2 = await _seed_crnn_cej(session)
        classify1 = await _classify_id_for_cej(session, cej1)
        classify2 = await _classify_id_for_cej(session, cej2)
        sources = [
            {
                "continuous_embedding_job_id": cej1.id,
                "event_classification_job_id": classify1,
            },
            {
                "continuous_embedding_job_id": cej2.id,
                "event_classification_job_id": classify2,
            },
        ]

        with pytest.raises(ValueError, match="does not support contrastive"):
            await create_masked_transformer_job(
                session,
                sources=sources,
                retrieval_head_enabled=True,
                sequence_construction_mode="mixed",
                event_centered_fraction=0.5,
                contrastive_loss_weight=0.1,
                contrastive_label_source="human_corrections",
            )
        with pytest.raises(ValueError, match="projection-head-only"):
            await create_masked_transformer_job(
                session,
                sources=sources,
                training_freeze_mode="transformer_frozen_projection_head_only",
                source_masked_transformer_job_id="source",
            )

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

    async def test_batch_size_participates_in_signature_when_non_default(self, session):
        cej = await _seed_crnn_cej(session)
        default, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
        )
        larger_batch, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            batch_size=16,
        )

        assert created is True
        assert default.id != larger_batch.id
        assert default.training_signature != larger_batch.training_signature
        assert larger_batch.batch_size == 16

    async def test_default_batch_size_preserves_existing_signature(self, session):
        cej = await _seed_crnn_cej(session)
        implicit, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
        )
        explicit, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            batch_size=8,
        )

        assert created is False
        assert implicit.id == explicit.id

    async def test_batch_size_validates_positive(self, session):
        cej = await _seed_crnn_cej(session)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                batch_size=0,
            )

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
        assert retrieval.retrieval_head_arch == "mlp"

    async def test_retrieval_head_arch_participates_in_signature(self, session):
        cej = await _seed_crnn_cej(session)
        mlp, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
        )
        linear, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            retrieval_head_arch="linear",
            retrieval_hidden_dim=512,
        )

        assert created is True
        assert linear.id != mlp.id
        assert linear.training_signature != mlp.training_signature
        assert linear.retrieval_head_arch == "linear"
        assert linear.retrieval_dim == 128
        assert linear.retrieval_hidden_dim is None

    async def test_default_mlp_arch_preserves_existing_signature(self, session):
        cej = await _seed_crnn_cej(session)
        implicit, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
        )
        explicit, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            retrieval_head_arch="mlp",
        )

        assert created is False
        assert explicit.id == implicit.id

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

    async def test_region_sequence_mode_preserves_existing_signature(self, session):
        cej = await _seed_crnn_cej(session)
        first, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
        )
        second, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            sequence_construction_mode="region",
            event_centered_fraction=0.9,
            pre_event_context_sec=5.0,
            post_event_context_sec=6.0,
        )

        assert created is False
        assert first.id == second.id
        assert second.sequence_construction_mode == "region"

    async def test_event_centered_sequence_config_changes_signature(self, session):
        cej = await _seed_crnn_cej(session)
        region, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
        )
        event_centered, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            sequence_construction_mode="event_centered",
        )

        assert created is True
        assert event_centered.id != region.id
        assert event_centered.training_signature != region.training_signature
        assert event_centered.sequence_construction_mode == "event_centered"
        assert event_centered.event_centered_fraction == pytest.approx(1.0)
        assert event_centered.pre_event_context_sec == pytest.approx(2.0)
        assert event_centered.post_event_context_sec == pytest.approx(2.0)

    async def test_mixed_sequence_fraction_participates_in_signature(self, session):
        cej = await _seed_crnn_cej(session)
        first, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            sequence_construction_mode="mixed",
            event_centered_fraction=0.25,
        )
        second, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            sequence_construction_mode="mixed",
            event_centered_fraction=0.75,
        )

        assert created is True
        assert first.id != second.id
        assert first.training_signature != second.training_signature

    async def test_mixed_sequence_mode_validates_fraction(self, session):
        cej = await _seed_crnn_cej(session)
        with pytest.raises(ValueError, match="0.0 < event_centered_fraction < 1.0"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                sequence_construction_mode="mixed",
                event_centered_fraction=1.0,
            )

    async def test_contrastive_config_changes_signature(self, session):
        cej = await _seed_crnn_cej(session)
        retrieval, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
        )
        contrastive, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            sequence_construction_mode="mixed",
            event_centered_fraction=0.7,
            contrastive_loss_weight=0.1,
            contrastive_label_source="human_corrections",
        )

        assert created is True
        assert contrastive.id != retrieval.id
        assert contrastive.training_signature != retrieval.training_signature
        assert contrastive.contrastive_loss_weight == pytest.approx(0.1)
        assert contrastive.contrastive_label_source == "human_corrections"
        assert contrastive.related_label_policy_json is not None
        assert contrastive.event_classification_job_id is not None

    async def test_contrastive_sampler_config_changes_signature_only_when_enabled(
        self, session
    ):
        cej = await _seed_crnn_cej(session)
        disabled, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            contrastive_labels_per_batch=2,
            contrastive_events_per_label=2,
            contrastive_max_unlabeled_fraction=0.1,
            contrastive_region_balance=False,
        )
        disabled_again, disabled_created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            contrastive_labels_per_batch=3,
            contrastive_events_per_label=3,
            contrastive_max_unlabeled_fraction=0.2,
            contrastive_region_balance=True,
        )

        assert disabled_created is False
        assert disabled.id == disabled_again.id

        first, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            sequence_construction_mode="mixed",
            event_centered_fraction=0.7,
            contrastive_loss_weight=0.1,
            contrastive_label_source="human_corrections",
            contrastive_labels_per_batch=2,
        )
        second, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            sequence_construction_mode="mixed",
            event_centered_fraction=0.7,
            contrastive_loss_weight=0.1,
            contrastive_label_source="human_corrections",
            contrastive_labels_per_batch=3,
        )

        assert created is True
        assert first.id != second.id
        assert first.training_signature != second.training_signature

    async def test_contrastive_signature_includes_classify_binding(self, session):
        cej = await _seed_crnn_cej(session)
        first, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            sequence_construction_mode="mixed",
            event_centered_fraction=0.7,
            contrastive_loss_weight=0.1,
            contrastive_label_source="human_corrections",
        )
        second_classify = EventClassificationJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=cej.event_segmentation_job_id,
        )
        session.add(second_classify)
        await session.commit()
        await session.refresh(second_classify)

        second, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            sequence_construction_mode="mixed",
            event_centered_fraction=0.7,
            contrastive_loss_weight=0.1,
            contrastive_label_source="human_corrections",
            event_classification_job_id=second_classify.id,
        )

        assert created is True
        assert first.id != second.id
        assert first.training_signature != second.training_signature

    async def test_contrastive_idempotency_still_excludes_k_values(self, session):
        cej = await _seed_crnn_cej(session)
        first, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            sequence_construction_mode="mixed",
            event_centered_fraction=0.7,
            contrastive_loss_weight=0.1,
            contrastive_label_source="human_corrections",
            k_values=[100],
        )
        second, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            sequence_construction_mode="mixed",
            event_centered_fraction=0.7,
            contrastive_loss_weight=0.1,
            contrastive_label_source="human_corrections",
            k_values=[200],
        )

        assert created is False
        assert first.id == second.id
        assert json.loads(second.k_values) == [100]

    async def test_contrastive_validates_required_retrieval_head(self, session):
        cej = await _seed_crnn_cej(session)
        with pytest.raises(ValueError, match="retrieval_head_enabled"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                contrastive_loss_weight=0.1,
                contrastive_label_source="human_corrections",
            )

    async def test_contrastive_validates_label_source(self, session):
        cej = await _seed_crnn_cej(session)
        with pytest.raises(ValueError, match="contrastive_label_source"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                retrieval_head_enabled=True,
                sequence_construction_mode="mixed",
                event_centered_fraction=0.7,
                contrastive_loss_weight=0.1,
                contrastive_label_source="none",
            )

    async def test_contrastive_validates_non_region_sequence_mode(self, session):
        cej = await _seed_crnn_cej(session)
        with pytest.raises(ValueError, match="event-centered or mixed"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                retrieval_head_enabled=True,
                sequence_construction_mode="region",
                contrastive_loss_weight=0.1,
                contrastive_label_source="human_corrections",
            )

    async def test_projection_head_ablation_accepts_region_sequence_mode(self, session):
        cej = await _seed_crnn_cej(session)
        source, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
        )
        source.status = JobStatus.complete.value
        await session.commit()

        no_policy, no_policy_created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            sequence_construction_mode="region",
            contrastive_loss_weight=1.0,
            contrastive_label_source="human_corrections",
            training_freeze_mode="transformer_frozen_projection_head_only",
            source_masked_transformer_job_id=source.id,
        )
        assert no_policy_created is True
        assert no_policy.negative_label_family_policy_json is None

        ablation, created = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            sequence_construction_mode="region",
            contrastive_loss_weight=1.0,
            contrastive_label_source="human_corrections",
            training_freeze_mode="transformer_frozen_projection_head_only",
            source_masked_transformer_job_id=source.id,
            negative_label_family_policy_json='{"families":{}}',
        )

        assert created is True
        assert (
            ablation.training_freeze_mode == "transformer_frozen_projection_head_only"
        )
        assert ablation.source_masked_transformer_job_id == source.id
        assert ablation.negative_label_family_policy_json == '{"families":{}}'
        assert ablation.training_signature != source.training_signature

    async def test_projection_head_ablation_rejects_invalid_negative_family_policy(
        self, session
    ):
        cej = await _seed_crnn_cej(session)
        source, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
        )
        source.status = JobStatus.complete.value
        await session.commit()

        with pytest.raises(ValueError, match="family labels must be lists"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                preset="small",
                retrieval_head_enabled=True,
                sequence_construction_mode="region",
                contrastive_loss_weight=1.0,
                contrastive_label_source="human_corrections",
                training_freeze_mode="transformer_frozen_projection_head_only",
                source_masked_transformer_job_id=source.id,
                negative_label_family_policy_json='{"families":{"bad":"Moan"}}',
            )

    async def test_projection_head_ablation_validates_source_job(self, session):
        cej = await _seed_crnn_cej(session)
        with pytest.raises(ValueError, match="source_masked_transformer_job_id"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                retrieval_head_enabled=True,
                contrastive_loss_weight=1.0,
                contrastive_label_source="human_corrections",
                training_freeze_mode="transformer_frozen_projection_head_only",
            )
        with pytest.raises(ValueError, match="source_masked_transformer_job not found"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                retrieval_head_enabled=True,
                contrastive_loss_weight=1.0,
                contrastive_label_source="human_corrections",
                training_freeze_mode="transformer_frozen_projection_head_only",
                source_masked_transformer_job_id="missing",
            )

    async def test_projection_head_ablation_requires_completed_matching_source(
        self, session
    ):
        cej = await _seed_crnn_cej(session)
        other_cej = await _seed_crnn_cej(session)
        source, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            retrieval_head_enabled=True,
        )
        with pytest.raises(ValueError, match="must be completed"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                retrieval_head_enabled=True,
                contrastive_loss_weight=1.0,
                contrastive_label_source="human_corrections",
                training_freeze_mode="transformer_frozen_projection_head_only",
                source_masked_transformer_job_id=source.id,
            )
        source.status = JobStatus.complete.value
        await session.commit()
        with pytest.raises(ValueError, match="same continuous_embedding_job_id"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=other_cej.id,
                retrieval_head_enabled=True,
                contrastive_loss_weight=1.0,
                contrastive_label_source="human_corrections",
                training_freeze_mode="transformer_frozen_projection_head_only",
                source_masked_transformer_job_id=source.id,
            )

    async def test_projection_head_ablation_requires_compatible_source_config(
        self, session
    ):
        cej = await _seed_crnn_cej(session)
        source, _ = await create_masked_transformer_job(
            session,
            continuous_embedding_job_id=cej.id,
            preset="small",
            retrieval_head_enabled=True,
            retrieval_dim=64,
            retrieval_hidden_dim=128,
            retrieval_l2_normalize=False,
        )
        source.status = JobStatus.complete.value
        await session.commit()

        with pytest.raises(ValueError, match="source preset"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                preset="default",
                retrieval_head_enabled=True,
                retrieval_dim=64,
                retrieval_hidden_dim=128,
                retrieval_l2_normalize=False,
                contrastive_loss_weight=1.0,
                contrastive_label_source="human_corrections",
                training_freeze_mode="transformer_frozen_projection_head_only",
                source_masked_transformer_job_id=source.id,
            )
        with pytest.raises(ValueError, match="source retrieval_dim"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                preset="small",
                retrieval_head_enabled=True,
                retrieval_dim=128,
                retrieval_hidden_dim=128,
                retrieval_l2_normalize=False,
                contrastive_loss_weight=1.0,
                contrastive_label_source="human_corrections",
                training_freeze_mode="transformer_frozen_projection_head_only",
                source_masked_transformer_job_id=source.id,
            )
        with pytest.raises(ValueError, match="source retrieval_hidden_dim"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                preset="small",
                retrieval_head_enabled=True,
                retrieval_dim=64,
                retrieval_hidden_dim=256,
                retrieval_l2_normalize=False,
                contrastive_loss_weight=1.0,
                contrastive_label_source="human_corrections",
                training_freeze_mode="transformer_frozen_projection_head_only",
                source_masked_transformer_job_id=source.id,
            )
        with pytest.raises(ValueError, match="source retrieval_l2_normalize"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                preset="small",
                retrieval_head_enabled=True,
                retrieval_dim=64,
                retrieval_hidden_dim=128,
                retrieval_l2_normalize=True,
                contrastive_loss_weight=1.0,
                contrastive_label_source="human_corrections",
                training_freeze_mode="transformer_frozen_projection_head_only",
                source_masked_transformer_job_id=source.id,
            )
        with pytest.raises(ValueError, match="source retrieval_head_arch"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                preset="small",
                retrieval_head_enabled=True,
                retrieval_dim=64,
                retrieval_hidden_dim=128,
                retrieval_head_arch="linear",
                retrieval_l2_normalize=False,
                contrastive_loss_weight=1.0,
                contrastive_label_source="human_corrections",
                training_freeze_mode="transformer_frozen_projection_head_only",
                source_masked_transformer_job_id=source.id,
            )

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
        with pytest.raises(ValueError, match="retrieval_head_arch"):
            await create_masked_transformer_job(
                session,
                continuous_embedding_job_id=cej.id,
                retrieval_head_enabled=True,
                retrieval_head_arch="wide",
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
