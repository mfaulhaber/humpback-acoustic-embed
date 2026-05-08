"""Tests for Event Encoder service idempotency and validation."""

from datetime import datetime, timezone

import pytest

from humpback.models.call_parsing import (
    EventBoundaryCorrection,
    EventSegmentationJob,
    RegionDetectionJob,
)
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.schemas.sequence_models import (
    EventEncoderDescriptorConfig,
    EventEncoderJobCreate,
)
from humpback.services.event_encoder_service import (
    CancelEventEncoderTerminalJobError,
    cancel_event_encoder_job,
    create_event_encoder_job,
    delete_event_encoder_job,
    get_event_encoder_job,
    list_event_encoder_jobs,
)
from humpback.storage import event_encoder_dir


async def _seed_seg_job(
    session,
    *,
    status: str = JobStatus.complete.value,
) -> tuple[RegionDetectionJob, EventSegmentationJob]:
    region = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=0.0,
        end_timestamp=300.0,
    )
    session.add(region)
    await session.flush()
    seg = EventSegmentationJob(
        status=status,
        region_detection_job_id=region.id,
    )
    session.add(seg)
    await session.commit()
    await session.refresh(region)
    await session.refresh(seg)
    return region, seg


async def _seed_crnn_embedding_job(
    session,
    seg_job: EventSegmentationJob,
    region_job: RegionDetectionJob,
    *,
    status: str = JobStatus.complete.value,
    model_version: str = "crnn-call-parsing-pytorch",
    signature: str = "cej-sig-1",
) -> ContinuousEmbeddingJob:
    job = ContinuousEmbeddingJob(
        status=status,
        event_segmentation_job_id=seg_job.id,
        event_source_mode="raw",
        region_detection_job_id=region_job.id,
        model_version=model_version,
        target_sample_rate=16000,
        encoding_signature=signature,
        crnn_checkpoint_sha256="abc123",
        chunk_size_seconds=0.25,
        chunk_hop_seconds=0.25,
        projection_kind="identity",
        projection_dim=1024,
        total_regions=1,
        total_chunks=8,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def test_create_returns_new_event_encoder_job(session):
    region, seg = await _seed_seg_job(session)
    continuous = await _seed_crnn_embedding_job(session, seg, region)

    payload = EventEncoderJobCreate(
        event_segmentation_job_id=seg.id,
        continuous_embedding_job_id=continuous.id,
    )
    job, created = await create_event_encoder_job(session, payload)

    assert created is True
    assert job.status == JobStatus.queued.value
    assert job.event_segmentation_job_id == seg.id
    assert job.continuous_embedding_job_id == continuous.id
    assert job.continuous_embedding_signature == continuous.encoding_signature
    assert job.tokenization_signature
    assert job.k_values_json == "[50,100,200]"


async def test_idempotent_returns_existing_active_or_complete(session):
    region, seg = await _seed_seg_job(session)
    continuous = await _seed_crnn_embedding_job(session, seg, region)
    payload = EventEncoderJobCreate(
        event_segmentation_job_id=seg.id,
        continuous_embedding_job_id=continuous.id,
    )
    first, _ = await create_event_encoder_job(session, payload)
    first.status = JobStatus.running.value
    await session.commit()

    second, created = await create_event_encoder_job(session, payload)
    assert created is False
    assert second.id == first.id

    first.status = JobStatus.complete.value
    await session.commit()

    third, created = await create_event_encoder_job(session, payload)
    assert created is False
    assert third.id == first.id
    assert len(await list_event_encoder_jobs(session)) == 1


async def test_resubmitting_failed_job_requeues_existing_row(session):
    region, seg = await _seed_seg_job(session)
    continuous = await _seed_crnn_embedding_job(session, seg, region)
    payload = EventEncoderJobCreate(
        event_segmentation_job_id=seg.id,
        continuous_embedding_job_id=continuous.id,
    )
    job, _ = await create_event_encoder_job(session, payload)
    job.status = JobStatus.failed.value
    job.error_message = "boom"
    job.event_vector_dim = 42
    job.event_vectors_path = "/tmp/old.parquet"
    job.event_tokens_path = "/tmp/old-tokens.parquet"
    job.token_sequences_path = "/tmp/old-sequences.parquet"
    job.manifest_path = "/tmp/manifest.json"
    job.report_path = "/tmp/report.json"
    job.total_events = 10
    job.encoded_events = 5
    job.skipped_events = 5
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()

    retried, created = await create_event_encoder_job(session, payload)

    assert created is False
    assert retried.id == job.id
    assert retried.status == JobStatus.queued.value
    assert retried.error_message is None
    assert retried.event_vector_dim is None
    assert retried.event_vectors_path is None
    assert retried.event_tokens_path is None
    assert retried.token_sequences_path is None
    assert retried.manifest_path is None
    assert retried.report_path is None
    assert retried.total_events is None
    assert retried.encoded_events is None
    assert retried.skipped_events is None


async def test_effective_mode_correction_revision_changes_signature(session):
    region, seg = await _seed_seg_job(session)
    continuous = await _seed_crnn_embedding_job(session, seg, region)
    payload = EventEncoderJobCreate(
        event_segmentation_job_id=seg.id,
        event_source_mode="effective",
        continuous_embedding_job_id=continuous.id,
    )
    first, _ = await create_event_encoder_job(session, payload)
    first.status = JobStatus.complete.value
    await session.commit()

    session.add(
        EventBoundaryCorrection(
            region_detection_job_id=region.id,
            event_segmentation_job_id=seg.id,
            region_id="region-1",
            source_event_id="event-1",
            correction_type="adjust",
            original_start_sec=1.0,
            original_end_sec=2.0,
            corrected_start_sec=1.1,
            corrected_end_sec=2.1,
        )
    )
    await session.commit()

    second, created = await create_event_encoder_job(session, payload)
    assert created is True
    assert second.id != first.id
    assert second.tokenization_signature != first.tokenization_signature


async def test_descriptor_config_changes_signature(session):
    region, seg = await _seed_seg_job(session)
    continuous = await _seed_crnn_embedding_job(session, seg, region)
    base_payload = EventEncoderJobCreate(
        event_segmentation_job_id=seg.id,
        continuous_embedding_job_id=continuous.id,
    )
    first, _ = await create_event_encoder_job(session, base_payload)
    first.status = JobStatus.complete.value
    await session.commit()

    changed_payload = EventEncoderJobCreate(
        event_segmentation_job_id=seg.id,
        continuous_embedding_job_id=continuous.id,
        descriptor=EventEncoderDescriptorConfig(ridge_max_frequency_hz=2500.0),
    )
    second, created = await create_event_encoder_job(session, changed_payload)

    assert created is True
    assert second.id != first.id
    assert second.tokenization_signature != first.tokenization_signature


async def test_rejects_invalid_sources(session):
    region, seg = await _seed_seg_job(session)
    continuous = await _seed_crnn_embedding_job(session, seg, region)

    with pytest.raises(ValueError, match="event_segmentation_job not found"):
        await create_event_encoder_job(
            session,
            EventEncoderJobCreate(
                event_segmentation_job_id="missing",
                continuous_embedding_job_id=continuous.id,
            ),
        )

    seg.status = JobStatus.running.value
    await session.commit()
    with pytest.raises(ValueError, match="completed event_segmentation_job"):
        await create_event_encoder_job(
            session,
            EventEncoderJobCreate(
                event_segmentation_job_id=seg.id,
                continuous_embedding_job_id=continuous.id,
            ),
        )


async def test_rejects_invalid_continuous_embedding_sources(session):
    region, seg = await _seed_seg_job(session)
    continuous = await _seed_crnn_embedding_job(session, seg, region)

    with pytest.raises(ValueError, match="continuous_embedding_job not found"):
        await create_event_encoder_job(
            session,
            EventEncoderJobCreate(
                event_segmentation_job_id=seg.id,
                continuous_embedding_job_id="missing",
            ),
        )

    continuous.status = JobStatus.running.value
    await session.commit()
    with pytest.raises(ValueError, match="completed continuous_embedding_job"):
        await create_event_encoder_job(
            session,
            EventEncoderJobCreate(
                event_segmentation_job_id=seg.id,
                continuous_embedding_job_id=continuous.id,
            ),
        )

    continuous.status = JobStatus.complete.value
    continuous.model_version = "surfperch-tensorflow2"
    await session.commit()
    with pytest.raises(ValueError, match="region_crnn"):
        await create_event_encoder_job(
            session,
            EventEncoderJobCreate(
                event_segmentation_job_id=seg.id,
                continuous_embedding_job_id=continuous.id,
            ),
        )


async def test_rejects_segmentation_mismatch(session):
    region, seg = await _seed_seg_job(session)
    _, other_seg = await _seed_seg_job(session)
    continuous = await _seed_crnn_embedding_job(session, seg, region)

    with pytest.raises(ValueError, match="does not match"):
        await create_event_encoder_job(
            session,
            EventEncoderJobCreate(
                event_segmentation_job_id=other_seg.id,
                continuous_embedding_job_id=continuous.id,
            ),
        )


async def test_cancel_and_delete(session, settings):
    region, seg = await _seed_seg_job(session)
    continuous = await _seed_crnn_embedding_job(session, seg, region)
    payload = EventEncoderJobCreate(
        event_segmentation_job_id=seg.id,
        continuous_embedding_job_id=continuous.id,
    )
    job, _ = await create_event_encoder_job(session, payload)

    canceled = await cancel_event_encoder_job(session, job.id)
    assert canceled is not None
    assert canceled.status == JobStatus.canceled.value

    canceled.status = JobStatus.complete.value
    await session.commit()
    with pytest.raises(CancelEventEncoderTerminalJobError):
        await cancel_event_encoder_job(session, job.id)

    artifact_dir = event_encoder_dir(settings.storage_root, job.id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "report.json").write_text("{}")

    assert await delete_event_encoder_job(session, job.id, settings) is True
    assert not artifact_dir.exists()
    assert await get_event_encoder_job(session, job.id) is None
    assert await delete_event_encoder_job(session, "missing", settings) is False


async def test_get_missing_returns_none(session):
    assert await get_event_encoder_job(session, "missing") is None
