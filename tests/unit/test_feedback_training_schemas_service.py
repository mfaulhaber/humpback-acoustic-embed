"""Tests for feedback training Pydantic schemas and service layer.

Covers Tasks 2-4 from the implementation plan: schema validation,
correction CRUD, training job CRUD, and classifier model management.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from humpback.models.call_parsing import (
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
)
from humpback.models.vocalization import VocalizationClassifierModel
from humpback.schemas.call_parsing import (
    BoundaryCorrection,
    CreateClassifierTrainingJobRequest,
    CreateSegmentationFeedbackTrainingJobRequest,
    TypeCorrection,
)
from humpback.services.call_parsing import (
    CallParsingFKError,
    CallParsingStateError,
    clear_boundary_corrections,
    clear_type_corrections,
    create_classifier_training_job,
    create_segmentation_feedback_training_job,
    delete_classifier_model,
    delete_classifier_training_job,
    delete_segmentation_feedback_training_job,
    list_boundary_corrections,
    list_classifier_models,
    list_type_corrections,
    upsert_boundary_corrections,
    upsert_type_corrections,
)


# ---- Task 2: Pydantic schema validation ----------------------------------


def test_boundary_correction_add_requires_start_end():
    with pytest.raises(ValidationError, match="start_sec and end_sec"):
        BoundaryCorrection(
            event_id="e1",
            region_id="r1",
            correction_type="add",
        )


def test_boundary_correction_add_with_start_end_succeeds():
    c = BoundaryCorrection(
        event_id="e1",
        region_id="r1",
        correction_type="add",
        start_sec=1.0,
        end_sec=2.0,
    )
    assert c.start_sec == 1.0


def test_boundary_correction_adjust_requires_start_end():
    with pytest.raises(ValidationError, match="start_sec and end_sec"):
        BoundaryCorrection(
            event_id="e1",
            region_id="r1",
            correction_type="adjust",
        )


def test_boundary_correction_delete_forbids_start_end():
    with pytest.raises(ValidationError, match="must not set"):
        BoundaryCorrection(
            event_id="e1",
            region_id="r1",
            correction_type="delete",
            start_sec=1.0,
            end_sec=2.0,
        )


def test_boundary_correction_delete_without_start_end_succeeds():
    c = BoundaryCorrection(
        event_id="e1",
        region_id="r1",
        correction_type="delete",
    )
    assert c.start_sec is None


def test_boundary_correction_end_must_be_after_start():
    with pytest.raises(ValidationError, match="strictly after"):
        BoundaryCorrection(
            event_id="e1",
            region_id="r1",
            correction_type="add",
            start_sec=2.0,
            end_sec=1.0,
        )


def test_boundary_correction_rejects_invalid_type():
    with pytest.raises(ValidationError):
        BoundaryCorrection(
            event_id="e1",
            region_id="r1",
            correction_type="invalid",
        )


def test_type_correction_accepts_null_type_name():
    c = TypeCorrection(event_id="e1", type_name=None)
    assert c.type_name is None


def test_type_correction_accepts_string_type_name():
    c = TypeCorrection(event_id="e1", type_name="upcall")
    assert c.type_name == "upcall"


def test_create_segmentation_feedback_request_rejects_empty_ids():
    with pytest.raises(ValidationError):
        CreateSegmentationFeedbackTrainingJobRequest(source_job_ids=[])


def test_create_classifier_request_rejects_empty_ids():
    with pytest.raises(ValidationError):
        CreateClassifierTrainingJobRequest(source_job_ids=[])


# ---- Task 3: Service layer — corrections ----------------------------------


def _make_complete_segmentation_job():
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    es = EventSegmentationJob(region_detection_job_id="placeholder", status="complete")
    return rd, es


def _make_complete_classification_job():
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    es = EventSegmentationJob(region_detection_job_id="placeholder", status="complete")
    ec = EventClassificationJob(
        event_segmentation_job_id="placeholder", status="complete"
    )
    return rd, es, ec


async def test_upsert_boundary_creates_new(session):
    rd, es = _make_complete_segmentation_job()
    session.add(rd)
    await session.flush()
    es.region_detection_job_id = rd.id
    session.add(es)
    await session.commit()

    corrections = [
        BoundaryCorrection(
            event_id="e1",
            region_id="r1",
            correction_type="adjust",
            start_sec=1.0,
            end_sec=2.0,
        )
    ]
    count = await upsert_boundary_corrections(session, es.id, corrections)
    assert count == 1

    rows = await list_boundary_corrections(session, es.id)
    assert len(rows) == 1
    assert rows[0].event_id == "e1"
    assert rows[0].start_sec == 1.0


async def test_upsert_boundary_updates_on_repeat(session):
    rd, es = _make_complete_segmentation_job()
    session.add(rd)
    await session.flush()
    es.region_detection_job_id = rd.id
    session.add(es)
    await session.commit()

    corrections1 = [
        BoundaryCorrection(
            event_id="e1",
            region_id="r1",
            correction_type="adjust",
            start_sec=1.0,
            end_sec=2.0,
        )
    ]
    await upsert_boundary_corrections(session, es.id, corrections1)

    corrections2 = [
        BoundaryCorrection(
            event_id="e1",
            region_id="r1",
            correction_type="adjust",
            start_sec=1.5,
            end_sec=3.0,
        )
    ]
    await upsert_boundary_corrections(session, es.id, corrections2)

    rows = await list_boundary_corrections(session, es.id)
    assert len(rows) == 1
    assert rows[0].start_sec == 1.5


async def test_upsert_boundary_rejects_nonexistent_job(session):
    corrections = [
        BoundaryCorrection(event_id="e1", region_id="r1", correction_type="delete")
    ]
    with pytest.raises(CallParsingFKError):
        await upsert_boundary_corrections(session, "nonexistent", corrections)


async def test_upsert_boundary_rejects_noncomplete_job(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="queued")
    session.add(es)
    await session.commit()

    corrections = [
        BoundaryCorrection(event_id="e1", region_id="r1", correction_type="delete")
    ]
    with pytest.raises(CallParsingStateError):
        await upsert_boundary_corrections(session, es.id, corrections)


async def test_clear_boundary_removes_all(session):
    rd, es = _make_complete_segmentation_job()
    session.add(rd)
    await session.flush()
    es.region_detection_job_id = rd.id
    session.add(es)
    await session.commit()

    corrections = [
        BoundaryCorrection(
            event_id="e1",
            region_id="r1",
            correction_type="adjust",
            start_sec=1.0,
            end_sec=2.0,
        ),
        BoundaryCorrection(event_id="e2", region_id="r1", correction_type="delete"),
    ]
    await upsert_boundary_corrections(session, es.id, corrections)
    assert len(await list_boundary_corrections(session, es.id)) == 2

    await clear_boundary_corrections(session, es.id)
    assert len(await list_boundary_corrections(session, es.id)) == 0


async def test_list_boundary_empty_for_no_corrections(session):
    rd, es = _make_complete_segmentation_job()
    session.add(rd)
    await session.flush()
    es.region_detection_job_id = rd.id
    session.add(es)
    await session.commit()

    rows = await list_boundary_corrections(session, es.id)
    assert rows == []


async def test_upsert_type_creates_and_overwrites(session):
    rd, es, ec = _make_complete_classification_job()
    session.add(rd)
    await session.flush()
    es.region_detection_job_id = rd.id
    session.add(es)
    await session.flush()
    ec.event_segmentation_job_id = es.id
    session.add(ec)
    await session.commit()

    corrections1 = [TypeCorrection(event_id="e1", type_name="upcall")]
    count = await upsert_type_corrections(session, ec.id, corrections1)
    assert count == 1

    rows = await list_type_corrections(session, ec.id)
    assert len(rows) == 1
    assert rows[0].type_name == "upcall"

    corrections2 = [TypeCorrection(event_id="e1", type_name="moan")]
    await upsert_type_corrections(session, ec.id, corrections2)

    rows = await list_type_corrections(session, ec.id)
    assert len(rows) == 1
    assert rows[0].type_name == "moan"


async def test_upsert_type_accepts_null_type_name(session):
    rd, es, ec = _make_complete_classification_job()
    session.add(rd)
    await session.flush()
    es.region_detection_job_id = rd.id
    session.add(es)
    await session.flush()
    ec.event_segmentation_job_id = es.id
    session.add(ec)
    await session.commit()

    corrections = [TypeCorrection(event_id="e1", type_name=None)]
    await upsert_type_corrections(session, ec.id, corrections)

    rows = await list_type_corrections(session, ec.id)
    assert len(rows) == 1
    assert rows[0].type_name is None


async def test_upsert_type_rejects_nonexistent_job(session):
    corrections = [TypeCorrection(event_id="e1", type_name="upcall")]
    with pytest.raises(CallParsingFKError):
        await upsert_type_corrections(session, "nonexistent", corrections)


async def test_upsert_type_rejects_noncomplete_job(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.flush()
    ec = EventClassificationJob(event_segmentation_job_id=es.id, status="running")
    session.add(ec)
    await session.commit()

    corrections = [TypeCorrection(event_id="e1", type_name="upcall")]
    with pytest.raises(CallParsingStateError):
        await upsert_type_corrections(session, ec.id, corrections)


async def test_clear_type_removes_all(session):
    rd, es, ec = _make_complete_classification_job()
    session.add(rd)
    await session.flush()
    es.region_detection_job_id = rd.id
    session.add(es)
    await session.flush()
    ec.event_segmentation_job_id = es.id
    session.add(ec)
    await session.commit()

    corrections = [
        TypeCorrection(event_id="e1", type_name="upcall"),
        TypeCorrection(event_id="e2", type_name="moan"),
    ]
    await upsert_type_corrections(session, ec.id, corrections)
    assert len(await list_type_corrections(session, ec.id)) == 2

    await clear_type_corrections(session, ec.id)
    assert len(await list_type_corrections(session, ec.id)) == 0


# ---- Task 4: Service layer — training jobs + model management --------------


async def test_create_segmentation_feedback_training_job_happy(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.commit()

    req = CreateSegmentationFeedbackTrainingJobRequest(source_job_ids=[es.id])
    job = await create_segmentation_feedback_training_job(session, req)
    await session.commit()

    assert job.status == "queued"
    assert json.loads(job.source_job_ids) == [es.id]


async def test_create_segmentation_feedback_training_404_missing(session):
    req = CreateSegmentationFeedbackTrainingJobRequest(source_job_ids=["nonexistent"])
    with pytest.raises(CallParsingFKError):
        await create_segmentation_feedback_training_job(session, req)


async def test_create_segmentation_feedback_training_409_not_complete(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="running")
    session.add(es)
    await session.commit()

    req = CreateSegmentationFeedbackTrainingJobRequest(source_job_ids=[es.id])
    with pytest.raises(CallParsingStateError):
        await create_segmentation_feedback_training_job(session, req)


async def test_delete_segmentation_feedback_training_job(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.commit()

    req = CreateSegmentationFeedbackTrainingJobRequest(source_job_ids=[es.id])
    job = await create_segmentation_feedback_training_job(session, req)
    await session.commit()

    assert await delete_segmentation_feedback_training_job(session, job.id) is True
    assert await delete_segmentation_feedback_training_job(session, job.id) is False


async def test_create_classifier_training_job_happy(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.flush()
    ec = EventClassificationJob(event_segmentation_job_id=es.id, status="complete")
    session.add(ec)
    await session.commit()

    req = CreateClassifierTrainingJobRequest(source_job_ids=[ec.id])
    job = await create_classifier_training_job(session, req)
    await session.commit()

    assert job.status == "queued"
    assert json.loads(job.source_job_ids) == [ec.id]


async def test_create_classifier_training_job_404_missing(session):
    req = CreateClassifierTrainingJobRequest(source_job_ids=["nonexistent"])
    with pytest.raises(CallParsingFKError):
        await create_classifier_training_job(session, req)


async def test_create_classifier_training_job_409_not_complete(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.flush()
    ec = EventClassificationJob(event_segmentation_job_id=es.id, status="queued")
    session.add(ec)
    await session.commit()

    req = CreateClassifierTrainingJobRequest(source_job_ids=[ec.id])
    with pytest.raises(CallParsingStateError):
        await create_classifier_training_job(session, req)


async def test_delete_classifier_training_job(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.flush()
    ec = EventClassificationJob(event_segmentation_job_id=es.id, status="complete")
    session.add(ec)
    await session.commit()

    req = CreateClassifierTrainingJobRequest(source_job_ids=[ec.id])
    job = await create_classifier_training_job(session, req)
    await session.commit()

    assert await delete_classifier_training_job(session, job.id) is True
    assert await delete_classifier_training_job(session, job.id) is False


async def test_list_classifier_models_only_pytorch_event_cnn(session):
    session.add(
        VocalizationClassifierModel(
            name="sklearn-model",
            model_dir_path="/tmp/sklearn",
            vocabulary_snapshot="[]",
            per_class_thresholds="{}",
            model_family="sklearn_perch_embedding",
        )
    )
    session.add(
        VocalizationClassifierModel(
            name="event-model",
            model_dir_path="/tmp/event",
            vocabulary_snapshot="[]",
            per_class_thresholds="{}",
            model_family="pytorch_event_cnn",
            input_mode="segmented_event",
        )
    )
    await session.commit()

    models = await list_classifier_models(session)
    assert len(models) == 1
    assert models[0].model_family == "pytorch_event_cnn"


async def test_delete_classifier_model_409_in_flight_classification(session, settings):
    model = VocalizationClassifierModel(
        name="event-model",
        model_dir_path="/tmp/event",
        vocabulary_snapshot="[]",
        per_class_thresholds="{}",
        model_family="pytorch_event_cnn",
        input_mode="segmented_event",
    )
    session.add(model)
    await session.flush()

    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.flush()
    ec = EventClassificationJob(
        event_segmentation_job_id=es.id,
        vocalization_model_id=model.id,
        status="running",
    )
    session.add(ec)
    await session.commit()

    with pytest.raises(CallParsingStateError, match="in-flight classification"):
        await delete_classifier_model(session, model.id, settings)


async def test_delete_classifier_model_rejects_non_pytorch(session, settings):
    model = VocalizationClassifierModel(
        name="sklearn-model",
        model_dir_path="/tmp/sklearn",
        vocabulary_snapshot="[]",
        per_class_thresholds="{}",
        model_family="sklearn_perch_embedding",
    )
    session.add(model)
    await session.commit()

    result = await delete_classifier_model(session, model.id, settings)
    assert result is False
