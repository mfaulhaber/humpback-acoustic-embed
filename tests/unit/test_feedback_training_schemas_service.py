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
    CreateClassifierTrainingJobRequest,
    EventBoundaryCorrectionItem,
    VocalizationCorrectionItem,
)
from humpback.services.call_parsing import (
    CallParsingCorrectionConflict,
    CallParsingFKError,
    CallParsingStateError,
    clear_event_boundary_corrections,
    clear_vocalization_corrections,
    create_classifier_training_job,
    delete_classifier_model,
    delete_classifier_training_job,
    list_event_boundary_corrections,
    list_classifier_models,
    list_vocalization_corrections,
    upsert_event_boundary_corrections,
    upsert_vocalization_corrections,
)
from humpback.call_parsing.segmentation.extraction import load_effective_events
from humpback.call_parsing.storage import segmentation_job_dir, write_events
from humpback.call_parsing.types import Event


# ---- Task 2: Pydantic schema validation ----------------------------------


def test_boundary_correction_add_requires_corrected():
    with pytest.raises(
        ValidationError, match="corrected_start_sec and corrected_end_sec"
    ):
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="add",
        )


def test_boundary_correction_add_rejects_original():
    with pytest.raises(ValidationError, match="must not set original"):
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="add",
            original_start_sec=1.0,
            original_end_sec=2.0,
            corrected_start_sec=1.0,
            corrected_end_sec=2.0,
        )


def test_boundary_correction_add_succeeds():
    c = EventBoundaryCorrectionItem(
        region_id="r1",
        correction_type="add",
        corrected_start_sec=1.0,
        corrected_end_sec=2.0,
    )
    assert c.corrected_start_sec == 1.0
    assert c.original_start_sec is None


def test_boundary_correction_adjust_requires_both_pairs():
    with pytest.raises(
        ValidationError, match="original_start_sec and original_end_sec"
    ):
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="adjust",
            corrected_start_sec=1.0,
            corrected_end_sec=2.0,
        )
    with pytest.raises(
        ValidationError, match="corrected_start_sec and corrected_end_sec"
    ):
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="adjust",
            original_start_sec=1.0,
            original_end_sec=2.0,
        )


def test_boundary_correction_adjust_succeeds():
    c = EventBoundaryCorrectionItem(
        region_id="r1",
        correction_type="adjust",
        original_start_sec=1.0,
        original_end_sec=2.0,
        corrected_start_sec=1.5,
        corrected_end_sec=3.0,
    )
    assert c.original_start_sec == 1.0
    assert c.corrected_start_sec == 1.5


def test_boundary_correction_delete_requires_original():
    with pytest.raises(
        ValidationError, match="original_start_sec and original_end_sec"
    ):
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="delete",
        )


def test_boundary_correction_delete_rejects_corrected():
    with pytest.raises(ValidationError, match="must not set corrected"):
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="delete",
            original_start_sec=1.0,
            original_end_sec=2.0,
            corrected_start_sec=1.0,
            corrected_end_sec=2.0,
        )


def test_boundary_correction_delete_succeeds():
    c = EventBoundaryCorrectionItem(
        region_id="r1",
        correction_type="delete",
        original_start_sec=1.0,
        original_end_sec=2.0,
    )
    assert c.corrected_start_sec is None


def test_boundary_correction_end_must_be_after_start():
    with pytest.raises(ValidationError, match="strictly after"):
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="add",
            corrected_start_sec=2.0,
            corrected_end_sec=1.0,
        )


def test_boundary_correction_rejects_invalid_type():
    with pytest.raises(ValidationError):
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="invalid",
        )


def test_vocalization_correction_rejects_invalid_correction_type():
    with pytest.raises(ValidationError):
        VocalizationCorrectionItem(
            start_sec=1.0, end_sec=2.0, type_name="Whup", correction_type="invalid"
        )


def test_vocalization_correction_accepts_add():
    c = VocalizationCorrectionItem(
        start_sec=1.0, end_sec=2.0, type_name="Whup", correction_type="add"
    )
    assert c.correction_type == "add"


def test_vocalization_correction_accepts_remove():
    c = VocalizationCorrectionItem(
        start_sec=1.0, end_sec=2.0, type_name="Whup", correction_type="remove"
    )
    assert c.correction_type == "remove"


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
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.commit()

    corrections = [
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="adjust",
            original_start_sec=1.0,
            original_end_sec=2.0,
            corrected_start_sec=1.5,
            corrected_end_sec=2.5,
        )
    ]
    rows = await upsert_event_boundary_corrections(session, rd.id, es.id, corrections)
    assert len(rows) == 1
    assert rows[0].event_segmentation_job_id == es.id
    assert rows[0].original_start_sec == 1.0
    assert rows[0].corrected_start_sec == 1.5


async def test_upsert_boundary_updates_on_repeat(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.commit()

    corrections1 = [
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="adjust",
            original_start_sec=1.0,
            original_end_sec=2.0,
            corrected_start_sec=1.5,
            corrected_end_sec=2.5,
        )
    ]
    await upsert_event_boundary_corrections(session, rd.id, es.id, corrections1)

    corrections2 = [
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="adjust",
            original_start_sec=1.0,
            original_end_sec=2.0,
            corrected_start_sec=1.8,
            corrected_end_sec=3.0,
        )
    ]
    rows = await upsert_event_boundary_corrections(session, rd.id, es.id, corrections2)
    assert len(rows) == 1
    assert rows[0].corrected_start_sec == 1.8


async def test_upsert_boundary_updates_by_source_event_id(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.commit()

    await upsert_event_boundary_corrections(
        session,
        rd.id,
        es.id,
        [
            EventBoundaryCorrectionItem(
                region_id="r1",
                source_event_id="event-1",
                correction_type="adjust",
                original_start_sec=1.0,
                original_end_sec=2.0,
                corrected_start_sec=1.5,
                corrected_end_sec=2.5,
            )
        ],
    )
    rows = await upsert_event_boundary_corrections(
        session,
        rd.id,
        es.id,
        [
            EventBoundaryCorrectionItem(
                region_id="r1",
                source_event_id="event-1",
                correction_type="adjust",
                original_start_sec=1.0,
                original_end_sec=2.0,
                corrected_start_sec=1.7,
                corrected_end_sec=2.7,
            )
        ],
    )

    assert len(rows) == 1
    assert rows[0].source_event_id == "event-1"
    assert rows[0].corrected_start_sec == 1.7


async def test_upsert_boundary_updates_saved_add_by_id(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.commit()

    rows = await upsert_event_boundary_corrections(
        session,
        rd.id,
        es.id,
        [
            EventBoundaryCorrectionItem(
                region_id="r1",
                correction_type="add",
                original_start_sec=None,
                original_end_sec=None,
                corrected_start_sec=1.0,
                corrected_end_sec=2.0,
            )
        ],
    )

    rows = await upsert_event_boundary_corrections(
        session,
        rd.id,
        es.id,
        [
            EventBoundaryCorrectionItem(
                id=rows[0].id,
                region_id="r1",
                correction_type="add",
                original_start_sec=None,
                original_end_sec=None,
                corrected_start_sec=1.2,
                corrected_end_sec=2.2,
            )
        ],
    )

    assert len(rows) == 1
    assert rows[0].corrected_start_sec == 1.2


async def test_load_effective_events_preserves_adjusted_event_id(session, tmp_path):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.commit()

    seg_dir = segmentation_job_dir(tmp_path, es.id)
    seg_dir.mkdir(parents=True, exist_ok=True)
    write_events(
        seg_dir / "events.parquet",
        [
            Event("event-1", "r1", 1.0, 2.0, 1.5, 0.9),
            Event("event-2", "r1", 3.0, 4.0, 3.5, 0.8),
        ],
    )

    await upsert_event_boundary_corrections(
        session,
        rd.id,
        es.id,
        [
            EventBoundaryCorrectionItem(
                region_id="r1",
                source_event_id="event-1",
                correction_type="adjust",
                original_start_sec=1.0,
                original_end_sec=2.0,
                corrected_start_sec=1.1,
                corrected_end_sec=2.1,
            )
        ],
        storage_root=tmp_path,
    )

    events = await load_effective_events(
        session, event_segmentation_job_id=es.id, storage_root=tmp_path
    )
    by_id = {event.event_id: event for event in events}
    assert by_id["event-1"].start_sec == 1.1
    assert by_id["event-1"].end_sec == 2.1
    assert by_id["event-2"].start_sec == 3.0


async def test_upsert_boundary_rejects_overlapping_add(session, tmp_path):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.commit()

    seg_dir = segmentation_job_dir(tmp_path, es.id)
    seg_dir.mkdir(parents=True, exist_ok=True)
    write_events(
        seg_dir / "events.parquet",
        [Event("event-1", "r1", 1.0, 2.0, 1.5, 0.9)],
    )

    with pytest.raises(CallParsingCorrectionConflict) as exc:
        await upsert_event_boundary_corrections(
            session,
            rd.id,
            es.id,
            [
                EventBoundaryCorrectionItem(
                    region_id="r1",
                    correction_type="add",
                    original_start_sec=None,
                    original_end_sec=None,
                    corrected_start_sec=1.5,
                    corrected_end_sec=2.5,
                )
            ],
            storage_root=tmp_path,
        )

    assert exc.value.detail["conflicting_event_id"].startswith("added-")


async def test_upsert_boundary_rejects_same_batch_overlap(session, tmp_path):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.commit()

    seg_dir = segmentation_job_dir(tmp_path, es.id)
    seg_dir.mkdir(parents=True, exist_ok=True)
    write_events(seg_dir / "events.parquet", [])

    with pytest.raises(CallParsingCorrectionConflict):
        await upsert_event_boundary_corrections(
            session,
            rd.id,
            es.id,
            [
                EventBoundaryCorrectionItem(
                    region_id="r1",
                    correction_type="add",
                    original_start_sec=None,
                    original_end_sec=None,
                    corrected_start_sec=1.0,
                    corrected_end_sec=2.0,
                ),
                EventBoundaryCorrectionItem(
                    region_id="r1",
                    correction_type="add",
                    original_start_sec=None,
                    original_end_sec=None,
                    corrected_start_sec=1.5,
                    corrected_end_sec=2.5,
                ),
            ],
            storage_root=tmp_path,
        )


async def test_upsert_boundary_rejects_nonexistent_job(session):
    corrections = [
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="delete",
            original_start_sec=1.0,
            original_end_sec=2.0,
        )
    ]
    with pytest.raises(CallParsingFKError):
        await upsert_event_boundary_corrections(
            session, "nonexistent", "seg-missing", corrections
        )


async def test_upsert_boundary_rejects_noncomplete_job(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="queued")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.commit()

    corrections = [
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="delete",
            original_start_sec=1.0,
            original_end_sec=2.0,
        )
    ]
    with pytest.raises(CallParsingStateError):
        await upsert_event_boundary_corrections(session, rd.id, es.id, corrections)


async def test_clear_boundary_removes_all(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.flush()
    es = EventSegmentationJob(region_detection_job_id=rd.id, status="complete")
    session.add(es)
    await session.commit()

    corrections = [
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="adjust",
            original_start_sec=1.0,
            original_end_sec=2.0,
            corrected_start_sec=1.5,
            corrected_end_sec=2.5,
        ),
        EventBoundaryCorrectionItem(
            region_id="r1",
            correction_type="delete",
            original_start_sec=3.0,
            original_end_sec=4.0,
        ),
    ]
    await upsert_event_boundary_corrections(session, rd.id, es.id, corrections)
    assert (
        len(
            await list_event_boundary_corrections(
                session, region_detection_job_id=rd.id
            )
        )
        == 2
    )

    await clear_event_boundary_corrections(session, rd.id)
    assert (
        len(
            await list_event_boundary_corrections(
                session, region_detection_job_id=rd.id
            )
        )
        == 0
    )


async def test_list_boundary_empty_for_no_corrections(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.commit()

    rows = await list_event_boundary_corrections(session, region_detection_job_id=rd.id)
    assert rows == []


async def test_upsert_vocalization_creates_new(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.commit()

    corrections = [
        VocalizationCorrectionItem(
            start_sec=94.0, end_sec=99.0, type_name="Whup", correction_type="add"
        )
    ]
    rows = await upsert_vocalization_corrections(session, rd.id, corrections)
    assert len(rows) == 1
    assert rows[0].type_name == "Whup"
    assert rows[0].correction_type == "add"


async def test_upsert_vocalization_updates_on_same_key(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.commit()

    corrections1 = [
        VocalizationCorrectionItem(
            start_sec=94.0, end_sec=99.0, type_name="Whup", correction_type="add"
        )
    ]
    await upsert_vocalization_corrections(session, rd.id, corrections1)

    corrections2 = [
        VocalizationCorrectionItem(
            start_sec=94.0, end_sec=99.0, type_name="Whup", correction_type="remove"
        )
    ]
    rows = await upsert_vocalization_corrections(session, rd.id, corrections2)
    assert len(rows) == 1
    assert rows[0].correction_type == "remove"


async def test_list_vocalization_filtered_by_detection_job(session):
    rd1 = RegionDetectionJob(audio_file_id="af-1", status="complete")
    rd2 = RegionDetectionJob(audio_file_id="af-2", status="complete")
    session.add_all([rd1, rd2])
    await session.commit()

    await upsert_vocalization_corrections(
        session,
        rd1.id,
        [
            VocalizationCorrectionItem(
                start_sec=1.0, end_sec=2.0, type_name="Whup", correction_type="add"
            )
        ],
    )
    await upsert_vocalization_corrections(
        session,
        rd2.id,
        [
            VocalizationCorrectionItem(
                start_sec=3.0, end_sec=4.0, type_name="Moan", correction_type="add"
            )
        ],
    )

    rows = await list_vocalization_corrections(session, rd1.id)
    assert len(rows) == 1
    assert rows[0].type_name == "Whup"


async def test_clear_vocalization_removes_all(session):
    rd = RegionDetectionJob(audio_file_id="af-1", status="complete")
    session.add(rd)
    await session.commit()

    corrections = [
        VocalizationCorrectionItem(
            start_sec=1.0, end_sec=2.0, type_name="Whup", correction_type="add"
        ),
        VocalizationCorrectionItem(
            start_sec=3.0, end_sec=4.0, type_name="Moan", correction_type="remove"
        ),
    ]
    await upsert_vocalization_corrections(session, rd.id, corrections)
    assert len(await list_vocalization_corrections(session, rd.id)) == 2

    await clear_vocalization_corrections(session, rd.id)
    assert len(await list_vocalization_corrections(session, rd.id)) == 0


async def test_upsert_vocalization_rejects_nonexistent_job(session):
    corrections = [
        VocalizationCorrectionItem(
            start_sec=1.0, end_sec=2.0, type_name="Whup", correction_type="add"
        )
    ]
    with pytest.raises(CallParsingFKError):
        await upsert_vocalization_corrections(session, "nonexistent", corrections)


# ---- Task 4: Service layer — training jobs + model management --------------


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
