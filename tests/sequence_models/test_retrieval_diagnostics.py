"""Tests for masked-transformer retrieval diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from humpback.call_parsing.storage import (
    classification_job_dir,
    segmentation_job_dir,
    write_events,
    write_typed_events,
)
from humpback.call_parsing.types import Event, TypedEvent
from humpback.models.call_parsing import (
    EventBoundaryCorrection,
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
    VocalizationCorrection,
)
from humpback.models.processing import JobStatus
from humpback.sequence_models import retrieval_diagnostics as diag

REGION_START_UTC = 1_000.0


def _row(
    idx: int,
    *,
    region_id: str,
    event_id: str,
    label: str,
    duration: float = 1.0,
) -> dict:
    return {
        "idx": idx,
        "region_id": region_id,
        "chunk_index": idx,
        "start_timestamp": float(idx),
        "end_timestamp": float(idx) + 0.25,
        "center_timestamp": float(idx) + 0.125,
        "tier": "event_core",
        "hydrophone_id": "hydrophone",
        "token": idx,
        "token_confidence": 1.0,
        "call_probability": 0.9,
        "event_overlap_fraction": 1.0,
        "nearest_event_id": event_id,
        "event_id": event_id,
        "event_duration": duration,
        "human_types": (label,),
    }


def _event(event_id: str, region_id: str, start: float, end: float) -> Event:
    return Event(
        event_id=event_id,
        region_id=region_id,
        start_sec=start,
        end_sec=end,
        center_sec=(start + end) / 2.0,
        segmentation_confidence=0.9,
    )


def _typed(event_id: str, type_name: str, score: float, above: bool) -> TypedEvent:
    return TypedEvent(
        event_id=event_id,
        start_sec=0.0,
        end_sec=0.0,
        type_name=type_name,
        score=score,
        above_threshold=above,
    )


async def _seed_chain(session, storage_root: Path) -> tuple[str, str, str]:
    rdj = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=REGION_START_UTC,
        end_timestamp=REGION_START_UTC + 300.0,
    )
    session.add(rdj)
    await session.flush()

    seg = EventSegmentationJob(
        status=JobStatus.complete.value,
        region_detection_job_id=rdj.id,
    )
    session.add(seg)
    await session.flush()

    cls = EventClassificationJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id=seg.id,
    )
    session.add(cls)
    await session.commit()
    await session.refresh(rdj)
    await session.refresh(seg)
    await session.refresh(cls)

    segmentation_job_dir(storage_root, seg.id).mkdir(parents=True, exist_ok=True)
    classification_job_dir(storage_root, cls.id).mkdir(parents=True, exist_ok=True)
    return rdj.id, seg.id, cls.id


def test_exclude_same_event_and_region_masks_local_neighbors():
    rows = [
        _row(0, region_id="same-region", event_id="event-a", label="Moan"),
        _row(1, region_id="same-region", event_id="event-b", label="Moan"),
        _row(2, region_id="other-region", event_id="event-c", label="Moan"),
        _row(3, region_id="third-region", event_id="event-d", label="Growl"),
    ]
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    _summaries, neighbors = diag.analyze_neighbors(
        rows,
        vectors,
        topn=2,
        sample_indices=[0],
        mode="exclude_same_event_and_region",
    )

    assert [record["neighbor_idx"] for record in neighbors] == [2, 3]
    assert all(not record["same_region"] for record in neighbors)
    assert all(not record["same_event"] for record in neighbors)


def test_exclude_same_event_allows_other_regions_in_different_events():
    rows = [
        _row(0, region_id="r1", event_id="event-a", label="Moan"),
        _row(1, region_id="r2", event_id="event-a", label="Moan"),
        _row(2, region_id="r2", event_id="event-b", label="Moan"),
    ]
    vectors = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]], dtype=np.float32)

    _summaries, neighbors = diag.analyze_neighbors(
        rows,
        vectors,
        topn=1,
        sample_indices=[0],
        mode="exclude_same_event",
    )

    assert neighbors[0]["neighbor_idx"] == 2
    assert not neighbors[0]["same_event"]


def test_pc_removal_and_whitening_variants_are_normalized():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(24, 8)).astype(np.float32)

    variants = diag.build_embedding_variants(
        raw,
        seed=0,
        variants=("raw_l2", "centered_l2", "remove_pc1", "remove_pc3", "whiten_pca"),
    )

    assert set(variants) == {
        "raw_l2",
        "centered_l2",
        "remove_pc1",
        "remove_pc3",
        "whiten_pca",
    }
    for values in variants.values():
        assert values.shape[0] == raw.shape[0]
        np.testing.assert_allclose(
            np.linalg.norm(values, axis=1),
            np.ones(raw.shape[0]),
            atol=1e-5,
        )


def test_variant_matrix_reuses_sample_indices_across_modes_and_variants():
    rows = [
        _row(0, region_id="r0", event_id="e0", label="Moan"),
        _row(1, region_id="r1", event_id="e1", label="Moan"),
        _row(2, region_id="r2", event_id="e2", label="Growl"),
        _row(3, region_id="r3", event_id="e3", label="Growl"),
    ]
    vectors = np.eye(4, dtype=np.float32)
    options = diag.RetrievalReportOptions(
        samples=2,
        topn=1,
        seed=12,
        retrieval_modes=("unrestricted", "exclude_same_event"),
        embedding_variants=("raw_l2", "centered_l2"),
    )

    results, _queries, _neighbors, sample_indices, human_pool = diag.run_variant_matrix(
        rows, vectors, options=options
    )

    assert human_pool == 4
    assert len(sample_indices) == 2
    assert set(results) == {"unrestricted", "exclude_same_event"}
    assert set(results["unrestricted"]) == {"raw_l2", "centered_l2"}


def test_label_specific_metric_aggregates_by_query_label():
    rows = [
        _row(0, region_id="r0", event_id="e0", label="Moan"),
        _row(1, region_id="r1", event_id="e1", label="Moan"),
        _row(2, region_id="r2", event_id="e2", label="Growl"),
    ]
    vectors = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    results, *_ = diag.run_variant_matrix(
        rows,
        vectors,
        options=diag.RetrievalReportOptions(
            samples=3,
            topn=1,
            seed=0,
            retrieval_modes=("unrestricted",),
            embedding_variants=("raw_l2",),
        ),
    )

    label_metrics = results["unrestricted"]["raw_l2"]["label_specific_same_human_label"]
    assert "Moan" in label_metrics
    assert label_metrics["Moan"]["same_human_label"] == 1.0


async def test_human_corrections_support_multiple_labels(session, tmp_storage):
    rdj_id, seg_id, _cls_id = await _seed_chain(session, tmp_storage)
    write_events(
        segmentation_job_dir(tmp_storage, seg_id) / "events.parquet",
        [_event("E1", "R1", 10.0, 12.0)],
    )
    session.add_all(
        [
            VocalizationCorrection(
                region_detection_job_id=rdj_id,
                start_sec=10.0,
                end_sec=12.0,
                type_name="moan",
                correction_type="add",
            ),
            VocalizationCorrection(
                region_detection_job_id=rdj_id,
                start_sec=10.0,
                end_sec=12.0,
                type_name="song",
                correction_type="add",
            ),
        ]
    )
    await session.commit()

    events, meta = await diag.load_human_correction_events(
        session,
        storage_root=tmp_storage,
        event_segmentation_job_id=seg_id,
        region_detection_job_id=rdj_id,
        region_start_timestamp=REGION_START_UTC,
    )

    assert events[0].human_types == ("moan", "song")
    assert meta["event_label_counts"] == {"moan": 1, "song": 1}


async def test_remove_correction_subtracts_added_type(session, tmp_storage):
    rdj_id, seg_id, _cls_id = await _seed_chain(session, tmp_storage)
    write_events(
        segmentation_job_dir(tmp_storage, seg_id) / "events.parquet",
        [_event("E1", "R1", 10.0, 12.0)],
    )
    session.add_all(
        [
            VocalizationCorrection(
                region_detection_job_id=rdj_id,
                start_sec=10.0,
                end_sec=12.0,
                type_name="moan",
                correction_type="add",
            ),
            VocalizationCorrection(
                region_detection_job_id=rdj_id,
                start_sec=10.5,
                end_sec=11.5,
                type_name="moan",
                correction_type="remove",
            ),
        ]
    )
    await session.commit()

    events, _meta = await diag.load_human_correction_events(
        session,
        storage_root=tmp_storage,
        event_segmentation_job_id=seg_id,
        region_detection_job_id=rdj_id,
        region_start_timestamp=REGION_START_UTC,
    )

    assert events[0].human_types == ()


async def test_model_classify_labels_are_ignored(session, tmp_storage):
    rdj_id, seg_id, cls_id = await _seed_chain(session, tmp_storage)
    write_events(
        segmentation_job_dir(tmp_storage, seg_id) / "events.parquet",
        [_event("E1", "R1", 10.0, 12.0)],
    )
    write_typed_events(
        classification_job_dir(tmp_storage, cls_id) / "typed_events.parquet",
        [_typed("E1", "moan", 0.99, above=True)],
    )

    events, meta = await diag.load_human_correction_events(
        session,
        storage_root=tmp_storage,
        event_segmentation_job_id=seg_id,
        region_detection_job_id=rdj_id,
        region_start_timestamp=REGION_START_UTC,
    )

    assert events[0].human_types == ()
    assert meta["events_with_human_labels"] == 0


async def test_boundary_adjusted_event_controls_row_membership(session, tmp_storage):
    rdj_id, seg_id, _cls_id = await _seed_chain(session, tmp_storage)
    write_events(
        segmentation_job_dir(tmp_storage, seg_id) / "events.parquet",
        [_event("E1", "R1", 10.0, 12.0)],
    )
    session.add_all(
        [
            EventBoundaryCorrection(
                region_detection_job_id=rdj_id,
                event_segmentation_job_id=seg_id,
                region_id="R1",
                source_event_id="E1",
                correction_type="adjust",
                original_start_sec=10.0,
                original_end_sec=12.0,
                corrected_start_sec=15.0,
                corrected_end_sec=17.0,
            ),
            VocalizationCorrection(
                region_detection_job_id=rdj_id,
                start_sec=15.0,
                end_sec=17.0,
                type_name="moan",
                correction_type="add",
            ),
        ]
    )
    await session.commit()

    events, _meta = await diag.load_human_correction_events(
        session,
        storage_root=tmp_storage,
        event_segmentation_job_id=seg_id,
        region_detection_job_id=rdj_id,
        region_start_timestamp=REGION_START_UTC,
    )
    rows = [
        {
            "start_timestamp": REGION_START_UTC + 10.25,
            "end_timestamp": REGION_START_UTC + 10.5,
        },
        {
            "start_timestamp": REGION_START_UTC + 15.25,
            "end_timestamp": REGION_START_UTC + 15.5,
        },
    ]

    assigned = diag._assign_events_to_rows(rows, events)

    assert events[0].start_utc == REGION_START_UTC + 15.0
    assert assigned[0] is None
    assert assigned[1] is not None
    assert assigned[1].human_types == ("moan",)
