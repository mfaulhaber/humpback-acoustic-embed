"""Tests for human-correction contrastive label loading."""

from __future__ import annotations

from dataclasses import dataclass

from humpback.models.call_parsing import VocalizationCorrection
from humpback.sequence_models.contrastive_labels import apply_human_correction_labels


@dataclass(frozen=True)
class _Event:
    event_id: str
    region_id: str
    start_sec: float
    end_sec: float


def _correction(
    *,
    start: float,
    end: float,
    type_name: str,
    correction_type: str = "add",
) -> VocalizationCorrection:
    return VocalizationCorrection(
        region_detection_job_id="rd-1",
        start_sec=start,
        end_sec=end,
        type_name=type_name,
        correction_type=correction_type,
    )


def test_apply_human_correction_labels_supports_multilabel_events() -> None:
    events, meta = apply_human_correction_labels(
        effective_events=[_Event("event-1", "region-a", 10.0, 12.0)],
        corrections=[
            _correction(start=9.5, end=10.5, type_name="Moan"),
            _correction(start=11.0, end=13.0, type_name="Whup"),
        ],
        region_start_timestamp=1000.0,
    )

    assert events[0].event_id == "event-1"
    assert events[0].start_timestamp == 1010.0
    assert events[0].end_timestamp == 1012.0
    assert events[0].human_types == ("Moan", "Whup")
    assert meta["events_with_human_labels"] == 1


def test_apply_human_correction_labels_applies_add_remove_overlay() -> None:
    events, _ = apply_human_correction_labels(
        effective_events=[
            _Event("event-1", "region-a", 10.0, 12.0),
            _Event("event-2", "region-a", 20.0, 22.0),
        ],
        corrections=[
            _correction(start=10.0, end=12.0, type_name="Moan"),
            _correction(
                start=10.0,
                end=12.0,
                type_name="Moan",
                correction_type="remove",
            ),
            _correction(
                start=20.0,
                end=22.0,
                type_name="Whup",
                correction_type="remove",
            ),
        ],
        region_start_timestamp=0.0,
    )

    assert events[0].human_types == ()
    assert events[1].human_types == ()


def test_apply_human_correction_labels_ignores_non_overlapping_rows() -> None:
    events, meta = apply_human_correction_labels(
        effective_events=[_Event("event-1", "region-a", 10.0, 12.0)],
        corrections=[_correction(start=12.0, end=13.0, type_name="Moan")],
        region_start_timestamp=None,
    )

    assert events[0].human_types == ()
    assert meta["total_correction_rows"] == 1
    assert meta["events_with_human_labels"] == 0
