"""Unit tests for call_parsing types and schemas."""

from humpback.call_parsing.types import (
    EVENT_SCHEMA,
    REGION_SCHEMA,
    TRACE_SCHEMA,
    TYPED_EVENT_SCHEMA,
    Event,
    Region,
    TypedEvent,
    WindowScore,
    new_uuid,
)


def test_new_uuid_produces_unique_strings() -> None:
    ids = {new_uuid() for _ in range(1000)}
    assert len(ids) == 1000
    # UUID4 canonical form: 36 chars with 4 hyphens.
    for id_ in ids:
        assert len(id_) == 36
        assert id_.count("-") == 4


def test_dataclasses_are_frozen() -> None:
    region = Region(
        region_id="r1",
        start_sec=0.0,
        end_sec=5.0,
        padded_start_sec=0.0,
        padded_end_sec=5.5,
        max_score=0.9,
        mean_score=0.75,
        n_windows=5,
    )
    try:
        region.start_sec = 99.0  # type: ignore[misc]
    except Exception:  # dataclasses.FrozenInstanceError
        return
    raise AssertionError("Region should be frozen")


def test_schemas_have_expected_fields() -> None:
    assert TRACE_SCHEMA.names == ["time_sec", "score"]
    assert REGION_SCHEMA.names == [
        "region_id",
        "start_sec",
        "end_sec",
        "padded_start_sec",
        "padded_end_sec",
        "max_score",
        "mean_score",
        "n_windows",
    ]
    assert EVENT_SCHEMA.names == [
        "event_id",
        "region_id",
        "start_sec",
        "end_sec",
        "center_sec",
        "segmentation_confidence",
    ]
    assert TYPED_EVENT_SCHEMA.names == [
        "event_id",
        "start_sec",
        "end_sec",
        "type_name",
        "score",
        "above_threshold",
    ]


def test_schema_fields_are_not_nullable() -> None:
    for schema in (TRACE_SCHEMA, REGION_SCHEMA, EVENT_SCHEMA, TYPED_EVENT_SCHEMA):
        for name in schema.names:
            assert not schema.field(name).nullable, (
                f"{schema} field {name} unexpectedly nullable"
            )


def test_dataclass_instances_construct_as_expected() -> None:
    score = WindowScore(time_sec=1.0, score=0.5)
    assert score.time_sec == 1.0

    event = Event(
        event_id="e1",
        region_id="r1",
        start_sec=1.0,
        end_sec=2.0,
        center_sec=1.5,
        segmentation_confidence=0.8,
    )
    assert event.center_sec == 1.5

    typed = TypedEvent(
        event_id="e1",
        start_sec=1.0,
        end_sec=2.0,
        type_name="whup",
        score=0.91,
        above_threshold=True,
    )
    assert typed.above_threshold is True
