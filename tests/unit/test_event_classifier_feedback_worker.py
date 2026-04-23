"""Tests for _resolve_event_labels in the event classifier feedback worker."""

from __future__ import annotations

from types import SimpleNamespace

from humpback.call_parsing.event_classifier.trainer import (
    EventClassifierTrainingConfig as TrainerConfig,
)
from humpback.schemas.call_parsing import EventClassifierTrainingConfig as SchemaConfig
from humpback.workers.event_classifier_feedback_worker import _resolve_event_labels


def _typed_row(event_id: str, type_name: str, score: float, above: bool):
    return SimpleNamespace(
        event_id=event_id,
        type_name=type_name,
        score=score,
        above_threshold=above,
    )


def test_corrections_only_excludes_uncorrected_events():
    typed = {
        "e1": [_typed_row("e1", "Moan", 0.9, True)],
        "e2": [_typed_row("e2", "Moan", 0.8, True)],
        "e3": [_typed_row("e3", "Moan", 0.7, True)],
    }
    corrections: dict[str, dict[str, str]] = {"e1": {"Cry": "add"}}
    bounds = {"e1": (1.0, 2.0), "e2": (3.0, 4.0), "e3": (5.0, 6.0)}
    labels = _resolve_event_labels(typed, corrections, bounds, corrections_only=True)
    assert labels["e1"] == "Cry"
    assert labels["e2"] is None
    assert labels["e3"] is None


def test_corrections_only_false_includes_inference_labels():
    typed = {
        "e1": [_typed_row("e1", "Moan", 0.9, True)],
        "e2": [_typed_row("e2", "Moan", 0.8, True)],
        "e3": [_typed_row("e3", "Growl", 0.3, False)],
    }
    corrections: dict[str, dict[str, str]] = {"e1": {"Cry": "add"}}
    bounds = {"e1": (1.0, 2.0), "e2": (3.0, 4.0), "e3": (5.0, 6.0)}
    labels = _resolve_event_labels(typed, corrections, bounds, corrections_only=False)
    assert labels["e1"] == "Cry"
    assert labels["e2"] == "Moan"  # inference fallback
    assert labels["e3"] is None  # below threshold


def test_corrections_only_default_is_true():
    typed = {
        "e1": [_typed_row("e1", "Moan", 0.9, True)],
    }
    corrections: dict[str, dict[str, str]] = {}
    bounds = {"e1": (1.0, 2.0)}
    labels = _resolve_event_labels(typed, corrections, bounds)
    assert labels["e1"] is None


def test_remove_correction_suppresses_type():
    typed = {
        "e1": [_typed_row("e1", "Moan", 0.9, True)],
    }
    corrections: dict[str, dict[str, str]] = {"e1": {"Moan": "remove"}}
    bounds = {"e1": (1.0, 2.0)}
    labels = _resolve_event_labels(typed, corrections, bounds, corrections_only=True)
    assert labels["e1"] is None


def test_pydantic_schema_corrections_only_default():
    config = SchemaConfig()
    assert config.corrections_only is True


def test_trainer_dataclass_corrections_only_default():
    config = TrainerConfig()
    assert config.corrections_only is True
