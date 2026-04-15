"""Tests for the Pass 3 event classifier inference module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from humpback.call_parsing.event_classifier.inference import (
    classify_events,
    load_event_classifier,
)
from humpback.call_parsing.event_classifier.model import EventClassifierCNN
from humpback.call_parsing.event_classifier.trainer import (
    EventClassifierTrainingConfig,
    train_event_classifier,
)
from humpback.call_parsing.types import Event
from humpback.schemas.call_parsing import SegmentationFeatureConfig


@dataclass
class FakeTrainSample:
    start_sec: float
    end_sec: float
    type_index: int
    audio_file_id: str = "file-0"


def _make_audio(duration_sec: float = 5.0, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(duration_sec * sr)).astype(np.float32)


def _make_event(event_id: str, start: float, end: float) -> Event:
    return Event(
        event_id=event_id,
        region_id="r1",
        start_sec=start,
        end_sec=end,
        center_sec=(start + end) / 2.0,
        segmentation_confidence=0.9,
    )


def _train_and_save(tmp_path: Path) -> Path:
    """Train a tiny model and return the model directory."""
    audio = _make_audio(5.0)

    def _load(_s: object) -> tuple[np.ndarray, float]:
        return audio, 0.0

    samples = [
        FakeTrainSample(0.0, 1.5, type_index=0, audio_file_id="f1"),
        FakeTrainSample(1.0, 2.5, type_index=0, audio_file_id="f1"),
        FakeTrainSample(0.5, 2.0, type_index=1, audio_file_id="f2"),
        FakeTrainSample(1.5, 3.0, type_index=1, audio_file_id="f2"),
        FakeTrainSample(0.0, 1.0, type_index=0, audio_file_id="f3"),
        FakeTrainSample(2.0, 3.5, type_index=1, audio_file_id="f3"),
    ]
    model_dir = tmp_path / "model"
    train_event_classifier(
        samples=samples,
        vocabulary=["upcall", "downcall"],
        feature_config=SegmentationFeatureConfig(),
        audio_loader=_load,
        config=EventClassifierTrainingConfig(
            epochs=2, batch_size=4, min_examples_per_type=2, val_fraction=0.34
        ),
        model_dir=model_dir,
    )
    return model_dir


class TestLoadEventClassifier:
    def test_loads_model_and_config(self, tmp_path: Path) -> None:
        model_dir = _train_and_save(tmp_path)
        model, vocab, thresholds, feat_config = load_event_classifier(model_dir)

        assert isinstance(model, EventClassifierCNN)
        assert vocab == ["upcall", "downcall"]
        assert set(thresholds.keys()) == {"upcall", "downcall"}
        assert isinstance(feat_config, SegmentationFeatureConfig)

    def test_model_produces_output(self, tmp_path: Path) -> None:
        model_dir = _train_and_save(tmp_path)
        model, vocab, _, _ = load_event_classifier(model_dir)
        model.eval()

        x = torch.randn(1, 1, 64, 30)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, len(vocab))

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        try:
            load_event_classifier(empty_dir)
            assert False, "Should have raised"  # noqa: B011
        except FileNotFoundError:
            pass


class TestClassifyEvents:
    def test_produces_typed_events(self, tmp_path: Path) -> None:
        model_dir = _train_and_save(tmp_path)
        model, vocab, thresholds, feat_config = load_event_classifier(model_dir)

        audio = _make_audio(5.0)

        def _load(_e: object) -> tuple[np.ndarray, float]:
            return audio, 0.0

        events = [
            _make_event("ev1", 0.5, 2.0),
            _make_event("ev2", 2.0, 3.5),
        ]

        typed = classify_events(
            model=model,
            events=events,
            audio_loader=_load,
            feature_config=feat_config,
            vocabulary=vocab,
            thresholds=thresholds,
        )

        assert len(typed) >= 2
        event_ids = {te.event_id for te in typed}
        assert "ev1" in event_ids
        assert "ev2" in event_ids

    def test_propagates_event_times(self, tmp_path: Path) -> None:
        model_dir = _train_and_save(tmp_path)
        model, vocab, thresholds, feat_config = load_event_classifier(model_dir)

        audio = _make_audio(5.0)

        def _load(_e: object) -> tuple[np.ndarray, float]:
            return audio, 0.0

        events = [_make_event("ev1", 1.0, 2.5)]
        typed = classify_events(
            model=model,
            events=events,
            audio_loader=_load,
            feature_config=feat_config,
            vocabulary=vocab,
            thresholds=thresholds,
        )

        for te in typed:
            assert te.start_sec == 1.0
            assert te.end_sec == 2.5

    def test_scores_in_valid_range(self, tmp_path: Path) -> None:
        model_dir = _train_and_save(tmp_path)
        model, vocab, thresholds, feat_config = load_event_classifier(model_dir)

        audio = _make_audio(5.0)

        def _load(_e: object) -> tuple[np.ndarray, float]:
            return audio, 0.0

        events = [_make_event("ev1", 0.5, 1.5)]
        typed = classify_events(
            model=model,
            events=events,
            audio_loader=_load,
            feature_config=feat_config,
            vocabulary=vocab,
            thresholds=thresholds,
        )

        for te in typed:
            assert 0.0 <= te.score <= 1.0

    def test_above_threshold_consistency(self, tmp_path: Path) -> None:
        model_dir = _train_and_save(tmp_path)
        model, vocab, thresholds, feat_config = load_event_classifier(model_dir)

        audio = _make_audio(5.0)

        def _load(_e: object) -> tuple[np.ndarray, float]:
            return audio, 0.0

        events = [_make_event("ev1", 0.5, 2.0)]
        typed = classify_events(
            model=model,
            events=events,
            audio_loader=_load,
            feature_config=feat_config,
            vocabulary=vocab,
            thresholds=thresholds,
        )

        for te in typed:
            thresh = thresholds.get(te.type_name, 0.5)
            if te.above_threshold:
                assert te.score >= thresh

    def test_empty_events(self, tmp_path: Path) -> None:
        model_dir = _train_and_save(tmp_path)
        model, vocab, thresholds, feat_config = load_event_classifier(model_dir)

        typed = classify_events(
            model=model,
            events=[],
            audio_loader=lambda _e: (_make_audio(), 0.0),
            feature_config=feat_config,
            vocabulary=vocab,
            thresholds=thresholds,
        )

        assert typed == []

    def test_fallback_row_when_nothing_above_threshold(self, tmp_path: Path) -> None:
        model_dir = _train_and_save(tmp_path)
        model, vocab, _, feat_config = load_event_classifier(model_dir)

        audio = _make_audio(5.0)

        def _load(_e: object) -> tuple[np.ndarray, float]:
            return audio, 0.0

        very_high_thresholds = {t: 1.0 for t in vocab}

        events = [_make_event("ev1", 0.5, 2.0)]
        typed = classify_events(
            model=model,
            events=events,
            audio_loader=_load,
            feature_config=feat_config,
            vocabulary=vocab,
            thresholds=very_high_thresholds,
        )

        assert len(typed) == 1
        assert typed[0].above_threshold is False
        assert typed[0].type_name in vocab
