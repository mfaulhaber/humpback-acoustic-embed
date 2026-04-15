"""Tests for the Pass 3 event classifier training driver."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from humpback.call_parsing.event_classifier.trainer import (
    EventClassifierTrainingConfig,
    _filter_by_min_examples,
    _optimize_threshold_for_type,
    compute_per_type_pos_weight,
    train_event_classifier,
)
from humpback.schemas.call_parsing import SegmentationFeatureConfig


@dataclass
class FakeSample:
    start_sec: float
    end_sec: float
    type_index: int
    audio_file_id: str = "file-0"


def _make_audio(duration_sec: float, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(duration_sec * sr)).astype(np.float32)


def _const_loader(audio: np.ndarray):  # noqa: ANN202
    def _load(_sample: object) -> tuple[np.ndarray, float]:
        return audio, 0.0

    return _load


# ---- compute_per_type_pos_weight -------------------------------------------


class TestPosWeight:
    def test_balanced_two_types(self) -> None:
        samples = [
            FakeSample(0, 1, type_index=0),
            FakeSample(0, 1, type_index=1),
        ]
        pw = compute_per_type_pos_weight(samples, n_types=2)
        assert pw.shape == (2,)
        assert pw[0] == pytest.approx(1.0)
        assert pw[1] == pytest.approx(1.0)

    def test_imbalanced(self) -> None:
        samples = [
            FakeSample(0, 1, type_index=0),
            FakeSample(0, 1, type_index=0),
            FakeSample(0, 1, type_index=0),
            FakeSample(0, 1, type_index=1),
        ]
        pw = compute_per_type_pos_weight(samples, n_types=2)
        assert pw[0] == pytest.approx(1 / 3)
        assert pw[1] == pytest.approx(3.0)

    def test_unseen_type_gets_weight_one(self) -> None:
        samples = [FakeSample(0, 1, type_index=0)]
        pw = compute_per_type_pos_weight(samples, n_types=3)
        assert pw[0] == pytest.approx(0.0)
        assert pw[1] == pytest.approx(1.0)
        assert pw[2] == pytest.approx(1.0)


# ---- _filter_by_min_examples -----------------------------------------------


class TestFilterByMinExamples:
    def test_drops_rare_type(self) -> None:
        vocab = ["upcall", "downcall", "moan"]
        samples = [
            FakeSample(0, 1, type_index=0),
            FakeSample(0, 1, type_index=0),
            FakeSample(0, 1, type_index=1),
            FakeSample(0, 1, type_index=1),
            FakeSample(0, 1, type_index=2),  # only 1 example
        ]
        filtered, new_vocab, _ = _filter_by_min_examples(samples, vocab, min_examples=2)
        assert new_vocab == ["upcall", "downcall"]
        assert len(filtered) == 4
        assert all(s.type_index in (0, 1) for s in filtered)

    def test_remaps_indices(self) -> None:
        vocab = ["a", "b", "c"]
        samples = [
            FakeSample(0, 1, type_index=2),
            FakeSample(0, 1, type_index=2),
        ]
        filtered, new_vocab, _ = _filter_by_min_examples(samples, vocab, min_examples=2)
        assert new_vocab == ["c"]
        assert filtered[0].type_index == 0

    def test_all_below_min_raises(self) -> None:
        vocab = ["a", "b"]
        samples = [FakeSample(0, 1, type_index=0)]
        with pytest.raises(ValueError, match="No types have"):
            _filter_by_min_examples(samples, vocab, min_examples=5)


# ---- _optimize_threshold_for_type ------------------------------------------


class TestThresholdOptimization:
    def test_perfect_separation(self) -> None:
        scores = np.array([0.9, 0.8, 0.1, 0.05])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        thresh, f1 = _optimize_threshold_for_type(scores, labels)
        assert 0.1 <= thresh <= 0.8
        assert f1 == pytest.approx(1.0)

    def test_returns_value_in_range(self) -> None:
        rng = np.random.default_rng(99)
        scores = rng.random(20).astype(np.float32)
        labels = (rng.random(20) > 0.5).astype(np.float32)
        thresh, f1 = _optimize_threshold_for_type(scores, labels)
        assert 0.0 < thresh < 1.0
        assert 0.0 <= f1 <= 1.0


# ---- train_event_classifier (end-to-end) -----------------------------------


class TestTrainEventClassifier:
    def test_tiny_training_run(self, tmp_path: Path) -> None:
        audio = _make_audio(5.0)
        samples = [
            FakeSample(0.0, 1.5, type_index=0, audio_file_id="f1"),
            FakeSample(1.0, 2.5, type_index=0, audio_file_id="f1"),
            FakeSample(0.5, 2.0, type_index=1, audio_file_id="f2"),
            FakeSample(1.5, 3.0, type_index=1, audio_file_id="f2"),
            FakeSample(0.0, 1.0, type_index=0, audio_file_id="f3"),
            FakeSample(2.0, 3.5, type_index=1, audio_file_id="f3"),
        ]
        vocab = ["upcall", "downcall"]
        config = EventClassifierTrainingConfig(
            epochs=2,
            batch_size=4,
            min_examples_per_type=2,
            val_fraction=0.34,
        )
        model_dir = tmp_path / "model"

        result = train_event_classifier(
            samples=samples,
            vocabulary=vocab,
            feature_config=SegmentationFeatureConfig(),
            audio_loader=_const_loader(audio),
            config=config,
            model_dir=model_dir,
        )

        assert len(result.train_losses) > 0
        assert result.n_train_samples > 0
        assert result.vocabulary == ["upcall", "downcall"]

        assert (model_dir / "model.pt").exists()
        assert (model_dir / "config.json").exists()
        assert (model_dir / "thresholds.json").exists()
        assert (model_dir / "metrics.json").exists()

        cfg = json.loads((model_dir / "config.json").read_text())
        assert cfg["model_type"] == "EventClassifierCNN"
        assert cfg["vocabulary"] == ["upcall", "downcall"]

        thresholds = json.loads((model_dir / "thresholds.json").read_text())
        assert set(thresholds.keys()) == {"upcall", "downcall"}
        assert all(0.0 < v < 1.0 for v in thresholds.values())

    def test_types_below_min_excluded(self, tmp_path: Path) -> None:
        audio = _make_audio(3.0)
        samples = [
            FakeSample(0.0, 1.0, type_index=0, audio_file_id="f1"),
            FakeSample(0.5, 1.5, type_index=0, audio_file_id="f1"),
            FakeSample(1.0, 2.0, type_index=0, audio_file_id="f2"),
            FakeSample(0.0, 1.0, type_index=1, audio_file_id="f2"),
        ]
        vocab = ["upcall", "rare_type"]
        config = EventClassifierTrainingConfig(
            epochs=2,
            batch_size=4,
            min_examples_per_type=2,
        )

        result = train_event_classifier(
            samples=samples,
            vocabulary=vocab,
            feature_config=SegmentationFeatureConfig(),
            audio_loader=_const_loader(audio),
            config=config,
            model_dir=tmp_path / "model",
        )

        assert result.vocabulary == ["upcall"]
        assert "rare_type" not in result.per_type_thresholds

    def test_result_summary_serializable(self, tmp_path: Path) -> None:
        audio = _make_audio(3.0)
        samples = [
            FakeSample(0.0, 1.0, type_index=0, audio_file_id=f"f{i}") for i in range(4)
        ]
        vocab = ["upcall"]
        config = EventClassifierTrainingConfig(
            epochs=1, batch_size=4, min_examples_per_type=1
        )

        result = train_event_classifier(
            samples=samples,
            vocabulary=vocab,
            feature_config=SegmentationFeatureConfig(),
            audio_loader=_const_loader(audio),
            config=config,
            model_dir=tmp_path / "model",
        )

        summary = result.to_summary()
        serialized = json.dumps(summary)
        assert isinstance(serialized, str)
