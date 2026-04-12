"""Tests for EventCropDataset and collate_fn (Pass 3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from humpback.call_parsing.event_classifier.dataset import (
    EventCropDataset,
    collate_fn,
)
from humpback.schemas.call_parsing import SegmentationFeatureConfig


@dataclass
class FakeSample:
    start_sec: float
    end_sec: float
    type_index: int
    audio_file_id: str = "test-audio"


def _make_config() -> SegmentationFeatureConfig:
    return SegmentationFeatureConfig()


def _make_audio(duration_sec: float, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(duration_sec * sr)).astype(np.float32)


def _const_loader(audio: np.ndarray) -> AudioLoader:
    def _load(_sample: object) -> np.ndarray:
        return audio

    return _load


AudioLoader = Callable[[object], np.ndarray]


class TestEventCropDataset:
    def test_returns_correct_shapes(self) -> None:
        config = _make_config()
        audio = _make_audio(5.0)
        samples = [FakeSample(start_sec=1.0, end_sec=3.0, type_index=0)]

        ds = EventCropDataset(samples, config, _const_loader(audio), n_types=3)
        feat, label = ds[0]

        assert feat.ndim == 3
        assert feat.shape[0] == 1
        assert feat.shape[1] == config.n_mels
        assert feat.shape[2] > 0
        assert label.shape == (3,)
        assert label[0] == 1.0
        assert label[1] == 0.0
        assert label[2] == 0.0

    def test_type_index_sets_correct_label(self) -> None:
        config = _make_config()
        audio = _make_audio(3.0)
        samples = [FakeSample(start_sec=0.0, end_sec=2.0, type_index=2)]

        ds = EventCropDataset(samples, config, _const_loader(audio), n_types=4)
        _, label = ds[0]

        assert label[2] == 1.0
        assert label.sum() == 1.0

    def test_short_event_produces_features(self) -> None:
        config = _make_config()
        audio = _make_audio(2.0)
        samples = [FakeSample(start_sec=0.5, end_sec=0.8, type_index=0)]

        ds = EventCropDataset(samples, config, _const_loader(audio), n_types=2)
        feat, _ = ds[0]

        assert feat.shape[2] >= 1

    def test_event_at_audio_start(self) -> None:
        config = _make_config()
        audio = _make_audio(3.0)
        samples = [FakeSample(start_sec=0.0, end_sec=1.0, type_index=0)]

        ds = EventCropDataset(samples, config, _const_loader(audio), n_types=2)
        feat, _ = ds[0]

        assert feat.shape[2] > 0

    def test_event_at_audio_end(self) -> None:
        config = _make_config()
        duration = 3.0
        audio = _make_audio(duration)
        samples = [FakeSample(start_sec=2.5, end_sec=duration, type_index=1)]

        ds = EventCropDataset(samples, config, _const_loader(audio), n_types=2)
        feat, label = ds[0]

        assert feat.shape[2] > 0
        assert label[1] == 1.0

    def test_event_beyond_audio_clipped(self) -> None:
        config = _make_config()
        audio = _make_audio(2.0)
        samples = [FakeSample(start_sec=1.5, end_sec=3.0, type_index=0)]

        ds = EventCropDataset(samples, config, _const_loader(audio), n_types=2)
        feat, _ = ds[0]

        assert feat.shape[2] > 0

    def test_len(self) -> None:
        config = _make_config()
        audio = _make_audio(3.0)
        samples = [
            FakeSample(start_sec=0.0, end_sec=1.0, type_index=0),
            FakeSample(start_sec=1.0, end_sec=2.0, type_index=1),
        ]

        ds = EventCropDataset(samples, config, _const_loader(audio), n_types=2)
        assert len(ds) == 2


class TestCollate:
    def test_pads_to_max_time(self) -> None:
        feat_short = torch.randn(1, 64, 10)
        feat_long = torch.randn(1, 64, 30)
        label_a = torch.tensor([1.0, 0.0])
        label_b = torch.tensor([0.0, 1.0])

        batch = [(feat_short, label_a), (feat_long, label_b)]
        features, labels = collate_fn(batch)

        assert features.shape == (2, 1, 64, 30)
        assert labels.shape == (2, 2)

    def test_preserves_content(self) -> None:
        feat = torch.randn(1, 64, 15)
        label = torch.tensor([1.0, 0.0, 0.0])

        features, labels = collate_fn([(feat, label)])

        assert torch.equal(features[0, :, :, :15], feat)
        assert torch.equal(labels[0], label)

    def test_padding_is_zero(self) -> None:
        feat = torch.randn(1, 64, 10)
        label = torch.tensor([1.0])

        batch = [(feat, label), (torch.randn(1, 64, 20), torch.tensor([0.0]))]
        features, _ = collate_fn(batch)

        assert (features[0, :, :, 10:] == 0).all()

    def test_empty_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            collate_fn([])

    def test_same_length_no_padding(self) -> None:
        feat_a = torch.randn(1, 64, 25)
        feat_b = torch.randn(1, 64, 25)
        label = torch.tensor([0.0, 1.0])

        features, labels = collate_fn([(feat_a, label), (feat_b, label)])

        assert features.shape == (2, 1, 64, 25)
        assert labels.shape == (2, 2)
